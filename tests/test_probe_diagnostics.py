import pytest
import torch
from torch import nn

from experiments.synbios_moe.probe_diagnostics import (
    GroundTruthFirstWholeDataset,
    build_ground_truth_first_input,
    evaluate_ground_truth_first_whole_by_source_position,
    insert_oracle_first_token,
    pairwise_route_summary,
    first_token_text,
    summarize_oracle_rows,
)
from experiments.synbios_moe.probes import AttributeProbe, GPT2Codec, ProbeBatchItem


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = type("Config", (), {"hidden_size": 4})()
        self.embedding = nn.Embedding(128, 4)

    def hidden_states(self, input_ids, embedding_delta=None):
        hidden = self.embedding(input_ids)
        return hidden if embedding_delta is None else hidden + embedding_delta


class _ItemDataset:
    def __init__(self, item):
        self.item = item
        self.class_names = ["a", "b"]
        self.sample_indices = [0]
        self.profile_indices = [0]
        self.positions = [item.positions]
        self.offsets = [0, len(item.input_ids)]

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return self.item


def test_oracle_insertion_preserves_final_eos_readout():
    assert insert_oracle_first_token([99, 4, 5, 99], 17, 99) == [99, 4, 5, 17, 99]


def test_ground_truth_first_p_reads_the_inserted_token_itself():
    item = ProbeBatchItem(
        input_ids=[99, 10, 11, 12, 13, 14, 15],
        positions=[0, 1, 2, 3, 4, 5],
        label=7,
    )
    rebuilt = build_ground_truth_first_input(
        kind="p",
        item=item,
        source_position=2,
        token_id=42,
        eos_id=99,
    )
    assert rebuilt.input_ids == [99, 10, 11, 42]
    assert rebuilt.positions == [3]
    assert rebuilt.label == 7


def test_ground_truth_first_q_reads_the_eos_after_inserted_token():
    item = ProbeBatchItem(input_ids=[99, 10, 11, 99], positions=[3], label=7)
    rebuilt = build_ground_truth_first_input(
        kind="q",
        item=item,
        source_position=0,
        token_id=42,
        eos_id=99,
    )
    assert rebuilt.input_ids == [99, 10, 11, 42, 99]
    assert rebuilt.positions == [4]


def test_ground_truth_dataset_uses_cached_label_as_inserted_token_class():
    first = _ItemDataset(
        ProbeBatchItem(
            input_ids=[99, 10, 11, 12, 13, 14, 15],
            positions=[0, 1, 2, 3, 4, 5],
            label=1,
        )
    )
    whole = _ItemDataset(
        ProbeBatchItem(
            input_ids=[99, 10, 11, 12, 13, 14, 15],
            positions=[0, 1, 2, 3, 4, 5],
            label=7,
        )
    )
    dataset = GroundTruthFirstWholeDataset(
        kind="p",
        first_data=first,
        whole_data=whole,
        token_ids_by_class=(41, 42),
        eos_id=99,
    )
    rebuilt = dataset[2]
    assert rebuilt.input_ids == [99, 10, 11, 42]
    assert rebuilt.positions == [3]
    assert rebuilt.label == 7


def test_first_token_recovers_exact_leading_space_text():
    pytest.importorskip("tiktoken")
    codec = GPT2Codec()
    token_id = codec.encode(" Cambridge")[0]
    recovered_id, text = first_token_text(str(token_id), codec)
    assert recovered_id == token_id
    assert text.startswith(" ")
    assert codec.encode(text) == [token_id]


def test_ground_truth_first_whole_probe_uses_rank_matched_input_delta():
    backbone = _TinyBackbone()
    backbone.cfg.vocab_size = 128
    probe = AttributeProbe(backbone, 3, rank=2, kind="p").train()
    logits = probe(torch.tensor([[1, 2, 3]]), torch.tensor([[2]]))
    logits.sum().backward()
    assert logits.shape == (1, 1, 3)
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert probe.classifier.weight.grad is not None
    assert probe.delta.a.weight.grad is not None
    assert probe.delta.b.weight.grad is not None


def test_ground_truth_validation_returns_overall_and_position_counts_in_one_pass():
    first = _ItemDataset(
        ProbeBatchItem(input_ids=[99, 10, 99], positions=[2], label=1)
    )
    whole = _ItemDataset(
        ProbeBatchItem(input_ids=[99, 10, 99], positions=[2], label=1)
    )
    dataset = GroundTruthFirstWholeDataset(
        kind="q",
        first_data=first,
        whole_data=whole,
        token_ids_by_class=(41, 42),
        eos_id=99,
    )
    backbone = _TinyBackbone()
    backbone.cfg.vocab_size = 128
    probe = AttributeProbe(backbone, 2, rank=2, kind="q")
    with torch.no_grad():
        probe.classifier.weight.zero_()
        probe.classifier.bias.copy_(torch.tensor([0.0, 1.0]))
    result = evaluate_ground_truth_first_whole_by_source_position(
        probe,
        dataset,
        device=torch.device("cpu"),
        batch_size=1,
    )
    assert result["accuracy"] == 1.0
    assert result["accuracy_by_position"] == [1.0]
    assert result["correct_by_position"] == [1]
    assert result["total_by_position"] == [1]


def test_oracle_summary_separates_recovery_and_harm():
    rows = [
        {"whole_before_correct": False, "whole_after_correct": True},
        {"whole_before_correct": False, "whole_after_correct": False},
        {"whole_before_correct": True, "whole_after_correct": False},
        {"whole_before_correct": True, "whole_after_correct": True},
    ]
    summary = summarize_oracle_rows(rows)
    assert summary["accuracy_before"] == 0.5
    assert summary["accuracy_after"] == 0.5
    assert summary["recovery_rate"] == 0.5
    assert summary["harm_rate"] == 0.5


def test_pairwise_route_summary_detects_second_token_branching():
    cases = []
    for case_id, t2, t2_route in (
        (0, 10, [[2, 3], [2, 3]]),
        (1, 10, [[2, 3], [2, 3]]),
        (2, 11, [[4, 5], [4, 5]]),
        (3, 11, [[4, 5], [4, 5]]),
    ):
        cases.append(
            {
                "case_id": case_id,
                "attribute": "university",
                "t1_id": 7,
                "t2_id": t2,
                "routes": [
                    [[0, 1], [0, 1]],
                    t2_route,
                ],
            }
        )
    rows = pairwise_route_summary(cases, layers=2, pair_limit=20, seed=3)
    same = [row for row in rows if row["pair_group"] == "same_t2"]
    different = [row for row in rows if row["pair_group"] == "different_t2"]
    assert all(row["t1_route_overlap"] == 1.0 for row in same + different)
    assert all(row["t2_route_overlap"] == 1.0 for row in same)
    assert all(row["t2_route_overlap"] == 0.0 for row in different)
    assert all(row["branching_score"] == 1.0 for row in different)


def test_pairwise_route_summary_allows_no_eligible_control_pairs():
    cases = [
        {
            "attribute": "company",
            "t1_id": 7,
            "t2_id": 10,
            "routes": [[[0, 1]], [[2, 3]]],
        }
    ]
    assert pairwise_route_summary(cases, layers=1, pair_limit=10) == []
