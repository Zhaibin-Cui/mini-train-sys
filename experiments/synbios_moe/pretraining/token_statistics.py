"""Measure GPT-2 token distributions for every SynBioS attribute."""


import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np

from experiments.synbios_moe.pretraining.dataset import ATTRIBUTES, generate_profiles
from experiments.synbios_moe.probes.model import GPT2Codec


def _entropy(counts: Counter[int]) -> float:
    total = sum(counts.values())
    return float(
        -sum((count / total) * math.log2(count / total) for count in counts.values())
    )


def _percentile(values: list[int], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def analyze(
    *,
    num_people: int,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    codec = GPT2Codec()
    profiles = generate_profiles(num_people, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    length_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []

    for attribute in ATTRIBUTES:
        values = [str(getattr(profile, attribute)) for profile in profiles]
        weighted_sequences = [tuple(codec.encode(" " + value)) for value in values]
        unique_values = sorted(set(values))
        unique_sequences = {
            value: tuple(codec.encode(" " + value)) for value in unique_values
        }
        bare_sequences = {value: tuple(codec.encode(value)) for value in unique_values}
        weighted_lengths = [len(sequence) for sequence in weighted_sequences]
        unique_lengths = [len(sequence) for sequence in unique_sequences.values()]
        weighted_length_counts = Counter(weighted_lengths)
        unique_length_counts = Counter(unique_lengths)
        max_length = max(weighted_lengths)

        position_unique_counts: dict[str, int] = {}
        position_entropies: dict[str, float] = {}
        for position in range(max_length):
            weighted_tokens = Counter(
                sequence[position]
                for sequence in weighted_sequences
                if len(sequence) > position
            )
            unique_tokens: dict[int, int] = defaultdict(int)
            for sequence in unique_sequences.values():
                if len(sequence) > position:
                    unique_tokens[sequence[position]] += 1
            present = sum(weighted_tokens.values())
            position_unique_counts[str(position + 1)] = len(weighted_tokens)
            position_entropies[str(position + 1)] = _entropy(weighted_tokens)
            for token_id, count in weighted_tokens.most_common():
                token_bytes = codec.encoding.decode_single_token_bytes(token_id)
                position_rows.append(
                    {
                        "attribute": attribute,
                        "token_position": position + 1,
                        "token_id": token_id,
                        "token_text": token_bytes.decode("utf-8", errors="replace"),
                        "profile_count": count,
                        "profile_probability": count / num_people,
                        "conditional_probability_when_present": count / present,
                        "unique_value_count": unique_tokens[token_id],
                    }
                )

        first_groups = Counter(sequence[0] for sequence in unique_sequences.values())
        bare_first = {sequence[0] for sequence in bare_sequences.values()}
        space_first = set(first_groups)
        changed_sequences = sum(
            unique_sequences[value] != bare_sequences[value] for value in unique_values
        )
        summary = {
            "profile_count": num_people,
            "unique_values": len(unique_values),
            "unique_token_sequences": len(set(unique_sequences.values())),
            "leading_space": True,
            "token_length": {
                "minimum": min(weighted_lengths),
                "mean_profile_weighted": mean(weighted_lengths),
                "median_profile_weighted": median(weighted_lengths),
                "p95_profile_weighted": _percentile(weighted_lengths, 95),
                "maximum": max(weighted_lengths),
                "mean_unique_values": mean(unique_lengths),
                "median_unique_values": median(unique_lengths),
            },
            "first_token": {
                "unique_tokens_profile_weighted": len(
                    {sequence[0] for sequence in weighted_sequences}
                ),
                "unique_tokens_unique_values": len(space_first),
                "entropy_bits_profile_weighted": _entropy(
                    Counter(sequence[0] for sequence in weighted_sequences)
                ),
                "whole_classes_per_first_token": {
                    "minimum": min(first_groups.values()),
                    "mean": mean(first_groups.values()),
                    "median": median(first_groups.values()),
                    "maximum": max(first_groups.values()),
                },
            },
            "position_unique_token_counts": position_unique_counts,
            "position_entropy_bits_profile_weighted": position_entropies,
            "leading_space_comparison": {
                "bare_first_token_classes": len(bare_first),
                "space_first_token_classes": len(space_first),
                "unique_values_with_changed_token_sequence": changed_sequences,
                "changed_fraction": changed_sequences / len(unique_values),
                "bare_mean_unique_value_length": mean(
                    len(sequence) for sequence in bare_sequences.values()
                ),
                "space_mean_unique_value_length": mean(unique_lengths),
            },
        }
        summaries[attribute] = summary
        for weighting, counts in (
            ("profile_weighted", weighted_length_counts),
            ("unique_values", unique_length_counts),
        ):
            denominator = num_people if weighting == "profile_weighted" else len(unique_values)
            for token_length, count in sorted(counts.items()):
                length_rows.append(
                    {
                        "attribute": attribute,
                        "weighting": weighting,
                        "token_length": token_length,
                        "count": count,
                        "probability": count / denominator,
                    }
                )

    with (output_dir / "attribute_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "attribute",
            "profiles",
            "unique_values",
            "space_first_token_classes",
            "minimum_tokens",
            "mean_tokens_profile_weighted",
            "median_tokens_profile_weighted",
            "p95_tokens_profile_weighted",
            "maximum_tokens",
            "first_token_entropy_bits",
            "mean_whole_classes_per_first_token",
            "space_vs_bare_changed_fraction",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for attribute in ATTRIBUTES:
            summary = summaries[attribute]
            writer.writerow(
                {
                    "attribute": attribute,
                    "profiles": summary["profile_count"],
                    "unique_values": summary["unique_values"],
                    "space_first_token_classes": summary["first_token"][
                        "unique_tokens_unique_values"
                    ],
                    "minimum_tokens": summary["token_length"]["minimum"],
                    "mean_tokens_profile_weighted": summary["token_length"][
                        "mean_profile_weighted"
                    ],
                    "median_tokens_profile_weighted": summary["token_length"][
                        "median_profile_weighted"
                    ],
                    "p95_tokens_profile_weighted": summary["token_length"][
                        "p95_profile_weighted"
                    ],
                    "maximum_tokens": summary["token_length"]["maximum"],
                    "first_token_entropy_bits": summary["first_token"][
                        "entropy_bits_profile_weighted"
                    ],
                    "mean_whole_classes_per_first_token": summary["first_token"][
                        "whole_classes_per_first_token"
                    ]["mean"],
                    "space_vs_bare_changed_fraction": summary[
                        "leading_space_comparison"
                    ]["changed_fraction"],
                }
            )

    for filename, rows in (
        ("length_distribution.csv", length_rows),
        ("position_token_distribution.csv", position_rows),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    result = {
        "protocol": "synbios_attribute_leading_space_token_distribution_v1",
        "tokenizer": "tiktoken:gpt2",
        "encoding_rule": 'encoding.encode(" " + attribute_value)',
        "num_people": num_people,
        "seed": seed,
        "data_conditions": {
            "single": "one biography per profile",
            "multi5_permute": (
                "five biographies per profile; normalized attribute-value token "
                "distribution is identical because the same profiles are repeated"
            ),
        },
        "attributes": summaries,
        "artifacts": {
            "summary_table": "attribute_summary.csv",
            "length_distribution": "length_distribution.csv",
            "position_token_distribution": "position_token_distribution.csv",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-people", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        num_people=args.num_people,
        seed=args.seed,
        output_dir=args.output,
    )
    print(json.dumps(result["attributes"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
