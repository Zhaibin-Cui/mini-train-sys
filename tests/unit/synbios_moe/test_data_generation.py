import json
import tempfile
import unittest
from pathlib import Path

from experiments.synbios_moe.pretraining.dataset import (
    ATTRIBUTES,
    candidate_pools,
    entries_per_person,
    generate_profiles,
    iter_biographies,
    render_biography,
    split_for_person,
    write_dataset,
)


class SynBioSDataGenerationTest(unittest.TestCase):
    def test_profiles_are_deterministic_and_unique(self):
        first = generate_profiles(32, seed=1337)
        second = generate_profiles(32, seed=1337)

        self.assertEqual(first, second)
        self.assertEqual(len({profile.full_name for profile in first}), 32)

    def test_candidate_pool_cardinalities_match_the_experiment(self):
        pools = candidate_pools()

        self.assertEqual(len(pools["birth_city"]), 200)
        self.assertEqual(len(pools["university"]), 300)
        self.assertEqual(len(pools["major"]), 100)
        self.assertEqual(len(pools["company"]), 263)
        self.assertEqual(len(pools["company_city"]), 263)

    def test_company_and_company_city_share_an_indexed_relation(self):
        pools = candidate_pools()
        profiles = generate_profiles(64, seed=7)
        company_to_city = dict(zip(pools["company"], pools["company_city"]))

        for profile in profiles:
            self.assertEqual(profile.company_city, company_to_city[profile.company])

    def test_rendered_spans_recover_every_attribute(self):
        profile = generate_profiles(1, seed=11)[0]
        biography = render_biography(
            profile,
            variant="multi5_permute",
            sample=0,
            seed=11,
        )

        for attribute in ATTRIBUTES:
            start, end = biography.attribute_spans[attribute]
            self.assertEqual(biography.text[start:end], getattr(profile, attribute))
        self.assertEqual(biography.text.count(profile.full_name), 1)

    def test_variant_controls_biographies_per_person(self):
        profiles = generate_profiles(3, seed=5)

        self.assertEqual(entries_per_person("single"), 1)
        self.assertEqual(entries_per_person("multi5_permute"), 5)
        self.assertEqual(len(list(iter_biographies(profiles, "single", 5))), 3)
        self.assertEqual(len(list(iter_biographies(profiles, "multi5_permute", 5))), 15)

    def test_person_split_is_deterministic(self):
        first = [split_for_person(person_id, seed=1337) for person_id in range(100)]
        second = [split_for_person(person_id, seed=1337) for person_id in range(100)]

        self.assertEqual(first, second)
        self.assertEqual(set(first), {"train", "validation"})

    def test_write_dataset_records_files_and_generation_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_dataset(
                root,
                num_people=4,
                variant="multi5_permute",
                seed=17,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["num_people"], 4)
            self.assertEqual(manifest["biographies"], 20)
            self.assertEqual(manifest["variant"], "multi5_permute")
            self.assertEqual(manifest["seed"], 17)
            for name in ("profiles.jsonl", "biographies.jsonl", "biographies.txt"):
                self.assertTrue((root / name).is_file())
                self.assertGreater(manifest["files"][name]["bytes"], 0)
                self.assertEqual(len(manifest["files"][name]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
