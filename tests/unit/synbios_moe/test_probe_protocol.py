import unittest

from experiments.synbios_moe.probes.dataset import paper_probe_tasks
from experiments.synbios_moe.probes.pipeline import (
    ProbeJob,
    ProbeRuntimeConfig,
    all_probe_jobs,
    jobs_for_stage,
    steps_for_job,
)


class ProbeProtocolTest(unittest.TestCase):
    def test_task_matrix_contains_six_first_and_five_whole_tasks(self):
        tasks = paper_probe_tasks()
        first = [task for task in tasks if task.target == "first"]
        whole = [task for task in tasks if task.target == "whole"]

        self.assertEqual(len(first), 6)
        self.assertEqual(len(whole), 5)
        self.assertNotIn("birth_date_whole", {task.key for task in tasks})

    def test_pipeline_expands_each_task_to_p_and_q(self):
        jobs = all_probe_jobs()

        self.assertEqual(len(jobs), 22)
        self.assertEqual(len({job.key for job in jobs}), 22)

    def test_runtime_config_separates_p_and_q_batches(self):
        config = {
            "runtime": {
                "training_batch_sizes": {"p": 32, "q": 64},
                "validation_batch_sizes": {"p": 128, "q": 256},
                "log_interval_steps": 25,
                "heartbeat_seconds": 5,
                "checkpoint_interval_steps": 200,
                "evaluate_train": False,
            }
        }

        runtime = ProbeRuntimeConfig.from_config(config)

        self.assertEqual(runtime.p_batch_size, 32)
        self.assertEqual(runtime.q_batch_size, 64)
        self.assertEqual(runtime.p_validation_batch_size, 128)
        self.assertEqual(runtime.q_validation_batch_size, 256)
        self.assertFalse(runtime.evaluate_train)

    def test_stage_validation_rejects_duplicate_jobs(self):
        config = {
            "stages": {
                "duplicate": {
                    "steps": 10,
                    "tasks": [
                        {"kind": "p", "attribute": "major", "target": "first"},
                        {"kind": "p", "attribute": "major", "target": "first"},
                    ],
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "duplicate probe jobs"):
            jobs_for_stage(config, "duplicate")

    def test_target_specific_schedule_is_resolved_per_job(self):
        schedule = {"p_first": 10, "p_whole": 20, "q_first": 30, "q_whole": 40}

        self.assertEqual(steps_for_job(schedule, ProbeJob("p", "major", "first")), 10)
        self.assertEqual(steps_for_job(schedule, ProbeJob("q", "major", "whole")), 40)


if __name__ == "__main__":
    unittest.main()
