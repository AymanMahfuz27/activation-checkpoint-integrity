"""Allocated-only checks against selective inference and invalid pairing."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import selector  # Enforce the existing real-allocation numerical admission.
from .analyze import analyze


class AnalysisFixtures(unittest.TestCase):
    def data(self):
        config = {"experiment_id": "SYNTHETIC_ANALYSIS_QUALIFICATION", "seeds": [1, 2, 3],
                  "administrative_wall_horizon_seconds": 10.0, "endpoint_clock": "checker_cold_elapsed_seconds",
                  "arms": [{"name": "candidate"}, {"name": "reference"}],
                  "comparisons": [{"candidate": "candidate", "reference": "reference"}]}
        rows = []
        for seed in config["seeds"]:
            for arm, seconds in (("candidate", 8.0), ("reference", 10.0)):
                rows.append({"seed": seed, "arm": {"name": arm},
                    "initial_full_state_sha256": str(seed), "fault_schedule_sha256": str(seed),
                    "failure": None, "endpoint": {"valid": True,
                        "horizon_seconds": 10.0, "clock": "checker_cold_elapsed_seconds",
                        "restricted_seconds": seconds, "event_observed": arm == "candidate",
                        "confirmed_target_elapsed_seconds": 8.0 if arm == "candidate" else None,
                        "observed_stop_elapsed_seconds": seconds,
                        "common_horizon_right_censored": arm == "reference",
                        "administrative_censor_elapsed_seconds": 10.0 if arm == "reference" else None,
                        "stop_reason": "TARGET_CONFIRMED" if arm == "candidate" else "COMMON_WALL_HORIZON",
                        "failure": None}})
        return {"experiment_id": config["experiment_id"], "results": rows}, config

    def call(self, summary, config):
        with tempfile.TemporaryDirectory() as directory:
            result = analyze(summary, config, Path(directory) / "result")
            retained = json.loads((Path(directory) / "result/analysis.json").read_text())
            self.assertEqual(result, retained)
            return result["contrasts"][0]

    def test_censored_histories_stay_in_the_registered_mean(self):
        summary, config = self.data()
        result = self.call(summary, config)
        self.assertEqual(result["differences_seconds"], [-2.0, -2.0, -2.0])
        self.assertEqual(result["mean_candidate_minus_reference_seconds"], -2.0)
        self.assertEqual(result["paired_student_t_interval"], [-2.0, -2.0])
        self.assertEqual(result["exact_two_sided_sign_p"], 0.25)
        self.assertEqual(result["candidate_target_successes"], 3)
        self.assertEqual(result["reference_target_successes"], 0)

    def test_one_early_invalid_endpoint_disables_whole_contrast(self):
        summary, config = self.data()
        summary["results"][0]["endpoint"]["valid"] = False
        summary["results"][0]["failure"] = "EARLY_NATIVE_FAILURE"
        result = self.call(summary, config)
        self.assertEqual(result["status"], "INCOMPLETE_NO_COMPLETE_CASE_INFERENCE")
        self.assertIsNone(result["estimate"])
        self.assertEqual(len(result["pairs"]), 3)

    def test_duplicate_missing_unpaired_and_different_horizon_are_refused(self):
        summary, config = self.data()
        mutations = []
        duplicate = copy.deepcopy(summary)
        duplicate["results"].append(copy.deepcopy(duplicate["results"][0]))
        mutations.append(duplicate)
        missing = copy.deepcopy(summary)
        missing["results"].pop()
        mutations.append(missing)
        unpaired = copy.deepcopy(summary)
        unpaired["results"][0]["initial_full_state_sha256"] = "different"
        mutations.append(unpaired)
        horizon = copy.deepcopy(summary)
        horizon["results"][0]["endpoint"]["horizon_seconds"] = 11.0
        mutations.append(horizon)
        regime = copy.deepcopy(summary)
        regime["results"][0]["arm"]["fault_probability"] = 0.5
        mutations.append(regime)
        experiment = copy.deepcopy(summary)
        experiment["experiment_id"] = "ANOTHER_EXPERIMENT"
        mutations.append(experiment)
        for mutated in mutations:
            with self.subTest(mutated=mutated), self.assertRaises(ValueError):
                self.call(mutated, config)

    def test_false_validity_cannot_hide_failure_or_short_followup(self):
        summary, config = self.data()
        mutations = []
        failed = copy.deepcopy(summary)
        failed["results"][0]["failure"] = "EARLY_NATIVE_FAILURE"
        mutations.append(failed)
        short = copy.deepcopy(summary)
        short["results"][1]["endpoint"]["observed_stop_elapsed_seconds"] = 4.0
        mutations.append(short)
        late = copy.deepcopy(summary)
        late["results"][0]["endpoint"]["confirmed_target_elapsed_seconds"] = 12.0
        mutations.append(late)
        for mutated in mutations:
            with self.subTest(mutated=mutated), self.assertRaises(ValueError):
                self.call(mutated, config)
