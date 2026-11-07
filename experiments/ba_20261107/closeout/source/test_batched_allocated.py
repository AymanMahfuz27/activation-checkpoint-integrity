"""Minimal changed-path fixtures; numerical execution is allocation-only."""

import copy
import json
import math
from pathlib import Path
import unittest

from analysis_core import require_allocation
from analysis_e2 import analyze, canonical, interval, evaluation

PROOF = {
    "valid": True,
    "all_finite": True,
    "byte_equal": True,
    "value_equal": True,
    "tensors": [{"name": "synthetic-full-state-rng"}],
}


class Reader:
    def __init__(self):
        self.values = {}
        self.states = {}
        self.members = {}

    def put(self, name, value):
        self.values[name] = value
        self.members[name] = {"sha256": "d" * 64, "bytes": 1}

    def json(self, name):
        return copy.deepcopy(self.values[name])

    def journal(self, name):
        return {"rows": self.json(name), "partial_final_write": None}

    def tensor(self, name):
        return copy.deepcopy(self.states[name])


def fixture(incomplete=False):
    require_allocation()
    protocol = json.loads(
        (Path(__file__).parent.parent / "configs/repair-contract.json").read_text()
    )["e2"]
    spec = {"synthetic": "independently supplied fixture spec"}
    reader = Reader()
    initial = {}

    def measured(ce):
        return {
            "cross_entropy": ce,
            "perplexity": math.exp(ce),
            "heldout_tokens": 1048576,
            "unscored_available_targets": 0,
            "evaluation_spec": spec,
            "evaluation_spec_sha256": canonical(spec),
            "state_preservation": copy.deepcopy(PROOF),
            "status": "PASS",
            "windows": [
                {
                    "window_id": n,
                    "target_begin": n * 512 + 1,
                    "target_end_exclusive": (n + 1) * 512 + 1,
                    "target_count": 512,
                    "nll_sum": ce * 512,
                    "cross_entropy": ce,
                }
                for n in range(2048)
            ],
        }

    def fault(stratum, seed, enabled):
        return {
            "case_id": stratum["case_id"],
            "location": stratum["location"],
            "layer": 0,
            "kind": stratum["kind"],
            "magnitude": stratum["magnitude"] or 0.001,
            "bit": stratum["bit"] if stratum["bit"] is not None else 31,
            "seed": seed,
            "onset": 1,
            "duration": 1,
            "period": 1,
            "enabled": enabled,
        }

    def injection(stratum, seed, specification):
        return {
            "case_id": stratum["case_id"],
            "spec": specification,
            "element": next(
                x["element"] for x in stratum["sites"] if x["seed"] == seed
            ),
            "tensor": stratum["identity"],
            "eligible_elements": stratum["eligible_elements"],
            "update": 1,
        }

    for index, h in enumerate(protocol["histories"]):
        root = f"E2/history-{h}/"
        initial[f"corrected210-history{h}"] = {
            "updates": 210,
            "cursor": 210 * 65536,
            "identity": {"history": h},
            "source_sha256": "e" * 64,
        }
        baseline = measured(3.0)
        reader.put(root + "initial-evaluation.json", baseline)
        control = [
            {
                "kind": "clean-observer",
                "status": "ACCEPTED",
                "phase": "ACCEPTED",
                "branch_update": 1,
                "scope": "checkpoint_boundaries",
                "policy": "every",
                "seed": 1729,
                "preserve_rng": True,
                "spec": None,
                "injections": [],
                "state_vs_clean": PROOF,
                "gradient_vs_clean": PROOF,
            }
        ]
        for stratum in protocol["strata"]:
            for seed in protocol["site_seeds"]:
                off = fault(stratum, seed, False)
                control.append(
                    {
                        "kind": "trigger-off",
                        "status": "ACCEPTED",
                        "phase": "ACCEPTED",
                        "branch_update": 1,
                        "scope": "checkpoint_boundaries",
                        "policy": "disabled",
                        "seed": 1729,
                        "preserve_rng": True,
                        "spec": off,
                        "injections": [],
                        "state_vs_clean": PROOF,
                    }
                )
                if stratum["stratum"] == "smaller":
                    on = fault(stratum, seed, True)
                    control.append(
                        {
                            "kind": "guarded-replay",
                            "status": "REJECTED_PRE_OPTIMIZER",
                            "phase": "BACKWARD_COMPLETE",
                            "branch_update": 1,
                            "scope": "checkpoint_boundaries",
                            "policy": "every",
                            "seed": 1729,
                            "preserve_rng": True,
                            "spec": on,
                            "injections": [injection(stratum, seed, on)],
                            "protected_state": PROOF,
                            "guard_outcome": "DETECTED_REJECTED",
                        }
                    )
        reader.put(root + "controls.jsonl", control)
        reader.put(root + "controls.json", control)
        summaries = []
        for plan in [x for x in protocol["arms"] if x["history"] == h]:
            prefix = root + plan["id"] + "/"
            end = 50 if incomplete and plan["id"] == "E2-H2-largest-S1729" else 100
            summary = {
                "arm": plan,
                "status": "COMPLETE"
                if end == 100
                else "INCOMPLETE_RETAINED_NO_REPLACEMENT",
                "source_input_sha256": "e" * 64,
                "final_state_sha256": "d" * 64,
                "accepted_updates": end,
                "accepted_targets": end * 65536,
            }
            summaries.append(summary)
            reader.put(prefix + "summary.json", summary)
            reader.members[prefix + "final-state.pt"] = {"sha256": "d" * 64, "bytes": 1}
            reader.states[prefix + "final-state.pt"] = {
                "updates": 210 + end,
                "cursor": (210 + end) * 65536,
                "identity": {"history": h},
            }
            rows = []
            evaluations = [
                {
                    "branch_update": 0,
                    "shared_initial": "../initial-evaluation.json",
                    **{k: v for k, v in baseline.items() if k != "windows"},
                }
            ]
            stratum = next(
                (x for x in protocol["strata"] if x["case_id"] == plan["fault"]), None
            )
            for n in range(1, end + 1):
                specification = (
                    fault(stratum, plan["site_seed"], True) if stratum else None
                )
                row = {
                    "arm_id": plan["id"],
                    "history": h,
                    "branch_update": n,
                    "status": "ACCEPTED",
                    "phase": "ACCEPTED",
                    "preserve_rng": True,
                    "policy": "disabled",
                    "scope": "checkpoint_boundaries",
                    "seed": 1729,
                    "spec": specification,
                    "injections": [injection(stratum, plan["site_seed"], specification)]
                    if stratum and n == 1
                    else [],
                    "matched_post_rng": PROOF,
                    "update": {
                        "update": 210 + n,
                        "accepted_tokens": (210 + n) * 65536,
                        "tokens_this_update": 65536,
                    },
                }
                if stratum and n in (1, 10, 50, 100):
                    row.update(
                        state_vs_clean={
                            x: PROOF
                            for x in (
                                "model",
                                "adam_exp_avg",
                                "adam_exp_avg_sq",
                                "full_optimizer",
                            )
                        },
                        gradient_vs_clean=PROOF,
                        post_clip_gradient_vs_clean=PROOF,
                    )
                rows.append(row)
                if n in (1, 10, 50, 100):
                    difference = (
                        (index + 1)
                        * (0.1 if plan.get("stratum") == "largest" else 0.002)
                        if stratum
                        else 0
                    )
                    exact = measured(3.0 + difference)
                    reader.put(prefix + f"evaluation-{n:03}.json", exact)
                    evaluations.append(
                        {
                            "branch_update": n,
                            **{k: v for k, v in exact.items() if k != "windows"},
                        }
                    )
            if end < 100:
                terminal = copy.deepcopy(rows[-1])
                terminal.update(status="ERROR", branch_update=end + 1)
                terminal.pop("update")
                rows.append(terminal)
            reader.put(prefix + "updates.jsonl", rows)
            reader.put(prefix + "evaluations.jsonl", evaluations)
        reader.put(
            root + "completion.json",
            {"history": h, "no_replacements_or_extension": True, "arms": summaries},
        )
    return reader, protocol, initial, spec


class BatchedFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_allocation()

    def test_three_history_interval_independent_df2_and_exp(self):
        r = interval([0.1, 0.2, 0.3], 0.975)
        u = 0.975
        q = math.sqrt(2 * u * u / (1 - u * u))
        half = q * 0.1 / math.sqrt(3)
        self.assertAlmostEqual(r["ce_difference"], 0.2)
        self.assertAlmostEqual(r["lower"], 0.2 - half)
        self.assertAlmostEqual(r["relative_ppl_upper"], math.exp(0.2 + half))
        self.assertIsNone(interval([0.1, None, 0.3], 0.975))

    def test_full15_arm_reader_nested_aggregation_and_all75_endpoints(self):
        r = analyze(*fixture())
        self.assertTrue(r["complete_registration"])
        self.assertEqual(len(r["endpoint_rows"]), 75)
        self.assertEqual(len(r["controls"]), 21)
        self.assertEqual(len(r["individual_effects"]), 48)
        self.assertAlmostEqual(
            r["strata"]["largest"]["points"][-1]["primary975"]["ce_difference"], 0.2
        )

    def test_missing_endpoint_suppresses_all_population_intervals(self):
        r = analyze(*fixture(incomplete=True))
        self.assertFalse(r["complete_registration"])
        self.assertTrue(
            all(
                x["recovery_interval"] is None and x["primary975"] is None
                for d in r["strata"].values()
                for x in d["points"]
            )
        )

    def test_wrong_bit_site_and_duplicate_control_rejected(self):
        for change in ("bit", "element", "control"):
            args = fixture()
            r = args[0]
            if change == "control":
                r.values["E2/history-1234/controls.jsonl"][1] = copy.deepcopy(
                    r.values["E2/history-1234/controls.jsonl"][2]
                )
            else:
                row = r.values["E2/history-1234/E2-H1234-largest-S1729/updates.jsonl"][
                    0
                ]
                if change == "bit":
                    row["spec"]["bit"] = 29
                else:
                    row["injections"][0]["element"] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(*args)

    def test_trailing_update_duplicate_endpoint_and_missing_diagnostic_rejected(self):
        for change in ("trailing", "endpoint", "diagnostic"):
            args = fixture()
            r = args[0]
            prefix = "E2/history-1234/E2-H1234-largest-S1729/"
            if change == "trailing":
                r.values[prefix + "updates.jsonl"].append(
                    copy.deepcopy(r.values[prefix + "updates.jsonl"][-1])
                )
            elif change == "endpoint":
                r.values[prefix + "evaluations.jsonl"].append(
                    copy.deepcopy(r.values[prefix + "evaluations.jsonl"][-1])
                )
            else:
                r.values[prefix + "updates.jsonl"][0].pop("post_clip_gradient_vs_clean")
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(*args)

    def test_window_count_range_nll_and_evaluator_state_rejected(self):
        for change in ("range", "nll", "state", "count"):
            args = fixture()
            r = args[0]
            row = r.values["E2/history-1234/initial-evaluation.json"]
            if change == "range":
                row["windows"][0]["target_begin"] = 2
            elif change == "nll":
                row["windows"][0]["nll_sum"] += 1
            elif change == "count":
                row["windows"].pop()
            else:
                row["state_preservation"]["byte_equal"] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(*args)

    def test_finite_ce_with_unrepresentable_ppl_retains_ce(self):
        row = {
            "cross_entropy": 710.0,
            "perplexity": None,
            "status": "PASS",
            "heldout_tokens": 1048576,
            "evaluation_spec": {},
            "evaluation_spec_sha256": canonical({}),
            "state_preservation": copy.deepcopy(PROOF),
        }
        self.assertEqual(evaluation(row, {}), 710.0)
