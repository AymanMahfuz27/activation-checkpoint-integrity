"""Partial-journal/missing-outcome ingestion fixtures; allocated only."""

import json
import unittest

from analysis_core import ARMS, HISTORIES, analyze_e5, require_allocation
from evidence_loading import journal_bytes, load_original
from analysis_original import analyze_e3
from test_original_allocated import error_cells, finite_cell


class ReaderFixture:
    def __init__(self, journals=None, objects=None, members=None):
        self.journals = journals or {}
        self.objects = objects or {}
        self.members = {
            **(members or {}),
            **{name: {} for name in self.journals},
            **{name: {} for name in self.objects},
        }

    def journal(self, name):
        if name not in self.journals:
            return {"rows": [], "partial_final_write": None, "status": "NOT_RECORDED"}
        return {
            **journal_bytes(self.journals[name]),
            "status": "RETAINED",
            "member": name,
        }

    def json(self, name):
        return self.objects[name]


class EvidenceLoadingFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_allocation()

    def test_real_journal_prefix_without_summary_survives_loader_and_reducer(self):
        schedules, records, initial, members, native = finite_cell()
        identity = native["summary"]["id"]
        path = f"E3/worker-0/{identity}/attempts.jsonl"
        raw = json.dumps(native["attempts"][0]).encode() + b'\n{"branch_update":'
        loaded = load_original(
            {
                "3520038": ReaderFixture(),
                "3520048": ReaderFixture({path: raw}, members=members),
            },
            schedules,
        )
        self.assertEqual(len(loaded["e3_cells"]), 1)
        self.assertIsNone(loaded["e3_cells"][0]["summary"])
        result = analyze_e3(schedules, loaded["e3_cells"], initial, members)
        cell = next(row for row in result["all_cells"] if row["id"] == identity)
        self.assertEqual(cell["status"], "MISSING_TERMINAL_SUMMARY_RETAINED")
        self.assertEqual(len(cell["attempts"]), 1)
        self.assertEqual(cell["attempts"][0]["authoritative_status"], "ACCEPTED")
        self.assertFalse(cell["attempts"][0]["validated_pre_optimizer_rejection"])
        self.assertEqual(cell["observed_prefix"]["attempts"], 1)
        self.assertEqual(cell["observed_prefix"]["accepted_updates"], 1)
        self.assertIsNone(cell["terminal_duration_or_completed_endpoint"])
        self.assertEqual(
            cell["journal_evidence"]["partial_final_write"]["decoded_text"],
            '{"branch_update":',
        )

    def test_foreign_duplicate_or_split_e3_journal_locations_are_explicit_errors(self):
        schedules, records, _, _ = error_cells()
        native = records[0]
        identity = native["summary"]["id"]
        raw = json.dumps(native["attempts"][0]).encode() + b"\n"
        for journals, objects in (
            ({"E3/worker-0/foreign/attempts.jsonl": raw}, {}),
            (
                {
                    f"E3/worker-0/{identity}/attempts.jsonl": raw,
                    f"E3/worker-1/{identity}/attempts.jsonl": raw,
                },
                {},
            ),
            (
                {f"E3/worker-1/{identity}/attempts.jsonl": raw},
                {f"E3/worker-0/{identity}/summary.json": native["summary"]},
            ),
        ):
            with self.assertRaises(ValueError):
                load_original(
                    {
                        "3520038": ReaderFixture(),
                        "3520048": ReaderFixture(journals, objects),
                    },
                    schedules,
                )

    def test_summary_without_journal_keeps_native_summary_and_unknown_endpoint(self):
        schedules, records, initial, members = error_cells()
        native = records[0]
        identity = native["summary"]["id"]
        loaded = load_original(
            {
                "3520038": ReaderFixture(),
                "3520048": ReaderFixture(
                    objects={f"E3/worker-0/{identity}/summary.json": native["summary"]}
                ),
            },
            schedules,
        )
        result = analyze_e3(schedules, loaded["e3_cells"], initial, members)
        cell = next(row for row in result["all_cells"] if row["id"] == identity)
        self.assertEqual(cell["status"], "MISSING_ATTEMPT_JOURNAL_RETAINED")
        self.assertEqual(cell["summary"], native["summary"])
        self.assertEqual(cell["attempts"], [])

    def test_newline_committed_rows_and_partial_tail_are_distinct(self):
        raw = b'{"a":1}\n{"a":2}'
        result = journal_bytes(raw)
        self.assertEqual(result["rows"], [{"a": 1}])
        self.assertEqual(result["partial_final_write"]["parseable_candidate"], {"a": 2})
        self.assertEqual(result["source_bytes"], len(raw))

    def test_incomplete_json_tail_is_preserved_without_fabricated_row(self):
        result = journal_bytes(b'{"a":1}\n{"a":')
        self.assertEqual(len(result["rows"]), 1)
        self.assertIsNone(result["partial_final_write"]["parseable_candidate"])
        self.assertEqual(result["partial_final_write"]["decoded_text"], '{"a":')

    def test_invalid_interior_duplicate_key_and_nonfinite_records_fail(self):
        for raw in (
            b'{"a":1}\nbroken\n{"a":2}\n',
            b'{"a":1,"a":2}\n',
            b'{"a":NaN}\n',
            b"\n",
            b"[]\n",
        ):
            with self.assertRaises(ValueError):
                journal_bytes(raw)

    def test_empty_journal_is_explicit_zero_prefix(self):
        result = journal_bytes(b"")
        self.assertEqual(result["rows"], [])
        self.assertIsNone(result["partial_final_write"])

    def test_all12_missing_outcomes_survive_without_times_or_population_intervals(self):
        counters = {
            history: dict(updates=210, cursor=210 * 65536) for history in HISTORIES
        }
        result = analyze_e5([], counters)
        self.assertEqual(
            {(row["history"], row["arm"]) for row in result["outcomes"]},
            {(history, arm) for history in HISTORIES for arm in ARMS},
        )
        self.assertTrue(
            all(row["confirmed_seconds"] is None for row in result["outcomes"])
        )
        self.assertTrue(
            all(
                row["population_interval"] is None
                for row in result["contrasts"].values()
            )
        )

    def test_missing_summary_keeps_only_actual_journal_clock_lower_bound(self):
        counters = {
            history: dict(updates=210, cursor=210 * 65536) for history in HISTORIES
        }
        records = [
            dict(
                history=1234,
                arm="buggy-disabled",
                summary=None,
                evaluations=[{"elapsed_seconds": 5.0}],
                attempts=[{"elapsed_seconds": 7.0}],
            )
        ]
        result = analyze_e5(records, counters)
        row = next(
            row
            for row in result["outcomes"]
            if row["history"] == 1234 and row["arm"] == "buggy-disabled"
        )
        self.assertEqual(row["last_journal_elapsed_seconds"], 7.0)
        self.assertIsNone(row["observed_bound_seconds"])
        self.assertIsNone(row["confirmed_seconds"])
        self.assertEqual(row["status"], "MISSING_ARM_SUMMARY_RETAINED")


if __name__ == "__main__":
    require_allocation()
    unittest.main()
