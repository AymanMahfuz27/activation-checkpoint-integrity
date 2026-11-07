"""Additional subset admission fixtures; run only in the reviewed allocation."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis_core import require_allocation
from run_subset import duration, initial_semantics, source_gate
from subset_analysis import e4_extra_validation


class SubsetMetadataFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_allocation()

    def test_owned_remaining_time_units_and_malformed_values(self):
        self.assertEqual(duration("59:22"), 3562)
        self.assertEqual(duration("01:00:00"), 3600)
        self.assertEqual(duration("1-00:00:00"), 86400)
        for value in ("UNLIMITED", "None", "30", "1:2:3:4", "a-00:01:00"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                duration(value)

    def test_exact_source_census_and_known_byte_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source").mkdir()
            (root / "configs").mkdir()
            file = root / "source/known.py"
            file.write_bytes(b"abc")
            known = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            manifest = root / "source-manifest.json"
            manifest.write_text(json.dumps({"files": {"source/known.py": known}}))
            contract = {
                "source_packet_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest()
            }
            self.assertEqual(
                source_gate(root, contract)["files"]["source/known.py"], known
            )
            file.write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "Source/config member differs"):
                source_gate(root, contract)
            file.write_bytes(b"abc")
            (root / "source/foreign.py").write_text("foreign")
            with self.assertRaisesRegex(ValueError, "census differs"):
                source_gate(root, contract)

    def test_four_role_semantics_use_independent_native_expectations(self):
        roles = {
            "early-history1234": 10,
            "corrected210-history1234": 210,
            "corrected210-history1": 210,
            "corrected210-history2": 210,
        }
        states = [
            {
                "role": role,
                "updates": count,
                "cursor": count * 65536,
                "identity": {"declared_role": role},
            }
            for role, count in roles.items()
        ]
        verification = {
            "jobs": {
                str(index): {"original_native_states": copy.deepcopy(states)}
                for index in range(4)
            }
        }
        inputs = {
            "initial_inputs": [
                {"role": role, "path": "/synthetic/" + role, "sha256": "e" * 64}
                for role in roles
            ]
        }

        def saved(path, **kwargs):
            self.assertEqual(kwargs, {"map_location": "cpu", "weights_only": False})
            role = Path(path).name
            return {
                "schema": 1,
                "updates": roles[role],
                "cursor": roles[role] * 65536,
                "identity": {"declared_role": role},
                **{
                    key: {"synthetic": True}
                    for key in (
                        "model",
                        "optimizer",
                        "scheduler",
                        "rng",
                        "model_config",
                        "recipe",
                    )
                },
            }

        with (
            tempfile.TemporaryDirectory() as folder,
            patch("torch.load", side_effect=saved),
        ):
            results = initial_semantics(inputs, verification, Path(folder))
            self.assertEqual(set(results), set(roles))
            self.assertFalse(
                any(
                    row["live_model_restore"] or row["process_rng_restore"]
                    for row in results.values()
                )
            )
        for field, bad in (
            ("schema", True),
            ("updates", True),
            ("cursor", 655361),
            ("identity", {"foreign": True}),
            ("model", {}),
        ):

            def changed(path, field=field, bad=bad, **kwargs):
                value = saved(path, **kwargs)
                value[field] = bad
                return value

            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as folder,
                patch("torch.load", side_effect=changed),
                self.assertRaises(ValueError),
            ):
                initial_semantics(inputs, verification, Path(folder))
        contradictory = copy.deepcopy(verification)
        contradictory["jobs"]["3"]["original_native_states"][0]["cursor"] += 1
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("torch.load") as loader,
            self.assertRaisesRegex(ValueError, "disagree"),
        ):
            initial_semantics(inputs, contradictory, Path(folder))
        loader.assert_not_called()

    def test_setup_scope_precision_and_isolation_negatives(self):
        from test_e4_provenance_allocated import source_fixture

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        original, pin = source_fixture(temporary.name)
        raw = {"setup": {}, "arms": []}
        for number, job in enumerate(("3520049", "3520050", "3520051"), start=1):
            raw["setup"][job] = {
                "isolation": {
                    "slurm_job": f"JobId={job} UserId=aymanmahfuz(893489) Account=CCR26063 Partition=gpu-a100 JobState=RUNNING NumNodes=1 OverSubscribe=NO Restarts=0 Requeue=0",
                    "cuda_pids": [],
                },
                "scope": "checkpoint_boundaries",
                "forensics_in_timers": False,
                "production_batch": {
                    "microbatch": 4,
                    "sequence": 512,
                    "accumulation": 32,
                },
                "extension": {"source_sha256": pin, "architecture": "sm_80"},
                "seconds": 2.0,
            }
            raw["arms"].append(
                {
                    "block": {"job": number},
                    "isolation": {"cuda_pids": [100 + number]},
                    "reference_residency": "CPU only, equal GPU residency across paired arms",
                    "warmup_seconds": 2.0,
                    "inner_update_seconds": 3.0,
                    "experimental_restore_seconds": 1.0,
                    "peak_allocated_bytes": 1000,
                    "peak_reserved_bytes": 2000,
                    "warmup_peak_allocated_bytes": 1000,
                    "measured_targets": 65536,
                    "full_loop_seconds": 4.0,
                    "targets_per_second": 16384.0,
                }
            )
        self.assertEqual(len(e4_extra_validation(raw, {}, original)), 3)
        for kind in (
            "competing_pid",
            "forensics",
            "extension",
            "negative_memory",
            "throughput",
            "residency",
        ):
            value = copy.deepcopy(raw)
            if kind == "competing_pid":
                value["arms"][0]["isolation"]["cuda_pids"] = [101, 999]
            elif kind == "forensics":
                value["setup"]["3520049"]["forensics_in_timers"] = True
            elif kind == "extension":
                value["setup"]["3520049"]["extension"]["source_sha256"] = "b" * 64
            elif kind == "negative_memory":
                value["arms"][0]["peak_allocated_bytes"] = -1
            elif kind == "throughput":
                value["arms"][0]["targets_per_second"] += 1
            else:
                value["arms"][0]["reference_residency"] = "GPU reference"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                e4_extra_validation(value, {}, original)
