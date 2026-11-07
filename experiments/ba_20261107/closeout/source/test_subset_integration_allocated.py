"""Real allocated export/ingestion/custody path on independent synthetic data."""

import csv
import hashlib
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis_core import require_allocation
from evidence_loading import journal_bytes
from run_subset import interrupted, retain
from subset_analysis import load_e3, run
from test_analysis_core_allocated import e4_fixture
from test_baselines_allocated import fixture as baseline_fixture
from test_original_allocated import error_cells
from test_e4_provenance_allocated import source_fixture


class DiskReader:
    """Synthetic inputs with real serialization, hashes and verified file reads."""

    def __init__(self, root):
        self.root, self.members = root, {}
        root.mkdir()

    def store(self, name, value, journal=False):
        raw = (
            b"".join(json.dumps(row).encode() + b"\n" for row in value)
            if journal
            else json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        )
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        self.members[name] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def file(self, name):
        p = self.root / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != self.members[name]["sha256"]:
            raise ValueError("Synthetic source bytes differ")
        return p

    def json(self, name, required=True):
        if name not in self.members and not required:
            return None
        return json.loads(self.file(name).read_text())

    def journal(self, name):
        return {
            **journal_bytes(self.file(name).read_bytes()),
            "member": name,
            "status": "RETAINED",
        }


def population(root):
    require_allocation()
    schedules, cells, early, tape_members = error_cells()
    e3 = DiskReader(root / "e3")
    e3.members.update(tape_members)
    for index, cell in enumerate(cells):
        prefix = f"E3/worker-{index % 3}/{cell['summary']['id']}/"
        e3.store(prefix + "summary.json", cell["summary"])
        e3.store(prefix + "attempts.jsonl", cell["attempts"], journal=True)
    baselines, _, _ = baseline_fixture()
    for name, obj in baselines.objects.items():
        e3.store(name, obj)
    for name, rows in baselines.journals.items():
        e3.store(name, rows, journal=True)
    # Baseline reducer needs recorded operand/product membership; no tensor load
    # occurs in this profile. Sentinel bytes are explicitly synthetic fixtures.
    for name in baselines.members:
        if name not in e3.members:
            path = e3.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic-no-tensor-payload")
            e3.members[name] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    pairs, arms, blocks, _ = e4_fixture()
    for pair in pairs:
        pair["memory_delta_bytes"] = 100
    readers = {"3520048": e3}
    original, pin = source_fixture(root)
    for number, job in enumerate(("3520049", "3520050", "3520051"), start=1):
        reader = DiskReader(root / job)
        selected = [row for row in arms if row["block"]["job"] == number]
        for arm in selected:
            arm.update(
                isolation={"cuda_pids": [number + 100]},
                reference_residency="CPU only, equal GPU residency across paired arms",
                warmup_seconds=2.0,
                inner_update_seconds=3.0,
                experimental_restore_seconds=1.0,
                peak_allocated_bytes=1000 if arm["arm"] == "disabled" else 1100,
                peak_reserved_bytes=2000,
                warmup_peak_allocated_bytes=1000,
                targets_per_second=arm["measured_targets"] / arm["full_loop_seconds"],
            )
        reader.store(
            "E4/pairs.jsonl", [p for p in pairs if p["job"] == number], journal=True
        )
        reader.store("E4/arms.jsonl", selected, journal=True)
        reader.store(
            "E4/setup.json",
            {
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
            },
        )
        reader.original_source_contract = original
        readers[job] = reader
    return readers, schedules, blocks, early, pin


class SubsetIntegrationFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_allocation()

    def test_real_loader_exports_figures_and_custody_retain_incomplete_census(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            readers, _schedules, blocks, early, pin = population(root)
            stage, out = root / "stage", root / "out"
            stage.mkdir()
            (stage / "configs").mkdir()
            out.mkdir()
            source = Path(__file__).resolve().parents[1] / "configs"
            for name in (
                "e3-schedules.json",
                "e3-abft-products.json",
                "Subset-capability-policy.json",
            ):
                shutil.copyfile(source / name, stage / "configs" / name)
            (stage / "configs/e4-blocks.json").write_text(json.dumps(blocks))
            (stage / "configs/original-contract.json").write_text(
                json.dumps(
                    readers["3520049"].original_source_contract
                )
            )
            result = run(stage, out, readers, {**early, "source_sha256": "a" * 64})
            # A 39-row incomplete synthetic census must NEVER pass actual2973 gate.
            self.assertEqual(result["status"], "ANALYSIS_VALIDATION_FAILURE_RETAINED")
            self.assertFalse(result["scientific_acceptance"])
            self.assertEqual(set(result["errors"]), {"e3"})
            expected = {
                "e3-all39-cells.csv": 39,
                "e3-all2973-attempts.csv": 39,
                "e3-all276-abft-objects.csv": 276,
                "e4-all100-arms.csv": 100,
                "e4-all50-blocks.csv": 50,
                "e4-all4400-measured-updates.csv": 4400,
                "e4-registered-policy-estimates.csv": 5,
            }
            for name, count in expected.items():
                with (out / name).open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), count, name)
            for item in result["e4"]["primary"].values():
                self.assertAlmostEqual(item["geometric_ratio"], 2.0)
                self.assertAlmostEqual(item["throughput_percent"], -50.0)
                self.assertEqual(item["gpu_memory_delta_bytes"]["minimum"], 100)
                self.assertEqual(item["pairs_per_job"], [4, 3, 3])
            saved = json.loads((out / "results.json").read_text())
            self.assertEqual(saved["e3"]["attempt_rows_validated"], 39)
            with tarfile.open(out / "e3-all39-verified-journals.tar.gz") as archive:
                self.assertEqual(len(archive.getmembers()), 78)
                for member in archive:
                    self.assertEqual(
                        archive.extractfile(member).read(),
                        readers["3520048"].file(member.name).read_bytes(),
                    )
            for prefix in ("e3-wrong-updates", "e4-ratios"):
                self.assertEqual(
                    (out / (prefix + ".png")).read_bytes()[:8], b"\x89PNG\r\n\x1a\n"
                )
                self.assertTrue(
                    (out / (prefix + ".pdf")).read_bytes().startswith(b"%PDF")
                )
            self.assertIn("three fresh jobs", (out / "Report.md").read_text())
            originals = {
                str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.rglob("*")
                if p.is_file()
            }
            contract = {"campaign_id": "synthetic-output"}
            with (
                patch("run_subset.WORK", root / "work"),
                patch("run_subset.storage_gate") as quota,
                patch("run_subset.signal.setitimer"),
                patch("run_subset.signal.signal"),
            ):
                retain(out, "synthetic-job", contract)
                quota.assert_called_once()
            target = root / "work/completion/synthetic-output/synthetic-job"
            receipt = json.loads((target / "output-custody.json").read_text())
            self.assertFalse(receipt["scientific_acceptance"])
            self.assertEqual(
                (out / "output-custody.json").read_bytes(),
                (target / "output-custody.json").read_bytes(),
            )
            for name, sha in originals.items():
                self.assertEqual(
                    hashlib.sha256((target / name).read_bytes()).hexdigest(), sha
                )
                self.assertEqual(receipt["files"][name]["sha256"], sha)

    def test_subset_missing_split_and_corrupt_journal_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            readers, schedules, _, _, _ = population(Path(folder))
            reader = readers["3520048"]
            name = next(n for n in reader.members if n.endswith("attempts.jsonl"))
            reader.members[name]["sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "source bytes differ"):
                load_e3(reader, schedules)
            reader.members.pop(name)
            with self.assertRaisesRegex(ValueError, "Missing or split"):
                load_e3(reader, schedules)

    def test_cancellation_escapes_section_handlers_and_unittest(self):
        with self.assertRaises(KeyboardInterrupt):
            try:
                interrupted(None, None)
            except Exception as error:  # noqa: BLE001 - prove section recovery cannot consume cancellation
                self.fail("Control cancellation consumed:" + str(error))

        class InterruptedCase(unittest.TestCase):
            def runTest(self):
                interrupted(None, None)

        with self.assertRaises(KeyboardInterrupt):
            InterruptedCase().run(unittest.TestResult())
