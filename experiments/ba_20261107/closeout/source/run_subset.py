"""Physical Slurm admission, bounded fixtures and batched E1-E5 analysis."""

import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import traceback
import unittest
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/scratch/09999/aymanmahfuz/activation-checkpoint-integrity")
WORK = Path("/work/09999/aymanmahfuz/ls6/activation-checkpoint-integrity")
CUTOFF = None


def require(value, message):
    if not value:
        raise ValueError(message)


def check_deadline():
    if CUTOFF is not None and time.monotonic() >= CUTOFF:
        raise TimeoutError("Subset processing cutoff; retain outputs without retry")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            check_deadline()
            h.update(block)
    return h.hexdigest()


def read(path, pin=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Regular pinned file required")
    if pin:
        require(digest(path) == pin, "Input hash differs: " + str(path))

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        path.read_text(),
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("Nonfinite JSON:" + value)
        ),
    )


def write(path, value):
    path = Path(path)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def retained_command(out, name, args):
    start = time.monotonic()
    p = subprocess.run(args, capture_output=True, timeout=30, check=False)
    (out / (name + ".stdout")).write_bytes(p.stdout)
    (out / (name + ".stderr")).write_bytes(p.stderr)
    write(
        out / (name + ".json"),
        {
            "args": args,
            "returncode": p.returncode,
            "elapsed_seconds": time.monotonic() - start,
            "stdout_sha256": hashlib.sha256(p.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(p.stderr).hexdigest(),
        },
    )
    require(p.returncode == 0, "Native command failed: " + name)
    return p.stdout.decode()


def descriptor_fields(text):
    return dict(token.split("=", 1) for token in text.split() if "=" in token)


def duration(value):
    days, separator, clock = value.partition("-")
    if not separator:
        clock, days = days, "0"
    parts = clock.split(":")
    require(
        len(parts) in (2, 3)
        and all(part.isdecimal() for part in parts)
        and days.isdecimal(),
        "Malformed scheduler time left",
    )
    nums = list(map(int, parts))
    return int(days) * 86400 + (
        nums[0] * 3600 + nums[1] * 60 + nums[2]
        if len(nums) == 3
        else nums[0] * 60 + nums[1]
    )


def physical_context(stage, out, contract, release):
    global CUTOFF
    job = os.environ.get("SLURM_JOB_ID", "")
    require(
        job.isdecimal() and sys.platform == "linux", "Real allocated Linux job required"
    )
    text = retained_command(out, "owned-job", ["scontrol", "show", "job", job, "-o"])
    fields = descriptor_fields(text)
    require(
        fields.get("JobId") == job
        and fields.get("UserId") == "aymanmahfuz(893489)"
        and os.getuid() == 893489
        and fields.get("Account") == "CCR26063"
        and fields.get("Partition") == "normal"
        and fields.get("JobState") == "RUNNING"
        and fields.get("JobName") == "vg-batched-analysis"
        and fields.get("NumNodes") == "1"
        and fields.get("NumTasks") == "1"
        and fields.get("CPUs/Task") == "12"
        and fields.get("OverSubscribe") == "NO"
        and fields.get("Requeue") == "0"
        and fields.get("Restarts") == "0"
        and fields.get("TimeLimit") == "01:00:00"
        and fields.get("SuspendTime") == "None"
        and fields.get("SecsPreSuspend") == "0",
        "Owned normal CPU resources differ",
    )
    host = socket.gethostname().split(".")[0]
    nodes = retained_command(
        out, "owned-nodes", ["scontrol", "show", "hostnames", fields["NodeList"]]
    ).splitlines()
    require(nodes == [host], "Host is not this owned allocation")
    actual = retained_command(
        out, "actual-batch", ["scontrol", "write", "batch_script", job, "-"]
    )
    require(
        hashlib.sha256(actual.encode()).hexdigest()
        == release["packet_files"]["analysis.sbatch"],
        "Actual batch differs from independent release",
    )
    left = retained_command(
        out, "time-left", ["squeue", "-h", "-j", job, "-o", "%L"]
    ).strip()
    seconds = duration(left)
    require(1200 <= seconds <= 3600, "Insufficient or invalid owned time left")
    CUTOFF = time.monotonic() + seconds - 600
    signal.setitimer(signal.ITIMER_REAL, seconds - 600)
    require(
        contract["max_su"] == 1
        and contract["shared_analysis_ceiling_su"] == 3
        and contract["later_analysis_max_su"] == 1
        and contract["prior_analysis_reserved_su"] == 1
        and contract["campaign_conservative_bound_su"] == 102
        and contract["campaign_ceiling_su"] == 105,
        "Frozen shared budget differs",
    )
    require(
        fields.get("Command") == str(stage / "analysis.sbatch"),
        "Owned batch path differs",
    )
    return {
        "job_id": job,
        "hostname": host,
        "time_left_seconds": seconds,
        "retention_reserve_seconds": 600,
        "processing_cutoff_monotonic": CUTOFF,
        "state": "RUNNING",
        "account": "CCR26063",
        "partition": "normal",
        "native_admission_scope": "E1_E5_RECORD_ANALYSIS_ONLY",
    }


def source_gate(stage, contract):
    manifest = read(stage / "source-manifest.json", contract["source_packet_sha256"])
    require(set(manifest) == {"files"}, "Source manifest schema differs")
    wanted = manifest["files"]
    actual = {
        str(path.relative_to(stage))
        for folder in (stage / "source", stage / "configs")
        for path in folder.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    require(actual == set(wanted), "Exact source/config file census differs")
    for name, pin in wanted.items():
        read_path = stage / name
        require(
            not read_path.is_symlink() and digest(read_path) == pin,
            "Source/config member differs:" + name,
        )
    return manifest


def storage_gate(out):
    original = ROOT / "slurm/e1e4-full-20261019-v2/source"
    sys.path.insert(0, str(original / "src"))
    from ac_integrity.research.storage import query_work_quota

    q = query_work_quota()
    require(
        q["usage_bytes"] + 200_000_000 < q["effective_limit_bytes"] * 0.9
        and q["headroom_bytes"] >= 50_000_000_000 + 200_000_000
        and q["file_headroom"] >= 2000,
        "Fresh WORK quota/reserve fails",
    )
    s = os.statvfs(ROOT)
    tmp = os.statvfs("/tmp")
    require(
        s.f_bavail * s.f_frsize >= 451_000_000_000
        and tmp.f_bavail * tmp.f_frsize >= 20_000_000_000,
        "Scratch/tmp floors fail",
    )
    write(out / "fresh-storage.json", q)


def accounting_gate(stage, out, context):
    names = {
        "3520038": "vgfull-M4",
        "3520048": "vgfull-M5",
        "3520049": "vgfull-E4-1",
        "3520050": "vgfull-E4-2",
        "3520051": "vgfull-E4-3",
        "3520353": "vgfull-E5-1",
        "3520355": "vgfull-E5-2",
        "3520356": "vgfull-E5-3",
        "3533911": "vgfull-M4-repair",
        "3535049": "vg-E3E4-analysis",
    }
    text = retained_command(
        out,
        "campaign-accounting",
        [
            "sacct",
            "-X",
            "-j",
            ",".join(names),
            "-n",
            "-P",
            "--format=JobIDRaw,Account,JobName,State,ElapsedRaw,NNodes,ExitCode",
        ],
    )
    rows = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.strip().split("|")
        require(
            len(values) == 7
            and values[0] in names
            and values[0] not in rows
            and values[1].upper() == "CCR26063"
            and values[2] == names[values[0]],
            "Missing/foreign/duplicate campaign accounting",
        )
        rows[values[0]] = values
    require(set(rows) == set(names), "All ten prior campaign records required")
    old = read(stage / "configs/original-budget-release.json")["original_jobs"]
    total = 0
    for job, record in old.items():
        row = rows[job]
        require(
            row[3] == record["state"]
            and row[6] == record["exit_code"]
            and int(row[4]) == record["elapsed_seconds"]
            and int(row[5]) == 1,
            "Original terminal accounting changed",
        )
        total += math.ceil(max(int(row[4]), 900) / 900) * 0.75
    require(
        total == 36 and total + 45 + 18 + 1 + 1 + 1 == 102,
        "Actual conservative campaign or split analysis bound fails",
    )
    require(
        rows["3535049"][3:] == ["FAILED", "367", "1", "1:0"],
        "Prior failed analysis must remain recorded",
    )
    for job in ("3520353", "3520355", "3520356", "3533911"):
        require(
            rows[job][3] == "COMPLETED"
            and rows[job][6] == "0:0"
            and int(rows[job][5]) == 1,
            "Scientific jobs must be terminal",
        )
    queue = retained_command(
        out, "owned-queue", ["squeue", "--me", "-h", "-o", "%i|%a|%j"]
    )
    for line in queue.splitlines():
        values = line.split("|")
        require(
            len(values) == 3
            and values[1] == "CCR26063"
            and (values[0] == context["job_id"] and values[2] == "vg-batched-analysis"),
            "Unreconciled owned queued/running work",
        )
    write(
        out / "budget-admission.json",
        {
            "original_conservative_su": total,
            "e5_reserved_su": 45,
            "m4_repair_reserved_su": 18,
            "this_analysis_max_su": 1,
            "prior_analysis_reserved_su": 1,
            "later_analysis_max_su": 1,
            "shared_analysis_max_su": 3,
            "campaign_bound_su": 102,
            "approved_ceiling_su": 105,
            "no_additional_reservation_granted": True,
        },
    )


def verify_payloads(stage, inputs, out):
    require(
        {job["job_id"] for job in inputs["jobs"]}
        == {
            "3520038",
            "3520048",
            "3520049",
            "3520050",
            "3520051",
            "3520353",
            "3520355",
            "3520356",
            "3533911",
        }
        and len(inputs["jobs"]) == 9,
        "Nine exact scientific jobs required",
    )
    verified = {"status": "PASS_RAW_ARCHIVE_AND_PERSISTENT_MEMBER_LINEAGE", "jobs": {}}
    contracts = {}
    for name in ("original-contract.json", "e5-contract.json", "repair-contract.json"):
        original = read(stage / "configs" / name)
        original_root = Path(original["source_path"])
        require(
            subprocess.check_output(
                ["git", "-C", str(original_root), "rev-parse", "HEAD"], text=True
            ).strip()
            == original["execution_source_revision"],
            "Frozen experiment HEAD differs",
        )
        for member, pin in original["source_files"].items():
            require(
                digest(original_root / member) == pin,
                "Frozen scientific source byte differs",
            )
        contracts[digest(stage / "configs" / name)] = original
    for job in inputs["jobs"]:
        check_deadline()
        root = Path(job["payload_root"])
        members = job["members"]
        proof = read(
            stage / "configs" / job["ranch_proof_file"], job["ranch_proof_sha256"]
        )
        require(
            proof["files_verified"] == len(members)
            and {
                name: {"bytes": proof["member_bytes"][name], "sha256": pin}
                for name, pin in proof["sha256"].items()
            }
            == members
            and proof["archive_sha256"] == job["full_archive_sha256"]
            and proof["archive_bytes"] == job["full_archive_bytes"],
            "Historical persistent membership differs",
        )
        actual = {
            str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
        }
        require(actual == set(members), "Raw payload census differs")
        for name, item in members.items():
            p = root / name
            require(
                not p.is_symlink()
                and p.stat().st_size == item["bytes"]
                and digest(p) == item["sha256"],
                "Raw member differs:" + name,
            )
        archive = Path(job["full_archive"])
        require(
            not archive.is_symlink()
            and archive.stat().st_size == job["full_archive_bytes"]
            and digest(archive) == job["full_archive_sha256"],
            "Frozen full archive differs",
        )
        seen = {}
        with tarfile.open(archive, "r|") as tar:
            for member in tar:
                require(
                    member.isfile()
                    and member.name in members
                    and member.name not in seen,
                    "Archive member census differs",
                )
                h = hashlib.sha256()
                with tar.extractfile(member) as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        check_deadline()
                        h.update(block)
                seen[member.name] = {"bytes": member.size, "sha256": h.hexdigest()}
        require(seen == members, "All-member archive verification failed")
        native = read(root / "control/model-work-release.json")
        require(
            native.get("status") == "PASS"
            and native.get("job_id") == job["job_id"]
            and native.get("account") == "CCR26063"
            and native.get("source_revision") == job["source_revision"]
            and native.get("contract_sha256") == job["contract_sha256"]
            and native.get("torch") == "2.13.0+cu126"
            and native.get("model_work_release") is True,
            "Original native lineage differs",
        )
        fixture = (root / "control/fixtures.log").read_text()
        require(
            digest(root / "control/fixtures.log") == native["fixture_receipt_sha256"]
            and ("116 passed" if job["job_id"].startswith("352035") else "102 passed")
            in fixture
            and " skipped" not in fixture,
            "Original native fixture proof differs",
        )
        require(
            job["contract_sha256"] in contracts
            and read(root / "control/contract.json")
            == contracts[job["contract_sha256"]],
            "Original saved contract differs",
        )
        if job["job_id"] == "3533911":
            repair = native["m4_repair_preflight"]
            require(
                repair["status"] == "PASS"
                and repair["clean_controls"] == 4
                and repair["full_state_includes_rng"] is True
                and repair["fixture"]
                == {"passed": 10, "failed": 0, "skipped": 0, "cuda_placement_passed": 2}
                and repair["original_fixture"]
                == {"passed": 102, "failed": 0, "skipped": 0},
                "M4 strict native repair controls differ",
            )
            require(
                digest(root / "control/reference-fixtures.log")
                == native["reference_fixture_receipt_sha256"],
                "M4 native reference fixture receipt differs",
            )
        verified["jobs"][job["job_id"]] = {
            "payload_root": job["payload_root"],
            "members_verified": len(members),
            "ranch_proof_sha256": job["ranch_proof_sha256"],
            "member_manifest_sha256": hashlib.sha256(
                json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "original_native_states": native["states"],
        }
    for item in inputs["initial_inputs"]:
        p = Path(item["path"])
        require(
            not p.is_symlink()
            and p.stat().st_size == item["bytes"]
            and digest(p) == item["sha256"],
            "Initial/data opaque hash differs",
        )
    write(out / "all-input-byte-lineage.json", verified)
    return verified


def initial_semantics(inputs, verification, out):
    """Trusted allocated serialized inspection; never instantiates/restores models."""
    import torch

    require(torch.__version__ == "2.13.0+cu126", "Pinned checkpoint reader differs")
    expected_roles = {
        "early-history1234": 10,
        "corrected210-history1234": 210,
        "corrected210-history1": 210,
        "corrected210-history2": 210,
    }
    expected = None
    for proof in verification["jobs"].values():
        require(
            len(proof["original_native_states"]) == 4, "Duplicate original native role"
        )
        states = {item["role"]: item for item in proof["original_native_states"]}
        require(
            set(states) == set(expected_roles),
            "Original native four-role semantic census differs",
        )
        if expected is None:
            expected = states
        else:
            require(
                states == expected,
                "Original native role identities/counters disagree across scientific jobs",
            )
    results = {}
    input_roles = [
        item["role"]
        for item in inputs["initial_inputs"]
        if item["role"] in expected_roles
    ]
    require(
        len(input_roles) == 4 and set(input_roles) == set(expected_roles),
        "Exact four serialized inputs required",
    )
    for item in inputs["initial_inputs"]:
        if item["role"] not in expected_roles:
            continue
        check_deadline()
        saved = torch.load(item["path"], map_location="cpu", weights_only=False)
        role = item["role"]
        native = expected[role]
        required = {
            "schema",
            "identity",
            "updates",
            "cursor",
            "model",
            "optimizer",
            "scheduler",
            "rng",
            "model_config",
            "recipe",
        }
        require(
            required <= set(saved)
            and type(saved["schema"]) is int
            and saved["schema"] == 1
            and type(saved["updates"]) is int
            and type(saved["cursor"]) is int
            and saved["updates"] == native["updates"] == expected_roles[role]
            and saved["cursor"] == native["cursor"] == expected_roles[role] * 65536
            and saved["identity"] == native["identity"],
            "Frozen expected serialized role/schema/identity/counters differ",
        )
        for key in ("model", "optimizer", "scheduler", "rng", "model_config", "recipe"):
            require(
                isinstance(saved[key], dict) and bool(saved[key]),
                "Required serialized object absent:" + key,
            )
        results[role] = {
            "updates": saved["updates"],
            "cursor": saved["cursor"],
            "identity": saved["identity"],
            "source_sha256": item["sha256"],
            "serialized_required_fields_present": True,
            "live_model_restore": False,
            "process_rng_restore": False,
        }
        del saved
    require(set(results) == set(expected_roles), "Four-role inspected census differs")
    write(out / "initial-serialized-role-checks.json", results)
    return results


def issue(stage, out, contract, context, scope, name):
    record = {
        "status": "PASS",
        "scope": scope,
        "job_id": context["job_id"],
        "account": "CCR26063",
        "source_packet_sha256": contract["source_packet_sha256"],
        "contract_sha256": os.environ["ACI_ANALYSIS_CONTRACT_SHA256"],
        "inputs_sha256": contract["inputs_sha256"],
        "capability_profile": "E1_E5_RECORD_ANALYSIS_ONLY",
        "whole_program_semantic_admission": False,
        "live_model_restore": False,
        "process_rng_restore": False,
    }
    path = out / name
    write(path, record)
    os.environ["ACI_ANALYSIS_RELEASE"] = str(path)


def fixtures(stage, out, contract):
    modules = contract["fixture_modules"]
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)

    def flatten(test):
        if isinstance(test, unittest.TestSuite):
            for child in test:
                yield from flatten(child)
        else:
            yield test.id()

    ids = list(flatten(suite))
    require(
        len(ids) == len(set(ids)) and set(ids) == set(contract["fixture_ids"]),
        "Exact fixture definition census differs",
    )
    with (out / "fixtures.log").open("x") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    proof = {
        "tests_run": result.testsRun,
        "test_ids": ids,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
        "scope": "SYNTHETIC_ALLOCATED_METHOD_FIXTURES_ONLY",
    }
    write(out / "fixture-result.json", proof)
    require(
        result.wasSuccessful() and result.testsRun == len(ids) and not result.skipped,
        "Exact allocated fixtures failed/skipped",
    )


def retain(out, job, contract):
    """Persist this allocation's own bounded outputs; never changes old evidence."""
    global CUTOFF
    CUTOFF = None
    signal.setitimer(signal.ITIMER_REAL, 0)
    for signum in (signal.SIGALRM, signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum, signal.SIG_IGN)
    fresh = out / "retention-storage"
    fresh.mkdir()
    storage_gate(fresh)
    target = WORK / "completion" / contract["campaign_id"] / job
    target.mkdir(parents=True, exist_ok=False)
    files = [path for path in out.rglob("*") if path.is_file()]
    require(not any(path.is_symlink() for path in files), "Output symlink forbidden")
    require(
        sum(path.stat().st_size for path in files) <= 200_000_000
        and len(files) <= 2000,
        "Output reservation exceeded; originals retained",
    )
    manifest = {}
    for path in files:
        relative = path.relative_to(out)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        pin = digest(path)
        require(digest(destination) == pin, "Durable output byte differs")
        manifest[str(relative)] = {"sha256": pin, "bytes": path.stat().st_size}
    receipt = {
        "status": "VERIFIED_OUTPUT_CUSTODY_PENDING_INDEPENDENT_ACTUAL_REVIEW",
        "job_id": job,
        "source": str(out),
        "durable": str(target),
        "files": manifest,
        "source_removed": False,
        "scientific_acceptance": False,
    }
    write(out / "output-custody.json", receipt)
    write(target / "output-custody.json", receipt)


def interrupted(signum, frame):
    """Control cancellation must escape unittest and recoverable section errors."""
    raise KeyboardInterrupt(
        "Processing interrupted; preserve failure and bounded outputs"
    )


def main():
    for signum in (signal.SIGALRM, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, interrupted)
    stage = Path(os.environ["ACI_SUBSET_STAGE"])
    job = os.environ.get("SLURM_JOB_ID", "")
    require(
        sys.platform == "linux" and job.isdecimal(),
        "Real Slurm Linux context required before output creation",
    )
    out = ROOT / "artifacts" / job / "batched-analysis"
    out.mkdir(parents=True, exist_ok=False)
    contract = read(stage / "contract.json", os.environ["ACI_ANALYSIS_CONTRACT_SHA256"])
    release = read(
        stage / "execution-release.json",
        os.environ["ACI_SUBSET_EXECUTION_RELEASE_SHA256"],
    )
    require(
        release["verdict"] == "PROMOTE"
        and release["scope"] == "PRECHECK_ONLY"
        and release["contract_sha256"] == os.environ["ACI_ANALYSIS_CONTRACT_SHA256"]
        and release["source_packet_sha256"] == contract["source_packet_sha256"],
        "Exact independent subset release absent",
    )
    os.environ["ACI_ANALYSIS_SOURCE_SHA256"] = contract["source_packet_sha256"]
    os.environ["ACI_ANALYSIS_INPUTS_SHA256"] = contract["inputs_sha256"]
    try:
        source_gate(stage, contract)
        context = physical_context(stage, out, contract, release)
        write(out / "physical-admission.json", context)
        for name in ("contract.json", "execution-release.json", "source-manifest.json"):
            with (
                (stage / name).open("rb") as incoming,
                (out / name).open("xb") as outgoing,
            ):
                shutil.copyfileobj(incoming, outgoing)
        source_packet = read(stage / "source-manifest.json")
        with tarfile.open(
            out / "exact-analysis-source-and-config.tar.gz", "x:gz"
        ) as archive:
            for name in source_packet["files"]:
                archive.add(stage / name, arcname=name, recursive=False)
        write(
            out / "lineage.json",
            {
                "source_packet_sha256": contract["source_packet_sha256"],
                "contract_sha256": os.environ["ACI_ANALYSIS_CONTRACT_SHA256"],
                "execution_release_sha256": os.environ[
                    "ACI_SUBSET_EXECUTION_RELEASE_SHA256"
                ],
                "inputs_sha256": contract["inputs_sha256"],
                "scope": "E1_E5_RECORD_ANALYSIS_ONLY",
                "old_whole_program_semantic_admission": False,
            },
        )
        storage_gate(out)
        accounting_gate(stage, out, context)
        inputs = read(stage / "configs/nine-job-inputs.json", contract["inputs_sha256"])
        verification = verify_payloads(stage, inputs, out)
        import matplotlib
        import numpy
        import scipy
        import torch

        require(
            (
                numpy.__version__,
                scipy.__version__,
                torch.__version__,
                matplotlib.__version__,
            )
            == ("2.5.2", "1.18.1", "2.13.0+cu126", "3.11.2"),
            "Locked analysis runtime differs",
        )
        write(
            out / "runtime-versions.json",
            {
                "python": sys.version,
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
                "matplotlib": matplotlib.__version__,
                "node": context["hostname"],
            },
        )
        issue(
            stage,
            out,
            contract,
            context,
            "ALLOCATED_FIXTURES_ONLY",
            "fixture-release.json",
        )
        fixtures(stage, out, contract)
        repeated = out / "post-fixture-admission"
        repeated.mkdir()
        source_gate(stage, contract)
        context = physical_context(stage, repeated, contract, release)
        check_deadline()
        initial = initial_semantics(inputs, verification, out)
        storage_gate_name = out / "pre-analysis-storage"
        storage_gate_name.mkdir()
        storage_gate(storage_gate_name)
        issue(
            stage,
            out,
            contract,
            context,
            "ALLOCATED_ANALYSIS",
            "subset-analysis-release.json",
        )
        from evidence_loading import PayloadReader
        from batch_analysis import run

        readers = {
            item["job_id"]: PayloadReader(item, verification) for item in inputs["jobs"]
        }
        results = run(stage, out, readers, initial, inputs)
        write(
            out / "terminal.json",
            {
                "status": results["status"],
                "job_id": job,
                "scientific_acceptance": False,
                "at_utc": datetime.now(UTC).isoformat(),
            },
        )
        require(not results["errors"], "Actual data validation issues retained")
    except BaseException:
        write(
            out / "failure.json",
            {
                "status": "FAILED_RETAINED_NO_RETRY",
                "traceback": traceback.format_exc(),
                "scientific_acceptance": False,
            },
        )
        raise
    finally:
        retain(out, job, contract)


if __name__ == "__main__":
    main()
