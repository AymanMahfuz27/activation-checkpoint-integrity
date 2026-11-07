"""New allocated E1-E5 closeout execution, preserving historical scientific gates.

The new source/contract/review/batch must be sealed by the orchestrator. Old
once-only dispatch/release authorities are not consumed or reused. All evidence
loading, serialized inspection, estimators, and retention are inherited unchanged.
"""

import hashlib
import os
import shutil
import signal
import socket
import sys
import tarfile
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import run_subset as inherited


def physical_context(stage, out, contract, release):
    """Bind this process to the new reviewed one-node CPU allocation and batch."""
    job = os.environ["SLURM_JOB_ID"]
    text = inherited.retained_command(
        out, "owned-job", ["scontrol", "show", "job", job, "-o"]
    )
    fields = inherited.descriptor_fields(text)
    wanted = {
        "JobId": job,
        "UserId": "aymanmahfuz(893489)",
        "Account": "CCR26063",
        "Partition": "normal",
        "JobState": "RUNNING",
        "JobName": "vg-ba-closeout",
        "NumNodes": "1",
        "NumTasks": "1",
        "CPUs/Task": "12",
        "OverSubscribe": "NO",
        "Requeue": "0",
        "Restarts": "0",
        "TimeLimit": "01:00:00",
        "SuspendTime": "None",
        "SecsPreSuspend": "0",
        "Command": str(stage / "analysis.sbatch"),
    }
    inherited.require(
        os.getuid() == 893489 and all(fields.get(k) == v for k, v in wanted.items()),
        "New closeout owned CPU resources differ",
    )
    host = socket.gethostname().split(".")[0]
    nodes = inherited.retained_command(
        out, "owned-nodes", ["scontrol", "show", "hostnames", fields["NodeList"]]
    ).splitlines()
    inherited.require(nodes == [host], "Process host is outside owned allocation")
    actual = inherited.retained_command(
        out, "actual-batch", ["scontrol", "write", "batch_script", job, "-"]
    )
    inherited.require(
        hashlib.sha256(actual.encode()).hexdigest() == release["batch_sha256"],
        "Actual batch is not the independently reviewed new batch",
    )
    remaining = inherited.duration(
        inherited.retained_command(
            out, "time-left", ["squeue", "-h", "-j", job, "-o", "%L"]
        ).strip()
    )
    inherited.require(1200 <= remaining <= 3600, "Insufficient owned processing time")
    inherited.require(
        contract["campaign_id"] == "ba-closeout-20261107-v1"
        and contract["scope"] == "E1_E5_EXISTING_DATA_ONLY"
        and contract["max_su"] == 1
        and contract["wall_minutes"] == 60
        and contract["retry_authority"] is False
        and contract["new_training_or_scoring"] is False,
        "New bounded closeout authority differs",
    )
    inherited.CUTOFF = time.monotonic() + remaining - 600
    signal.setitimer(signal.ITIMER_REAL, remaining - 600)
    return {
        "job_id": job,
        "hostname": host,
        "account": "CCR26063",
        "partition": "normal",
        "time_left_seconds": remaining,
        "retention_reserve_seconds": 600,
        "this_allocation_max_su": 1,
        "historical_release_reused": False,
    }


def reconcile_queue(out, context):
    """Require the nine historical scientific jobs terminal and no duplicate work."""
    expected = {
        "3520038": ("FAILED", "1:0"),
        "3520048": ("COMPLETED", "0:0"),
        "3520049": ("COMPLETED", "0:0"),
        "3520050": ("COMPLETED", "0:0"),
        "3520051": ("COMPLETED", "0:0"),
        "3520353": ("COMPLETED", "0:0"),
        "3520355": ("COMPLETED", "0:0"),
        "3520356": ("COMPLETED", "0:0"),
        "3533911": ("COMPLETED", "0:0"),
        "3535049": ("FAILED", "1:0"),
        "3540071": ("FAILED", "1:0"),
    }
    accounting = inherited.retained_command(
        out,
        "historical-accounting",
        [
            "sacct",
            "-X",
            "-j",
            ",".join(expected),
            "-n",
            "-P",
            "--format=JobIDRaw,Account,State,ExitCode",
        ],
    )
    observed = {}
    for line in accounting.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split("|")
        inherited.require(
            len(parts) == 4
            and parts[0] in expected
            and parts[0] not in observed
            and parts[1].upper() == "CCR26063",
            "Historical accounting contains foreign/duplicate records",
        )
        observed[parts[0]] = tuple(parts[2:])
    inherited.require(observed == expected, "Historical terminal outcomes differ")
    queue = inherited.retained_command(
        out, "owned-queue", ["squeue", "--me", "-h", "-o", "%i|%a|%j"]
    )
    for line in queue.splitlines():
        inherited.require(
            line.strip().split("|")
            == [context["job_id"], "CCR26063", "vg-ba-closeout"],
            "Other owned work must be reconciled before closeout",
        )


def main():
    """Verify new execution identity, old data, fixtures, analysis, then custody."""
    job = os.environ.get("SLURM_JOB_ID", "")
    inherited.require(
        sys.platform == "linux" and job.isdecimal(), "Real Slurm Linux job required"
    )
    stage = Path(os.environ["ACI_SUBSET_STAGE"])
    contract = inherited.read(
        stage / "contract.json", os.environ["ACI_ANALYSIS_CONTRACT_SHA256"]
    )
    release = inherited.read(
        stage / "execution-release.json",
        os.environ["ACI_SUBSET_EXECUTION_RELEASE_SHA256"],
    )
    inherited.require(
        release["verdict"] == "PROMOTE"
        and release["scope"] == "PRECHECK_ONLY"
        and release["campaign_id"] == contract["campaign_id"]
        and release["contract_sha256"] == os.environ["ACI_ANALYSIS_CONTRACT_SHA256"]
        and release["source_packet_sha256"] == contract["source_packet_sha256"],
        "New independent closeout execution release is absent",
    )
    for signum in (signal.SIGALRM, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, inherited.interrupted)
    out = inherited.ROOT / "artifacts" / job / "ba-closeout"
    out.mkdir(parents=True, exist_ok=False)
    os.environ["ACI_ANALYSIS_SOURCE_SHA256"] = contract["source_packet_sha256"]
    os.environ["ACI_ANALYSIS_INPUTS_SHA256"] = contract["inputs_sha256"]
    try:
        inherited.source_gate(stage, contract)
        context = physical_context(stage, out, contract, release)
        inherited.write(out / "physical-admission.json", context)
        for name in ("contract.json", "execution-release.json", "source-manifest.json"):
            shutil.copyfile(stage / name, out / name)
        packet = inherited.read(stage / "source-manifest.json")
        with tarfile.open(
            out / "exact-analysis-source-and-config.tar.gz", "x:gz"
        ) as archive:
            for name in sorted(packet["files"]):
                archive.add(stage / name, arcname=name, recursive=False)
        inherited.write(
            out / "lineage.json",
            {
                "campaign_id": contract["campaign_id"],
                "source_packet_sha256": contract["source_packet_sha256"],
                "contract_sha256": os.environ["ACI_ANALYSIS_CONTRACT_SHA256"],
                "execution_release_sha256": os.environ[
                    "ACI_SUBSET_EXECUTION_RELEASE_SHA256"
                ],
                "inputs_sha256": contract["inputs_sha256"],
                "scope": "E1_E5_RECORD_ANALYSIS_ONLY",
                "historical_releases_reused": False,
                "scientific_acceptance": False,
            },
        )
        inherited.storage_gate(out)
        reconcile_queue(out, context)
        inputs = inherited.read(
            stage / "configs/nine-job-inputs.json", contract["inputs_sha256"]
        )
        verification = inherited.verify_payloads(stage, inputs, out)
        import matplotlib
        import numpy
        import scipy
        import torch

        versions = (
            numpy.__version__,
            scipy.__version__,
            torch.__version__,
            matplotlib.__version__,
        )
        inherited.require(
            versions == ("2.5.2", "1.18.1", "2.13.0+cu126", "3.11.2"),
            "Locked analysis runtime differs",
        )
        inherited.write(
            out / "runtime-versions.json",
            {
                "python": sys.version,
                "numpy": versions[0],
                "scipy": versions[1],
                "torch": versions[2],
                "matplotlib": versions[3],
                "node": context["hostname"],
            },
        )
        inherited.issue(
            stage,
            out,
            contract,
            context,
            "ALLOCATED_FIXTURES_ONLY",
            "fixture-release.json",
        )
        inherited.fixtures(stage, out, contract)
        repeated = out / "post-fixture-admission"
        repeated.mkdir()
        inherited.source_gate(stage, contract)
        context = physical_context(stage, repeated, contract, release)
        initial = inherited.initial_semantics(inputs, verification, out)
        storage = out / "pre-analysis-storage"
        storage.mkdir()
        inherited.storage_gate(storage)
        inherited.issue(
            stage, out, contract, context, "ALLOCATED_ANALYSIS", "analysis-release.json"
        )
        from batch_analysis import run
        from evidence_loading import PayloadReader

        readers = {
            item["job_id"]: PayloadReader(item, verification) for item in inputs["jobs"]
        }
        results = run(stage, out, readers, initial, inputs)
        inherited.write(
            out / "terminal.json",
            {
                "status": results["status"],
                "job_id": job,
                "scientific_acceptance": False,
                "at_utc": datetime.now(UTC).isoformat(),
                "historical_e1_validation_gates_preserved": True,
            },
        )
        inherited.require(
            not results["errors"],
            "Actual data validation issues retained; no blanket promotion",
        )
    except BaseException:
        inherited.write(
            out / "failure.json",
            {
                "status": "FAILED_RETAINED_NO_RETRY",
                "traceback": traceback.format_exc(),
                "scientific_acceptance": False,
            },
        )
        raise
    finally:
        inherited.retain(out, job, contract)


if __name__ == "__main__":
    main()
