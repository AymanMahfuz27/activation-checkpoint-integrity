"""Recover interrupted read-only custody collection into a new exclusive folder.

Adapted from independently reviewed BA source recovery helper (a5c190db).
No experiment is run or retried. Recheck owned terminal accounting and identical
original/Scratch/WORK receipts, reuse only verified partial members, then fetch
only missing/different members and verify every declared byte/hash.
"""

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[2]
DERIVED_FROM_SHA256 = "a5c190dbc2d037bef099a020734fac9c23812507ae62d50b3e4307a5c7e21b9b"
TERMINAL_STATES = (
    "COMPLETED",
    "FAILED",
    "TIMEOUT",
    "CANCELLED",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument(
        "study",
        choices=(
            "bb-selector",
            "bb-e1",
            "bb-selector-environment-setup",
            "bb-e1-environment-setup",
        ),
    )
    parser.add_argument("original", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected-state", required=True, choices=TERMINAL_STATES)
    parser.add_argument("--expected-exit", required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9]+", args.job):
        parser.error("job must contain ASCII decimal digits only")
    if not re.fullmatch(r"[0-9]+:[0-9]+", args.expected_exit):
        parser.error("expected exit must be the exact numeric Slurm code:signal")
    args.original = args.original.resolve()
    args.destination = args.destination.resolve()
    if (
        args.original == args.destination
        or args.original in args.destination.parents
        or args.destination in args.original.parents
    ):
        parser.error("original and destination must be disjoint directories")
    return args


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def accounting_row(text, args):
    rows = [line.split("|") for line in text.splitlines() if line.strip()]
    if (
        len(rows) != 1
        or len(rows[0]) != 7
        or rows[0][0] != args.job
        or rows[0][1].upper() != "CCR26063"
        or rows[0][3] != args.expected_state
        or rows[0][5] != "1"
        or rows[0][6] != args.expected_exit
    ):
        raise RuntimeError(
            "Exact owned terminal accounting differs from requested outcome"
        )
    return rows[0]


def regular_previous(previous, original_root, pin, collector):
    """Reject symlink members/parents; only verified regular bytes can be reused."""
    if not previous.is_file() or previous.is_symlink():
        return False
    for part in (previous.parent, *previous.parent.parents):
        if part.is_symlink():
            return False
        if part == original_root:
            break
    return (
        previous.resolve().is_relative_to(original_root)
        and previous.stat().st_size == pin["bytes"]
        and collector.digest(previous) == pin["sha256"]
    )


def main(argv=None):
    args = parse_args(argv)
    spec = importlib.util.spec_from_file_location(
        "collector", REPO / "experiments/bb_20261108/retrieve_results.py"
    )
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    if not args.original.is_dir():
        raise RuntimeError("Original partial collection is absent")
    original_receipt = args.original / "saved-output-custody.json"
    if original_receipt.is_symlink() or not original_receipt.is_file():
        raise RuntimeError("Original regular custody receipt is absent")
    original_bytes = original_receipt.read_bytes()
    original_accounting = accounting_row(
        (args.original / "native-accounting.txt").read_text(), args
    )
    args.destination.mkdir(exist_ok=False)
    write_json(
        args.destination / "retrieval-intent.json",
        {
            "job_id": args.job,
            "study": args.study,
            "automatic_retry": False,
            "expected_state": args.expected_state,
            "expected_exit": args.expected_exit,
            "scope": "READ_ONLY_CUSTODY_RECOVERY_NO_SCIENTIFIC_EXECUTION",
            "original_partial_collection": str(args.original),
            "original_receipt_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "orchestrator_sha256": collector.digest(__file__),
            "collector_sha256": collector.digest(Path(collector.__file__)),
            "derived_from_orchestrator_sha256": DERIVED_FROM_SHA256,
        },
    )
    socket = (
        f"/tmp/tacc-control-{os.getuid()}/ctrl-ls6-activation-checkpoint-integrity.sock"
    )
    ssh = [
        "ssh",
        "-tt",
        "-S",
        socket,
        "-o",
        "BatchMode=yes",
        "aymanmahfuz@ls6.tacc.utexas.edu",
    ]
    accounting = collector.remote_text(
        ssh,
        [
            "sacct",
            "--user=aymanmahfuz",
            "-X",
            "-j",
            args.job,
            "-n",
            "-P",
            "--format=JobIDRaw,Account,JobName,State,ElapsedRaw,NNodes,ExitCode",
        ],
        args.destination,
        "accounting",
    )
    (args.destination / "native-accounting.txt").write_text(accounting)
    row = accounting_row(accounting, args)
    if row != original_accounting:
        raise RuntimeError("Fresh and original terminal accounting rows differ")
    source = f"{collector.ROOT}/artifacts/{args.job}/{args.study}"
    receipt = collector.remote_text(
        ssh,
        ["cat", source + "/output-custody.json"],
        args.destination,
        "source-receipt",
    )
    if receipt.encode() != original_bytes:
        raise RuntimeError("Original and fresh source receipts differ")
    custody = json.loads(receipt)
    durable = custody["durable"]
    prefix = "/work/09999/aymanmahfuz/ls6/activation-checkpoint-integrity/completion/"
    if (
        custody["source"] != source
        or custody["source_removed"] is not False
        or not isinstance(durable, str)
        or not durable.startswith(prefix)
    ):
        raise RuntimeError("Custody namespace differs")
    durable_member = collector.safe_member(durable[len(prefix) :])
    if len(durable_member.parts) != 2 or durable_member.parts[-1] != args.job:
        raise RuntimeError("Durable campaign/job namespace differs")
    if not isinstance(custody["files"], dict) or not custody["files"]:
        raise RuntimeError("Declared custody member census is empty or malformed")
    for name, pin in custody["files"].items():
        collector.safe_member(name)
        if (
            not isinstance(pin, dict)
            or type(pin.get("bytes")) is not int
            or pin["bytes"] < 0
            or not isinstance(pin.get("sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", pin["sha256"])
        ):
            raise RuntimeError("Malformed custody member pin")
    durable_receipt = collector.remote_text(
        ssh,
        ["cat", durable + "/output-custody.json"],
        args.destination,
        "durable-receipt",
    )
    if durable_receipt != receipt:
        raise RuntimeError("Scratch and WORK receipts differ")
    (args.destination / "saved-output-custody.json").write_bytes(original_bytes)
    folder = args.destination / "extracted" / args.study
    folder.mkdir(parents=True)
    original_root = args.original / "extracted" / args.study
    reused, fetched, commands = [], [], []
    for name, pin in custody["files"].items():
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        previous = original_root / name
        if regular_previous(previous, original_root, pin, collector):
            shutil.copyfile(previous, target)
            reused.append(name)
        else:
            commands.append(
                "get " + json.dumps(source + "/" + name) + " " + json.dumps(str(target))
            )
            fetched.append(name)
    transfer_timeout = max(240, min(900, len(commands) * 2))
    provenance = {
        "reused_verified_members": reused,
        "newly_requested_members": fetched,
        "original_files_modified": False,
        "scientific_jobs_submitted": 0,
        "automatic_retry": False,
        "transfer_timeout_seconds": transfer_timeout,
        "source_receipt_sha256": hashlib.sha256(original_bytes).hexdigest(),
    }
    write_json(args.destination / "transfer-recovery.json", provenance)
    if commands:
        try:
            transfer = subprocess.run(
                [
                    "sftp",
                    "-b",
                    "-",
                    "-o",
                    "ControlPath=" + socket,
                    "-o",
                    "BatchMode=yes",
                    "aymanmahfuz@ls6.tacc.utexas.edu",
                ],
                input=("\n".join(commands + ["quit"]) + "\n").encode(),
                capture_output=True,
                timeout=transfer_timeout,
            )
        except BaseException as error:
            (args.destination / "sftp-exception.stdout.raw").write_bytes(
                getattr(error, "stdout", None) or b""
            )
            (args.destination / "sftp-exception.stderr.raw").write_bytes(
                getattr(error, "stderr", None) or b""
            )
            write_json(
                args.destination / "sftp-exception.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "timeout_seconds": transfer_timeout,
                    "automatic_retry": False,
                    "partial_files_preserved": True,
                },
            )
            raise
        (args.destination / "sftp-transfer.stdout.raw").write_bytes(transfer.stdout)
        (args.destination / "sftp-transfer.stderr.raw").write_bytes(transfer.stderr)
        (args.destination / "sftp-transfer.stdout").write_text(
            transfer.stdout.decode(errors="replace")
        )
        (args.destination / "sftp-transfer.stderr").write_text(
            transfer.stderr.decode(errors="replace")
        )
        write_json(
            args.destination / "sftp-transfer-result.json",
            {
                "returncode": transfer.returncode,
                "timeout_seconds": transfer_timeout,
                "requested_members": len(commands),
                "automatic_retry": False,
            },
        )
        if transfer.returncode:
            raise RuntimeError("Targeted SFTP transfer failed; partial bytes preserved")
    else:
        write_json(
            args.destination / "sftp-not-required.json", {"requested_members": 0}
        )
    for name, pin in custody["files"].items():
        target = folder / name
        if (
            target.is_symlink()
            or not target.is_file()
            or target.stat().st_size != pin["bytes"]
            or collector.digest(target) != pin["sha256"]
        ):
            raise RuntimeError("Member differs: " + name)
    if original_receipt.read_bytes() != original_bytes:
        raise RuntimeError("Original custody receipt changed during recovery")
    (folder / "output-custody.json").write_bytes(original_bytes)
    prefix = args.study.removesuffix("-environment-setup")
    suffixes = (
        ("-runner.log", "-modules.txt", "-exit-code.txt")
        if args.study.endswith("-environment-setup")
        else ("-runner.log", "-installed.txt", "-modules.txt", "-exit-code.txt")
    )
    names = [prefix + suffix for suffix in suffixes]
    control_script = """import base64,hashlib,json
from pathlib import Path
root=Path(ROOT)
records={}
for name in NAMES:
    p=root/name
    if p.is_symlink() or not p.is_file() or p.stat().st_size>2000000:
        raise RuntimeError('Missing or oversized shell control')
    data=p.read_bytes()
    records[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                   'base64':base64.b64encode(data).decode('ascii')}
print(json.dumps(records))
""".replace("ROOT", repr(f"{collector.ROOT}/artifacts/{args.job}")).replace(
        "NAMES", repr(names)
    )
    records = json.loads(
        collector.remote_text(
            ssh, ["python3", "-c", control_script], args.destination, "shell-controls"
        )
    )
    if sorted(records) != sorted(names):
        raise RuntimeError("Shell control census differs")
    controls = args.destination / "control-metadata"
    controls.mkdir()
    manifest = {}
    for name, record in records.items():
        data = base64.b64decode(record["base64"], validate=True)
        if (
            len(data) != record["bytes"]
            or hashlib.sha256(data).hexdigest() != record["sha256"]
        ):
            raise RuntimeError("Shell control transfer differs")
        (controls / name).write_bytes(data)
        manifest[name] = {"bytes": len(data), "sha256": record["sha256"]}
    write_json(controls / "manifest.json", manifest)
    archive = args.destination / "job-output.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        stream.add(folder, arcname=args.study)
    verification = {
        "job_id": args.job,
        "study": args.study,
        "native_accounting": row,
        "status": "ALL_RECEIPTED_MEMBERS_BYTE_VERIFIED_LOCALLY",
        "members": len(custody["files"]),
        "archive_sha256": collector.digest(archive),
        "archive_origin": "LOCAL_CONTAINER_OF_VERIFIED_SFTP_MEMBERS",
        "source": source,
        "durable": durable,
        "source_removed": False,
        "shell_controls": manifest,
        "read_only_recovery": provenance,
        "orchestrator_sha256": collector.digest(__file__),
        "collector_sha256": collector.digest(Path(collector.__file__)),
    }
    write_json(args.destination / "retrieval-verification.json", verification)
    print(
        json.dumps(
            {
                "status": verification["status"],
                "members": verification["members"],
                "reused": len(reused),
                "fetched": len(fetched),
                "archive_sha256": verification["archive_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
