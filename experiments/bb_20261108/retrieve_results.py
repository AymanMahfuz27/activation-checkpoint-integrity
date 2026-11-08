"""Retrieve one terminal job and verify its saved durable-custody manifest.

This is byte verification and metadata retrieval, with no statistical reduction.
An incomplete job or missing durable receipt stops collection without retry.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import secrets
import subprocess
import tarfile


ROOT = "/scratch/09999/aymanmahfuz/activation-checkpoint-integrity"


def safe_member(name):
    """Known custody names are canonical relative ASCII, without SFTP glob syntax."""
    if not isinstance(name, str) or not name.isascii():
        raise RuntimeError("Custody member must be ASCII text")
    member = PurePosixPath(name)
    if (
        member.is_absolute()
        or ".." in member.parts
        or not member.parts
        or str(member) != name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
        or any(character in name for character in "*?[]\\")
    ):
        raise RuntimeError("Unsafe custody member")
    return member


def extract_envelope(raw, marker):
    """Terminal chatter cannot become command output without one exact frame."""
    matches = [
        line[len(marker) :]
        for line in raw.decode("utf-8", errors="replace").splitlines()
        if line.startswith(marker)
    ]
    if len(matches) != 1:
        raise RuntimeError("Missing or duplicate remote-text frame")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate remote-text JSON key")
            result[key] = value
        return result

    envelope = json.loads(matches[0], object_pairs_hook=unique)
    required = {"returncode", "stdoutbase64", "stderrbase64"}
    if (
        not isinstance(envelope, dict)
        or not required <= set(envelope)
        or set(envelope) - required - {"timed_out"}
        or type(envelope.get("returncode")) is not int
        or any(
            not isinstance(envelope.get(key), str)
            for key in ("stdoutbase64", "stderrbase64")
        )
        or ("timed_out" in envelope and type(envelope["timed_out"]) is not bool)
    ):
        raise RuntimeError("Invalid remote-text envelope schema")
    return envelope


def remote_text(ssh, argv, destination, label):
    """Capture one argv command remotely and separate its bytes from PTY notices.

    Raw SSH streams, the nonce envelope and both decoded payloads are retained.
    A timeout or nonzero remote/transport status fails without retrying a call.
    """
    marker = "ACI_REMOTE_TEXT_" + secrets.token_hex(24) + ":"
    script = """import base64,json,subprocess
argv=ARGV
marker=MARKER
try:
    result=subprocess.run(argv,capture_output=True,timeout=30,check=False)
    envelope={'returncode':result.returncode,'stdoutbase64':base64.b64encode(result.stdout).decode('ascii'),'stderrbase64':base64.b64encode(result.stderr).decode('ascii')}
except subprocess.TimeoutExpired as error:
    envelope={'returncode':124,'timed_out':True,'stdoutbase64':base64.b64encode(error.stdout or b'').decode('ascii'),'stderrbase64':base64.b64encode(error.stderr or b'').decode('ascii')}
print('\\n'+marker+json.dumps(envelope),flush=True)
""".replace("ARGV", repr(list(argv)), 1).replace("MARKER", repr(marker), 1)
    request = {
        "argv": list(argv),
        "response_marker": marker,
        "remote_timeout_seconds": 30,
        "transport_timeout_seconds": 40,
        "automatic_retry": False,
    }
    (destination / (label + ".request.json")).write_text(
        json.dumps(request, indent=2) + "\n"
    )
    try:
        result = subprocess.run(
            ssh + ["python3 -c " + shlex.quote(script)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=40,
        )
    except subprocess.TimeoutExpired as error:
        (destination / (label + ".transport.stdout.raw")).write_bytes(
            error.stdout or b""
        )
        (destination / (label + ".transport.stderr.raw")).write_bytes(
            error.stderr or b""
        )
        (destination / (label + ".transport-error.json")).write_text(
            json.dumps(
                {
                    "status": "TRANSPORT_TIMEOUT_COMMAND_OUTCOME_UNKNOWN",
                    "automatic_retry": False,
                },
                indent=2,
            )
            + "\n"
        )
        raise
    (destination / (label + ".transport.stdout.raw")).write_bytes(result.stdout)
    (destination / (label + ".transport.stderr.raw")).write_bytes(result.stderr)
    (destination / (label + ".transport-returncode.txt")).write_text(
        str(result.returncode) + "\n"
    )
    envelope = extract_envelope(result.stdout, marker)
    (destination / (label + ".envelope.json")).write_text(
        json.dumps(envelope, indent=2) + "\n"
    )
    stdout = base64.b64decode(envelope["stdoutbase64"].encode("ascii"), validate=True)
    stderr = base64.b64decode(envelope["stderrbase64"].encode("ascii"), validate=True)
    (destination / (label + ".payload.stdout")).write_bytes(stdout)
    (destination / (label + ".payload.stderr")).write_bytes(stderr)
    if (
        result.returncode != 0
        or envelope["returncode"] != 0
        or envelope.get("timed_out") is True
    ):
        raise RuntimeError("Remote text command failed: " + label)
    return stdout.decode("utf-8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument(
        "study",
        choices=[
            "bb-selector",
            "bb-e1",
            "bb-selector-environment-setup",
            "bb-e1-environment-setup",
        ],
    )
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if not args.job.isdecimal():
        raise ValueError("Numeric job ID required")
    args.destination.mkdir(parents=True, exist_ok=False)
    orchestrator = Path(__file__).resolve()
    (args.destination / "retrieval-intent.json").write_text(
        json.dumps(
            {
                "job_id": args.job,
                "study": args.study,
                "automatic_retry": False,
                "orchestrator_path": str(orchestrator),
                "orchestrator_sha256": digest(orchestrator),
            },
            indent=2,
        )
        + "\n"
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
    accounting = remote_text(
        ssh,
        [
            "sacct",
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
    rows = [line.split("|") for line in accounting.splitlines() if line.strip()]
    terminal = {
        "COMPLETED",
        "FAILED",
        "TIMEOUT",
        "CANCELLED",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
    }
    if (
        len(rows) != 1
        or len(rows[0]) < 7
        or rows[0][0] != args.job
        or rows[0][1].upper() != "CCR26063"
        or not rows[0][3].split()
        or rows[0][3].split()[0] not in terminal
    ):
        raise RuntimeError("Exact owned job is not terminal; no data copied")
    source = f"{ROOT}/artifacts/{args.job}/{args.study}"
    receipt = remote_text(
        ssh,
        ["cat", source + "/output-custody.json"],
        args.destination,
        "source-receipt",
    )
    custody = json.loads(receipt)
    (args.destination / "saved-output-custody.json").write_text(receipt)
    if custody.get("source") != source or custody.get("source_removed") is not False:
        raise RuntimeError("Unexpected output custody source")
    durable = custody["durable"]
    expected_prefix = (
        "/work/09999/aymanmahfuz/ls6/activation-checkpoint-integrity/completion/"
    )
    if not durable.startswith(expected_prefix) or not durable.endswith("/" + args.job):
        raise RuntimeError("Durable custody leaves project namespace")
    actual = remote_text(
        ssh,
        ["cat", durable + "/output-custody.json"],
        args.destination,
        "durable-receipt",
    )
    if actual != receipt:
        raise RuntimeError("Scratch and durable custody receipts differ")
    extracted = args.destination / "extracted"
    extracted.mkdir()
    folder = extracted / args.study
    folder.mkdir()
    transfers = []
    for name in custody["files"]:
        safe_member(name)
        local = folder / name
        local.parent.mkdir(parents=True, exist_ok=True)
        transfers.append(
            "get "
            + json.dumps(source + "/" + name)
            + " "
            + json.dumps(str(local.resolve()))
        )
    # SFTP bypasses shell startup and transports binary bytes without a PTY.
    # Copy only declared members; no recursive traversal or remote mutation.
    # A native-build archive has hundreds of small files. Bound the transfer
    # by its declared member count, retaining a hard ceiling and partial logs.
    transfer_timeout = max(240, min(900, len(transfers) * 2))
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
            input="\n".join(transfers + ["quit"]) + "\n",
            capture_output=True,
            text=True,
            timeout=transfer_timeout,
        )
    except subprocess.TimeoutExpired as error:
        (args.destination / "sftp-timeout.stdout.raw").write_bytes(error.stdout or b"")
        (args.destination / "sftp-timeout.stderr.raw").write_bytes(error.stderr or b"")
        (args.destination / "sftp-timeout.json").write_text(
            json.dumps({"timeout_seconds": transfer_timeout,
                        "declared_members": len(transfers),
                        "status": "PARTIAL_TRANSFER_PRESERVED_NO_AUTOMATIC_RETRY"},
                       indent=2) + "\n"
        )
        raise
    (args.destination / "sftp-transfer.stdout").write_text(transfer.stdout)
    (args.destination / "sftp-transfer.stderr").write_text(transfer.stderr)
    if transfer.returncode:
        raise RuntimeError("Declared-member transfer failed; preserve partial bytes")
    (folder / "output-custody.json").write_text(receipt)
    for name, pin in custody["files"].items():
        safe_member(name)
        path = folder / name
        if (
            path.is_symlink()
            or path.stat().st_size != pin["bytes"]
            or digest(path) != pin["sha256"]
        ):
            raise RuntimeError("Copied output differs: " + name)
    archive_path = args.destination / "job-output.tar.gz"
    with tarfile.open(archive_path, "x:gz") as archive:
        archive.add(folder, arcname=args.study)
    # Preserve only declared, nonsecret shell controls. They live outside the
    # scientific folder because the final exit record is written by the shell.
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
    path=root/name
    if path.is_symlink() or not path.is_file() or path.stat().st_size>2000000:
        raise RuntimeError('Missing or oversized shell control: '+name)
    data=path.read_bytes()
    records[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                   'base64':base64.b64encode(data).decode('ascii')}
print(json.dumps(records))
""".replace("ROOT", repr(f"{ROOT}/artifacts/{args.job}")).replace("NAMES", repr(names))
    controls = remote_text(
        ssh, ["python3", "-c", control_script], args.destination, "shell-controls"
    )
    records = json.loads(controls)
    if sorted(records) != sorted(names):
        raise RuntimeError("Shell control census differs")
    control_folder = args.destination / "control-metadata"
    control_folder.mkdir()
    control_manifest = {}
    for name, record in records.items():
        data = base64.b64decode(record["base64"], validate=True)
        if (
            len(data) != record["bytes"]
            or hashlib.sha256(data).hexdigest() != record["sha256"]
        ):
            raise RuntimeError("Shell control transfer differs: " + name)
        (control_folder / name).write_bytes(data)
        control_manifest[name] = {"bytes": len(data), "sha256": record["sha256"]}
    (control_folder / "manifest.json").write_text(
        json.dumps(control_manifest, indent=2) + "\n"
    )
    verified = {
        "job_id": args.job,
        "native_accounting": rows[0],
        "status": "ALL_RECEIPTED_MEMBERS_BYTE_VERIFIED_LOCALLY",
        "members": len(custody["files"]),
        "archive_sha256": digest(archive_path),
        "archive_origin": "LOCAL_CONTAINER_OF_VERIFIED_SFTP_MEMBERS",
        "source": source,
        "durable": durable,
        "shell_controls_locally_verified": len(control_manifest),
        "scientific_acceptance": False,
    }
    (args.destination / "retrieval-verification.json").write_text(
        json.dumps(verified, indent=2) + "\n"
    )
    print(json.dumps(verified))


if __name__ == "__main__":
    main()
