"""Draft allocated input gate: hashes and full persistent member proof only.

The final manifest is not available until E5 terminal custody. This module does
not submit, copy, delete, authenticate, score, or load tensor contents.
"""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from analysis_core import require_allocation


SCRATCH = Path("/scratch/09999/aymanmahfuz/activation-checkpoint-integrity")
WORK = Path("/work/09999/aymanmahfuz/ls6/activation-checkpoint-integrity")
ORIGINAL = ("3520038", "3520048", "3520049", "3520050", "3520051")
SUPPLEMENTAL = ("3520353", "3520355", "3520356")
SOURCE = {
    **dict.fromkeys(ORIGINAL, "b23f86b2d9337fe5f28f035b07707ce0f56bda81"),
    **dict.fromkeys(SUPPLEMENTAL, "d1daebd4022e860bf2175ef3d4495b2ea5319c74"),
}
CONTRACT = {
    **dict.fromkeys(
        ORIGINAL, "8880e76d174dd22fea27852f648dd846034c41f3a937811e4cefd043e1d11406"
    ),
    **dict.fromkeys(
        SUPPLEMENTAL, "fd9fa291d28ae8b57819180c6586164ea2acb0b3c340b55c9ec9a042785fded8"
    ),
}
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=_unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("Nonfinite JSON constant: " + value)
        ),
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path, root):
    path = Path(path)
    if not path.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Evidence path leaves registered root")
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Evidence path contains a symlink")
    if path.resolve() != path or not path.is_file():
        raise ValueError("Evidence path is not a canonical regular file")
    return path


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected lowercase SHA256")
    return value


def _verify_file(path, digest, *, root, size=None):
    path = _regular(path, root)
    if size is not None and (
        type(size) is not int or size < 0 or path.stat().st_size != size
    ):
        raise ValueError("Evidence byte count differs: " + str(path))
    if sha256(path) != _digest(digest):
        raise ValueError("Evidence hash differs: " + str(path))
    return path


def verify_manifest(path, expected_sha256):
    """All eight outcomes must be terminal and have member-complete Ranch proof.

    Persistent proof is reused by exact saved hash; no repeated Ranch scan or
    download is performed. The allocation independently hashes raw Scratch
    members and its existing staged full archive against that same proof.
    """
    release = require_allocation()
    if release["inputs_sha256"] != expected_sha256:
        raise ValueError("Input digest differs from this job-bound release")
    return _verify_manifest_bytes(path, expected_sha256)


def verify_manifest_before_fixtures(stage, path, expected_sha256):
    """Opaque evidence only, after physical owned-job and reviewed-stage pins.

    This permits the byte gate before a fixture-only release, avoiding a
    circular requirement that fixtures already passed to verify their inputs.
    It grants no tensor, numerical, fixture or scientific execution scope.
    """
    from native_collection import collection_context

    context = collection_context(stage)
    if context["inputs_sha256"] != expected_sha256:
        raise ValueError("Input digest differs from native reviewed context")
    result = _verify_manifest_bytes(path, expected_sha256)
    return {
        **result,
        "verification_scope": "PREFIXTURE_OPAQUE_BYTES_ONLY",
        "analysis_job_id": context["job_id"],
        "source_packet_sha256": context["source_packet_sha256"],
        "contract_sha256": context["contract_sha256"],
        "execution_release_sha256": context["execution_release_sha256"],
        "fixture_execution_authorized": False,
        "tensor_semantics_verified": False,
        "scientific_acceptance": False,
    }


def _verify_manifest_bytes(path, expected_sha256):
    """Shared byte validator; entry points above own the execution-context gate."""
    path = _verify_file(path, expected_sha256, root=SCRATCH)
    manifest = read_json(path)
    if manifest.get("status") != "FROZEN_AFTER_ALL_MEMBER_CUSTODY":
        raise ValueError("Draft or partial custody cannot open analysis")
    jobs = manifest["jobs"]
    if len(jobs) != 8 or {job["job_id"] for job in jobs} != set(
        ORIGINAL + SUPPLEMENTAL
    ):
        raise ValueError("Analysis requires all eight exact registered outcomes")
    result = {}
    for job in jobs:
        job_id = job["job_id"]
        if (
            job["source_revision"] != SOURCE[job_id]
            or job["contract_sha256"] != CONTRACT[job_id]
            or job["scheduler_state"] not in TERMINAL
        ):
            raise ValueError("Outcome source/contract/terminal identity differs")
        campaign = (
            "e1e4-full-20261019-v2"
            if job_id in ORIGINAL
            else "e5-time-quality-20261019-v1"
        )
        source_root = SCRATCH / "artifacts" / job_id / "completion"
        if job["payload_root"] != str(source_root):
            raise ValueError("Unexpected raw payload root")
        proof_path = WORK / "completion" / campaign / job_id / "ranch-proof.json"
        proof = read_json(
            _verify_file(proof_path, job["ranch_proof_sha256"], root=WORK)
        )
        members = job["members"]
        if not members:
            raise ValueError("Empty registered evidence census")
        hashes, sizes = {}, {}
        for relative, record in members.items():
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
                raise ValueError("Unsafe registered member name")
            _verify_file(
                source_root / relative,
                record["sha256"],
                root=source_root,
                size=record["bytes"],
            )
            hashes[relative], sizes[relative] = record["sha256"], record["bytes"]
        if (
            proof.get("status") != "VERIFIED_PERSISTENT_FINAL_EVIDENCE_ARCHIVE"
            or proof.get("files_verified") != len(members)
            or proof.get("sha256") != hashes
            or proof.get("member_bytes") != sizes
            or proof.get("archive_sha256") != job["archive_sha256"]
            or proof.get("archive_bytes") != job["archive_bytes"]
            or proof.get("source_removal_released") is not False
        ):
            raise ValueError("Full persistent Ranch proof differs from input census")
        archive = (
            SCRATCH
            / "completion"
            / campaign
            / "archives"
            / job_id
            / "full-evidence.tar"
        )
        # Actual staging filenames are sealed explicitly; only this job's
        # existing archive directory is eligible, never a newly restored copy.
        archive = Path(job["scratch_archive_path"])
        allowed_archive_root = SCRATCH / "completion" / campaign / "archives" / job_id
        _verify_file(
            archive,
            job["archive_sha256"],
            root=allowed_archive_root,
            size=job["archive_bytes"],
        )
        result[job_id] = {
            "payload_root": str(source_root),
            "members_verified": len(members),
            "archive_sha256": job["archive_sha256"],
            "ranch_proof_sha256": job["ranch_proof_sha256"],
            "member_manifest_sha256": hashlib.sha256(
                json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    return {
        "status": "PASS_RAW_ARCHIVE_AND_PERSISTENT_MEMBER_LINEAGE",
        "inputs_sha256": expected_sha256,
        "jobs": result,
        "limit": "Scheduler/source/preflight receipt semantics require separate native admission",
    }
