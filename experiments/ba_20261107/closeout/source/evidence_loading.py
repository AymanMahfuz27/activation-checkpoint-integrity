"""Allocated-only ingestion of verified completion members, with partial tails.

Only exact manifest members are eligible. This module never imports a model,
performs training/scoring, opens a network connection or changes source evidence.
"""

import hashlib
import json

from analysis_core import ARMS, HISTORIES, require_allocation
from input_binding import SCRATCH, _unique, _verify_file, read_json


def _decode(raw):
    return json.loads(
        raw,
        object_pairs_hook=_unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("Nonfinite JSON: " + value)
        ),
    )


def journal_bytes(raw):
    """Retain every valid row and record an incomplete final write explicitly.

    Unterminated but parseable final JSON is preserved as uncommitted candidate
    bytes. It is not admitted as a committed journal row. Interior malformed
    records fail ingestion instead of silently shortening the population.
    """
    require_allocation()
    rows, partial = [], None
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        final = index == len(lines) - 1
        if final and not line.endswith(b"\n"):
            try:
                candidate = _decode(line)
            except (ValueError, UnicodeDecodeError):
                candidate = None
            partial = {
                "status": "UNCOMMITTED_OR_INCOMPLETE_FINAL_WRITE",
                "bytes": len(line),
                "sha256": hashlib.sha256(line).hexdigest(),
                "decoded_text": line.decode("utf-8", errors="replace"),
                "parseable_candidate": candidate,
            }
            break
        if not line.strip():
            raise ValueError("Blank/interior journal record cannot be silently skipped")
        row = _decode(line)
        if not isinstance(row, dict):
            raise ValueError("Journal record must be an object")
        rows.append(row)
    return {
        "rows": rows,
        "partial_final_write": partial,
        "source_bytes": len(raw),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }


class PayloadReader:
    def __init__(self, job, verification):
        require_allocation()
        checked = verification.get("jobs", {}).get(job["job_id"], {})
        if (
            verification.get("status")
            != "PASS_RAW_ARCHIVE_AND_PERSISTENT_MEMBER_LINEAGE"
            or checked.get("payload_root") != job["payload_root"]
            or checked.get("members_verified") != len(job["members"])
            or checked.get("ranch_proof_sha256") != job["ranch_proof_sha256"]
            or checked.get("member_manifest_sha256")
            != hashlib.sha256(
                json.dumps(
                    job["members"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        ):
            raise ValueError("Payload reader requires successful exact input gate")
        from pathlib import Path

        self.root = Path(job["payload_root"])
        self.members = job["members"]
        self.job = job["job_id"]

    def file(self, name, *, required=True):
        require_allocation()
        if name not in self.members:
            if required:
                raise FileNotFoundError("Registered member absent: " + name)
            return None
        item = self.members[name]
        return _verify_file(
            self.root / name, item["sha256"], root=self.root, size=item["bytes"]
        )

    def json(self, name, *, required=True):
        path = self.file(name, required=required)
        return read_json(path) if path is not None else None

    def journal(self, name):
        path = self.file(name, required=False)
        if path is None:
            return {
                "rows": [],
                "partial_final_write": None,
                "status": "NOT_RECORDED",
                "source_bytes": None,
            }
        return {
            **journal_bytes(path.read_bytes()),
            "member": name,
            "status": "RETAINED",
        }

    def tensor(self, name):
        path = self.file(name)
        import torch

        # Only trusted, registered project checkpoints verified above are loaded.
        return torch.load(path, map_location="cpu", weights_only=False)


def load_original(readers, registration):
    require_allocation()
    m4, e3 = readers["3520038"], readers["3520048"]
    e1_rows, journals = [], []
    for name in sorted(m4.members):
        if name.startswith("E1/worker-") and name.endswith("/attempts.jsonl"):
            parsed = m4.journal(name)
            e1_rows.extend(parsed["rows"])
            journals.append(parsed)
    e2_members = [name for name in m4.members if name.startswith("E2/")]
    import re

    expected = {
        f"E3-{cell['policy']}-S{cell['seed']}-{cell['onset']}" for cell in registration
    }
    locations = {}
    for name in e3.members:
        if name.startswith("E3/worker-") and name.endswith(
            ("/summary.json", "/attempts.jsonl")
        ):
            match = re.fullmatch(
                r"E3/worker-[012]/([^/]+)/(summary\.json|attempts\.jsonl)", name
            )
            if match is None or match[1] not in expected:
                raise ValueError("Foreign/unmapped E3 cell member: " + name)
            key = (match[1], match[2])
            if key in locations:
                raise ValueError("Duplicate E3 cell member locations: " + name)
            locations[key] = name
    cells = []
    for cell in registration:
        identity = f"E3-{cell['policy']}-S{cell['seed']}-{cell['onset']}"
        summary_name = locations.get((identity, "summary.json"))
        journal_name = locations.get((identity, "attempts.jsonl"))
        if summary_name is None and journal_name is None:
            continue
        if (
            summary_name is not None
            and journal_name is not None
            and summary_name.rsplit("/", 1)[0] != journal_name.rsplit("/", 1)[0]
        ):
            raise ValueError(
                "E3 summary/journal split across worker locations: " + identity
            )
        parsed = (
            e3.journal(journal_name)
            if journal_name is not None
            else {"rows": [], "partial_final_write": None, "status": "NOT_RECORDED"}
        )
        cells.append(
            {
                "cell_id": identity,
                "summary": e3.json(summary_name) if summary_name is not None else None,
                "attempts": parsed["rows"],
                "journal_evidence": parsed,
            }
        )
    # Keep every baseline/ABFT JSON output available for its separate census.
    baselines = {
        name: e3.json(name)
        for name in sorted(e3.members)
        if name.startswith("E3/baselines/") and name.endswith(".json")
    }
    baseline_journals = {
        name: e3.journal(name)
        for name in sorted(e3.members)
        if name.startswith("E3/baselines/") and name.endswith(".jsonl")
    }
    return {
        "e1_rows": e1_rows,
        "e1_journals": journals,
        "e2_members": e2_members,
        "e3_cells": cells,
        "baseline_json": baselines,
        "baseline_journals": baseline_journals,
    }


def load_e4(readers):
    require_allocation()
    pairs, arms, setup, journals = [], [], {}, []
    for job in ("3520049", "3520050", "3520051"):
        reader = readers[job]
        p, a = reader.journal("E4/pairs.jsonl"), reader.journal("E4/arms.jsonl")
        pairs.extend(p["rows"])
        arms.extend(a["rows"])
        journals.extend((p, a))
        setup[job] = reader.json("E4/setup.json", required=False)
    if any(row["partial_final_write"] is not None for row in journals):
        raise ValueError(
            "E4 complete timing census has an uncommitted final journal record"
        )
    return {"pairs": pairs, "arms": arms, "setup": setup, "journal_evidence": journals}


def load_e5(readers):
    require_allocation()
    records, histories, finals, load_issues = [], {}, {}, []
    for history, job in zip(HISTORIES, ("3520353", "3520355", "3520356")):
        reader = readers[job]
        setup = reader.json("E5/setup.json", required=False)
        if setup is not None and setup.get("history") != history:
            raise ValueError("E5 setup history differs from fixed job mapping")
        parity = reader.json("E5/parity.json", required=False)
        histories[history] = {"setup": setup, "parity": parity or {}, "job": job}
        finals[history] = {}
        for arm in ARMS:
            root = "E5/" + arm + "/"
            summary = reader.json(root + "summary.json", required=False)
            evaluations, attempts = (
                reader.journal(root + "evaluations.jsonl"),
                reader.journal(root + "attempts.jsonl"),
            )
            if summary is not None and (
                summary.get("history") != history or summary.get("arm") != arm
            ):
                raise ValueError(
                    "E5 summary history/arm differs from registered location"
                )
            record = {
                "history": history,
                "arm": arm,
                "summary": summary,
                "evaluations": evaluations["rows"],
                "attempts": attempts["rows"],
                "journal_evidence": {"evaluations": evaluations, "attempts": attempts},
                "terminal_evidence": {
                    "failure_json": reader.json(root + "failure.json", required=False)
                    or {},
                    "journal_partial_writes": {
                        "evaluations": evaluations["partial_final_write"],
                        "attempts": attempts["partial_final_write"],
                    },
                },
            }
            records.append(record)
            if summary is None:
                continue
            try:
                state_member, buffer_member = (
                    root + "final-state.pt",
                    root + "final-buffers.pt",
                )
                if (
                    state_member not in reader.members
                    or buffer_member not in reader.members
                ):
                    raise FileNotFoundError("Final state or named buffers not recorded")
                if (
                    reader.members[state_member]["sha256"]
                    != summary["final_state_sha256"]
                ):
                    raise ValueError(
                        "Final-state summary hash differs from actual member"
                    )
                finals[history][arm] = (
                    reader.tensor(state_member),
                    reader.tensor(buffer_member),
                )
            except Exception as error:
                load_issues.append(
                    {
                        "history": history,
                        "arm": arm,
                        "stage": "FINAL_RETAINED_STATE_LOAD",
                        "error": type(error).__name__ + ": " + str(error),
                        "source_status_retained": summary.get("status"),
                    }
                )
    return {
        "records": records,
        "histories": histories,
        "finals": finals,
        "load_issues": load_issues,
    }


def load_initial_counters(items):
    """Load only registered starting checkpoints; never instantiate a model."""
    require_allocation()
    import torch

    expected = {
        "early-history1234",
        "corrected210-history1234",
        "corrected210-history1",
        "corrected210-history2",
    }
    if len(items) != 4 or {item["role"] for item in items} != expected:
        raise ValueError(
            "All four frozen starting checkpoints required for analysis counters"
        )
    result = {}
    for item in items:
        path = _verify_file(
            item["path"], item["sha256"], root=SCRATCH, size=item["bytes"]
        )
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (
            saved.get("schema") != 1
            or type(saved.get("updates")) is not int
            or type(saved.get("cursor")) is not int
        ):
            raise ValueError("Starting checkpoint counter schema differs")
        result[item["role"]] = {
            "updates": saved["updates"],
            "cursor": saved["cursor"],
            "source_sha256": item["sha256"],
        }
    return result
