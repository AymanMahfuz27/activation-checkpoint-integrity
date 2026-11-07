"""Fail-closed correspondence and noninterference checks for four fresh arms."""

import hashlib
import json
import os
from pathlib import Path


TAG = "ba-layer6-activation"
REVISION = "146d6bb523cf8364bbf2788cb319aef1a46f966b"
SHAPE = [4, 512, 3072]
NUMEL = 6291456
NBYTES = NUMEL * 4
EXPECTED_RANGES = [
    [0, 6291456],
    [6291456, 12582912],
    [12582912, 18874368],
    [18874368, 25165824],
]


def require(value, message):
    if not value:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def parse(raw):
    return json.loads(
        raw,
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("Nonfinite JSON")
        ),
    )


def read(path):
    path = Path(path)
    require(
        path.is_file() and not path.is_symlink(), "Regular retained record required"
    )
    return parse(path.read_bytes())


def signature(summary):
    require(
        isinstance(summary, dict) and not summary.get("note"),
        "Fallback or absent tensor summary",
    )
    stats = summary.get("stats")
    require(isinstance(stats, dict), "Tensor numerical statistics absent")
    require(
        summary.get("shape") == SHAPE
        and stats.get("dtype") == "torch.float32"
        and stats.get("device") == "cuda:0"
        and stats.get("numel") == NUMEL,
        "Configured activation tensor metadata differs",
    )
    require(
        stats.get("xor_sig_mode") == "bytes",
        "Metadata placeholder cannot qualify input observation",
    )
    require(
        stats.get("xor_sig_layout")
        == {
            "scheme": "flat-u32-le-v1",
            "regions": 4,
            "logical_nbytes": NBYTES,
            "byte_ranges": EXPECTED_RANGES,
        },
        "Regional byte layout differs",
    )
    values = stats.get("xor_signature")
    require(
        isinstance(values, list)
        and len(values) == 4
        and all(type(value) is int and 0 <= value < 2**32 for value in values),
        "Four finalized uint32 signatures required",
    )
    return values


def traces(arm):
    """Read every newline-committed raw official record; reject incomplete drains."""
    arm = Path(arm)
    records, members = [], {}
    for path in sorted((arm / "official-trace").rglob("*.jsonl")):
        require(not path.is_symlink(), "Official raw trace symlink forbidden")
        raw = path.read_bytes()
        require(raw.endswith(b"\n"), "Official trace has uncommitted trailing record")
        members[str(path.relative_to(arm))] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for line in raw.splitlines():
            require(line.strip(), "Empty official JSONL record")
            record = parse(line)
            if record.get("tag_prefix") == TAG:
                records.append(record)
    require(len(records) == 8, "Exactly eight official activation records required")
    require(
        len({row.get("counter") for row in records}) == 8
        and all(
            type(row.get("counter")) is int and row["counter"] > 0 for row in records
        ),
        "Official counters missing or duplicated",
    )
    records.sort(key=lambda row: row["counter"])
    result, order = {}, []
    for row in records:
        meta = row.get("meta") or {}
        role, ordinal = meta.get("ba_role"), meta.get("ba_ordinal")
        require(
            role in ("original", "recompute")
            and type(ordinal) is int
            and 0 <= ordinal < 4
            and meta.get("ba_region") == "gpt_neox.layers.6"
            and meta.get("ba_layer") == 6,
            "Official metadata lacks actual intervention correspondence",
        )
        key = (role, ordinal)
        require(key not in result, "Duplicate official role/microbatch")
        require(
            row.get("gpu_rank") == 0
            and row.get("dump_mode") == "summary"
            and row.get("is_async") is False,
            "Unexpected official rank or dump mode",
        )
        args = row.get("inputs", {}).get("args")
        require(
            isinstance(args, list) and len(args) == 2,
            "Official wrapper callable/tensor argument census differs",
        )
        result[key] = {
            "input": signature(args[1]),
            "output": signature(row.get("outputs")),
        }
        order.append(
            {
                "role": role,
                "ordinal": ordinal,
                "region": "gpt_neox.layers.6",
                "layer": 6,
            }
        )
    require(
        set(result)
        == {
            (role, ordinal)
            for role in ("original", "recompute")
            for ordinal in range(4)
        },
        "Official original/replay call census incomplete",
    )
    return result, order, members


def aggregate(arm_dirs, qualification, contract=None):
    """Compare exact whole-state hashes before inspecting the scoped signatures.

    Parent must call only on allocated compute after each fresh child returns
    normally. Build smoke and doctor receipts remain separate raw evidence.
    """
    wanted = {"clean-native", "clean-official", "fault-native", "fault-official"}
    require(set(arm_dirs) == wanted, "Four exact matched arms required")
    if contract is not None:
        require(
            contract.get("experiment_id") == "BA-OPGUARD-L6-20261107-v1"
            and contract.get("opguard_revision") == REVISION,
            "Wrong baseline experiment contract",
        )
    require(
        qualification.get("status") == "PASS"
        and qualification.get("opguard_revision") == REVISION
        and qualification.get("doctor", {}).get("native_ready") is True
        and qualification.get("native_smoke", {}).get("verified") is True,
        "Official native build/smoke qualification is incomplete",
    )
    arms = {}
    for key, folder in arm_dirs.items():
        outcome = read(Path(folder) / "child-outcome.json")
        require(
            outcome.get("returncode") == 0 and outcome.get("normal_exit") is True,
            "Fresh child did not finish normally; trace drain is unqualified",
        )
        row = read(Path(folder) / "arm-result.json")
        if contract is not None:
            require(
                row.get("contract_sha256") == os.environ.get("BA_CONTRACT_SHA256")
                and bool(row.get("contract_sha256")),
                "Arm contract does not bind parent admission",
            )
        require(
            row.get("status") == "ARM_COMPLETE_AWAITING_NORMAL_PROCESS_EXIT"
            and row.get("experiment_id") == "BA-OPGUARD-L6-20261107-v1"
            and row.get("fault") is key.startswith("fault-")
            and row.get("official") is key.endswith("-official"),
            "Arm treatment or completion differs",
        )
        require(len(row.get("calls", [])) == 8, "Native replay call census absent")
        arms[key] = row
    parities = {}
    for treatment in ("clean", "fault"):
        native, observed = arms[treatment + "-native"], arms[treatment + "-official"]
        equal = {}
        for field in ("initial_state", "post_state", "gradients", "calls", "contexts"):
            require(
                bool(native.get(field)) and native[field] == observed.get(field),
                "Official instrumentation changes or omits matched "
                + treatment
                + " "
                + field,
            )
            equal[field] = True
        require(
            native["row"].get("injections") == observed["row"].get("injections"),
            "Official instrumentation changes injection evidence",
        )
        forwards = observed.get("official_forward_receipts", [])
        require(
            len(forwards) == 8
            and all(
                x.get("original_calls") == 1 and x.get("same_result_object") is True
                for x in forwards
            ),
            "Official wrapper did not execute original exactly once",
        )
        require(
            observed.get("native_backend", {}).get("fallback_reported") is False,
            "Native XOR backend did not remain active",
        )
        require(
            [
                {key: receipt[key] for key in ("role", "ordinal", "region", "layer")}
                for receipt in forwards
            ]
            == observed["calls"],
            "Official forward receipts lack native correspondence",
        )
        parities[treatment] = equal
    require(
        all(
            arm["initial_state"] == arms["clean-native"]["initial_state"]
            for arm in arms.values()
        ),
        "Fresh processes did not begin from one exact full prestate",
    )
    native_hashes = [
        arms[name]["native_backend"].get("native_sha256")
        for name in ("clean-official", "fault-official")
    ]
    require(
        native_hashes[0] == native_hashes[1]
        and isinstance(native_hashes[0], str)
        and len(native_hashes[0]) == 64
        and all(character in "0123456789abcdef" for character in native_hashes[0]),
        "Official native extension identity differs across matched arms",
    )
    clean, clean_order, clean_members = traces(arm_dirs["clean-official"])
    faulty, fault_order, fault_members = traces(arm_dirs["fault-official"])
    require(
        clean_order == arms["clean-official"]["calls"]
        and fault_order == arms["fault-official"]["calls"],
        "Official counters do not preserve observed native call correspondence",
    )
    differences = []
    for key in clean:
        differences.append(
            {
                "role": key[0],
                "ordinal": key[1],
                "input_xor_delta": [
                    a ^ b for a, b in zip(clean[key]["input"], faulty[key]["input"])
                ],
                "output_xor_delta": [
                    a ^ b for a, b in zip(clean[key]["output"], faulty[key]["output"])
                ],
            }
        )
    expected_primary = next(
        row for row in differences if row["role"] == "recompute" and row["ordinal"] == 0
    )
    expected_control = all(
        row["input_xor_delta"]
        == ([0, 0, 0, 1] if row is expected_primary else [0, 0, 0, 0])
        for row in differences
    )
    output_equal = all(row["output_xor_delta"] == [0, 0, 0, 0] for row in differences)
    return {
        "experiment_id": "BA-OPGUARD-L6-20261107-v1",
        "execution_complete": True,
        "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW"
        if expected_control
        else "UNSUPPORTED_SIGNATURE_CONTROL",
        "qualified": expected_control,
        "noninterference": parities,
        "primary_input_signature_changed": expected_primary["input_xor_delta"]
        != [0, 0, 0, 0],
        "all_output_signatures_equal": output_equal,
        "differences": differences,
        "raw_trace_members": {"clean": clean_members, "fault": fault_members},
        "conclusion": "Official configured activation input observation exposes this output-equal gap"
        if expected_control and output_equal
        else "No output-equal input-gap conclusion",
        "scientific_acceptance": False,
        "limits": [
            "Configured official layer6 activation I/O; not full automatic OpGuard or backward coverage",
            "One fixed software bit fault, one state/history; no population, overhead or learning-impact claim",
            "Regional XOR is not collision-free; a single-bit delta is the registered positive control",
        ],
    }
