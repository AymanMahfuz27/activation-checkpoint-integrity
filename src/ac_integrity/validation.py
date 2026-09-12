"""Bounded causal experiments from a common, warmed pre-update snapshot.

This uses the research trainer's real step and recorder. Every arm gets a fresh
interpreter. It preserves one snapshot and one outcome per arm, without redundant
resume snapshots. Full capture retains all visible dense operator outputs.
"""

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch
from ac_integrity import state
from ac_integrity.capture.compare import validate_artifacts
from ac_integrity.capture.runtime import CaptureRuntime
from ac_integrity.config import load_config, write_config
from ac_integrity.data import PackedCorpus
from ac_integrity.distributed import IntegrityError
from ac_integrity.train import (build, collect_batches, compare_state, execute_step,
                                snapshot_identity)


def verify_snapshot(config, corpus, snapshot):
    """Reject a replay with changed code, data, device or optimizer semantics."""
    expected = snapshot_identity(config, corpus.digest)
    if snapshot["provenance"] != expected:
        differences = compare_state(snapshot["provenance"], expected)
        raise ValueError(f"Snapshot provenance mismatch: {differences}")
    batches, _ = collect_batches(corpus, snapshot["cursor"], config)
    if compare_state(batches, snapshot["batches"]):
        raise ValueError("Snapshot batches differ from the verified corpus")


def run_arm(config, snapshot_path, root):
    """Execute and retain one arm; an expected enforcement abort is still evidence."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write_config(config, root / "config.toml")
    started = datetime.now(timezone.utc).isoformat()
    model, optimizer, scheduler = build(config)
    environment = state.environment()
    state.write_json(root / "environment.json", environment)
    corpus = PackedCorpus(config)
    snapshot = state.load_snapshot(snapshot_path, model, optimizer, scheduler)
    verify_snapshot(config, corpus, snapshot)
    runtime = None
    if config.capture.mode != "off":
        runtime = CaptureRuntime(root / "capture", config)
        runtime.step = snapshot["step"]
        runtime.attach(model)
        model.runtime = runtime
    updates = []
    handle = optimizer.register_step_pre_hook(lambda *unused: updates.append(True))
    result, error, status = None, None, "PASS"
    start = time.monotonic()
    try:
        result = execute_step(model, optimizer, scheduler, snapshot["batches"], config, runtime)
        if result["incorrect_update_observed"]:
            status = "OBSERVED_MISMATCH"
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception),
                 "traceback": traceback.format_exc()}
        status = "ENFORCED_ABORT" if isinstance(exception, IntegrityError) else "FAILED"
    finally:
        handle.remove()
        model.runtime = None
        if runtime:
            try:
                runtime.close()
            except Exception as exception:
                status = "FAILED"
                error = {"type": type(exception).__name__, "message": str(exception)}
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    outcome = {"model": state.cpu_tree(model.state_dict()),
               "optimizer": state.cpu_tree(optimizer.state_dict()),
               "scheduler": scheduler.state_dict(),
               "gradients": (result["gradients"] if result else
                             {n: state.cpu_tree(p.grad) for n, p in model.named_parameters()}),
               "rng": state.rng_state(), "losses": result["losses"] if result else None,
               "cursor": snapshot["cursor"] + (config.training.accumulation if updates else 0)}
    state.save_tensor_file(root / "outcome.pt", outcome)
    summary = {"status": status, "started_at": started,
               "finished_at": datetime.now(timezone.utc).isoformat(),
               "seconds": time.monotonic() - start, "error": error,
               "optimizer_step_calls": len(updates),
               "execution_backend": state.backend_state(),
               "snapshot_sha256": state.sha256(snapshot_path),
               "result": {k: v for k, v in (result or {}).items() if k != "gradients"},
               "capture": runtime.summary() if runtime else None,
               "peak_cuda_allocated": torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None}
    if runtime and (runtime.root / "failure.json").exists():
        summary["failure"] = json.loads((runtime.root / "failure.json").read_text())
    if runtime and runtime.writer:
        # Capture completion and optimizer commitment are different facts.
        # An enforced abort has a complete trace but no optimizer STEP_COMMIT.
        commit = {"step": runtime.step, "rank": 0, "tensor_count": runtime.events,
                  "index_sha256": state.sha256(runtime.root / "events/rank_0.jsonl")}
        state.write_json(runtime.root / "CAPTURE_COMMIT", commit)
        if updates and status != "FAILED":
            state.write_json(runtime.root / "steps" / str(runtime.step) / "STEP_COMMIT", commit)
        state.write_json(runtime.root / "summary.json", runtime.summary())
        summary["artifact_validation"] = validate_artifacts(runtime.root, require_step_commit=bool(updates))
        if not summary["artifact_validation"]["valid"]:
            summary["status"] = "FAILED"
    state.write_json(root / "summary.json", summary)
    return summary


def fresh_arm(config, snapshot, root):
    path = root.with_suffix(".toml")
    write_config(config, path)
    command = [sys.executable, "-m", "ac_integrity.validation", "arm",
               "--config", str(path), "--snapshot", str(snapshot), "--output", str(root)]
    with root.with_suffix(".stdout").open("w") as out, root.with_suffix(".stderr").open("w") as err:
        process = subprocess.run(command, stdout=out, stderr=err, timeout=7200)
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Arm failed without evidence: {root}, exit={process.returncode}")
    summary = json.loads(summary_path.read_text())
    summary["command"] = command
    summary["returncode"] = process.returncode
    return summary


def outcomes_equal(left, right):
    """Retain full quantitative differences only for actual changed state."""
    a = torch.load(left / "outcome.pt", map_location="cpu", weights_only=False)
    b = torch.load(right / "outcome.pt", map_location="cpu", weights_only=False)
    return {key: compare_state(a[key], b[key], key) for key in a}


def suite(config, root, budget_bytes, stages="all"):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    c = copy.deepcopy(config)
    c.adapter.name, c.adapter.trigger = "pytorch_84864", False
    c.capture.mode = "off"
    c.capture.policy = "observe"
    c.training.snapshot_steps = []
    c.training.checkpoint_every = 0
    c.training.stop_after = 0
    c.validate()
    model, optimizer, scheduler = build(c)
    environment = state.environment()
    state.write_json(root / "environment.json", environment)
    state.archive_source(root / "source.zip", environment)
    corpus = PackedCorpus(c)
    batches, cursor = collect_batches(corpus, 0, c)
    # Populate real Adam moments before checking that an abort preserves them.
    execute_step(model, optimizer, scheduler, batches, c, collect_evidence=False)
    batches, _ = collect_batches(corpus, cursor, c)
    snapshot = root / "pre_step_2.pt"
    state.save_snapshot(snapshot, model, optimizer, scheduler, cursor, 2, batches,
                        snapshot_identity(c, corpus.digest), trigger_state=asdict(c.adapter))
    del model, optimizer, scheduler, corpus, batches
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    state.write_json(root / "experiment.json", {"seed": c.seed,
        "snapshot_sha256": state.sha256(snapshot), "warmup_updates": 1,
        "scope": "all microbatches of one real next-token step from warmed Adam state",
        "parameter_count": 39985664 if c.model.width == 512 and c.model.layers == 8 else None,
        "evidence_class": c.evidence_class, "budget_bytes": budget_bytes})
    arms = {}
    specifications = [
        ("trigger_reference", False, True, "off", "observe"),
        ("trigger_candidate", True, True, "off", "observe"),
        ("trigger_repeat", True, True, "off", "observe"),
        ("trigger_off_reference", False, False, "off", "observe"),
        ("trigger_off_candidate", True, False, "off", "observe"),
    ]
    if stages == "all":
        specifications += [
            ("recorded_reference", False, True, "full", "observe"),
            ("recorded_candidate", True, True, "full", "observe"),
            ("enforced_candidate", True, True, "full", "enforce"),
            ("clean_enforced", True, False, "full", "enforce"),
        ]
    for name, checkpointed, trigger, capture, policy in specifications:
        arm = copy.deepcopy(c)
        arm.run_id = name
        arm.checkpoint.enabled, arm.adapter.trigger = checkpointed, trigger
        arm.capture.mode, arm.capture.policy = capture, policy
        arm.capture.encoding, arm.capture.deduplicate = "zlib", True
        arm.capture.max_bytes = budget_bytes
        if capture == "full":
            if budget_bytes <= 0:
                raise ValueError("Full suite needs an explicit verified storage budget")
            # A complete census is mandatory before full materialization. This
            # conservative bound makes no assumption about compression savings.
            census_config = copy.deepcopy(arm)
            census_config.capture.mode = "census"
            measured = fresh_arm(census_config, snapshot, root / (name + "_census"))
            if measured["status"] != "PASS":
                raise RuntimeError(f"Census failed: {name}")
            raw = measured["capture"]["payload_bytes"]
            needed = int(1.25 * (raw + measured["capture"]["captured_tensors"] * 8192
                                  + 2 * snapshot.stat().st_size))
            available = min(budget_bytes, shutil.disk_usage(root).free - arm.capture.reserve_bytes)
            state.write_json(root / (name + "_storage.json"),
                             {"raw_bytes": raw, "needed_with_margin": needed, "available": available})
            if needed > available:
                raise OSError(f"Complete {name} capture requires {needed}; budget {available}")
        print(json.dumps({"starting_arm": name, "root": str(root)}), flush=True)
        arms[name] = fresh_arm(arm, snapshot, root / name)
        state.write_json(root / "progress.json", arms)
        if arms[name]["status"] == "FAILED":
            raise RuntimeError(f"Arm {name} failed; retained at {root}")
    comparisons = {}
    for label, left, right in [
        ("trigger", "trigger_reference", "trigger_candidate"),
        ("repeatability", "trigger_candidate", "trigger_repeat"),
        ("trigger_off", "trigger_off_reference", "trigger_off_candidate"),
        ("reference_recording_effect", "trigger_reference", "recorded_reference"),
        ("candidate_recording_effect", "trigger_candidate", "recorded_candidate"),
        ("clean_recording_effect", "trigger_off_candidate", "clean_enforced"),
    ]:
        if left in arms and right in arms:
            comparisons[label] = outcomes_equal(root / left, root / right)
            state.write_json(root / (label + "_comparison.json"), comparisons[label])
    gates = {
        "silent_gpu_or_cpu_reproduction": bool(comparisons["trigger"]["gradients"]
                    and comparisons["trigger"]["model"] and not comparisons["trigger"]["losses"]
                    and all(arms[n]["status"] == "PASS" for n in ["trigger_reference", "trigger_candidate"])),
        "repeatable": not any(comparisons["repeatability"].values()),
        "trigger_off_restores_equality": not any(comparisons["trigger_off"].values()),
    }
    if stages == "all":
        before = state.load_snapshot(snapshot)
        after = torch.load(root / "enforced_candidate/outcome.pt", weights_only=False)
        preserved = {key: not compare_state(before[key], after[key])
                     for key in ("model", "optimizer", "scheduler", "cursor")}
        state.write_json(root / "enforcement_preservation.json", preserved)
        comparison = arms["recorded_candidate"]["result"]["comparison"]
        first = comparison["first_divergence"]
        gates.update({
            "reference_recording_preserves_outcome": not any(comparisons["reference_recording_effect"].values()),
            "candidate_recording_preserves_outcome": not any(comparisons["candidate_recording_effect"].values()),
            "clean_recording_preserves_outcome": not any(comparisons["clean_recording_effect"].values()),
            "complete_pairing": comparison["pair_coverage"] == 1.0,
            "first_same_metadata_mismatch": bool(first and first["status"] == "value_mismatch"
                                                  and first["metadata_equal"]),
            "bad_update_blocked": arms["enforced_candidate"]["status"] == "ENFORCED_ABORT"
                    and arms["enforced_candidate"]["optimizer_step_calls"] == 0 and all(preserved.values())
                    and arms["enforced_candidate"].get("failure", {}).get("error") is None
                    and bool(arms["enforced_candidate"].get("failure", {}).get("comparison", {}).get("failed")),
            "clean_update_allowed": arms["clean_enforced"]["status"] == "PASS"
                    and arms["clean_enforced"]["optimizer_step_calls"] == 1,
        })
    result = {"status": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
              "device": c.device, "seed": c.seed, "arms": arms,
              "differing_gradients": len(comparisons["trigger"]["gradients"]),
              "first_gradient": comparisons["trigger"]["gradients"][0] if comparisons["trigger"]["gradients"] else None,
              "scope": "controlled known-bug transplant; not discovery in untouched training"}
    state.write_json(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["suite", "arm"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget-bytes", type=int, default=0)
    parser.add_argument("--stages", choices=["all", "off"], default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
        config.expected_torch = torch.__version__
    if args.seed is not None:
        config.seed = args.seed
    result = (run_arm(config, args.snapshot, args.output) if args.action == "arm"
              else suite(config, args.output, args.budget_bytes, args.stages))
    print(json.dumps({"output": str(args.output), "status": result["status"]}), flush=True)
    return 0 if result["status"] in {"PASS", "OBSERVED_MISMATCH", "ENFORCED_ABORT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
