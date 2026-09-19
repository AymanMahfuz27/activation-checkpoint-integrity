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
import statistics
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
    if runtime:
        state.write_json(runtime.root / "summary.json", runtime.summary())
    if runtime and runtime.writer:
        # Capture completion and optimizer commitment are different facts.
        # An enforced abort has a complete trace but no optimizer STEP_COMMIT.
        commit = {"step": runtime.step, "rank": 0, "tensor_count": runtime.events,
                  "index_sha256": state.sha256(runtime.root / "events/rank_0.jsonl")}
        state.write_json(runtime.root / "CAPTURE_COMMIT", commit)
        if updates and status != "FAILED":
            state.write_json(runtime.root / "steps" / str(runtime.step) / "STEP_COMMIT", commit)
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


def arm_specifications(stages):
    """Return the minimum independent arms needed by one suite profile.

    ``fingerprint`` deliberately excludes every census and full-capture arm. It
    rechecks the current-code causal control, then measures clean and failing
    fingerprints from the same warmed snapshot.
    """
    causal = [
        ("trigger_reference", False, True, "off", "observe"),
        ("trigger_candidate", True, True, "off", "observe"),
        ("trigger_off_candidate", True, False, "off", "observe"),
    ]
    repeated_controls = [
        ("trigger_repeat", True, True, "off", "observe"),
        ("trigger_off_reference", False, False, "off", "observe"),
    ]
    full_capture = [
        ("recorded_reference", False, True, "full", "observe"),
        ("recorded_candidate", True, True, "full", "observe"),
        ("enforced_candidate", True, True, "full", "enforce"),
        ("clean_enforced", True, False, "full", "enforce"),
    ]
    fingerprints = [
        ("fingerprinted_candidate", True, True, "fingerprint", "enforce"),
        ("clean_fingerprinted", True, False, "fingerprint", "enforce"),
    ]
    if stages == "fingerprint_ablation":
        arms = []
        for repeat in range(1, 4):
            suffix = f"_{repeat}"
            arms.extend([
                (f"ablation_off{suffix}", True, False, "off", "observe"),
                (f"ablation_bookkeeping{suffix}", True, False,
                 "fingerprint", "observe"),
                (f"ablation_launch{suffix}", True, False,
                 "fingerprint", "observe"),
                (f"ablation_full{suffix}", True, False,
                 "fingerprint", "enforce"),
            ])
        return arms
    if stages == "fingerprint":
        repeated_clean = [
            ("trigger_off_candidate_repeat_2", True, False, "off", "observe"),
            ("clean_fingerprinted_repeat_2", True, False, "fingerprint", "enforce"),
            ("trigger_off_candidate_repeat_3", True, False, "off", "observe"),
            ("clean_fingerprinted_repeat_3", True, False, "fingerprint", "enforce"),
        ]
        # Alternate clean capture-off/fingerprint arms to reduce order bias;
        # leave the injected failure last because it is a correctness gate, not
        # the production-overhead baseline.
        return causal + [fingerprints[1]] + repeated_clean + [fingerprints[0]]
    if stages == "fingerprint_boundary":
        # Alternate capture-off and checkpoint-boundary arms so thermal or
        # scheduler drift cannot systematically favor either measurement.
        return causal + [
            ("boundary_clean_1", True, False, "fingerprint", "enforce"),
            ("trigger_off_candidate_repeat_2", True, False, "off", "observe"),
            ("boundary_clean_2", True, False, "fingerprint", "enforce"),
            ("trigger_off_candidate_repeat_3", True, False, "off", "observe"),
            ("boundary_clean_3", True, False, "fingerprint", "enforce"),
            ("boundary_candidate", True, True, "fingerprint", "enforce"),
        ]
    if stages == "off":
        return causal + repeated_controls
    if stages == "all":
        return causal + repeated_controls + full_capture + fingerprints
    raise ValueError(f"Unknown validation stage: {stages}")


def load_historical_oracle(path):
    """Load only the immutable facts needed to compare a compact replay."""
    if path is None:
        return None
    path = Path(path).resolve()
    summary = json.loads(path.read_text())
    recorded = summary["arms"]["recorded_candidate"]
    comparison = recorded["result"]["comparison"]
    first = comparison["first_divergence"]
    return {
        "path": str(path),
        "sha256": state.sha256(path),
        "full_capture_seconds": recorded["seconds"],
        "capture_off_seconds": summary["arms"]["trigger_candidate"]["seconds"],
        "eligible_pairs": comparison["eligible_pairs"],
        "mismatches": comparison["counts"].get("value_mismatch", 0),
        "first_pair_id": first["pair_id"],
        "first_operator": first["operator"],
    }


def load_fingerprint_baseline(path):
    """Load the prior same-job clean timing used by the optimization gate."""
    if path is None:
        return None
    path = Path(path).resolve()
    summary = json.loads(path.read_text())
    return {
        "path": str(path),
        "sha256": state.sha256(path),
        "clean_fingerprint_seconds": summary["arms"]["clean_fingerprinted"]["seconds"],
        "clean_capture_off_seconds": summary["arms"]["trigger_off_candidate"]["seconds"],
        "clean_overhead_ratio": summary["arms"]["clean_fingerprinted"]["seconds"]
            / summary["arms"]["trigger_off_candidate"]["seconds"],
    }


def suite(config, root, budget_bytes, stages="all", oracle_summary=None,
          fingerprint_baseline_summary=None):
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
    historical_oracle = load_historical_oracle(oracle_summary)
    fingerprint_baseline = load_fingerprint_baseline(fingerprint_baseline_summary)
    state.write_json(root / "experiment.json", {"seed": c.seed,
        "snapshot_sha256": state.sha256(snapshot), "warmup_updates": 1,
        "scope": "all microbatches of one real next-token step from warmed Adam state",
        "parameter_count": 39985664 if c.model.width == 512 and c.model.layers == 8 else None,
        "evidence_class": c.evidence_class, "budget_bytes": budget_bytes,
        "stages": stages, "historical_oracle": historical_oracle,
        "fingerprint_baseline": fingerprint_baseline})
    arms = {}
    specifications = arm_specifications(stages)
    for name, checkpointed, trigger, capture, policy in specifications:
        arm = copy.deepcopy(c)
        arm.run_id = name
        arm.checkpoint.enabled, arm.adapter.trigger = checkpointed, trigger
        arm.capture.mode, arm.capture.policy = capture, policy
        if "ablation_bookkeeping" in name:
            arm.capture.fingerprint_backend = "bookkeeping"
        elif "ablation_launch" in name:
            arm.capture.fingerprint_backend = "launch"
        else:
            arm.capture.fingerprint_backend = "full"
        arm.capture.fingerprint_scope = (
            "checkpoint_boundaries"
            if name.startswith("boundary_")
            else "all_operators"
        )
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
        ("clean_fingerprint_effect", "trigger_off_candidate", "clean_fingerprinted"),
        ("clean_fingerprint_effect_repeat_2", "trigger_off_candidate_repeat_2",
         "clean_fingerprinted_repeat_2"),
        ("clean_fingerprint_effect_repeat_3", "trigger_off_candidate_repeat_3",
         "clean_fingerprinted_repeat_3"),
        ("boundary_effect_1", "trigger_off_candidate", "boundary_clean_1"),
        ("boundary_effect_2", "trigger_off_candidate_repeat_2", "boundary_clean_2"),
        ("boundary_effect_3", "trigger_off_candidate_repeat_3", "boundary_clean_3"),
    ]:
        if left in arms and right in arms:
            comparisons[label] = outcomes_equal(root / left, root / right)
            state.write_json(root / (label + "_comparison.json"), comparisons[label])
    if stages == "fingerprint_ablation":
        backend_names = {
            backend: [f"ablation_{backend}_{repeat}" for repeat in range(1, 4)]
            for backend in ("off", "bookkeeping", "launch", "full")
        }
        for repeat in range(1, 4):
            off_name = f"ablation_off_{repeat}"
            for backend in ("bookkeeping", "launch", "full"):
                name = f"ablation_{backend}_{repeat}"
                label = f"ablation_{backend}_effect_{repeat}"
                comparisons[label] = outcomes_equal(root / off_name, root / name)
                state.write_json(root / (label + "_comparison.json"), comparisons[label])

        times = {
            backend: [arms[name]["seconds"] for name in names]
            for backend, names in backend_names.items()
        }
        medians = {
            backend: statistics.median(values)
            for backend, values in times.items()
        }
        surcharges = {
            backend: medians[backend] - medians["off"]
            for backend in ("bookkeeping", "launch", "full")
        }
        launch_increment = medians["launch"] - medians["bookkeeping"]
        scan_increment = medians["full"] - medians["launch"]
        observer_plus_launch = max(0.0, medians["launch"] - medians["off"])
        full_surcharge = medians["full"] - medians["off"]
        gates = {
            "all_arms_completed": all(
                arm["status"] == "PASS" for arm in arms.values()
            ),
            "all_outcomes_match_capture_off": all(
                not any(comparisons[f"ablation_{backend}_effect_{repeat}"].values())
                for backend in ("bookkeeping", "launch", "full")
                for repeat in range(1, 4)
            ),
            "all_fingerprint_pairs_complete": all(
                not arms[name]["result"]["comparison"]["failed"]
                and arms[name]["result"]["comparison"]["pair_coverage"] == 1.0
                and arms[name]["result"]["comparison"]["decision_host_checks"] == 1
                for backend in ("bookkeeping", "launch", "full")
                for name in backend_names[backend]
            ),
            "full_backend_exact": all(
                arms[name]["result"]["comparison"]["exact_pairs"]
                == arms[name]["result"]["comparison"]["eligible_pairs"]
                for name in backend_names["full"]
            ),
            "observer_plus_launch_at_least_half_of_full_surcharge":
                full_surcharge > 0 and observer_plus_launch >= 0.5 * full_surcharge,
        }
        timing = {
            "seconds": times,
            "median_seconds": medians,
            "surcharge_over_off_seconds": surcharges,
            "launch_increment_over_bookkeeping_seconds": launch_increment,
            "payload_scan_increment_over_launch_seconds": scan_increment,
            "observer_plus_launch_fraction_of_full_surcharge":
                observer_plus_launch / full_surcharge if full_surcharge > 0 else None,
            "peak_cuda_allocated": {
                backend: [arms[name]["peak_cuda_allocated"] for name in names]
                for backend, names in backend_names.items()
            },
        }
        result = {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "gates": gates,
            "device": c.device,
            "seed": c.seed,
            "arms": arms,
            "timing": timing,
            "scope": "diagnostic timing ablation; non-full backends cannot enforce",
        }
        state.write_json(root / "summary.json", result)
        return result
    gates = {
        "silent_gpu_or_cpu_reproduction": bool(comparisons["trigger"]["gradients"]
                    and comparisons["trigger"]["model"] and not comparisons["trigger"]["losses"]
                    and all(arms[n]["status"] == "PASS" for n in ["trigger_reference", "trigger_candidate"])),
    }
    if "repeatability" in comparisons:
        gates["repeatable"] = not any(comparisons["repeatability"].values())
    if "trigger_off" in comparisons:
        gates["trigger_off_restores_equality"] = not any(comparisons["trigger_off"].values())
    timing = None
    if stages == "all":
        before = state.load_snapshot(snapshot)
        after = torch.load(root / "enforced_candidate/outcome.pt", weights_only=False)
        preserved = {key: not compare_state(before[key], after[key])
                     for key in ("model", "optimizer", "scheduler", "cursor")}
        state.write_json(root / "enforcement_preservation.json", preserved)
        fingerprint_after = torch.load(
            root / "fingerprinted_candidate/outcome.pt", weights_only=False
        )
        fingerprint_preserved = {
            key: not compare_state(before[key], fingerprint_after[key])
            for key in ("model", "optimizer", "scheduler", "cursor")
        }
        state.write_json(root / "fingerprint_enforcement_preservation.json",
                         fingerprint_preserved)
        comparison = arms["recorded_candidate"]["result"]["comparison"]
        first = comparison["first_divergence"]
        fingerprint_comparison = arms["fingerprinted_candidate"]["failure"]["comparison"]
        fingerprint_first = fingerprint_comparison["first_divergence"]
        clean_fingerprint_comparison = arms["clean_fingerprinted"]["result"]["comparison"]
        gates.update({
            "reference_recording_preserves_outcome": not any(comparisons["reference_recording_effect"].values()),
            "candidate_recording_preserves_outcome": not any(comparisons["candidate_recording_effect"].values()),
            "clean_recording_preserves_outcome": not any(comparisons["clean_recording_effect"].values()),
            "clean_fingerprint_preserves_outcome": not any(comparisons["clean_fingerprint_effect"].values()),
            "complete_pairing": comparison["pair_coverage"] == 1.0,
            "first_same_metadata_mismatch": bool(first and first["status"] == "value_mismatch"
                                                  and first["metadata_equal"]),
            "bad_update_blocked": arms["enforced_candidate"]["status"] == "ENFORCED_ABORT"
                    and arms["enforced_candidate"]["optimizer_step_calls"] == 0 and all(preserved.values())
                    and arms["enforced_candidate"].get("failure", {}).get("error") is None
                    and bool(arms["enforced_candidate"].get("failure", {}).get("comparison", {}).get("failed")),
            "clean_update_allowed": arms["clean_enforced"]["status"] == "PASS"
                    and arms["clean_enforced"]["optimizer_step_calls"] == 1,
            "fingerprint_matches_exact_first_divergence": bool(
                    fingerprint_first and first
                    and fingerprint_first["pair_id"] == first["pair_id"]
                    and fingerprint_first["status"] == "value_mismatch"
                    and fingerprint_first["metadata_equal"]),
            "fingerprint_bad_update_blocked":
                    arms["fingerprinted_candidate"]["status"] == "ENFORCED_ABORT"
                    and arms["fingerprinted_candidate"]["optimizer_step_calls"] == 0
                    and all(fingerprint_preserved.values())
                    and fingerprint_comparison["failed"],
            "fingerprint_clean_update_allowed":
                    arms["clean_fingerprinted"]["status"] == "PASS"
                    and arms["clean_fingerprinted"]["optimizer_step_calls"] == 1
                    and not clean_fingerprint_comparison["failed"]
                    and clean_fingerprint_comparison["pair_coverage"] == 1.0
                    and clean_fingerprint_comparison["decision_host_checks"] == 1,
        })
    elif stages == "fingerprint":
        clean_off_names = [
            "trigger_off_candidate",
            "trigger_off_candidate_repeat_2",
            "trigger_off_candidate_repeat_3",
        ]
        clean_fingerprint_names = [
            "clean_fingerprinted",
            "clean_fingerprinted_repeat_2",
            "clean_fingerprinted_repeat_3",
        ]
        clean_effect_names = [
            "clean_fingerprint_effect",
            "clean_fingerprint_effect_repeat_2",
            "clean_fingerprint_effect_repeat_3",
        ]
        before = state.load_snapshot(snapshot)
        fingerprint_after = torch.load(
            root / "fingerprinted_candidate/outcome.pt", weights_only=False
        )
        fingerprint_preserved = {
            key: not compare_state(before[key], fingerprint_after[key])
            for key in ("model", "optimizer", "scheduler", "cursor")
        }
        state.write_json(root / "fingerprint_enforcement_preservation.json",
                         fingerprint_preserved)
        fingerprint_comparison = arms["fingerprinted_candidate"]["failure"]["comparison"]
        fingerprint_first = fingerprint_comparison["first_divergence"]
        clean_comparisons = [
            arms[name]["result"]["comparison"] for name in clean_fingerprint_names
        ]
        clean_off_times = [arms[name]["seconds"] for name in clean_off_names]
        clean_fingerprint_times = [arms[name]["seconds"] for name in clean_fingerprint_names]
        clean_off_median = statistics.median(clean_off_times)
        clean_fingerprint_median = statistics.median(clean_fingerprint_times)
        clean_overhead_ratio = clean_fingerprint_median / clean_off_median
        peak_memory_ratios = [
            (arms[fingerprint_name]["peak_cuda_allocated"]
             - arms[off_name]["peak_cuda_allocated"])
            / arms[off_name]["peak_cuda_allocated"]
            for off_name, fingerprint_name in zip(
                clean_off_names, clean_fingerprint_names, strict=True
            )
        ]
        gates.update({
            "clean_fingerprint_preserves_outcome":
                all(not any(comparisons[name].values()) for name in clean_effect_names),
            "fingerprint_bad_update_blocked":
                arms["fingerprinted_candidate"]["status"] == "ENFORCED_ABORT"
                and arms["fingerprinted_candidate"]["optimizer_step_calls"] == 0
                and all(fingerprint_preserved.values())
                and fingerprint_comparison["failed"],
            "fingerprint_clean_update_allowed":
                all(arms[name]["status"] == "PASS"
                        and arms[name]["optimizer_step_calls"] == 1
                        and not comparison["failed"]
                        and comparison["pair_coverage"] == 1.0
                        and comparison["decision_host_checks"] == 1
                        for name, comparison in zip(
                            clean_fingerprint_names, clean_comparisons, strict=True
                        )),
            "clean_fingerprint_median_ratio_at_most_1_25x":
                clean_overhead_ratio <= 1.25,
            "fingerprint_peak_memory_increase_at_most_0_5_percent":
                max(peak_memory_ratios) <= 0.005,
        })
        if fingerprint_baseline:
            gates["clean_fingerprint_at_least_2x_faster_than_prior"] = (
                clean_fingerprint_median
                <= fingerprint_baseline["clean_fingerprint_seconds"] / 2
            )
        if historical_oracle:
            gates.update({
                "historical_pair_coverage_matches":
                    fingerprint_comparison["eligible_pairs"]
                    == historical_oracle["eligible_pairs"],
                "historical_mismatch_count_matches":
                    fingerprint_comparison["counts"].get("value_mismatch", 0)
                    == historical_oracle["mismatches"],
                "historical_first_divergence_matches": bool(
                    fingerprint_first
                    and fingerprint_first["pair_id"]
                    == historical_oracle["first_pair_id"]
                    and fingerprint_first["operator"]
                    == historical_oracle["first_operator"]
                ),
                "historical_full_capture_speedup_at_least_10x":
                    historical_oracle["full_capture_seconds"]
                    >= 10 * arms["fingerprinted_candidate"]["seconds"],
            })
        timing = {
            "clean_capture_off_seconds": clean_off_times,
            "clean_fingerprint_seconds": clean_fingerprint_times,
            "clean_capture_off_median_seconds": clean_off_median,
            "clean_fingerprint_median_seconds": clean_fingerprint_median,
            "clean_fingerprint_median_overhead_ratio": clean_overhead_ratio,
            "clean_fingerprint_median_overhead_percent":
                100 * (clean_overhead_ratio - 1),
            "clean_peak_memory_increase_ratios": peak_memory_ratios,
        }
        if historical_oracle:
            fingerprint_seconds = arms["fingerprinted_candidate"]["seconds"]
            capture_off_seconds = arms["trigger_candidate"]["seconds"]
            timing.update({
                "historical_full_capture_seconds": historical_oracle["full_capture_seconds"],
                "historical_capture_off_seconds": historical_oracle["capture_off_seconds"],
                "current_capture_off_seconds": capture_off_seconds,
                "fingerprint_seconds": fingerprint_seconds,
                "full_capture_speedup":
                    historical_oracle["full_capture_seconds"] / fingerprint_seconds,
                "fingerprint_overhead_ratio": fingerprint_seconds / capture_off_seconds,
                "fingerprint_overhead_percent":
                    100 * (fingerprint_seconds / capture_off_seconds - 1),
            })
    elif stages == "fingerprint_boundary":
        clean_off_names = [
            "trigger_off_candidate",
            "trigger_off_candidate_repeat_2",
            "trigger_off_candidate_repeat_3",
        ]
        clean_boundary_names = [
            "boundary_clean_1",
            "boundary_clean_2",
            "boundary_clean_3",
        ]
        clean_effect_names = [
            "boundary_effect_1",
            "boundary_effect_2",
            "boundary_effect_3",
        ]
        expected_pairs = (
            len(c.checkpoint.blocks) if c.checkpoint.blocks else c.model.layers
        ) * c.training.accumulation

        before = state.load_snapshot(snapshot)
        boundary_after = torch.load(
            root / "boundary_candidate/outcome.pt", weights_only=False
        )
        preserved = {
            key: not compare_state(before[key], boundary_after[key])
            for key in ("model", "optimizer", "scheduler", "cursor")
        }
        state.write_json(root / "boundary_enforcement_preservation.json", preserved)

        failing = arms["boundary_candidate"]["failure"]["comparison"]
        first = failing["first_divergence"]
        clean_comparisons = [
            arms[name]["result"]["comparison"] for name in clean_boundary_names
        ]
        clean_off_times = [arms[name]["seconds"] for name in clean_off_names]
        clean_boundary_times = [arms[name]["seconds"] for name in clean_boundary_names]
        clean_off_median = statistics.median(clean_off_times)
        clean_boundary_median = statistics.median(clean_boundary_times)
        clean_overhead_ratio = clean_boundary_median / clean_off_median
        peak_memory_ratios = [
            (arms[boundary_name]["peak_cuda_allocated"]
             - arms[off_name]["peak_cuda_allocated"])
            / arms[off_name]["peak_cuda_allocated"]
            for off_name, boundary_name in zip(
                clean_off_names, clean_boundary_names, strict=True
            )
            if arms[off_name]["peak_cuda_allocated"] not in (None, 0)
            and arms[boundary_name]["peak_cuda_allocated"] is not None
        ]

        gates.update({
            "clean_boundary_preserves_outcome":
                all(not any(comparisons[name].values()) for name in clean_effect_names),
            "boundary_bad_update_blocked":
                arms["boundary_candidate"]["status"] == "ENFORCED_ABORT"
                and arms["boundary_candidate"]["optimizer_step_calls"] == 0
                and all(preserved.values())
                and failing["failed"],
            "boundary_clean_update_allowed": all(
                arms[name]["status"] == "PASS"
                and arms[name]["optimizer_step_calls"] == 1
                and not comparison["failed"]
                and comparison["pair_coverage"] == 1.0
                and comparison["decision_host_checks"] == 1
                and comparison["eligible_pairs"] == expected_pairs
                and comparison["exact_pairs"] == expected_pairs
                for name, comparison in zip(
                    clean_boundary_names, clean_comparisons, strict=True
                )
            ),
            "boundary_failure_pairing_complete":
                failing["pair_coverage"] == 1.0
                and failing["decision_host_checks"] == 1
                and failing["eligible_pairs"] == expected_pairs,
            "boundary_first_divergence_is_first_block_output": bool(
                first
                and first["status"] == "value_mismatch"
                and first["metadata_equal"]
                and first["operator"] == "checkpoint_region_output"
                and first["region"] == "root/blocks.0:0"
            ),
            "clean_boundary_median_ratio_at_most_1_05x":
                clean_overhead_ratio <= 1.05,
            "boundary_peak_memory_increase_at_most_0_5_percent":
                not peak_memory_ratios or max(peak_memory_ratios) <= 0.005,
        })
        timing = {
            "clean_capture_off_seconds": clean_off_times,
            "clean_boundary_seconds": clean_boundary_times,
            "clean_capture_off_median_seconds": clean_off_median,
            "clean_boundary_median_seconds": clean_boundary_median,
            "clean_boundary_median_overhead_ratio": clean_overhead_ratio,
            "clean_boundary_median_overhead_percent":
                100 * (clean_overhead_ratio - 1),
            "production_target_at_most_1_02x": clean_overhead_ratio <= 1.02,
            "clean_peak_memory_increase_ratios": peak_memory_ratios,
            "expected_pairs": expected_pairs,
            "observations_per_audited_step": 2 * expected_pairs,
        }
        if historical_oracle:
            failing_seconds = arms["boundary_candidate"]["seconds"]
            timing.update({
                "historical_full_capture_seconds": historical_oracle["full_capture_seconds"],
                "boundary_failure_seconds": failing_seconds,
                "historical_full_capture_speedup":
                    historical_oracle["full_capture_seconds"] / failing_seconds,
            })
    result = {"status": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
              "device": c.device, "seed": c.seed, "arms": arms,
              "differing_gradients": len(comparisons["trigger"]["gradients"]),
              "first_gradient": comparisons["trigger"]["gradients"][0] if comparisons["trigger"]["gradients"] else None,
              "timing": timing, "historical_oracle": historical_oracle,
              "fingerprint_baseline": fingerprint_baseline,
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
    parser.add_argument(
        "--stages",
        choices=[
            "all", "off", "fingerprint", "fingerprint_ablation",
            "fingerprint_boundary",
        ],
        default="all",
    )
    parser.add_argument("--oracle-summary", type=Path)
    parser.add_argument("--fingerprint-baseline-summary", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
        config.expected_torch = torch.__version__
    if args.seed is not None:
        config.seed = args.seed
    result = (run_arm(config, args.snapshot, args.output) if args.action == "arm"
              else suite(config, args.output, args.budget_bytes, args.stages,
                         args.oracle_summary, args.fingerprint_baseline_summary))
    print(json.dumps({"output": str(args.output), "status": result["status"]}), flush=True)
    return 0 if result["status"] in {"PASS", "OBSERVED_MISMATCH", "ENFORCED_ABORT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
