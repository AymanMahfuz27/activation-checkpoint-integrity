"""Training lifecycle and one-step forensic arms from immutable pre-step states."""

from contextlib import nullcontext
from dataclasses import asdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np
import torch
from torch.nn import functional as F
from ac_integrity.config import write_config, from_dict
from ac_integrity.model import Decoder
from ac_integrity.data import PackedCorpus
from ac_integrity import state
from ac_integrity.capture.runtime import CaptureRuntime
from ac_integrity.capture.dispatch import CaptureMode
from ac_integrity.capture.compare import compare_events, compare_tensors
from ac_integrity.capture.report import report
from ac_integrity.distributed import optimizer_gate, IntegrityError
from ac_integrity.natural.pytorch_84864 import forward_context


def build(config):
    state.configure(config)
    model = Decoder(config).to(config.device)
    t = config.training
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.learning_rate,
                                 betas=(t.beta1, t.beta2), eps=t.epsilon,
                                 weight_decay=t.weight_decay, foreach=False, fused=False)
    def multiplier(index):
        step = index + 1
        if t.warmup_steps and step <= t.warmup_steps:
            return step / t.warmup_steps
        if t.decay_steps and step > t.steps - t.decay_steps:
            progress = min(1.0, (step - (t.steps - t.decay_steps)) / t.decay_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return model, optimizer, scheduler


def audited(config, step):
    selection = config.capture.audit_steps
    return config.capture.mode != "off" and (selection == "all" or step in selection)


def snapshot_identity(config, data_digest):
    environment = state.environment()
    return {"model": asdict(config.model), "training": asdict(config.training),
            "data_sha256": data_digest, "dtype": config.dtype,
            "adapter_name": config.adapter.name, "seed": config.seed,
            "code_sha256": environment["code_sha256"], "lock_sha256": environment["lock_sha256"],
            "torch": environment["torch"], "numpy": environment["numpy"],
            "machine": environment["machine"], "device": config.device}


def census_identity(config, snapshot):
    raw = asdict(config)
    for key in ("run_id", "artifact_root"):
        raw.pop(key)
    raw["capture"] = {"audit_steps": raw["capture"]["audit_steps"]}
    return {"configuration_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(),
            "snapshot_sha256": state.sha256(snapshot),
            "code_sha256": state.environment()["code_sha256"]}


def preflight(config, snapshot):
    if not config.capture.census_path:
        raise ValueError("Full capture requires a census from this exact snapshot and configuration")
    census = json.loads(Path(config.capture.census_path).read_text())
    if census["identity"] != census_identity(config, snapshot):
        raise ValueError("Census is stale or belongs to a different config/snapshot/code")
    needed = census["projected_bytes_with_margin"]
    root = Path(config.artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(root).free - config.capture.reserve_bytes
    if not config.capture.verified_quota_bytes:
        raise ValueError("Set verified_quota_bytes to a measured user budget; shared filesystem free space is not quota")
    allowed = min(config.capture.max_bytes, config.capture.verified_quota_bytes, available)
    if needed > allowed:
        raise OSError(f"Full capture requires {needed} bytes, available budget is {allowed}")
    return {"projected_bytes": needed, "available_budget": allowed, "census": str(Path(config.capture.census_path).resolve())}


def collect_batches(corpus, cursor, config):
    batches = []
    for _ in range(config.training.accumulation):
        batch, cursor = corpus.batch(cursor, config.training.microbatch_size, config.training.sequence_length)
        batches.append(batch)
    return batches, cursor


def execute_step(model, optimizer, scheduler, batches, config, runtime=None, collect_evidence=True):
    """Capture forward/backward, compare durably, then gate before clipping/update."""
    losses = []
    error = None
    default_check_raised = False
    comparison = None
    named_gradients = None
    phase = runtime.phase_context if runtime else lambda name: nullcontext()
    mode = CaptureMode(runtime) if runtime else nullcontext()
    with mode:
        try:
            with phase("zero_grad"):
                optimizer.zero_grad(set_to_none=True)
            for microbatch, batch in enumerate(batches):
                if runtime:
                    runtime.microbatch = microbatch
                with phase("forward"), forward_context(config):
                    logits = model(batch["input"].to(config.device))
                    loss = F.cross_entropy(logits.reshape(-1, config.model.vocab_size),
                                           batch["target"].to(config.device).reshape(-1))
                    scaled_loss = loss / config.training.accumulation
                with phase("backward"):
                    scaled_loss.backward()
                losses.append(float(loss.detach()))
        except Exception as exception:
            error = exception
            default_check_raised = isinstance(exception, torch.utils.checkpoint.CheckpointError)
        # Observer bookkeeping must never re-enter capture.
        from torch.utils._python_dispatch import _disable_current_modes
        with _disable_current_modes():
            if runtime:
                try:
                    runtime.flush()
                    if runtime.writer:
                        comparison = compare_events(runtime.root, runtime.step, runtime.rank)
                except Exception as exception:
                    error = error or exception
            failed = error is not None or bool(comparison and comparison["failed"])
            if runtime and failed:
                state.write_json(runtime.root / "failure.json", {
                    "comparison": comparison,
                    "default_check_raised": default_check_raised,
                    "error": str(error) if error else None,
                    "optimizer_updates": 0,
                })
            if error is not None or config.capture.policy == "enforce":
                optimizer_gate(failed, config.device)
            if collect_evidence:
                named_gradients = {name: p.grad.detach().cpu().clone() if p.grad is not None else None
                                   for name, p in model.named_parameters()}
        with phase("gradient_clipping"):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_norm,
                                                 error_if_nonfinite=True, foreach=False)
        with phase("optimizer"):
            optimizer.step()
        with phase("scheduler"):
            scheduler.step()
    if runtime:
        runtime.flush()
    return {"losses": losses, "gradients": named_gradients, "gradient_norm": float(norm),
            "comparison": comparison, "default_check_raised": default_check_raised,
            "incorrect_update_observed": bool(comparison and comparison["failed"]),
            "optimizer_updates": 1}


def run(config, snapshot=None, one_step=False, destination=None):
    config.validate()
    if config.capture.mode == "full":
        if snapshot is None:
            raise ValueError("Start full capture from an immutable snapshot after census")
        storage = preflight(config, snapshot)
    else:
        storage = None
    root = Path(destination) if destination else Path(config.artifact_root) / f"{config.run_id}-{uuid.uuid4().hex[:12]}"
    root.mkdir(parents=True, exist_ok=False)
    write_config(config, root / "config.resolved.toml")
    model, optimizer, scheduler = build(config)
    environment = state.environment()
    state.write_json(root / "environment.json", environment)
    state.archive_source(root / "source.zip", environment)
    state.write_json(root / "manifest.json", {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config.digest(), "environment_sha256": state.sha256(root / "environment.json"),
        "snapshot_sha256": state.sha256(snapshot) if snapshot else None, "storage_preflight": storage,
        "evidence_class": config.evidence_class, "rank": 0, "seed": config.seed,
        "cause": "unknown", "status": "RUNNING"})
    corpus = PackedCorpus(config)
    state.write_json(root / "data_manifest.json", corpus.manifest)
    provenance = snapshot_identity(config, corpus.digest)
    cursor, first_step, saved_batches = 0, 1, None
    if snapshot:
        saved = state.load_snapshot(snapshot, model, optimizer, scheduler)
        for key in ("model", "data_sha256", "dtype", "adapter_name", "seed", "code_sha256", "lock_sha256", "torch", "numpy", "machine", "device"):
            if saved["provenance"][key] != provenance[key]:
                raise ValueError(f"Snapshot provenance mismatch: {key}")
        # stop_after is operational; all schedule/batch/optimizer semantics must match.
        old_training = dict(saved["provenance"]["training"])
        new_training = dict(provenance["training"])
        for value in (old_training, new_training):
            value.pop("stop_after", None)
            value.pop("snapshot_steps", None)
            value.pop("checkpoint_every", None)
        if old_training != new_training:
            raise ValueError("Snapshot training/scheduler configuration differs")
        cursor, first_step, saved_batches = saved["cursor"], saved["step"], saved["batches"]
    metrics = []
    final = None
    summary = {"status": "RUNNING", "steps": metrics, "capture": [], "evidence_class": config.evidence_class}
    try:
        for step in range(first_step, config.training.steps + 1):
            start = time.monotonic()
            batches, next_cursor = collect_batches(corpus, cursor, config)
            if saved_batches is not None:
                for actual, expected in zip(batches, saved_batches, strict=True):
                    if actual["sample_ids"] != expected["sample_ids"] or not torch.equal(actual["input"], expected["input"]) or not torch.equal(actual["target"], expected["target"]):
                        raise ValueError("Snapshot token batch differs from corpus replay")
                saved_batches = None
            capture_step = audited(config, step)
            if step in config.training.snapshot_steps or capture_step:
                state.save_snapshot(root / "states" / f"pre_step_{step}.pt", model, optimizer, scheduler,
                                    cursor, step, batches, provenance,
                                    trigger_state=asdict(config.adapter))
            runtime = None
            if capture_step:
                # One rank shard series per audited step keeps shard/index ownership simple.
                capture_root = root / "captures" / f"step_{step}"
                capture_root.mkdir(parents=True)
                runtime = CaptureRuntime(capture_root, config)
                runtime.step = step
                runtime.attach(model)
                model.runtime = runtime
            try:
                result = execute_step(model, optimizer, scheduler, batches, config, runtime,
                                      collect_evidence=one_step or capture_step or step == config.training.steps or step == config.training.stop_after)
            finally:
                model.runtime = None
                if runtime:
                    try:
                        runtime.close()
                    finally:
                        summary["capture"].append(runtime.summary())
            cursor = next_cursor
            metric = {key: value for key, value in result.items() if key != "gradients"}
            metric.update({"step": step, "cursor": cursor, "seconds": time.monotonic() - start})
            metrics.append(metric)
            if result["gradients"] is not None:
                final = {"model": state.cpu_tree(model.state_dict()), "optimizer": state.cpu_tree(optimizer.state_dict()),
                         "scheduler": scheduler.state_dict(), "gradients": result["gradients"], "rng": state.rng_state(),
                         "cursor": cursor, "step": step, "losses": result["losses"]}
            state.write_json(root / "steps" / str(step) / "STEP_COMMIT", {"step": step, "rank": 0, "status": "committed"})
            if runtime:
                commit = {"step": step, "rank": 0, "tensor_count": runtime.events}
                if runtime.writer:
                    commit["index_sha256"] = state.sha256(runtime.root / "events/rank_0.jsonl")
                state.write_json(runtime.root / "steps" / str(step) / "STEP_COMMIT", commit)
                state.write_json(runtime.root / "summary.json", {**runtime.summary(), "comparison": result["comparison"]})
                report(runtime.root)
            interval = config.training.checkpoint_every
            stop = one_step or result["incorrect_update_observed"] or (config.training.stop_after and step >= config.training.stop_after)
            if (interval and step % interval == 0) or stop or step == config.training.steps:
                next_batches, _ = collect_batches(corpus, cursor, config)
                state.save_snapshot(root / "states" / f"resume_before_{step + 1}.pt", model, optimizer, scheduler,
                                    cursor, step + 1, next_batches, provenance,
                                    trigger_state=asdict(config.adapter))
            if stop:
                break
        summary["status"] = "OBSERVED_MISMATCH" if metrics and metrics[-1]["incorrect_update_observed"] else "PASS"
    except Exception as error:
        summary["status"] = "ENFORCED_ABORT" if isinstance(error, IntegrityError) else "FAILED"
        summary["error"] = f"{type(error).__name__}: {error}"
        final = {"model": state.cpu_tree(model.state_dict()), "optimizer": state.cpu_tree(optimizer.state_dict()),
                 "scheduler": scheduler.state_dict(), "gradients": {n: state.cpu_tree(p.grad) for n, p in model.named_parameters()},
                 "cursor": cursor, "step": len(metrics) + first_step - 1}
    if final is not None:
        state.save_tensor_file(root / "outcome.pt", final)
    state.write_json(root / "summary.json", summary)
    replay = {"config": str((root / "config.resolved.toml").resolve()),
              "config_sha256": state.sha256(root / "config.resolved.toml"),
              "snapshot": str(Path(snapshot).resolve()) if snapshot else "",
              "snapshot_sha256": state.sha256(snapshot) if snapshot else "", "one_step": one_step,
              "code_sha256": environment["code_sha256"]}
    (root / "replay.toml").write_text("\n".join(f"{k} = {json.dumps(v)}" for k, v in replay.items()) + "\n")
    report(root)
    return root, summary


def census(config, snapshot):
    candidate = copy.deepcopy(config)
    candidate.capture.mode = "census"
    root, summary = run(candidate, snapshot, one_step=True)
    if summary["status"] != "PASS":
        raise RuntimeError(f"Census execution failed: {summary}")
    counts = summary["capture"]
    if not counts:
        raise ValueError("Snapshot step is not selected by audit_steps")
    payload = sum(c["payload_bytes"] for c in counts)
    tensor_count = sum(c["captured_tensors"] for c in counts)
    snapshot_size = Path(snapshot).stat().st_size
    audited_steps = config.training.steps if config.capture.audit_steps == "all" else len(config.capture.audit_steps)
    # Both arms, full event metadata, snapshots/outcomes and 25% reserve. This is
    # deliberately conservative; there is no automatic coverage reduction.
    projected = math.ceil(1.25 * (2 * audited_steps * (payload + tensor_count * 8192 + 3 * snapshot_size)))
    result = {"identity": census_identity(config, snapshot), "per_step_payload_bytes": payload,
              "per_step_tensor_count": tensor_count, "audited_steps": audited_steps,
              "projected_bytes_with_margin": projected, "margin": 0.25,
              "scope": "shape-stable configured trainer; two arms and all selected steps",
              "run_dir": str(root.resolve())}
    state.write_json(root / "census.json", result)
    return root / "census.json", result


def compare_state(a, b, prefix=""):
    differences = []
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        result = compare_tensors(a, b)
        if not result["raw_equal"]:
            differences.append({"name": prefix, **result})
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if not np.array_equal(a, b):
            differences.append({"name": prefix, "type": "array"})
    elif isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            differences.append({"name": prefix, "type": "keys"})
        else:
            for key in a:
                differences.extend(compare_state(a[key], b[key], f"{prefix}.{key}"))
    elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            differences.append({"name": prefix, "type": "length"})
        else:
            for index, (x, y) in enumerate(zip(a, b)):
                differences.extend(compare_state(x, y, f"{prefix}.{index}"))
    elif type(a) != type(b) or a != b:
        differences.append({"name": prefix, "original": str(a), "candidate": str(b)})
    return differences


def pair(config, snapshot):
    """Exec a new interpreter for each arm; no model/cache state survives between arms."""
    root = Path(config.artifact_root) / f"pair-{config.run_id}-{uuid.uuid4().hex[:12]}"
    root.mkdir(parents=True)
    arms = {}
    for name, checkpoint_enabled in (("reference", False), ("candidate", True)):
        arm = copy.deepcopy(config)
        arm.checkpoint.enabled = checkpoint_enabled
        arm.run_id = f"{config.run_id}-{name}"
        if arm.capture.mode == "full":
            census_path, _ = census(arm, snapshot)
            arm.capture.census_path = str(census_path.resolve())
        config_path = root / f"{name}.toml"
        write_config(arm, config_path)
        command = [sys.executable, "-m", "ac_integrity.production_cli", "_arm", "--config", str(config_path),
                   "--snapshot", str(Path(snapshot).resolve()), "--destination", str(root / name)]
        with (root / f"{name}.stdout").open("w") as out, (root / f"{name}.stderr").open("w") as err:
            process = subprocess.run(command, stdout=out, stderr=err, timeout=1800)
        arms[name] = {"command": command, "returncode": process.returncode}
        if not (root / name / "outcome.pt").exists():
            state.write_json(root / "summary.json", {"status": "FAILED", "arms": arms})
            raise RuntimeError(f"Fresh arm {name} failed; see {root}")
    left = torch.load(root / "reference/outcome.pt", weights_only=False)
    right = torch.load(root / "candidate/outcome.pt", weights_only=False)
    diffs = {name: compare_state(left.get(name), right.get(name), name)
             for name in ("model", "optimizer", "scheduler", "gradients", "losses", "cursor", "rng")}
    clean = not any(diffs.values()) and all(arm["returncode"] == 0 for arm in arms.values())
    summary = {"status": "EXACT_MATCH" if clean else "DIVERGENCE_OR_ABORT", "arms": arms,
               "differences": diffs, "snapshot_sha256": state.sha256(snapshot),
               "evidence_class": config.evidence_class, "cause": "consistent_with" if config.adapter.trigger else "unknown"}
    state.write_json(root / "summary.json", summary)
    report(root)
    return root, summary
