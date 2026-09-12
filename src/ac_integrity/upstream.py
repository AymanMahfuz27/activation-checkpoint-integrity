"""Adapter-free compatibility screen using unmodified Transformers Llama code.

The model and native checkpoint API come from the pinned upstream package.
The training loop and optional observer integration belong to this project;
this is not a claim of unmodified Trainer/TorchTitan or fused-internal coverage.
"""

import argparse
import copy
from contextlib import nullcontext
from datetime import datetime, timezone
import functools
import inspect
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
from ac_integrity import state
from ac_integrity.capture.compare import compare_events, validate_artifacts
from ac_integrity.capture.dispatch import CaptureMode
from ac_integrity.capture.runtime import CaptureRuntime
from ac_integrity.config import load_config, write_config
from ac_integrity.data import PackedCorpus
from ac_integrity.train import collect_batches, compare_state


def build_model(config, attention="eager", dropout=0.0):
    import transformers
    from transformers import LlamaConfig, LlamaForCausalLM
    if transformers.__version__ != "5.16.1":
        raise RuntimeError(f"Expected transformers5.16.1, got {transformers.__version__}")
    m = config.model
    options = LlamaConfig(vocab_size=m.vocab_size, hidden_size=m.width,
        intermediate_size=m.intermediate, num_hidden_layers=m.layers,
        num_attention_heads=m.heads, num_key_value_heads=m.kv_heads,
        max_position_embeddings=max(2048, config.training.sequence_length),
        rms_norm_eps=m.norm_eps, attention_dropout=dropout,
        tie_word_embeddings=True, use_cache=False, attn_implementation=attention)
    model = LlamaForCausalLM(options).to(config.device).train()
    return model


def attach_observer(model, runtime):
    """Install per-invocation contexts at the native checkpoint callback boundary."""
    runtime.attach(model)
    for index, layer in enumerate(model.model.layers):
        def observed_checkpoint(function, *args, name=f"llama.layers.{index}", **kwargs):
            return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=True,
                              context_fn=runtime.checkpoint_contexts(name), **kwargs)
        layer._gradient_checkpointing_func = observed_checkpoint


def arm(config, root, *, cell, checkpointed, recorded, steps, backend):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    state.configure(config)
    state.write_json(root / "environment.json", state.environment())
    precision = "bfloat16" if "bf16" in cell else "float16" if "fp16" in cell else "float32"
    compiling = cell.startswith("compile")
    summary = {"cell": cell, "checkpointed": checkpointed, "recorded": recorded,
               "precision": precision, "compiler_backend": backend if compiling else None,
               "started_at": datetime.now(timezone.utc).isoformat(), "status": "RUNNING",
               "adapter": "none", "steps": [], "capture_scope": "none"}
    try:
        if recorded and compiling:
            raise NotImplementedError("Dispatch full capture is not supported inside compiled graphs")
        if config.device.startswith("cuda") and precision == "bfloat16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise NotImplementedError("This GPU does not support native BF16")
        if config.device.startswith("cuda") and compiling and backend == "inductor" and torch.cuda.get_device_capability()[0] < 7:
            raise NotImplementedError("Inductor/Triton GPU kernels require newer hardware than Pascal")
        attention = "sdpa" if "sdpa" in cell else "eager"
        model = build_model(config, attention, 0.1 if "dropout" in cell else 0.0)
        import transformers.models.llama.modeling_llama as source
        summary["upstream"] = {"package": "transformers", "version": "5.16.1",
            "model_source_sha256": state.sha256(inspect.getfile(source)),
            "model_source": "https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/llama/modeling_llama.py",
            "parameters": sum(p.numel() for p in model.parameters()), "attention": attention}
        if checkpointed:
            model.gradient_checkpointing_enable({"use_reentrant": False, "preserve_rng_state": True})
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                                      foreach=False, fused=False)
        scaler = torch.amp.GradScaler(config.device.split(":")[0], enabled=precision == "float16")
        corpus = PackedCorpus(config)
        initial = {"model": state.cpu_tree(model.state_dict()), "optimizer": optimizer.state_dict(),
                   "scaler": scaler.state_dict(), "rng": state.rng_state(), "corpus_sha256": corpus.digest}
        # Persist the initial snapshot for every independent execution; the matrix
        # verifies equality, rather than assuming seeds imply equal starting state.
        state.save_tensor_file(root / "initial.pt", initial)
        callable_model = torch.compile(model, backend=backend, fullgraph=True) if compiling else model
        cursor = 0
        for step in range(1, steps + 1):
            runtime = None
            if recorded:
                config.capture.mode, config.capture.policy = "full", "observe"
                config.capture.encoding, config.capture.deduplicate = "zlib", True
                runtime = CaptureRuntime(root / f"capture_{step}", config)
                runtime.step = step
                attach_observer(model, runtime)
                summary["capture_scope"] = "eager visible dense operator outputs; fused internals excluded"
            optimizer.zero_grad(set_to_none=True)
            batches, cursor = collect_batches(corpus, cursor, config)
            losses = []
            try:
                with CaptureMode(runtime) if runtime else nullcontext():
                    for microbatch, batch in enumerate(batches):
                        if runtime:
                            runtime.microbatch, runtime.phase = microbatch, "forward"
                        # AMP intentionally surrounds forward/loss, not backward.
                        # Native checkpointing is responsible for replaying it.
                        with set_checkpoint_early_stop(False), torch.autocast(
                                config.device.split(":")[0], dtype=getattr(torch, precision),
                                enabled=precision != "float32"):
                            logits = callable_model(input_ids=batch["input"].to(config.device), use_cache=False).logits
                            loss = F.cross_entropy(logits.float().flatten(0, 1),
                                                   batch["target"].to(config.device).flatten())
                            scaled = loss / config.training.accumulation
                        if runtime:
                            runtime.phase = "backward"
                        scaler.scale(scaled).backward()
                        losses.append(float(loss.detach()))
                if runtime:
                    runtime.flush()
                    comparison = compare_events(runtime.root, step)
                else:
                    comparison = None
                scaler.unscale_(optimizer)
                gradients = {n: state.cpu_tree(p.grad) for n, p in model.named_parameters()}
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                      error_if_nonfinite=True, foreach=False)
                count = []
                hook = optimizer.register_step_pre_hook(lambda *args: count.append(True))
                try:
                    scaler.step(optimizer)
                    scaler.update()
                finally:
                    hook.remove()
                metric = {"step": step, "losses": losses, "gradient_norm": float(norm),
                          "optimizer_step_calls": len(count), "comparison": comparison}
                summary["steps"].append(metric)
                outcome = {"model": state.cpu_tree(model.state_dict()),
                           "optimizer": state.cpu_tree(optimizer.state_dict()),
                           "gradients": gradients, "losses": losses, "rng": state.rng_state(),
                           "scaler": scaler.state_dict(), "cursor": cursor}
                state.save_tensor_file(root / f"step_{step}.pt", outcome)
            finally:
                if runtime:
                    runtime.close()
            if runtime:
                commit = {"step": step, "rank": 0, "tensor_count": runtime.events,
                          "index_sha256": state.sha256(runtime.root / "events/rank_0.jsonl")}
                state.write_json(runtime.root / "steps" / str(step) / "STEP_COMMIT", commit)
                state.write_json(runtime.root / "summary.json", runtime.summary())
                metric["artifact_validation"] = validate_artifacts(runtime.root)
                metric["capture"] = runtime.summary()
                if not metric["artifact_validation"]["valid"]:
                    raise RuntimeError("Captured evidence failed validation")
            state.write_json(root / "summary.json", summary)
        summary["status"] = "PASS"
        if compiling:
            from torch._dynamo.utils import counters
            summary["compiler_counters"] = {k: dict(v) for k, v in counters.items()}
        if torch.cuda.is_initialized():
            summary["peak_cuda_allocated"] = torch.cuda.max_memory_allocated()
    except NotImplementedError as error:
        summary.update(status="UNSUPPORTED", error=str(error))
    except Exception as error:
        summary.update(status="ERROR", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    state.write_json(root / "summary.json", summary)
    return summary


def matrix(config, root, cells, steps, backend, record):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_config(config, root / "config.toml")
    environment = state.environment()
    state.write_json(root / "environment.json", environment)
    state.archive_source(root / "source.zip", environment)
    results = {}
    for cell in cells:
        cell_root = root / cell
        cell_root.mkdir()
        results[cell] = {"arms": {}}
        variants = [("reference", False, False), ("reference_repeat", False, False),
                    ("checkpoint", True, False)]
        if record and not cell.startswith("compile"):
            variants.append(("recorded", True, True))
        for name, checkpointed, recorded in variants:
            command = [sys.executable, "-m", "ac_integrity.upstream", "arm", "--config", str(root / "config.toml"),
                "--output", str(cell_root / name), "--cells", cell, "--steps", str(steps), "--backend", backend]
            if checkpointed:
                command.append("--checkpoint")
            if recorded:
                command.append("--record")
            print(json.dumps({"cell": cell, "arm": name}), flush=True)
            with (cell_root / f"{name}.stdout").open("w") as out, (cell_root / f"{name}.stderr").open("w") as err:
                process = subprocess.run(command, stdout=out, stderr=err, timeout=7200)
            path = cell_root / name / "summary.json"
            result = json.loads(path.read_text()) if path.exists() else {"status": "PROCESS_FAILED"}
            results[cell]["arms"][name] = {**result, "command": command, "returncode": process.returncode}
        differences = {}
        for label, left, right in [("repeatability", "reference", "reference_repeat"),
                                   ("checkpoint_effect", "reference", "checkpoint"),
                                   ("recording_effect", "checkpoint", "recorded")]:
            if left not in results[cell]["arms"] or right not in results[cell]["arms"]:
                continue
            if any(results[cell]["arms"][n]["status"] != "PASS" for n in [left, right]):
                continue
            differences[label] = {}
            for filename in ["initial.pt"] + [f"step_{i}.pt" for i in range(1, steps + 1)]:
                a = torch.load(cell_root / left / filename, weights_only=False)
                b = torch.load(cell_root / right / filename, weights_only=False)
                differences[label][filename] = compare_state(a, b)
        state.write_json(cell_root / "differences.json", differences)
        exact = lambda key: key in differences and not any(differences[key].values())
        statuses = {v["status"] for v in results[cell]["arms"].values()}
        if statuses == {"UNSUPPORTED"}:
            verdict = "UNSUPPORTED"
        elif statuses != {"PASS"}:
            verdict = "EXECUTION_ERROR"
        elif not exact("repeatability"):
            verdict = "BASELINE_NOT_REPEATABLE"
        elif "recording_effect" in differences and not exact("recording_effect"):
            verdict = "OBSERVER_CHANGED_EXECUTION"
        elif not exact("checkpoint_effect"):
            verdict = "CHECKPOINT_DIFFERENCE_REQUIRES_ANALYSIS"
        elif any(step.get("comparison", {}).get("failed") for value in results[cell]["arms"].values()
                 for step in value.get("steps", []) if step.get("comparison")):
            verdict = "TRACE_MISMATCH_WITH_EQUAL_OUTCOME"
        else:
            verdict = "EXACT_MATCH"
        results[cell].update(verdict=verdict,
                             differing_entries={k: sum(len(rows) for rows in v.values()) for k, v in differences.items()})
        state.write_json(root / "summary.json", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["arm", "matrix"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--cells", default="eager_fp32,sdpa_fp32,dropout_fp32,compile_fp32")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--backend", default="aot_eager", choices=["aot_eager", "inductor"])
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config.device, config.expected_torch = args.device, torch.__version__
    if config.adapter.name != "none" or config.adapter.trigger:
        raise ValueError("Upstream screening forbids the reproduction adapter")
    if args.action == "arm":
        result = arm(config, args.output, cell=args.cells, checkpointed=args.checkpoint,
                     recorded=args.record, steps=args.steps, backend=args.backend)
        return 0 if result["status"] in {"PASS", "UNSUPPORTED"} else 1
    results = matrix(config, args.output, args.cells.split(","), args.steps, args.backend, args.record)
    return 1 if any(cell["verdict"] == "EXECUTION_ERROR" for cell in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
