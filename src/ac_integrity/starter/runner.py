"""Isolated arm execution, experiment gates, and reproducible artifact manifests."""

from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from ac_integrity.starter.capture import (
    ArtifactWriter,
    CapturedTensor,
    SavedTensorObserver,
    TensorRecorder,
    compare_tensors,
    pair_records,
    tensor_metadata,
)
from ac_integrity.starter.cases import VALID_CASES, make_controller
from ac_integrity.starter.fixture import (
    TAGS,
    FixtureTensors,
    fixture_forward,
    independent_reference,
    make_fixture,
)


SEED = 20260903
CHECKPOINT_ARGUMENTS = {
    "use_reentrant": False,
    "early_stop": False,
    "preserve_rng_state": True,
    "determinism_check": "default",
    "context_fn": "original/recompute phase contexts",
}


@dataclass
class RunOutcome:
    exit_code: int
    run_id: str
    run_dir: Path
    summary: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id(experiment: str, label: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{experiment.lower()}-{label}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _git_state() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--short")
        return {"commit": commit, "dirty": bool(status), "status_short": status.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": "unknown", "status_short": []}


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu_name": properties.name,
                "gpu_memory_bytes": properties.total_memory,
                "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    return result


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {name}")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    values = {"float64": torch.float64, "float32": torch.float32}
    if name not in values:
        raise ValueError(f"unsupported dtype: {name}")
    return values[name]


def _tensor_exact(left: torch.Tensor, right: torch.Tensor) -> bool:
    if tensor_metadata(left) != tensor_metadata(right):
        return False
    captured_left = CapturedTensor(left.detach().cpu(), tensor_metadata(left), "value", "left", "", "", 0)
    captured_right = CapturedTensor(right.detach().cpu(), tensor_metadata(right), "value", "right", "", "", 0)
    return bool(compare_tensors(captured_left, captured_right)["exact_equal"])


def _autograd_baseline(
    *, device: torch.device, dtype: torch.dtype, original_transform: Any
) -> dict[str, Any]:
    """Run the ordinary non-checkpoint graph to validate the closed-form oracle."""

    tensors = make_fixture(device=device, dtype=dtype)
    h = original_transform(tensors.x @ tensors.w1) + tensors.b
    g = h.square() + 0.5 * h
    y = g @ tensors.w2
    loss = (y * tensors.target).sum()
    loss.backward()
    gradients = {
        name: tensor.grad.detach().clone()
        for name, tensor in tensors.differentiable().items()
    }
    return {"h": h.detach(), "g": g.detach(), "y": y.detach(), "loss": loss.detach(), "gradients": gradients}


def _gradient_report(
    observed: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> dict[str, Any]:
    exact = {name: _tensor_exact(observed[name], expected[name]) for name in sorted(expected)}
    return {
        "exact_by_name": exact,
        "different_names": [name for name, matches in exact.items() if not matches],
        "observed": {name: observed[name].detach().cpu().tolist() for name in sorted(observed)},
        "expected": {name: expected[name].detach().cpu().tolist() for name in sorted(expected)},
    }


def _allclose(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.allclose(left, right, rtol=1e-6, atol=1e-6, equal_nan=True))


def _base_manifest(
    *,
    run_id: str,
    experiment: str,
    case: str,
    mode: str,
    variant: str | None,
    command: list[str],
    device: torch.device,
    dtype: torch.dtype,
    started_at: str,
    initial_state: dict[str, object],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment": experiment,
        "case": case,
        "mode": mode,
        "variant": variant,
        "exact_command": shlex.join(command),
        "git": _git_state(),
        "environment": _environment(device),
        "device": str(device),
        "dtype": str(dtype),
        "seed": SEED,
        "initial_state": initial_state,
        "checkpoint_arguments": CHECKPOINT_ARGUMENTS,
        "started_at": started_at,
        "ended_at": None,
        "duration_seconds": None,
        "exit_status": None,
        "condor": {
            "cluster_id": os.environ.get("CONDOR_CLUSTER_ID"),
            "process_id": os.environ.get("CONDOR_PROCESS_ID"),
        },
    }


def run_arm(
    *,
    experiment: str,
    case: str,
    mode: str,
    variant: str | None,
    device_name: str,
    dtype_name: str,
    artifact_root: Path,
    hooks: bool = False,
    tagged_save: bool = False,
    run_id: str | None = None,
    command: list[str] | None = None,
) -> RunOutcome:
    """Execute one cleanly initialized checkpoint arm and persist all evidence."""

    label = "-".join(part for part in (case, variant, mode, "hooks" if hooks else None) if part)
    run_id = run_id or _new_run_id(experiment, label)
    writer = ArtifactWriter(artifact_root, run_id)
    started_wall = time.monotonic()
    started_at = _utc_now()
    command = command or sys.argv
    manifest: dict[str, Any] | None = None
    try:
        device = resolve_device(device_name)
        dtype = resolve_dtype(dtype_name)
        if experiment == "E0" and device.type == "cpu" and dtype != torch.float64:
            raise ValueError("E0 CPU calibration requires float64")
        if case == "precision" and dtype != torch.float32:
            raise ValueError("the precision-policy case requires a float32 tagged boundary")

        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        controller = make_controller(case, mode=mode, seed=SEED, device=device, variant=variant)
        manifest = _base_manifest(
            run_id=run_id,
            experiment=experiment,
            case=case,
            mode=mode,
            variant=variant,
            command=command,
            device=device,
            dtype=dtype,
            started_at=started_at,
            initial_state=controller.initial_state(),
        )
        tensors = make_fixture(device=device, dtype=dtype)
        reference = independent_reference(tensors, controller.original_transform)
        baseline = _autograd_baseline(
            device=device, dtype=dtype, original_transform=controller.original_transform
        )
        reference_gradients = reference["gradients"]
        assert isinstance(reference_gradients, dict)
        baseline_gradient_report = _gradient_report(baseline["gradients"], reference_gradients)
        baseline_formula_equal = (
            _allclose(baseline["h"], reference["h"])
            and _allclose(baseline["g"], reference["g"])
            and _allclose(baseline["y"], reference["y"])
            and _allclose(baseline["loss"], reference["loss"])
            and all(
                _allclose(baseline["gradients"][name], reference_gradients[name])
                for name in reference_gradients
            )
        )

        recorder = TensorRecorder(run_id=run_id, experiment=experiment, case=case)
        observer = SavedTensorObserver(recorder) if hooks else None
        context_fn = observer.context_fn if observer is not None else recorder.context_fn

        def checkpointed_function(
            x: torch.Tensor, w1: torch.Tensor, b: torch.Tensor, w2: torch.Tensor
        ) -> torch.Tensor:
            current = FixtureTensors(x=x, w1=w1, b=b, w2=w2, target=tensors.target)
            return fixture_forward(
                current, recorder, controller.transform, tagged_save=tagged_save
            )

        default_check_raised = False
        backward_error: str | None = None
        with set_checkpoint_early_stop(False):
            output = checkpoint(
                checkpointed_function,
                tensors.x,
                tensors.w1,
                tensors.b,
                tensors.w2,
                use_reentrant=False,
                preserve_rng_state=True,
                determinism_check="default",
                context_fn=context_fn,
            )
        controller.after_forward()
        loss = (output * tensors.target).sum()
        try:
            loss.backward()
        except torch.utils.checkpoint.CheckpointError as error:
            default_check_raised = True
            backward_error = f"{type(error).__name__}: {error}"
        except Exception as error:  # recorded as environment/setup failure below
            backward_error = f"{type(error).__name__}: {error}"

        observed_gradients = {
            name: tensor.grad.detach().clone()
            for name, tensor in tensors.differentiable().items()
            if tensor.grad is not None
        }
        gradient_report = (
            _gradient_report(observed_gradients, baseline["gradients"])
            if len(observed_gradients) == len(baseline["gradients"])
            else {
                "exact_by_name": {},
                "different_names": sorted(set(baseline["gradients"]) - set(observed_gradients)),
                "observed": {},
                "expected": {name: value.detach().cpu().tolist() for name, value in baseline["gradients"].items()},
            }
        )

        pairs, pairing = pair_records(recorder, TAGS)
        tensor_paths: dict[tuple[str, str], Path] = {}
        for phase, tags in recorder.records.items():
            for tag, records in tags.items():
                if len(records) == 1:
                    tensor_paths[(phase, tag)] = writer.save_tensor(
                        case, tag, phase, records[0].tensor
                    )

        comparisons: list[dict[str, Any]] = []
        for original, recompute in pairs:
            comparison = compare_tensors(original, recompute)
            comparison["original_tensor_path"] = str(tensor_paths[("original", original.tag)])
            comparison["recompute_tensor_path"] = str(tensor_paths[("recompute", recompute.tag)])
            comparisons.append(comparison)

        exact_matches = sum(bool(row["exact_equal"]) for row in comparisons)
        value_mismatches = sum(
            bool(row["metadata_equal"] and not row["exact_equal"]) for row in comparisons
        )
        metadata_mismatches = sum(not bool(row["metadata_equal"]) for row in comparisons)
        first_mismatch = next(
            (row for tag in TAGS for row in comparisons if row["tag"] == tag and not row["exact_equal"]),
            None,
        )
        output_equal = _tensor_exact(output.detach(), baseline["y"])
        loss_equal = _tensor_exact(loss.detach(), baseline["loss"])
        common_gate = (
            backward_error is None
            and recorder.phase_counts == {"original": 1, "recompute": 1}
            and len(pairs) == 3
            and not pairing["missing_events"]
            and not pairing["duplicate_events"]
            and output_equal
            and loss_equal
            and (baseline_formula_equal if experiment == "E0" else True)
        )
        if mode == "broken":
            scientific_gate = (
                common_gate
                and first_mismatch is not None
                and first_mismatch["tag"] == "h"
                and bool(first_mismatch["metadata_equal"])
                and bool(gradient_report["different_names"])
                and not default_check_raised
            )
        else:
            scientific_gate = (
                common_gate
                and exact_matches == 3
                and value_mismatches == 0
                and metadata_mismatches == 0
                and not gradient_report["different_names"]
            )

        hook_summary = _summarize_hooks(observer) if observer is not None else None
        classification = "PASS" if scientific_gate else "FAIL"
        if experiment == "E2" and hooks:
            hook_gate = _e2_hook_gate(recorder, hook_summary)
            scientific_gate = scientific_gate and hook_gate
            classification = "PASS" if scientific_gate else "PUBLIC_HOOKS_INSUFFICIENT"

        summary: dict[str, Any] = {
            "run_id": run_id,
            "experiment": experiment,
            "case": case,
            "mode": mode,
            "variant": variant,
            "expected_tags": list(TAGS),
            "observed_tags": {
                phase: {tag: len(records) for tag, records in tags.items()}
                for phase, tags in recorder.records.items()
            },
            "phase_counts": recorder.phase_counts,
            "pair_count": len(pairs),
            "exact_matches": exact_matches,
            "same_metadata_value_mismatches": value_mismatches,
            "metadata_mismatches": metadata_mismatches,
            **pairing,
            "first_mismatch": first_mismatch,
            "output_matches_original_reference": output_equal,
            "loss_matches_original_reference": loss_equal,
            "no_checkpoint_matches_independent_formulas": baseline_formula_equal,
            "baseline_gradient_comparison": baseline_gradient_report,
            "gradient_comparison": gradient_report,
            "default_pytorch_check_raised": default_check_raised,
            "backward_error": backward_error,
            "trigger_removal_restored_equality": mode == "trigger_off" and scientific_gate,
            "hook_coverage": hook_summary,
            "result": classification,
        }
        events = recorder.events + (observer.events if observer is not None else [])
        writer.write_jsonl("events.jsonl", events)
        writer.write_jsonl("comparisons.jsonl", comparisons)
        writer.write_json("summary.json", summary)
        exit_code = 0 if scientific_gate else (2 if backward_error and not default_check_raised else 1)
    except (ValueError, RuntimeError, NotImplementedError) as error:
        summary = {
            "run_id": run_id,
            "experiment": experiment,
            "case": case,
            "mode": mode,
            "variant": variant,
            "result": "BLOCKED",
            "environment_error": f"{type(error).__name__}: {error}",
        }
        writer.write_jsonl("events.jsonl", [])
        writer.write_jsonl("comparisons.jsonl", [])
        writer.write_json("summary.json", summary)
        exit_code = 2
        if manifest is None:
            manifest = {
                "run_id": run_id,
                "experiment": experiment,
                "case": case,
                "mode": mode,
                "variant": variant,
                "exact_command": shlex.join(command),
                "git": _git_state(),
                "started_at": started_at,
            }

    ended_at = _utc_now()
    manifest["ended_at"] = ended_at
    manifest["duration_seconds"] = time.monotonic() - started_wall
    manifest["exit_status"] = exit_code
    writer.write_json("manifest.json", manifest)
    return RunOutcome(exit_code=exit_code, run_id=run_id, run_dir=writer.run_dir, summary=summary)


def _summarize_hooks(observer: SavedTensorObserver) -> dict[str, Any]:
    rows: dict[str, dict[str, dict[str, int]]] = {
        phase: {tag: {"pack": 0, "unpack": 0, "access": 0} for tag in TAGS}
        for phase in ("original", "recompute")
    }
    matches: dict[str, dict[str, int]] = {
        phase: {tag: 0 for tag in TAGS} for phase in ("original", "recompute")
    }
    for event in observer.events:
        phase = event.get("phase")
        tag = event.get("tag")
        if phase not in rows or tag not in TAGS:
            continue
        if event["kind"] == "hook_pack" and "semantic_token" in event:
            matches[phase][tag] += 1
            rows[phase][tag]["pack"] += 1
        elif event["kind"] == "hook_unpack":
            rows[phase][tag]["unpack"] += 1
        elif event["kind"] == "hook_access":
            rows[phase][tag]["access"] += 1
    return {"by_phase_tag": rows, "direct_matches": matches, "event_count": len(observer.events)}


def _e2_hook_gate(recorder: TensorRecorder, hook_summary: dict[str, Any]) -> bool:
    if recorder.phase_counts != {"original": 1, "recompute": 1}:
        return False
    for phase in ("original", "recompute"):
        for tag in TAGS:
            counts = hook_summary["by_phase_tag"][phase][tag]
            if hook_summary["direct_matches"][phase][tag] != 1:
                return False
            if counts["pack"] != 1 or counts["unpack"] < 1 or counts["access"] < 1:
                return False
    return True


def _read_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text())


def _spawn_arm(
    *,
    experiment: str,
    case: str,
    mode: str,
    variant: str | None,
    device: str,
    dtype: str,
    artifact_root: Path,
    hooks: bool = False,
    tagged_save: bool = False,
) -> RunOutcome:
    command = [
        sys.executable,
        "-m",
        "ac_integrity.cli",
        "_arm",
        experiment,
        "--case",
        case,
        "--mode",
        mode,
        "--device",
        device,
        "--dtype",
        dtype,
        "--artifact-root",
        str(artifact_root),
    ]
    if variant is not None:
        command.extend(["--variant", variant])
    if hooks:
        command.append("--hooks")
    if tagged_save:
        command.append("--tagged-save")
    completed = subprocess.run(command, capture_output=True, text=True)
    if not completed.stdout.strip():
        raise RuntimeError(
            f"isolated arm emitted no result (exit {completed.returncode}): {completed.stderr.strip()}"
        )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    run_dir = Path(payload["run_dir"])
    summary = _read_summary(run_dir)
    return RunOutcome(completed.returncode, payload["run_id"], run_dir, summary)


def _scientific_signature(summary: dict[str, Any]) -> dict[str, Any]:
    signature = {
        key: summary.get(key)
        for key in (
            "phase_counts",
            "pair_count",
            "exact_matches",
            "same_metadata_value_mismatches",
            "metadata_mismatches",
            "missing_events",
            "duplicate_events",
            "first_mismatch",
            "gradient_comparison",
            "default_pytorch_check_raised",
            "result",
        )
    }
    first_mismatch = signature.get("first_mismatch")
    if first_mismatch is not None:
        signature["first_mismatch"] = {
            key: value
            for key, value in first_mismatch.items()
            if key not in {"original_tensor_path", "recompute_tensor_path", "pair_id"}
        }
    return signature


def _originals_equal(left_dir: Path, right_dir: Path, case: str) -> bool:
    for tag in TAGS:
        left = torch.load(left_dir / "tensors" / case / f"{tag}.original.pt", weights_only=True)
        right = torch.load(right_dir / "tensors" / case / f"{tag}.original.pt", weights_only=True)
        if not _tensor_exact(left, right):
            return False
    return True


def _all_tensor_contents_equal(left_dir: Path, right_dir: Path, case: str) -> bool:
    for tag in TAGS:
        for phase in ("original", "recompute"):
            left = torch.load(left_dir / "tensors" / case / f"{tag}.{phase}.pt", weights_only=True)
            right = torch.load(right_dir / "tensors" / case / f"{tag}.{phase}.pt", weights_only=True)
            if not _tensor_exact(left, right):
                return False
    return True


def _aggregate_writer(
    *, artifact_root: Path, experiment: str, label: str, command: list[str]
) -> tuple[str, ArtifactWriter, dict[str, Any], float]:
    run_id = _new_run_id(experiment, label)
    writer = ArtifactWriter(artifact_root, run_id)
    start = time.monotonic()
    device = torch.device("cpu")
    manifest = {
        "run_id": run_id,
        "experiment": experiment,
        "exact_command": shlex.join(command),
        "git": _git_state(),
        "environment": _environment(device),
        "started_at": _utc_now(),
        "ended_at": None,
        "duration_seconds": None,
        "exit_status": None,
    }
    return run_id, writer, manifest, start


def run_e1(
    *,
    selected_case: str,
    variant: str | None,
    repetitions: int,
    device: str,
    artifact_root: Path,
    command: list[str] | None = None,
) -> RunOutcome:
    """Run correct, broken, and trigger-off arms in fresh processes for E1."""

    command = command or sys.argv
    run_id, writer, manifest, start = _aggregate_writer(
        artifact_root=artifact_root, experiment="E1", label=selected_case, command=command
    )
    scenarios: list[tuple[str, str | None]] = []
    cases = VALID_CASES if selected_case == "all" else (selected_case,)
    for case in cases:
        if case == "rng":
            variants = (variant,) if variant else ("python", "numpy")
            scenarios.extend((case, item) for item in variants)
        else:
            scenarios.append((case, None))

    results: list[dict[str, Any]] = []
    exit_code = 0
    for case, case_variant in scenarios:
        repeated: list[dict[str, RunOutcome]] = []
        for repetition in range(repetitions):
            arms = {
                mode: _spawn_arm(
                    experiment="E1",
                    case=case,
                    mode=mode,
                    variant=case_variant,
                    device=device,
                    dtype="float32",
                    artifact_root=artifact_root,
                )
                for mode in ("correct", "broken", "trigger_off")
            }
            repeated.append(arms)
            if any(outcome.exit_code == 2 for outcome in arms.values()):
                exit_code = 2
            elif any(outcome.exit_code != 0 for outcome in arms.values()) and exit_code == 0:
                exit_code = 1

        original_matches = all(
            _originals_equal(item["correct"].run_dir, item["broken"].run_dir, case)
            and _originals_equal(item["correct"].run_dir, item["trigger_off"].run_dir, case)
            for item in repeated
        )
        signatures_repeat = all(
            _scientific_signature(repeated[0][mode].summary)
            == _scientific_signature(item[mode].summary)
            and _all_tensor_contents_equal(repeated[0][mode].run_dir, item[mode].run_dir, case)
            for item in repeated[1:]
            for mode in ("correct", "broken", "trigger_off")
        )
        trigger_restored = all(
            item["trigger_off"].summary["result"] == "PASS"
            and _scientific_signature(item["correct"].summary)
            == _scientific_signature(item["trigger_off"].summary)
            for item in repeated
        )
        scenario_pass = (
            exit_code != 2
            and all(
                item[mode].summary["result"] == "PASS"
                for item in repeated
                for mode in ("correct", "broken", "trigger_off")
            )
            and original_matches
            and signatures_repeat
            and trigger_restored
        )
        if not scenario_pass and exit_code == 0:
            exit_code = 1
        results.append(
            {
                "case": case,
                "variant": case_variant,
                "repetitions": [
                    {
                        "index": index,
                        "arms": {
                            mode: {
                                "run_id": outcome.run_id,
                                "run_dir": str(outcome.run_dir),
                                "result": outcome.summary["result"],
                            }
                            for mode, outcome in item.items()
                        },
                    }
                    for index, item in enumerate(repeated)
                ],
                "broken_original_matches_correct": original_matches,
                "repeat_scientific_and_tensor_contents_match": signatures_repeat,
                "trigger_removal_restored_equality": trigger_restored,
                "result": "PASS" if scenario_pass else ("BLOCKED" if exit_code == 2 else "FAIL"),
            }
        )

    summary = {
        "run_id": run_id,
        "experiment": "E1",
        "selected_case": selected_case,
        "selected_variant": variant,
        "repetitions": repetitions,
        "scenario_count": len(results),
        "scenarios": results,
        "fresh_process_isolation": True,
        "result": "PASS" if exit_code == 0 else ("BLOCKED" if exit_code == 2 else "FAIL"),
    }
    writer.write_jsonl("events.jsonl", [])
    writer.write_jsonl("comparisons.jsonl", [])
    writer.write_json("summary.json", summary)
    manifest["ended_at"] = _utc_now()
    manifest["duration_seconds"] = time.monotonic() - start
    manifest["exit_status"] = exit_code
    writer.write_json("manifest.json", manifest)
    return RunOutcome(exit_code, run_id, writer.run_dir, summary)


def run_e2(
    *,
    device: str,
    dtype: str,
    artifact_root: Path,
    command: list[str] | None = None,
) -> RunOutcome:
    """Compare isolated no-hook and public-hook TaggedSave checkpoint runs."""

    command = command or sys.argv
    run_id, writer, manifest, start = _aggregate_writer(
        artifact_root=artifact_root, experiment="E2", label="coverage", command=command
    )
    baseline = _spawn_arm(
        experiment="E2",
        case="counter",
        mode="correct",
        variant=None,
        device=device,
        dtype=dtype,
        artifact_root=artifact_root,
        tagged_save=True,
    )
    candidate = _spawn_arm(
        experiment="E2",
        case="counter",
        mode="correct",
        variant=None,
        device=device,
        dtype=dtype,
        artifact_root=artifact_root,
        hooks=True,
        tagged_save=True,
    )
    behavior_equal = (
        baseline.summary.get("output_matches_original_reference")
        == candidate.summary.get("output_matches_original_reference")
        and baseline.summary.get("gradient_comparison")
        == candidate.summary.get("gradient_comparison")
        and baseline.summary.get("phase_counts") == candidate.summary.get("phase_counts")
        and _originals_equal(baseline.run_dir, candidate.run_dir, "counter")
    )
    exit_code = 0 if baseline.exit_code == 0 and candidate.exit_code == 0 and behavior_equal else 1
    if baseline.exit_code == 2 or candidate.exit_code == 2:
        exit_code = 2
    classification = "PASS" if exit_code == 0 else (
        "BLOCKED" if exit_code == 2 else "PUBLIC_HOOKS_INSUFFICIENT"
    )
    summary = {
        "run_id": run_id,
        "experiment": "E2",
        "baseline_run_id": baseline.run_id,
        "baseline_run_dir": str(baseline.run_dir),
        "candidate_run_id": candidate.run_id,
        "candidate_run_dir": str(candidate.run_dir),
        "baseline_result": baseline.summary["result"],
        "candidate_result": candidate.summary["result"],
        "hooks_preserve_behavior": behavior_equal,
        "public_hook_coverage": candidate.summary.get("hook_coverage"),
        "result": classification,
    }
    writer.write_jsonl("events.jsonl", [])
    writer.write_jsonl("comparisons.jsonl", [])
    writer.write_json("summary.json", summary)
    manifest["ended_at"] = _utc_now()
    manifest["duration_seconds"] = time.monotonic() - start
    manifest["exit_status"] = exit_code
    writer.write_json("manifest.json", manifest)
    return RunOutcome(exit_code, run_id, writer.run_dir, summary)
