"""Allocated GELU replay provenance; changes observations, never training rules.

Inputs are the pinned historical source, data and corrected update-210 state.
Outputs are every arm, exact consumer joins, gradient/state parity and failures.
No saved-tensor hook or eager read of autograd saved state is installed.
"""

from collections import Counter
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from ac_integrity.research import completion
from ac_integrity.research.m4m5 import FaultSpec


REGION = ContextVar("ba_gelu_region", default=None)
CASES = ((6, 0), (0, 0), (11, 0), (6, 22), (0, 22), (11, 22))
PRIMARY_EVENT = {"element": 6154933, "old_bits_u32": 3198802410,
                 "new_bits_u32": 3198802411, "role": "recompute"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, value):
    """Write a complete new record without replacing an earlier result."""
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def exact_equal(result):
    return result.get("valid") is True and result.get("byte_equal") is True


def tensor_record(value, element):
    """Snapshot logical FP32 bytes; transfers are diagnostic, outside timing claims."""
    with _disable_current_modes():
        require(value.dtype == torch.float32, "GELU trace requires FP32")
        plain = value.detach().contiguous()
        raw = plain.view(torch.uint8).cpu().numpy().tobytes()
        return {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "bytes": len(raw),
            "selected_bits_u32": int(plain.reshape(-1).view(torch.int32)[element].item()) & 0xFFFFFFFF,
        }


@contextmanager
def tracked_intervention(counter):
    """Preserve the original intervention; expose only its existing role context."""
    original_class = completion.Intervention

    class Tracked(original_class):
        @contextmanager
        def context(self, region, role):
            with super().context(region, role):
                counter[(region, role)] += 1
                token = REGION.set((region, role))
                try:
                    yield
                finally:
                    REGION.reset(token)

    completion.Intervention = Tracked
    try:
        yield
    finally:
        completion.Intervention = original_class


class GeluTrace(TorchDispatchMode):
    """Observe natural GELU calls and actual saved inputs passed to GELU backward.

    This outer mode intentionally ignores CaptureMode's private recursion flag:
    that flag is also set while natural model operators are forwarded. Snapshot
    bookkeeping disables all modes and cannot recursively enter either observer.
    """

    def __init__(self, layer, element, expected_shape=(4, 512, 3072)):
        super().__init__()
        self.region = f"gpt_neox.layers.{layer}"
        self.element = element
        self.expected_shape = tuple(expected_shape)
        self.forward = []
        self.backward = []
        self.backward_candidate_count = 0
        self.selected = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        context = REGION.get()
        forward = name == "aten.gelu.default" and context is not None and context[0] == self.region
        backward = (name == "aten.gelu_backward.default" and len(args) >= 2
                    and tuple(args[1].shape) == self.expected_shape)
        input_record = None
        if forward:
            require(tuple(args[0].shape) == self.expected_shape, "Target GELU shape changed")
            input_record = tensor_record(args[0], self.element)
        elif backward:
            self.backward_candidate_count += 1
            input_record = tensor_record(args[1], self.element)
        output = func(*args, **kwargs)
        if forward:
            role = context[1]
            ordinal = sum(row["role"] == role for row in self.forward)
            self.forward.append({"operator": name, "region": context[0], "role": role,
                                 "ordinal": ordinal, "input": input_record,
                                 "output": tensor_record(output, self.element),
                                 "approximate": kwargs.get("approximate", "none")})
            if ordinal == 0:
                with _disable_current_modes():
                    self.selected[role] = args[0].detach().clone()
        elif backward:
            matches = [{"role": row["role"], "ordinal": row["ordinal"]}
                       for row in self.forward if row["input"]["sha256"] == input_record["sha256"]]
            if matches:
                require(len({row["ordinal"] for row in matches}) == 1,
                        "Backward input hash ambiguously identifies multiple microbatches")
                ordinal = matches[0]["ordinal"]
                self.backward.append({"operator": name, "candidate_index": self.backward_candidate_count - 1,
                                      "ordinal": ordinal, "saved_input": input_record,
                                      "matching_forward_inputs": matches,
                                      "cotangent": tensor_record(args[0], self.element),
                                      "output": tensor_record(output, self.element),
                                      "approximate": kwargs.get("approximate", "none")})
                if ordinal == 0:
                    require("cotangent" not in self.selected, "Duplicate first-microbatch backward consumer")
                    with _disable_current_modes():
                        self.selected["cotangent"] = args[0].detach().clone()
        return output

    def summary(self):
        original = [row for row in self.forward if row["role"] == "original"]
        replay = [row for row in self.forward if row["role"] == "recompute"]
        require(len(original) == len(replay) == len(self.backward) == 4,
                "Expected all four original/replay/actual-backward GELU observations")
        require(all(row["approximate"] == "none" for row in self.forward + self.backward),
                "Same-cotangent reference requires the native exact GELU path")
        require(sorted(row["ordinal"] for row in self.backward) == list(range(4)),
                "Missing or duplicate target backward consumer")
        require(all(row['approximate'] == 'none' for row in self.forward + self.backward),
                'Registered local VJP requires native GELU approximate=none')
        pairs = [{"ordinal": left["ordinal"],
                  "input_bytes_equal": left["input"]["sha256"] == right["input"]["sha256"],
                  "output_bytes_equal": left["output"]["sha256"] == right["output"]["sha256"]}
                 for left, right in zip(original, replay)]
        consumers = {row["ordinal"]: row for row in self.backward}
        for pair in pairs:
            pair["backward_uses_replay_input"] = any(
                match["role"] == "recompute"
                for match in consumers[pair["ordinal"]]["matching_forward_inputs"]
            )
        require(set(self.selected) == {"original", "recompute", "cotangent"},
                "Actual matched cotangent and two input states required")
        # Same cotangent isolates the local VJP effect from upstream changes.
        with _disable_current_modes():
            left = torch.ops.aten.gelu_backward.default(self.selected["cotangent"], self.selected["original"], approximate="none")
            right = torch.ops.aten.gelu_backward.default(self.selected["cotangent"], self.selected["recompute"], approximate="none")
            vjp = completion.compare_trees({"vjp": right}, {"vjp": left}, distance=True)
        return {"forward": self.forward, "backward": self.backward,
                "backward_candidates": self.backward_candidate_count,
                "pairs": pairs, "same_cotangent_local_vjp": vjp,
                "coverage": "ACTUAL_ATEN_GELU_BACKWARD_INPUTS_WITH_UNIQUE_MICROBATCH_HASH_JOIN"}


def _arm(engine, stream, prestate, out, spec, trace_enabled, *, policy="disabled", scope="checkpoint_boundaries"):
    completion.cutoff()
    completion.restore(engine, prestate)
    counter = Counter()
    identity = f"gpt_neox.layers.{spec.layer}.mlp.dense_h_to_4h.output"
    element = int.from_bytes(hashlib.sha256(f"m4m5-site-v1:{spec.seed}:{identity}".encode()).digest()[:8], "big") % (4 * 512 * 3072)
    trace = GeluTrace(spec.layer, element) if trace_enabled else None
    with tracked_intervention(counter), (trace if trace is not None else nullcontext()):
        row, evidence = completion.execute(engine, stream, out, spec=spec, policy=policy,
                                           scope=scope, enforce=False, gradients=True,
                                           preserve_rng=True, branch_update=1)
    require(row["status"] == "ACCEPTED", "Provenance arm did not accept: " + json.dumps(row))
    target = f"gpt_neox.layers.{spec.layer}"
    require(counter[(target, "original")] == counter[(target, "recompute")] == 4,
            "Observer changed or failed to observe all four checkpoint replays")
    require(isinstance(evidence.get("gradients"), dict) and evidence["gradients"],
            "Pre-clipping gradient evidence missing")
    record = {"row": row, "trace_enabled": trace_enabled,
              "checkpoint_context_counts": [{"region": key[0], "role": key[1], "count": value}
                                            for key, value in sorted(counter.items())],
              "trace": trace.summary() if trace is not None else None,
              "element": element, "timing_claim": False}
    write(out / "arm.json", record)
    saved = completion.snapshot(engine)
    gradients = evidence["gradients"]
    del evidence, trace
    return record, saved, gradients


def _parity(saved, gradients, reference_saved, reference_gradients):
    return {"full_state": completion.compare_trees(saved, reference_saved),
            "gradients": completion.compare_trees(gradients, reference_gradients, distance=True)}


def _require_parity(result, label):
    require(all(exact_equal(value) for value in result.values()), label + " changed training state or gradients")


def run(out, contract):
    """Run six frozen cases and primary existing-detector comparators.

    Parent admission must validate Slurm ownership, source/input hashes, quota,
    environment and this contract before calling. All arms restart one immutable
    prestate; each new observer is compared with the same unobserved treatment.
    """
    completion.source_allocation_guard()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    require(contract["experiment_id"] == "BA-PROVENANCE-20261107-v1", "Wrong provenance contract")
    phase = contract.get('execution_phase', 'FULL_SIX_CASES')
    require(phase in ('FULL_SIX_CASES', 'DEV_PRIMARY_SMOKE'), 'Unknown execution phase')
    cases = CASES[:1] if phase == 'DEV_PRIMARY_SMOKE' else CASES
    require(contract["cases"] == [list(case) for case in cases], "Frozen phase case census changed")
    inputs = contract["inputs"]
    item = next(row for row in inputs if row["role"] == "corrected210-history1234")
    stream = completion.stream_from_inputs(inputs)
    engine = completion.load_engine(item)
    prestate = completion.snapshot(engine)
    started = time.monotonic()
    results = {"experiment_id": contract["experiment_id"], "execution_phase": phase, "status": "RUNNING",
               "independent_review": "PENDING", "parameter_count": sum(p.numel() for p in engine.model.parameters()),
               "parameter_count_delta": 0, "cases": [], "comparators": [],
               "limits": ["One history/state; fixed selected software bit faults, no incidence inference",
                          "Diagnostic transfers and snapshots invalidate performance claims",
                          "Input tracing comparator is not an execution of full OpGuard",
                          "No held-out learning consequence is measured"]}
    for layer, bit in cases:
        name = f"L{layer}-bit{bit}"
        folder = out / name
        folder.mkdir()
        spec = FaultSpec(f"recompute-L{layer}-fraction-{'low-bit0' if bit == 0 else 'high-bit22'}",
                         "recompute", layer, "bit", bit=bit, seed=1729)
        arms = {}
        clean_saved = clean_gradients = fault_saved = fault_gradients = None
        contrast = None
        # Keep only clean/fault references needed for exact comparisons.
        for enabled in (False, True):
            treatment = "fault" if enabled else "clean"
            arm_spec = replace(spec, enabled=enabled)
            references = None
            for traced in (False, True):
                arm_name = treatment + ("-trace" if traced else "-native")
                arm_out = folder / arm_name
                arm_out.mkdir()
                record, saved, gradients = _arm(engine, stream, prestate, arm_out, arm_spec, traced)
                arms[arm_name] = record
                if not traced:
                    references = (saved, gradients)
                    if enabled:
                        require(clean_saved is not None and clean_gradients is not None,
                                "Clean reference must precede the fault arm")
                        contrast = _parity(saved, gradients, clean_saved, clean_gradients)
                    else:
                        clean_saved, clean_gradients = saved, gradients
                else:
                    parity = _parity(saved, gradients, *references)
                    write(arm_out / "noninterference.json", parity)
                    _require_parity(parity, name + " " + treatment + " tracer")
                    arms[arm_name]["noninterference"] = parity
                    del saved, gradients
            if enabled:
                fault_saved, fault_gradients = references
            else:
                del references
        if (layer, bit) == (6, 0):
            events = arms["fault-native"]["row"]["injections"]
            require(len(events) == 1 and all(events[0].get(k) == v for k, v in PRIMARY_EVENT.items()),
                    "Primary event does not reproduce historical selected bytes")
            # Existing comparison scopes are preserved. Trace parity must also
            # hold when the old output observer is stacked above the new mode.
            for scope in ("checkpoint_boundaries", "all_operators"):
                baseline = None
                for traced in (False, True):
                    arm_out = folder / (scope + ("-trace" if traced else "-native"))
                    arm_out.mkdir()
                    record, saved, gradients = _arm(engine, stream, prestate, arm_out, spec, traced,
                                                    policy="every", scope=scope)
                    parity = _parity(saved, gradients, fault_saved, fault_gradients)
                    write(arm_out / "vs-unobserved-fault.json", parity)
                    _require_parity(parity, "Existing " + scope + " observation")
                    if traced:
                        require(comparison_signature(record) == comparison_signature(baseline),
                                "New tracer changed old detector comparison")
                    else:
                        baseline = record
                    results["comparators"].append(record)
                    del saved, gradients
        results["cases"].append({"layer": layer, "bit": bit,
                                  "fault_minus_clean": contrast, "arms": arms})
        write(folder / "case-results.json", results["cases"][-1])
        del clean_saved, clean_gradients, fault_saved, fault_gradients, references
    results.update(status="COMPLETE_PENDING_INDEPENDENT_REVIEW", elapsed_seconds=time.monotonic() - started,
                   arms_executed=4 * len(cases) + 4, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write(out / "results.json", results)
    return results


def comparison_signature(record):
    """Compare detector decisions without run-specific artifact paths/identifiers."""
    comparison = record["row"].get("comparison")
    require(isinstance(comparison, dict), "Existing detector comparison is absent")
    return {key: comparison.get(key) for key in ("failed", "counts", "eligible_pairs", "exact_pairs", "unsupported")}
