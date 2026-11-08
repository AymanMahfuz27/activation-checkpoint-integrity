"""Exact allocated CPU/CUDA correctness fixtures for the BB selector.

The parent admits hardware/source/runtime before this module imports numerical
libraries. CPU executes the thirteen shared tests plus its own published-byte
golden; CUDA executes those thirteen plus the native CUDA/CPU golden.
"""

from contextlib import nullcontext
import copy
import os
import random
import struct
import unittest

import selector as core

import numpy as np
import torch
from blake3 import blake3
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes


SHARED_METHODS = (
    "test_native_family_profiles_recover_declared_activation_demands",
    "test_duplicate_values_retain_all_class_members",
    "test_original_value_class_split_is_unsupported_and_blocks_optimizer",
    "test_uncheckpointed_registered_slots_are_frozen_outside_scope",
    "test_schema_options_metadata_order_and_count_drift_are_unsupported",
    "test_private_bookkeeping_exclusion_preserves_natural_gelu",
    "test_outside_scope_fault_request_is_refused_before_case",
    "test_profile_and_observation_preserve_gradients_full_state_and_rng",
    "test_consumer_entry_fault_distinguishes_forward_and_consumer_checks",
    "test_replay_forward_input_fault_is_seen_by_appropriate_policies",
    "test_native_execution_count_and_returned_values_are_unchanged",
    "test_missing_or_duplicate_fields_and_demands_fail_closed",
    "test_enforcement_is_preclip_preoptimizer_without_rng_rollback_claim",
)
CPU_FIXTURE_IDS = ["test_selector_allocated.SelectorFixtures." + name for name in
                   SHARED_METHODS + ("test_cpu_blake3_matches_canonical_published_bytes",)]
GPU_FIXTURE_IDS = ["test_selector_allocated.SelectorFixtures." + name for name in
                   ("test_full_cuda_blake3_matches_cpu_for_canonical_and_multichunk_bytes",) + SHARED_METHODS]


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or left.device != right.device:
            return False
        with _disable_current_modes():
            a = left.detach().clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8)
            b = right.detach().clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8)
            return bool(torch.equal(a, b))
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    if type(left) is float:
        return struct.pack("!d", left) == struct.pack("!d", right)
    return left == right


class _Toy(torch.nn.Module):
    def __init__(self, family="gelu", *, duplicate=False, external=False, split=False,
                 shape=(2, 17), approximate="none", overload_drift=False,
                 region="toy.block", external_eps=1e-5):
        super().__init__()
        self.family = family
        self.duplicate = duplicate
        self.external = external
        self.split = split
        self.shape = shape
        self.approximate = approximate
        self.overload_drift = overload_drift
        self.region = region
        self.external_eps = external_eps
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.shift = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("fixture_buffer", torch.tensor(0.0))
        self.body_calls = 0
        self.native_returns = []

    def activation(self, value):
        if self.family == "gelu":
            return F.gelu(value, approximate=self.approximate)
        if self.family == "native_layer_norm":
            return F.layer_norm(value, (value.shape[-1],), eps=1e-5)
        if self.family == "softmax":
            return F.softmax(value, dim=-1)
        if self.family == "relu":
            return F.relu(value)
        if self.family == "tanh":
            return torch.tanh(value)
        if self.family == "sigmoid":
            return torch.sigmoid(value)
        raise ValueError("Unknown fixture family")

    def body(self, source):
        self.body_calls += 1
        value = source * self.scale + self.shift
        first = self.activation(value)
        self.native_returns.append(first)
        if self.overload_drift:
            # An added family overload must not hide behind the expected default.
            with torch.no_grad():
                torch.ops.aten.gelu.out(value.detach(), out=torch.empty_like(value))
        if self.duplicate:
            second = self.activation(value + 0.125 if self.split else value)
            return first + second
        return first

    def forward(self, source):
        def contexts():
            return (core.region_context(self.region, "original"),
                    core.region_context(self.region, "recompute"))
        with set_checkpoint_early_stop(False):
            result = checkpoint(self.body, source, use_reentrant=False,
                                context_fn=contexts, preserve_rng_state=True,
                                determinism_check="default")
        if self.external == "mixed":
            result = result + F.gelu(source)
        elif self.external in ("layer_norm", "layer_norm_added"):
            # A residual keeps this final uncheckpointed input distinct from
            # registered checkpoint fields; its three native state slots are
            # then a clean all-external demand class, as in the model adapter.
            external_input = result + source * 0.125 + 0.375
            result = F.layer_norm(external_input, (external_input.shape[-1],), eps=self.external_eps)
            if self.external == "layer_norm_added":
                result = result + F.layer_norm(external_input + 0.125,
                                               (external_input.shape[-1],), eps=self.external_eps)
        elif self.external:
            result = result + torch.tanh(source + 0.375)
        return result


def _snapshot(model, optimizer, source, scheduler, cursor, updates, preclip=None):
    return _clone({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "buffers": dict(model.named_buffers()), "scheduler": scheduler.state_dict(),
                   "cursor": cursor, "updates": updates, "preclip_gradients": preclip,
                   "gradients": {name: parameter.grad for name, parameter in model.named_parameters()},
                   "input_gradient": source.grad,
                   "rng": {"cpu": torch.get_rng_state(),
                           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                           "python": random.getstate(), "numpy": np.random.get_state()}})


class _BitFault:
    def __init__(self, family="gelu", bit=22, element=None):
        self.family = family
        self.bit = bit
        self.element = element
        self.events = []

    def __call__(self, func, bound, description):
        if description["family"] != self.family or description["ordinal"] != 0:
            return
        if self.events:
            raise AssertionError("Duplicate independent fixture fault")
        registry = next(row for row in core.REGISTRY if row["family"] == self.family)
        name = registry["backward_activation_args"][0]["name"]
        tensor = bound[name]
        element = tensor.numel() // 2 if self.element is None else self.element
        with torch.no_grad(), _disable_current_modes():
            if not tensor.is_contiguous():
                raise AssertionError("Fixture fault requires a contiguous native argument")
            before = int(tensor.reshape(-1).view(torch.int32)[element].item()) & 0xFFFFFFFF
            tensor.reshape(-1).view(torch.int32)[element].bitwise_xor_(1 << self.bit if self.bit < 31 else -(1 << 31))
            after = int(tensor.reshape(-1).view(torch.int32)[element].item()) & 0xFFFFFFFF
        self.events.append({"family": self.family, "element": element,
                            "old_bits_u32": before, "new_bits_u32": after})


class _ForwardFault(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.events = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        context = core.REGION.get()
        if str(func) == "aten.gelu.default" and context is not None and context[1] == "recompute":
            with torch.no_grad(), _disable_current_modes():
                args[0].reshape(-1).view(torch.int32)[args[0].numel() // 2].bitwise_xor_(1 << 22)
            self.events += 1
        return func(*args, **(kwargs or {}))


def _run(device, family="gelu", *, policy="disabled", plan=None, profile=False,
         fault=None, forward_fault=None, enforce=False, seed=11, duplicate=False,
         external=False, split=False, shape=(2, 17), approximate="none", nested_mode=None,
         overload_drift=False, dtype=torch.float32, noncontiguous=False,
         region="toy.block", extra_root=False, external_eps=1e-5):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = _Toy(family, duplicate=duplicate, external=external, split=split,
                 shape=shape, approximate=approximate, overload_drift=overload_drift,
                 region=region, external_eps=external_eps).to(device=device, dtype=dtype)
    source = torch.randn(shape, device=device, dtype=dtype)
    if noncontiguous:
        source = source.t().contiguous().t()
    source[0, 0] = 1.0
    source.requires_grad_(True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda update: 1.0)
    cursor = updates = 0
    preclip = None
    initial = _snapshot(model, optimizer, source, scheduler, cursor, updates)
    observer = core.ProfileMode() if profile else core.CheckMode(plan, policy) if policy != "disabled" else None
    mode = observer if observer is not None else nullcontext()
    model_context = observer.track_model(model) if observer is not None else nullcontext()
    injector = core.ConsumerFaultMode(fault) if fault is not None else nullcontext()
    forward_injector = forward_fault if forward_fault is not None else nullcontext()
    optimizer_steps = 0
    clipping_calls = 0
    status = "ACCEPTED"
    error = None
    output = None
    try:
        with model_context, mode, injector, forward_injector, (nested_mode or nullcontext()):
            output = model(source)
            if extra_root:
                output = output + model(source)
            # A real draw makes RNG parity and the non-rollback boundary meaningful.
            loss = (output * torch.rand_like(output)).square().sum()
            loss.backward()
            with torch.no_grad(), _disable_current_modes():
                preclip = _clone({name: parameter.grad for name, parameter in model.named_parameters()})
            if observer is not None:
                core.make_pre_gate_callback(observer, enforce)(None, float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            clipping_calls += 1
            optimizer.step()
            optimizer_steps += 1
            scheduler.step()
            cursor += source.numel()
            updates += 1
    except core.UnsupportedPlan as caught:
        status, error = "UNSUPPORTED_PLAN", str(caught)
    except Exception as caught:
        from ac_integrity.distributed import IntegrityError
        if not isinstance(caught, IntegrityError):
            raise
        status, error = "REJECTED_PRE_OPTIMIZER", str(caught)
    return {"state": _snapshot(model, optimizer, source, scheduler, cursor, updates, preclip), "initial": initial,
            "status": status, "error": error, "optimizer_steps": optimizer_steps,
            "clipping_calls": clipping_calls,
            "observer": observer, "plan": observer.plan if profile and observer.gated else plan,
            "body_calls": model.body_calls,
            "same_result_object": bool(output is not None and not duplicate and not external
                                       and output is model.native_returns[0]),
            "output": None if output is None else output.detach().clone()}


class SelectorFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        requested = os.environ.get("BB_TEST_DEVICE")
        if requested not in ("cpu", "cuda:0"):
            raise RuntimeError("Parent must select CPU or qualified CUDA explicitly")
        cls.device = torch.device(requested)
        if cls.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Native CUDA fixture qualification is unavailable")

    def profile(self, family="gelu", **kwargs):
        result = _run(self.device, family, profile=True, **kwargs)
        self.assertEqual(result["status"], "ACCEPTED")
        return result

    def test_full_cuda_blake3_matches_cpu_for_canonical_and_multichunk_bytes(self):
        self.assertEqual(self.device.type, "cuda")
        values = [torch.empty(0, dtype=torch.float32),
                  torch.arange(255, dtype=torch.float32),
                  torch.arange(256, dtype=torch.float32),
                  torch.arange(257, dtype=torch.float32),
                  torch.ones((), dtype=torch.float32).expand(1),
                  torch.arange(35, dtype=torch.float32).reshape(5, 7).t(),
                  torch.arange(1_048_583, dtype=torch.float32)]
        for value in values:
            cpu = core.SignatureStore().capture(value)
            cuda = core.SignatureStore().capture(value.to(self.device)).cpu()
            self.assertTrue(torch.equal(cpu, cuda))

    def test_cpu_blake3_matches_canonical_published_bytes(self):
        self.assertEqual(self.device.type, "cpu")
        values = [torch.empty(0, dtype=torch.float32),
                  torch.arange(255, dtype=torch.float32),
                  torch.arange(256, dtype=torch.float32),
                  torch.arange(257, dtype=torch.float32),
                  torch.ones((), dtype=torch.float32).expand(1),
                  torch.ones((), dtype=torch.float32).expand(4),
                  torch.arange(35, dtype=torch.float32).reshape(5, 7).t(),
                  torch.arange(1_048_583, dtype=torch.float32)]
        for value in values:
            before = value.detach().clone()
            strides = value.stride()
            raw = value.detach().clone(memory_format=torch.contiguous_format).numpy().tobytes()
            expected = struct.unpack("<4q", blake3(raw).digest())
            actual = tuple(core.SignatureStore().capture(value).tolist())
            self.assertEqual(actual, expected)
            self.assertEqual(value.stride(), strides)
            self.assertTrue(torch.equal(value, before))

    def test_native_family_profiles_recover_declared_activation_demands(self):
        expected = {"gelu": "input.self", "native_layer_norm": "input.input",
                    "softmax": "output", "relu": "output", "tanh": "output", "sigmoid": "output"}
        for family, field in expected.items():
            result = self.profile(family)
            plan = result["plan"]
            self.assertEqual(plan["profile_counters"]["root_forwards"], 1)
            self.assertEqual(plan["profile_counters"]["backward"], {family: 1})
            self.assertEqual(sum(row["count"] for row in plan["profile_counters"]["forward"]), 2)
            registered = next(row for row in core.REGISTRY if row["family"] == family)
            self.assertEqual({row["argument"] for row in plan["bindings"]},
                             {row["name"] for row in registered["backward_activation_args"]})
            selected = [site for item in plan["classes"].values() if item["scope"] == "checkpoint"
                        for site in item["members"]]
            self.assertTrue(any(json_site[-1] == field for json_site in map(core.json.loads, selected)))
            if family == "native_layer_norm":
                paths = {core.json.loads(site)[-1] for site in selected}
                self.assertTrue({"input.input", "output[1]", "output[2]"} <= paths)
            native = _run(self.device, family, seed=37)
            for policy in core.POLICIES:
                held_out = _run(self.device, family, seed=37, policy=policy, plan=plan)
                self.assertEqual(held_out["status"], "ACCEPTED", (family, policy))
                self.assertEqual(held_out["observer"].finish()["status"], "PASS")
                self.assertTrue(_equal(native["state"], held_out["state"]), (family, policy))

    def test_duplicate_values_retain_all_class_members(self):
        plan = self.profile(duplicate=True)["plan"]
        self.assertTrue(any(len(item["members"]) >= 2 for item in plan["classes"].values()))
        result = _run(self.device, policy="selected_consumer", plan=plan, duplicate=True, seed=37)
        self.assertEqual(result["status"], "ACCEPTED")
        self.assertEqual(result["observer"].finish()["status"], "PASS")

    def test_original_value_class_split_is_unsupported_and_blocks_optimizer(self):
        plan = self.profile(duplicate=True)["plan"]
        result = _run(self.device, policy="selected_consumer", plan=plan, duplicate=True, split=True)
        self.assertEqual(result["status"], "UNSUPPORTED_PLAN")
        self.assertIn("class split", result["error"])
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertTrue(_equal(result["state"]["model"], result["initial"]["model"]))

    def test_uncheckpointed_registered_slots_are_frozen_outside_scope(self):
        plan = self.profile(external=True)["plan"]
        self.assertTrue(any(item["scope"] == "outside" for item in plan["classes"].values()))
        result = _run(self.device, policy="selected_consumer", plan=plan, external=True, seed=37)
        self.assertEqual(result["status"], "ACCEPTED")
        self.assertEqual(result["observer"].finish()["status"], "PASS")
        mixed = _run(self.device, profile=True, external="mixed")
        self.assertEqual(mixed["status"], "UNSUPPORTED_PLAN")
        self.assertIn("mixes checkpoint and external", mixed["error"])
        final_ln = self.profile(external="layer_norm")["plan"]
        outside = [binding for binding in final_ln["bindings"]
                   if final_ln["events"][binding["event"]]["family"] == "native_layer_norm"]
        self.assertEqual({binding["argument"] for binding in outside}, {"input", "mean", "rstd"})
        self.assertTrue(all(final_ln["classes"][binding["class_id"]]["scope"] == "outside"
                            for binding in outside))
        native = _run(self.device, seed=37, external="layer_norm")
        held_out = _run(self.device, seed=37, external="layer_norm",
                        policy="selected_consumer", plan=final_ln)
        self.assertEqual(held_out["status"], "ACCEPTED")
        self.assertTrue(_equal(native["state"], held_out["state"]))
        for settings in ({"external": False}, {"external": "layer_norm_added"},
                         {"external": "layer_norm", "external_eps": 1e-4}):
            changed = _run(self.device, policy="selected_consumer", plan=final_ln, **settings)
            self.assertEqual(changed["status"], "UNSUPPORTED_PLAN")
            self.assertEqual(changed["optimizer_steps"], 0)
            self.assertEqual(changed["clipping_calls"], 0)

    def test_schema_options_metadata_order_and_count_drift_are_unsupported(self):
        plan = self.profile()["plan"]
        for kwargs in [{"shape": (3, 33)}, {"approximate": "tanh"}, {"duplicate": True},
                       {"family": "tanh"}, {"overload_drift": True}, {"dtype": torch.float64},
                       {"noncontiguous": True}, {"region": "toy.changed"}, {"extra_root": True}]:
            result = _run(self.device, policy="selected_consumer", plan=plan, **kwargs)
            self.assertEqual(result["status"], "UNSUPPORTED_PLAN")
            self.assertEqual(result["optimizer_steps"], 0)

    def test_private_bookkeeping_exclusion_preserves_natural_gelu(self):
        class SharedFlagMode(TorchDispatchMode):
            active = False

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.active = True
                try:
                    return func(*args, **(kwargs or {}))
                finally:
                    self.active = False

        result = self.profile(nested_mode=SharedFlagMode())
        self.assertEqual(result["plan"]["profile_counters"]["backward"]["gelu"], 1)
        self.assertEqual(sum(row["count"] for row in result["plan"]["profile_counters"]["forward"]), 2)

    def test_outside_scope_fault_request_is_refused_before_case(self):
        plan = self.profile(external=True)["plan"]
        with self.assertRaises(core.UnsupportedPlan):
            core.require_fault_target(plan, "tanh", 0, "output")
        core.require_fault_target(plan, "gelu", 0, "self")

    def test_profile_and_observation_preserve_gradients_full_state_and_rng(self):
        native = _run(self.device)
        profiled = self.profile()
        self.assertTrue(_equal(native["state"], profiled["state"]))
        for policy in core.POLICIES:
            observed = _run(self.device, policy=policy, plan=profiled["plan"])
            self.assertEqual(observed["status"], "ACCEPTED")
            self.assertTrue(_equal(native["state"], observed["state"]), policy)

    def test_consumer_entry_fault_distinguishes_forward_and_consumer_checks(self):
        for family in (row["family"] for row in core.REGISTRY):
            plan = self.profile(family)["plan"]
            fault_options = {"bit": 31, "element": 0} if family == "relu" else {}
            native_fault = _BitFault(family, **fault_options)
            reference = _run(self.device, family, seed=37, fault=native_fault)
            self.assertEqual(len(native_fault.events), 1)
            for policy in core.POLICIES:
                fault = _BitFault(family, **fault_options)
                result = _run(self.device, family, seed=37, policy=policy, plan=plan, fault=fault)
                self.assertEqual(fault.events, native_fault.events)
                self.assertEqual(result["status"], "ACCEPTED")
                self.assertTrue(_equal(reference["state"], result["state"]), (family, policy))
                self.assertEqual(result["observer"].finish()["failed"], policy == "selected_consumer")

    def test_replay_forward_input_fault_is_seen_by_appropriate_policies(self):
        plan = self.profile()["plan"]
        reference = _run(self.device, forward_fault=_ForwardFault())
        for policy in ("full_forward_io", "selected_forward", "selected_consumer"):
            fault = _ForwardFault()
            result = _run(self.device, policy=policy, plan=plan, forward_fault=fault)
            self.assertEqual(fault.events, 1)
            self.assertEqual(result["status"], "ACCEPTED")
            self.assertTrue(_equal(reference["state"], result["state"]))
            self.assertTrue(result["observer"].finish()["failed"])

    def test_native_execution_count_and_returned_values_are_unchanged(self):
        profiled = self.profile()
        native = _run(self.device)
        self.assertTrue(profiled["same_result_object"])
        self.assertEqual(profiled["body_calls"], 2)
        for policy in core.POLICIES:
            result = _run(self.device, policy=policy, plan=profiled["plan"])
            self.assertEqual(result["body_calls"], native["body_calls"])
            self.assertTrue(result["same_result_object"])
            self.assertTrue(_equal(result["output"], native["output"]))

    def test_missing_or_duplicate_fields_and_demands_fail_closed(self):
        profiled = self.profile()
        for alteration in ("missing_field", "duplicate_binding", "missing_event", "empty_obligations"):
            plan = copy.deepcopy(profiled["plan"])
            if alteration == "missing_field":
                group = next(item for item in plan["classes"].values() if item["scope"] == "checkpoint")
                group["members"].append("nonexistent-required-site")
            elif alteration == "duplicate_binding":
                plan["bindings"].append(copy.deepcopy(plan["bindings"][0]))
            elif alteration == "empty_obligations":
                plan["bindings"] = []
                plan["classes"] = {}
            else:
                plan["events"].pop()
            plan["plan_sha256"] = core.plan_digest(plan)
            try:
                result = _run(self.device, policy="selected_consumer", plan=plan)
            except core.UnsupportedPlan:
                continue
            self.assertEqual(result["status"], "UNSUPPORTED_PLAN")
            self.assertEqual(result["optimizer_steps"], 0)

    def test_enforcement_is_preclip_preoptimizer_without_rng_rollback_claim(self):
        plan = self.profile("relu")["plan"]
        fault = _BitFault("relu", bit=31, element=0)
        result = _run(self.device, "relu", policy="selected_consumer", plan=plan,
                      fault=fault, enforce=True)
        self.assertEqual(result["status"], "REJECTED_PRE_OPTIMIZER")
        self.assertEqual(result["clipping_calls"], 0)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertTrue(_equal(result["state"]["model"], result["initial"]["model"]))
        self.assertTrue(_equal(result["state"]["optimizer"], result["initial"]["optimizer"]))
        for key in ("buffers", "scheduler", "cursor", "updates"):
            self.assertTrue(_equal(result["state"][key], result["initial"][key]), key)
        self.assertFalse(_equal(result["state"]["rng"], result["initial"]["rng"]))
        clean = _run(self.device, "relu", policy="selected_consumer", plan=plan, enforce=True)
        self.assertEqual(clean["status"], "ACCEPTED")
        self.assertEqual(clean["clipping_calls"], 1)
        self.assertEqual(clean["optimizer_steps"], 1)


if __name__ == "__main__":
    raise SystemExit("Use the independently admitted parent fixture driver")
