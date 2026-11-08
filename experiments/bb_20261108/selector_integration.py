"""Matched Pythia pilot and registered timing arms for BB consumer obligations.

The parent owns source, runtime, input, physical and storage admission. Every
diagnostic arm restores one immutable full prestate; timing arms also restore
before their complete eight-update sequence. The historical engine is unchanged.
"""

from collections import Counter
from contextlib import contextmanager, nullcontext
import gc
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import time

import selector as core

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from ac_integrity.research import completion
from ac_integrity.research.m4m5 import FaultSpec, Intervention


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def exact(comparison):
    return (comparison.get("valid") is True and comparison.get("byte_equal") is True
            and comparison.get("all_finite") is True
            and not comparison.get("changed_scalar_paths", []))


def _method(contract):
    method = contract["method_plan"]
    require(method["experiment_id"] == "BB-DEMAND-20261108-v1", "Wrong frozen method")
    return method


def _input(contract, state):
    role = "corrected210-history1234" if state == "later-update210" else "early-history1234"
    matches = [item for item in contract["inputs"] if item["role"] == role]
    require(len(matches) == 1, "One registered input is required for " + state)
    return matches[0]


def _admitted_binding(contract):
    """Freeze already-admitted source/runtime identity without model state bytes."""
    original = contract["historical_contract"]
    schemas = []
    for registered in core.REGISTRY:
        row = {"family": registered["family"]}
        for kind in ("forward", "backward"):
            namespace, name, overload = registered[kind].split(".")
            operator = getattr(getattr(getattr(torch.ops, namespace), name), overload)
            row[kind] = str(operator._schema)
        schemas.append(row)
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": properties.name,
                        "compute_capability": [properties.major, properties.minor],
                        "memory_bytes": properties.total_memory})
    runtime_environment = {key: os.environ.get(key) for key in contract["runtime_environment"]}
    require(runtime_environment == contract["runtime_environment"], "Admitted runtime environment changed")
    binding = {
        "source_file_pins": contract["files"],
        "historical_source_revision": original["execution_source_revision"],
        "historical_source_file_pins": original["source_files"],
        "requirements_lock_sha256": contract["files"][contract["requirements_lock"]],
        "native_extension_source_sha256": contract["expected_native_extension_source_sha256"],
        "runtime": {"python": ".".join(map(str, sys.version_info[:3])),
                    "torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "numpy": version("numpy"), "transformers": version("transformers"),
                    "runtime_environment": runtime_environment, "visible_devices": devices,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark},
        "registry": list(core.REGISTRY), "native_schemas": schemas,
    }
    # A canonical copy prevents later caller mutations of contract dictionaries.
    return json.loads(core.canonical_json(binding))


def _plan_envelope(binding, engine, recipe):
    """Recheck the allowed graph metadata on each profile or cached plan use.

    Config, module types and parameter metadata describe the graph. Parameter
    values, buffers, optimizer state, cursor and RNG are intentionally absent,
    so an optimizer update or the declared warm10 state can use the same graph.
    """
    require(binding is not None, "An admitted source/runtime binding is required")
    model = engine.model
    config = model.config.to_dict()
    for name in ("_name_or_path", "_commit_hash", "transformers_version"):
        config.pop(name, None)
    graph = {
        "config": config, "attention_implementation": model.config._attn_implementation,
        "modules": [{"name": name, "class": type(module).__module__ + "." + type(module).__qualname__}
                    for name, module in model.named_modules()],
        "parameters": [{"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                        "device": str(parameter.device), "requires_grad": parameter.requires_grad}
                       for name, parameter in model.named_parameters()],
        "execution": {"training": True, "use_cache": False, "use_reentrant": False,
                      "checkpoint_early_stop": False, "preserve_rng": True},
        "registered_families": list(core.REGISTRY),
    }
    return json.loads(core.canonical_json({"recipe": recipe, "admitted_identity": binding,
                                           "allowed_model_graph": graph}))


@contextmanager
def tracked_contexts(counter):
    """Expose existing role contexts without replacing the original intervention."""
    original = completion.Intervention

    class Tracked(original):
        @contextmanager
        def context(self, region, role):
            with super().context(region, role):
                counter[(region, role)] += 1
                with core.region_context(region, role):
                    yield

    completion.Intervention = Tracked
    try:
        yield
    finally:
        completion.Intervention = original


class _PilotCounter(TorchDispatchMode):
    """Independent metadata count and fault entry path shared by pilot policies.

    The pilot reference intentionally has this same metadata counter. Pilot
    timing is diagnostic only; clean performance arms do not install this mode.
    """

    def __init__(self, consumer_fault=None):
        super().__init__()
        self.consumer_fault = consumer_fault
        self.calls = Counter()
        self.backward = Counter()
        self.registry = {item["backward"]: item for item in core.REGISTRY}
        self.names = {item["forward"] for item in core.REGISTRY} | set(self.registry)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if name in self.names:
            context = core.REGION.get()
            role = context[1] if context is not None else "outside_context"
            self.calls[(name, role)] += 1
        if name in self.registry:
            family = self.registry[name]["family"]
            ordinal = self.backward[family]
            self.backward[family] += 1
            if self.consumer_fault is not None:
                with torch.no_grad(), _disable_current_modes():
                    self.consumer_fault(func, core._bind(func, args, kwargs),
                                        {"family": family, "ordinal": ordinal})
        return func(*args, **kwargs)

    def summary(self):
        return {"calls": [{"operator": key[0], "role": key[1], "count": value}
                           for key, value in sorted(self.calls.items())],
                "backward": dict(self.backward)}


class ConsumerInjection:
    """Fixed native GELU ordinal, independent of the selector's chosen fields."""

    def __init__(self, case, engine, branch_update=1):
        self.case = case
        spec = FaultSpec("bb-consumer-L%d-bit%d-seed%d" % (case["layer"], case["bit"], case["seed"]),
                         "recompute", case["layer"], "bit", bit=case["bit"], seed=case["seed"])
        self.intervention = Intervention(spec)
        self.intervention.before_forward(engine.model, engine.optimizer, branch_update)
        self.intervention.region = "gpt_neox.layers.%d" % case["layer"]
        self.intervention.role = "consumer-entry"
        self.branch_update = branch_update

    def __call__(self, func, bound, description):
        if description["family"] != "gelu" or description["ordinal"] != self.case["consumer_ordinal"]:
            return
        tensor = bound["self"]
        require(tuple(tensor.shape) == (4, 512, 3072) and tensor.dtype == torch.float32,
                "Independent GELU fault shape/dtype changed")
        require(not self.intervention.events, "Consumer fault fired twice")
        self.intervention.mutate(
            tensor, "gpt_neox.layers.%d.mlp.dense_h_to_4h.output" % self.case["layer"],
            event_key=[self.branch_update, "consumer-entry"])

    def summary(self):
        return {"events": completion.clone_tree(self.intervention.events),
                "target_selected_from_plan": False,
                "delivery": "after autograd unpack, before candidate fingerprint and native consumer"}


def _profile_records(mode):
    def rows(values):
        result = []
        for value in values:
            row = {key: item for key, item in value.items() if key != "signature"}
            with _disable_current_modes():
                row["blake3_int64_words"] = [int(word) for word in value["signature"].detach().cpu().tolist()]
            result.append(row)
        return result
    return {"events": mode.events, "forward_fields": rows(mode.forward_fields),
            "backward_fields": rows(mode.backward_fields), "counters": mode.counters(),
            "storage_alias_identity_proved": False}


def _gradient_copy(engine):
    return {name: parameter.grad.detach().clone() if parameter.grad is not None else None
            for name, parameter in engine.model.named_parameters()}


def _synchronize(engine):
    if engine.device.startswith("cuda"):
        torch.cuda.synchronize()


def _profile_cost(directory, elapsed):
    """Report the complete profiling call and its already-exported byte count."""
    files = [path for path in Path(directory).rglob("*") if path.is_file()]
    return {"complete_profile_call_wall_seconds": elapsed,
            "exported_profile_bytes": sum(path.stat().st_size for path in files),
            "exported_profile_files": len(files),
            "exported_profile_scope": "profile arm directory; compiled plan and admission records are separate",
            "includes_raw_profile_and_arm_serialization": True,
            "includes_transient_state_capture_when_enabled": True,
            "cost_report_serialization_outside_profile_call": True}


def run_arm(engine, stream, prestate, out, *, policy="disabled", plan=None,
            case=None, profile=False, enforce=False, batch=None, gradients=True,
            pilot_counter=True, restore=True, branch_update=1, capture_state=True,
            identity_binding=None):
    """Run one unchanged-engine update and return records plus transient tensors.

    Only the caller compares or releases the returned full state/gradients. The
    output directory contains metadata and comparisons, never model snapshots.
    A consumer fault is delivered in disabled and forward-only policies too.
    """
    completion.cutoff()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if restore:
        completion.restore(engine, prestate)
    # Include construction, hooks, hashes, comparisons and decision transfer.
    # Scientific snapshots and artifact serialization follow the timed interval.
    _synchronize(engine)
    step_started = time.perf_counter()
    contexts = Counter()
    envelope = None
    if profile or policy != "disabled":
        envelope = _plan_envelope(identity_binding, engine, batch or completion.DIAGNOSTIC)
        if not profile and (plan.get("declared_envelope") != envelope
                            or plan.get("registry") != list(core.REGISTRY)):
            write(out / "plan-envelope-invalid.json", {
                "status": "INVALID_PLAN_ENVELOPE", "expected": plan.get("declared_envelope"),
                "actual": envelope, "scientific_acceptance": False})
            raise RuntimeError("INVALID_PLAN_ENVELOPE: source/runtime/registry/model graph or recipe changed")
    observer = core.ProfileMode() if profile else core.CheckMode(plan, policy) if policy != "disabled" else None
    if case is None:
        spec = FaultSpec("bb-clean-context", "recompute", 0, "bit", bit=0, seed=1729, enabled=False)
    else:
        spec = FaultSpec("bb-%s-L%d-bit%d-seed%d" %
                         (case["window"], case["layer"], case["bit"], case["seed"]),
                         "recompute", case["layer"], "bit", bit=case["bit"], seed=case["seed"],
                         enabled=case["window"] == "replay_forward")
    injector = ConsumerInjection(case, engine, branch_update) if case is not None and case["window"] == "consumer_entry" else None
    counter = _PilotCounter(injector) if pilot_counter else None
    require(injector is None or counter is not None, "Consumer delivery requires its independent pilot entry path")
    callback = core.make_pre_gate_callback(observer, enforce, envelope) if observer is not None else None
    with tracked_contexts(contexts), (observer.track_model(engine.model) if observer is not None else nullcontext()), \
            (observer if observer is not None else nullcontext()), \
            (counter if counter is not None else nullcontext()):
        row, evidence = completion.execute(
            engine, stream, out, spec=spec, policy="disabled", scope="checkpoint_boundaries",
            enforce=False, gradients=gradients, preserve_rng=True,
            branch_update=branch_update, batch=batch, cheap_observer=callback)
    _synchronize(engine)
    step_elapsed = time.perf_counter() - step_started
    external = injector.summary() if injector is not None else {"events": []}
    unsupported = observer.unsupported if observer is not None else None
    if row.get("error", "").startswith("UnsupportedPlan:"):
        unsupported = unsupported or row["error"]
    if profile:
        result = {"status": "UNSUPPORTED_PLAN" if unsupported else "PROFILE_COMPLETE" if observer.gated else "INCOMPLETE",
                  "reason": unsupported,
                  "plan_sha256": observer.plan["plan_sha256"] if observer.gated else None,
                  "counters": observer.counters()}
        write(out / "profile-records.json", _profile_records(observer))
    else:
        result = observer.summary() if observer is not None else {"status": "DISABLED"}
        if unsupported:
            result.update(status="UNSUPPORTED_PLAN", reason=unsupported)
    record = {"method": policy, "profile": profile, "method_enforce": enforce,
              "historical_engine_policy": "disabled", "historical_engine_enforce": False,
              "case": case, "row": row, "observation": result,
              "external_consumer_injection": external,
              "checkpoint_context_counts": [{"region": key[0], "role": key[1], "count": value}
                                             for key, value in sorted(contexts.items())],
              "native_registered_counts": counter.summary() if counter is not None else None,
              "step_elapsed_seconds": step_elapsed,
              "timing_interval": "observer setup through execute, cleanup and final device synchronization",
              "validation_and_artifact_serialization_outside_timing": True,
              "timing_claim": False}
    write(out / "arm.json", record)
    if case is None:
        require(row["injections"] == [] and external["events"] == [], "Clean arm injected")
    elif not unsupported:
        if case["window"] == "consumer_entry":
            require(row["injections"] == [] and len(external["events"]) == 1,
                    "Independent consumer injection census differs")
        else:
            require(len(row["injections"]) == 1 and external["events"] == [],
                    "Unchanged replay-forward intervention census differs")
    saved = completion.snapshot(engine) if capture_state else None
    gradient_values = (evidence.get("gradients") if gradients else _gradient_copy(engine)) if capture_state else None
    if capture_state and unsupported and not isinstance(gradient_values, dict):
        gradient_values = _gradient_copy(engine)
    if capture_state and not unsupported:
        require(isinstance(gradient_values, dict) and gradient_values, "Gradient comparison evidence is missing")
    compiled = observer.plan if profile and observer.gated else None
    del evidence, observer
    return record, saved, gradient_values, compiled


def _parity(saved, gradients, reference_saved, reference_gradients):
    return {"state": completion.compare_trees(saved, reference_saved),
            "gradients": completion.compare_trees(gradients, reference_gradients, distance=True)}


def _require_parity(result, message):
    require(all(exact(value) for value in result.values()), message)


def _unsupported(record):
    return record["observation"]["status"] == "UNSUPPORTED_PLAN"


def _not_run(out, label, reason, case=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    record = {"method": label, "case": case, "row": {"status": "NOT_RUN_UNSUPPORTED"},
              "observation": {"status": "UNSUPPORTED_PLAN", "reason": reason},
              "scientific_acceptance": False}
    write(out / "arm.json", record)
    return record


def _validate_profile(plan, method, *, batch=None, out, validate_bindings=True):
    """Require the complete pinned Pythia graph before admitting its demand map.

    This gate does not learn its required census from the observed profile.
    Twelve checkpointed blocks each have two LayerNorms, one GELU and one
    softmax; one final LayerNorm is outside checkpoints. Consumer order stays
    learned. Every final-LayerNorm state slot must link to its own microbatch's
    registered external forward field, without assuming unique value classes.
    """
    recipe = batch or method["model_protocol"]["diagnostic_batch"]
    accumulation = recipe["accumulation"]
    regions = ["gpt_neox.layers.%d" % layer for layer in range(12)]
    calls_per_region = {"gelu": 1, "softmax": 1, "native_layer_norm": 2}
    activation_args = {"gelu": {"self"}, "softmax": {"output"},
                       "native_layer_norm": {"input", "mean", "rstd"}}
    operators = {item["family"]: item for item in core.REGISTRY if item["family"] in calls_per_region}
    expected_forward = {}
    for region in regions:
        for micro in range(accumulation):
            for role in ("original", "recompute"):
                for family, count in calls_per_region.items():
                    expected_forward[(family, region, role, micro)] = count
    for micro in range(accumulation):
        expected_forward[("native_layer_norm", "__external_forward__", "external_forward", micro)] = 1
    expected_backward = {"gelu": 12 * accumulation, "softmax": 12 * accumulation,
                         "native_layer_norm": 25 * accumulation}
    actual_forward, actual_backward, backward_by_micro = Counter(), Counter(), Counter()
    forward_ordinals = {}
    external_fields = {}
    backwards = {}
    unexpected_events = []
    checks = {
        "declared_recipe": recipe in (method["model_protocol"]["diagnostic_batch"],
                                       method["timing_protocol"]["production_batch"]),
        "fixed_microbatch_sequence": recipe["microbatch"] == 4 and recipe["sequence"] == 512,
        "profile_recipe": plan["declared_envelope"].get("recipe") == recipe,
        "root_forward_count": plan["profile_counters"]["root_forwards"] == accumulation,
        "only_required_model_families": True,
        "exact_registered_overloads": True,
        "forward_ordinal_sets": True,
        "every_backward_activation_slot": True,
        "outside_event_complete": True,
        "outside_binding_metadata_and_sites": True,
        "remaining_bindings_checkpoint_scoped": True,
        "native_activation_arguments_present": True,
    }
    for index, event in enumerate(plan["events"]):
        family, kind = event["family"], event["kind"]
        if family not in calls_per_region:
            checks["only_required_model_families"] = False
            unexpected_events.append({"event": index, "family": family, "kind": kind,
                                      "operator": event["operator"]})
            continue
        checks["exact_registered_overloads"] &= event["operator"] == operators[family][kind]
        if kind == "forward":
            key = (family, event["region"], event["role"], event["microbatch"])
            actual_forward[key] += 1
            forward_ordinals.setdefault(key, []).append(event["ordinal"])
            if event["role"] == "external_forward":
                fields = {"input.input": event["arguments"]["input"]["tensor"]} if family == "native_layer_norm" else {}
                fields.update(event.get("outputs", {}))
                for path, metadata in fields.items():
                    site = core.canonical_json([event["region"], event["microbatch"],
                                                event["ordinal"], event["schema"], path])
                    external_fields[site] = {"microbatch": event["microbatch"], "family": family,
                                             "path": path, "metadata": metadata}
        else:
            actual_backward[family] += 1
            backward_by_micro[(family, event["microbatch"])] += 1
            backwards[index] = event
            checks["native_activation_arguments_present"] &= all(
                isinstance(event["arguments"].get(argument, {}).get("tensor"), dict)
                for argument in activation_args[family])
    checks["complete_forward_census"] = dict(actual_forward) == expected_forward
    checks["complete_backward_census"] = dict(actual_backward) == expected_backward
    checks["backward_microbatch_census"] = dict(backward_by_micro) == {
        (family, micro): count for family, count in (("gelu", 12), ("softmax", 12),
                                                   ("native_layer_norm", 25))
        for micro in range(accumulation)}
    for key, count in expected_forward.items():
        checks["forward_ordinal_sets"] &= sorted(forward_ordinals.get(key, [])) == list(range(count))
    counter_forward = Counter()
    for row in plan["profile_counters"]["forward"]:
        counter_forward[(row["family"], row["region"], row["role"], row["microbatch"])] += row["count"]
    checks["counter_event_consistency"] = (counter_forward == actual_forward
                                            and plan["profile_counters"]["backward"] == dict(actual_backward))
    expected_external_fields = {"input.input": [4, 512, 768], "output[0]": [4, 512, 768],
                                "output[1]": [4, 512, 1], "output[2]": [4, 512, 1]}
    checks["complete_external_forward_fields"] = (
        len(external_fields) == 4 * accumulation
        and all(sum(source["microbatch"] == micro and source["path"] == path
                    for source in external_fields.values()) == 1
                for micro in range(accumulation) for path in expected_external_fields))
    checks["external_forward_metadata"] = all(
        source["family"] == "native_layer_norm"
        and source["metadata"].get("shape") == expected_external_fields.get(source["path"])
        and source["metadata"].get("dtype") == "torch.float32"
        and source["metadata"].get("device") == "cuda:0"
        for source in external_fields.values())

    def finish_census(outside_rows):
        record = {
            "status": "PASS" if all(checks.values()) else "INVALID_PROFILE_CENSUS",
            "validation_scope": "COMPILED_MODEL_PLAN" if validate_bindings else "RAW_ARCHITECTURE_ONLY_UNSUPPORTED_PROFILE",
            "outside_binding_map_validated": validate_bindings and all(checks.values()),
            "architecture": {"layers": 12, "hidden_size": 768, "intermediate_size": 3072},
            "recipe": recipe, "expected_backward": expected_backward,
            "expected_forward": [{"family": key[0], "region": key[1], "role": key[2],
                                  "microbatch": key[3], "count": value}
                                 for key, value in sorted(expected_forward.items())],
            "observed_forward": [{"family": key[0], "region": key[1], "role": key[2],
                                  "microbatch": key[3], "count": value}
                                 for key, value in sorted(actual_forward.items())],
            "observed_backward": dict(actual_backward), "outside_bindings": outside_rows,
            "unexpected_model_events": unexpected_events,
            "checks": checks, "invalid_reasons": [key for key, passed in checks.items() if not passed],
            "actual_native_backward_order_learned": True, "scientific_acceptance": False,
        }
        write(Path(out) / "model-census.json", record)
        require(record["status"] == "PASS", "INVALID_PROFILE_CENSUS: " + repr(record["invalid_reasons"]))
        return record

    if not validate_bindings:
        for name in ("every_backward_activation_slot", "outside_event_complete",
                     "outside_binding_metadata_and_sites", "remaining_bindings_checkpoint_scoped"):
            del checks[name]
        return finish_census([])
    bindings = {}
    outside_events = Counter()
    outside_rows = []
    for binding in plan["bindings"]:
        bindings.setdefault(binding["event"], []).append(binding)
    for index, event in backwards.items():
        rows = bindings.get(index, [])
        checks["every_backward_activation_slot"] &= (
            len(rows) == len(activation_args[event["family"]])
            and {row["argument"] for row in rows} == activation_args[event["family"]])
        outside = [row for row in rows if plan["classes"][row["class_id"]]["scope"] == "outside"]
        if not outside:
            checks["remaining_bindings_checkpoint_scoped"] &= all(
                plan["classes"][row["class_id"]]["scope"] == "checkpoint" for row in rows)
            continue
        micro = event["microbatch"]
        outside_events[micro] += 1
        checks["outside_event_complete"] &= (event["family"] == "native_layer_norm"
                                                and len(outside) == 3 and len(rows) == 3)
        paths = {"input": "input.input", "mean": "output[1]", "rstd": "output[2]"}
        for row in outside:
            group = plan["classes"][row["class_id"]]
            argument = row["argument"]
            metadata = event["arguments"].get(argument, {}).get("tensor", {})
            expected_shape = [4, 512, 768] if argument == "input" else [4, 512, 1]
            valid = (argument in paths and metadata.get("shape") == expected_shape
                     and metadata.get("dtype") == "torch.float32" and metadata.get("device") == "cuda:0"
                     and all(group.get(key) == metadata.get(key) for key in ("shape", "dtype", "device")))
            expected_site_found = False
            for site in group["members"]:
                source = external_fields.get(site)
                valid &= (source is not None and source["family"] == "native_layer_norm"
                          and all(source["metadata"].get(key) == group.get(key)
                                  for key in ("shape", "dtype", "device")))
                if source is not None and source["microbatch"] == micro and source["path"] == paths.get(argument):
                    expected_site_found = True
            valid &= expected_site_found and bool(group["members"])
            checks["outside_binding_metadata_and_sites"] &= valid
            outside_rows.append({"event": index, "actual_backward_ordinal": event["ordinal"],
                                 "microbatch": micro, "argument": argument,
                                 "metadata": metadata, "class_id": row["class_id"],
                                 "members": group["members"], "valid": bool(valid)})
    checks["only_existing_backward_events_bound"] = set(bindings) == set(backwards)
    checks["one_final_layernorm_backward_per_microbatch"] = dict(outside_events) == {
        micro: 1 for micro in range(accumulation)}
    checks["final_layernorm_outside_binding_count"] = len(outside_rows) == 3 * accumulation
    record = finish_census(outside_rows)
    for case in method["model_protocol"]["cases"]:
        event_index = core.require_fault_target(plan, "gelu", case["consumer_ordinal"], "self")
        binding = next(row for row in plan["bindings"]
                       if row["event"] == event_index and row["argument"] == "self")
        members = plan["classes"][binding["class_id"]]["members"]
        regions = {json.loads(site)[0] for site in members}
        require("gpt_neox.layers.%d" % case["layer"] in regions,
                "Independent native fault ordinal does not identify its registered layer")
    return record


def run_pilot(out, contract):
    """Execute the frozen census, retaining unsupported and falsified outcomes.

    Native execution, injection and parity failures invalidate the run and stop.
    Qualified scientific negatives retain their declared denominator and block
    timing; they never become an infrastructure error or a successful detection.
    """
    completion.source_allocation_guard()
    method = _method(contract)
    protocol = method["model_protocol"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    stream = completion.stream_from_inputs(contract["inputs"])
    engine = completion.load_engine(_input(contract, "later-update210"))
    prestate = completion.snapshot(engine)
    parameters = sum(parameter.numel() for parameter in engine.model.parameters())
    require(parameters == 162322944, "Registered model parameter census changed")
    binding_started = time.perf_counter()
    identity_binding = _admitted_binding(contract)
    binding_seconds = time.perf_counter() - binding_started
    started = time.perf_counter()
    reference, ref_saved, ref_grads, _ = run_arm(engine, stream, prestate, out / "profile-native")
    require(reference["row"]["status"] == "ACCEPTED", "Native clean profile reference failed")
    profile_started = time.perf_counter()
    profiled, prof_saved, prof_grads, plan = run_arm(engine, stream, prestate, out / "profile-observed", profile=True,
                                                 identity_binding=identity_binding)
    profile_cost = _profile_cost(out / "profile-observed", time.perf_counter() - profile_started)
    profile_cost["immutable_identity_binding_setup_seconds"] = binding_seconds
    profile_cost["profile_and_identity_binding_seconds"] = profile_cost["complete_profile_call_wall_seconds"] + binding_seconds
    write(out / "profile-cost.json", profile_cost)
    if _unsupported(profiled):
        raw = json.loads((out / "profile-observed" / "profile-records.json").read_text())
        census_started = time.perf_counter()
        census = _validate_profile({"events": raw["events"], "profile_counters": raw["counters"],
                                    "declared_envelope": {"recipe": protocol["diagnostic_batch"]}},
                                   method, out=out, validate_bindings=False)
        profile_cost["model_census_admission_seconds"] = time.perf_counter() - census_started
        profile_cost["complete_profile_admission_seconds"] = (profile_cost["profile_and_identity_binding_seconds"]
                                                               + profile_cost["model_census_admission_seconds"])
        write(out / "profile-qualified-cost.json", profile_cost)
        reason = profiled["observation"]["reason"]
        cases = []
        skipped = 0
        for state in ("later-update210", "early-warm10"):
            root = out / state
            root.mkdir()
            for policy in protocol["clean_arms_per_state"]:
                _not_run(root / ("clean-" + policy), policy, reason)
                skipped += 1
            for index, case in enumerate(row for row in protocol["cases"] if row["state"] == state):
                case_root = root / ("case-%02d-%s-L%d-bit%d" % (index, case["window"], case["layer"], case["bit"]))
                case_root.mkdir()
                for label in protocol["fault_arms_per_case"]:
                    _not_run(case_root / label, label, reason, case)
                    skipped += 1
                cases.append({"case": case, "status": "UNSUPPORTED", "reason": reason,
                              "detection_expectations_met": False, "counts_as_detection": False})
        require(2 + skipped == protocol["model_arms_planned"] == 84, "Declared arm census differs")
        return {"experiment_id": method["experiment_id"], "phase": "PILOT",
                "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW", "execution_complete": True,
                "evidence_complete": True, "scientific_outcome": "UNSUPPORTED",
                "arms_declared": 84, "arms_executed": 2, "arms_not_run_unsupported": skipped,
                "cases_declared": 12, "supported_cases": 0, "unsupported_cases": 12,
                "case_outcomes": cases, "timing_prerequisites_met": False,
                "all_declared_cases_supported": False, "all_detection_expectations_met": False,
                "strict_fingerprint_byte_reduction": False, "profile_unsupported_reason": reason,
                "profile_parity": "NOT_APPLICABLE_UNSUPPORTED_PROFILE",
                "profile_cost": profile_cost, "model_census": census,
                "parameter_count": parameters, "parameter_count_delta": 0,
                "scientific_acceptance": False, "limits": method["excluded_claims"]}
    require(profiled["row"]["status"] == "ACCEPTED", "Clean profile execution failed")
    profile_parity = _parity(prof_saved, prof_grads, ref_saved, ref_grads)
    write(out / "compiled-plan.json", plan)
    write(out / "profile-parity.json", profile_parity)
    _require_parity(profile_parity, "Clean profile changed native state or gradients")
    require(reference["native_registered_counts"] == profiled["native_registered_counts"], "Profiling changed native calls")
    census_started = time.perf_counter()
    census = _validate_profile(plan, method, out=out)
    profile_cost["model_census_admission_seconds"] = time.perf_counter() - census_started
    profile_cost["complete_profile_admission_seconds"] = (profile_cost["profile_and_identity_binding_seconds"]
                                                           + profile_cost["model_census_admission_seconds"])
    write(out / "profile-qualified-cost.json", profile_cost)
    profile_seconds = profile_cost["complete_profile_admission_seconds"]
    del ref_saved, ref_grads, prof_saved, prof_grads
    arms, skipped = 2, 0
    states = []
    outcomes = []
    for state in ("later-update210", "early-warm10"):
        if state == "early-warm10":
            del engine, prestate
            gc.collect()
            engine = completion.load_engine(_input(contract, state))
            prestate = completion.snapshot(engine)
        root = out / state
        root.mkdir()
        clean = {}
        base_saved = base_grads = None
        for policy in protocol["clean_arms_per_state"]:
            record, saved, gradients, _ = run_arm(engine, stream, prestate, root / ("clean-" + policy),
                                                policy=policy, plan=plan, identity_binding=identity_binding)
            arms += 1
            clean[policy] = record
            if _unsupported(record):
                del saved, gradients
                continue
            require(record["row"]["status"] == "ACCEPTED", "A clean policy did not complete")
            if policy == "disabled":
                base_saved, base_grads = saved, gradients
            else:
                parity = _parity(saved, gradients, base_saved, base_grads)
                write(root / ("clean-" + policy) / "parity.json", parity)
                _require_parity(parity, "A clean policy changed native execution")
                require(record["native_registered_counts"] == clean["disabled"]["native_registered_counts"],
                        "Clean observation changed registered native execution counts")
                del saved, gradients
        clean_expected = all(clean[policy]["observation"].get("status") == "PASS"
                             for policy in protocol["clean_arms_per_state"] if policy != "disabled")
        case_rows = []
        for index, case in enumerate(row for row in protocol["cases"] if row["state"] == state):
            case_root = root / ("case-%02d-%s-L%d-bit%d" % (index, case["window"], case["layer"], case["bit"]))
            case_root.mkdir()
            records = {}
            fault_saved = fault_grads = None
            for label in protocol["fault_arms_per_case"]:
                enforce = label.endswith("_enforce")
                policy = label.removesuffix("_enforce") if enforce else label
                unqualified = clean[policy] if _unsupported(clean[policy]) else None
                if enforce and _unsupported(records["selected_consumer"]):
                    unqualified = records["selected_consumer"]
                if unqualified is not None:
                    records[label] = _not_run(case_root / label, label,
                                             unqualified["observation"]["reason"], case)
                    skipped += 1
                    continue
                record, saved, gradients, _ = run_arm(engine, stream, prestate, case_root / label,
                                                    policy=policy, plan=plan, case=case, enforce=enforce,
                                                    identity_binding=identity_binding)
                records[label] = record
                arms += 1
                if _unsupported(record):
                    del saved, gradients
                    continue
                if label == "disabled":
                    require(record["row"]["status"] == "ACCEPTED", "Unobserved fault did not complete")
                    fault_saved, fault_grads = saved, gradients
                    continue
                require(record["native_registered_counts"] == records["disabled"]["native_registered_counts"],
                        "Observation changed registered native execution count")
                require(record["external_consumer_injection"] == records["disabled"]["external_consumer_injection"],
                        "Observation changed external intervention")
                require(record["row"]["injections"] == records["disabled"]["row"]["injections"],
                        "Observation changed historical intervention")
                if enforce and record["observation"].get("failed") is True:
                    require(record["row"]["status"] == "REJECTED_PRE_OPTIMIZER", "Enforcement failed its declared callback contract")
                    protected = {key: value for key, value in prestate.items() if key != "rng"}
                    observed = {key: value for key, value in saved.items() if key != "rng"}
                    protection = {"protected_non_rng_state": completion.compare_trees(observed, protected),
                                  "preclip_gradients": completion.compare_trees(gradients, fault_grads)}
                    write(case_root / label / "protection.json", protection)
                    _require_parity(protection, "Pre-optimizer rejection did not protect declared state")
                else:
                    require(record["row"]["status"] == "ACCEPTED", "Observation interrupted a supported arm")
                    parity = _parity(saved, gradients, fault_saved, fault_grads)
                    write(case_root / label / "parity.json", parity)
                    _require_parity(parity, "Observation changed a faulted execution")
                del saved, gradients
            effects = {"gradients": completion.compare_trees(fault_grads, base_grads, distance=True),
                       "post_state": completion.compare_trees(fault_saved, base_saved, distance=True)}
            write(case_root / "fault-effects.json", effects)
            detections = {label: record["observation"].get("failed") for label, record in records.items()}
            supported = not any(_unsupported(record) for record in records.values())
            expected = (supported and clean_expected
                        and detections["selected_consumer"] is True
                        and records["selected_consumer_enforce"]["row"]["status"] == "REJECTED_PRE_OPTIMIZER")
            if case["window"] == "consumer_entry":
                expected = expected and all(detections[policy] is False
                                           for policy in ("outputs_only", "full_forward_io", "selected_forward"))
            else:
                expected = expected and detections["full_forward_io"] is True and detections["selected_forward"] is True
            outcome = {"case": case, "status": "SUPPORTED" if supported else "UNSUPPORTED",
                       "scientific_outcome": "EXPECTED_PATTERN" if expected else "FALSIFIED" if supported else "UNSUPPORTED",
                       "detection_expectations_met": bool(expected), "detections": detections,
                       "unsupported_arms": {label: record["observation"].get("reason")
                                            for label, record in records.items() if _unsupported(record)},
                       "effects": effects, "output_only_result_not_forced": True,
                       "zero_effect_not_excluded": True}
            write(case_root / "case-outcome.json", outcome)
            case_rows.append(outcome)
            outcomes.append(outcome)
            del fault_saved, fault_grads
        consumer_counters = clean["selected_consumer"]["observation"].get("counters", {})
        full_counters = clean["full_forward_io"]["observation"].get("counters", {})
        consumer_bytes = consumer_counters.get("fingerprint_payload_bytes")
        full_bytes = full_counters.get("fingerprint_payload_bytes")
        strict_bytes = (not _unsupported(clean["selected_consumer"]) and not _unsupported(clean["full_forward_io"])
                        and consumer_bytes is not None and full_bytes is not None and consumer_bytes < full_bytes)
        states.append({"state": state, "cases": case_rows, "clean_expectations_met": clean_expected,
                       "consumer_fingerprint_payload_bytes": consumer_bytes,
                       "full_io_fingerprint_payload_bytes": full_bytes,
                       "strictly_fewer_fingerprinted_bytes": strict_bytes,
                       "calibration_plan_reused_without_reprofiling": True})
        del base_saved, base_grads
    require(arms + skipped == protocol["model_arms_planned"] == 84, "Frozen model-arm census changed")
    require(len(outcomes) == len(protocol["cases"]) == 12, "Declared case denominator changed")
    all_supported = all(row["status"] == "SUPPORTED" for row in outcomes)
    all_expected = all(row["detection_expectations_met"] for row in outcomes)
    strict_bytes = all(row["strictly_fewer_fingerprinted_bytes"] for row in states)
    return {"experiment_id": method["experiment_id"], "phase": "PILOT",
            "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW", "execution_complete": True,
            "scientific_outcome": "UNSUPPORTED" if not all_supported else
                                  "EXPECTED_PATTERN" if all_expected and strict_bytes else "FALSIFIED",
            "evidence_complete": True, "arms_declared": 84, "arms_executed": arms,
            "arms_not_run_unsupported": skipped, "cases_declared": 12,
            "supported_cases": sum(row["status"] == "SUPPORTED" for row in outcomes),
            "unsupported_cases": sum(row["status"] == "UNSUPPORTED" for row in outcomes),
            "all_declared_cases_supported": all_supported, "all_detection_expectations_met": all_expected,
            "strict_fingerprint_byte_reduction": strict_bytes,
            "timing_prerequisites_met": all_supported and all_expected and strict_bytes,
            "actual_independent_promotion_still_required": True,
            "parameter_count": parameters, "parameter_count_delta": 0,
            "case_outcomes": outcomes,
            "plan_sha256": plan["plan_sha256"], "model_census": census,
            "plan_profile_cost_seconds": profile_seconds,
            "profile_step_elapsed_seconds": profiled["step_elapsed_seconds"], "profile_cost": profile_cost,
            "profile_cost_is_not_free": True, "states": states,
            "elapsed_seconds_diagnostic_only": time.perf_counter() - started,
            "scientific_acceptance": False, "limits": method["excluded_claims"]}


def run_timing(out, contract):
    """Run one independent job's frozen paired timing blocks, without pooling jobs."""
    completion.source_allocation_guard()
    method = _method(contract)
    protocol = method["timing_protocol"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    stream = completion.stream_from_inputs(contract["inputs"])
    engine = completion.load_engine(_input(contract, "later-update210"))
    prestate = completion.snapshot(engine)
    batch = protocol["production_batch"]
    binding_started = time.perf_counter()
    identity_binding = _admitted_binding(contract)
    binding_seconds = time.perf_counter() - binding_started
    profile_started = time.perf_counter()
    profile_record, _, _, plan = run_arm(
        engine, stream, prestate, out / "production-profile", profile=True,
        batch=batch, gradients=False, pilot_counter=False, capture_state=False,
        identity_binding=identity_binding)
    profile_cost = _profile_cost(out / "production-profile", time.perf_counter() - profile_started)
    profile_cost["immutable_identity_binding_setup_seconds"] = binding_seconds
    profile_cost["profile_and_identity_binding_seconds"] = profile_cost["complete_profile_call_wall_seconds"] + binding_seconds
    write(out / "profile-cost.json", profile_cost)
    require(profile_record["row"]["status"] == "ACCEPTED", "Production profile failed")
    write(out / "production-plan.json", plan)
    census_started = time.perf_counter()
    census = _validate_profile(plan, method, batch=batch, out=out)
    profile_cost["model_census_admission_seconds"] = time.perf_counter() - census_started
    profile_cost["complete_profile_admission_seconds"] = (profile_cost["profile_and_identity_binding_seconds"]
                                                           + profile_cost["model_census_admission_seconds"])
    write(out / "profile-qualified-cost.json", profile_cost)
    records = []
    update_count = 0
    job_index = contract["timing_job_index"]
    require(job_index in (0, 1, 2), "Registered independent timing job index is absent")
    schedule = [row for row in protocol["frozen_pair_schedule"] if row["job_index"] == job_index]
    expected_pairs = {(policy, block) for policy in protocol["checked_policies"]
                      for block in range(protocol["paired_blocks_per_policy_per_job"])}
    require(len(schedule) == len(expected_pairs)
            and {(row["policy"], row["block_index"]) for row in schedule} == expected_pairs,
            "Frozen timing schedule has missing or duplicate pairs")
    for position, scheduled in enumerate(schedule):
        policy, block, order = scheduled["policy"], scheduled["block_index"], scheduled["arm_order"]
        require(sorted(order) == sorted(["disabled", policy]), "Frozen arm order is malformed")
        pair = {}
        saved_states, saved_gradients = {}, {}
        for arm_policy in order:
            completion.restore(engine, prestate)
            arm_root = out / ("%s-block%d-%s" % (policy, block, arm_policy))
            arm_root.mkdir()
            seconds, counts = [], []
            all_updates = protocol["warmups_per_arm"] + protocol["measured_updates_per_arm"]
            for update in range(all_updates):
                completion.cutoff()
                last_update = update == all_updates - 1
                record, saved, gradients, _ = run_arm(
                    engine, stream, prestate, arm_root / ("update-%02d" % update),
                    policy=arm_policy, plan=plan, batch=batch, gradients=False,
                    pilot_counter=False, restore=False, branch_update=update + 1,
                    capture_state=last_update, identity_binding=identity_binding)
                seconds.append(record["step_elapsed_seconds"])
                require(record["row"]["status"] == "ACCEPTED", "Timing arm did not complete")
                if arm_policy != "disabled":
                    require(record["observation"]["status"] == "PASS", "Clean timing observer failed")
                    counts.append(record["observation"]["counters"])
                if last_update:
                    saved_states[arm_policy], saved_gradients[arm_policy] = saved, gradients
                update_count += 1
            pair[arm_policy] = {"warmup_seconds": seconds[:protocol["warmups_per_arm"]],
                                "measured_seconds": seconds[protocol["warmups_per_arm"]:],
                                "measured_elapsed_seconds": sum(seconds[protocol["warmups_per_arm"]:]),
                                "last_observer_counters": counts[-1] if counts else None}
        parity = _parity(saved_states[policy], saved_gradients[policy],
                         saved_states["disabled"], saved_gradients["disabled"])
        write(out / ("%s-block%d-parity.json" % (policy, block)), parity)
        records.append({"schedule_position": position, "policy": policy, "block_index": block,
                        "arm_order": order, "arms": pair,
                        "secondary_checked_disabled_elapsed_ratio":
                            pair[policy]["measured_elapsed_seconds"] /
                            pair["disabled"]["measured_elapsed_seconds"],
                        "final_eight_update_parity": parity,
                        "final_gradient_phase": "post-clipping"})
        write(out / ("%s-block%d-pair.json" % (policy, block)), records[-1])
        _require_parity(parity, "Eight-update timing pair state/gradient parity failed")
        del saved_states, saved_gradients
    require(update_count == protocol["total_updates_per_job"] == 128, "Timing-update census changed")
    direct = []
    for block in range(protocol["paired_blocks_per_policy_per_job"]):
        elapsed = {row["policy"]: row["arms"][row["policy"]]["measured_elapsed_seconds"]
                   for row in records if row["block_index"] == block}
        ratio = elapsed["selected_consumer"] / elapsed["full_forward_io"]
        require(math.isfinite(ratio) and ratio > 0, "Direct timing contrast is not finite positive")
        direct.append({"block_index": block, "consumer_elapsed_seconds": elapsed["selected_consumer"],
                       "full_io_elapsed_seconds": elapsed["full_forward_io"],
                       "direct_consumer_full_io_ratio": ratio, "log_ratio": math.log(ratio)})
    return {"experiment_id": method["experiment_id"], "phase": "TIMING",
            "independent_job_index": job_index, "independent_unit": "Slurm job",
            "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW", "execution_complete": True,
            "updates_executed": update_count, "updates_executed_scope": "timed arms only",
            "timed_arm_updates": update_count, "production_profile_updates": 1,
            "total_updates_executed": update_count + 1, "pairs": records,
            "primary_direct_blocks": direct,
            "primary_job_mean_log_ratio": sum(row["log_ratio"] for row in direct) / len(direct),
            "primary_estimator": protocol["estimator"],
            "production_profile_elapsed_seconds": profile_cost["complete_profile_admission_seconds"],
            "production_profile_step_elapsed_seconds": profile_record["step_elapsed_seconds"],
            "profile_cost": profile_cost,
            "production_profile_counters": profile_record["observation"]["counters"],
            "production_plan_sha256": plan["plan_sha256"], "model_census": census,
            "profile_cost_is_not_free": True,
            "no_pooled_interval_from_one_job": True, "scientific_acceptance": False}
