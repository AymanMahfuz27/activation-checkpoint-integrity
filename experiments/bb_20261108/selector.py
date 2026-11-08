"""Profile and check original-forward activation obligations at native consumers.

This is a family-scoped experiment, not a general autograd verifier. Clean
profiling learns logical value classes and a fixed event stream. Runtime never
learns from faulted values: it checks every required original member, demands
class unanimity, and compares captured consumer values before optimizer entry.
"""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import hashlib
import json
import os
import sys


def require_allocated():
    job = os.environ.get("SLURM_JOB_ID", "")
    if (sys.platform != "linux" or os.getuid() != 893489 or not job.isdecimal()
            or os.environ.get("BB_NUMERICAL_ADMISSION") != "PASS:" + job):
        raise RuntimeError("BB numerical modules require fresh owned Slurm admission")


require_allocated()

import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes  # noqa: E402
from ac_integrity.capture.fingerprint import ALGORITHM, fingerprint_tensor  # noqa: E402


REGION = ContextVar("bb_checkpoint_region", default=None)
POLICIES = ("outputs_only", "full_forward_io", "selected_forward", "selected_consumer")
REGISTRY = (
    {"family": "gelu", "forward": "aten.gelu.default", "backward": "aten.gelu_backward.default",
     "forward_activation_inputs": ["self"], "forward_outputs": ["output"],
     "backward_activation_args": [{"name": "self", "position": 1}]},
    {"family": "native_layer_norm", "forward": "aten.native_layer_norm.default",
     "backward": "aten.native_layer_norm_backward.default",
     "forward_activation_inputs": ["input"],
     "forward_outputs": ["output[0]", "output[1]", "output[2]"],
     "backward_activation_args": [{"name": "input", "position": 1},
                                  {"name": "mean", "position": 3},
                                  {"name": "rstd", "position": 4}]},
    {"family": "softmax", "forward": "aten._softmax.default",
     "backward": "aten._softmax_backward_data.default",
     "forward_activation_inputs": ["self"], "forward_outputs": ["output"],
     "backward_activation_args": [{"name": "output", "position": 1}]},
    {"family": "relu", "forward": "aten.relu.default", "backward": "aten.threshold_backward.default",
     "forward_activation_inputs": ["self"], "forward_outputs": ["output"],
     "backward_activation_args": [{"name": "self", "position": 1}]},
    {"family": "tanh", "forward": "aten.tanh.default", "backward": "aten.tanh_backward.default",
     "forward_activation_inputs": ["self"], "forward_outputs": ["output"],
     "backward_activation_args": [{"name": "output", "position": 1}]},
    {"family": "sigmoid", "forward": "aten.sigmoid.default",
     "backward": "aten.sigmoid_backward.default",
     "forward_activation_inputs": ["self"], "forward_outputs": ["output"],
     "backward_activation_args": [{"name": "output", "position": 1}]},
)


class UnsupportedPlan(RuntimeError):
    """An unqualified plan must not be credited as successful detection."""


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def plan_digest(plan):
    body = {name: value for name, value in plan.items() if name != "plan_sha256"}
    return hashlib.sha256(canonical_json(body).encode()).hexdigest()


@contextmanager
def region_context(region, role):
    if role not in ("original", "recompute"):
        raise UnsupportedPlan("Unknown checkpoint role")
    token = REGION.set((str(region), role))
    try:
        yield
    finally:
        REGION.reset(token)


def _metadata(value):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or value.layout != torch.strided or value.is_quantized
            or value.is_conj() or value.is_neg() or value.device.type not in ("cpu", "cuda")):
        raise UnsupportedPlan("Only materialized dense FP32 activation values are qualified")
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "device": str(value.device), "stride": list(value.stride()),
            "storage_offset": value.storage_offset(), "layout": str(value.layout)}


def _scalar_metadata(value):
    if isinstance(value, torch.Tensor):
        return {"tensor": _metadata(value)}
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, (tuple, list)):
        return [_scalar_metadata(item) for item in value]
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    raise UnsupportedPlan("Unsupported operator argument metadata: " + type(value).__name__)


def _bind(func, args, kwargs):
    arguments = func._schema.arguments
    if len(args) > len(arguments) or any(key not in {item.name for item in arguments} for key in kwargs):
        raise UnsupportedPlan("Operator arguments do not match the native schema")
    bound = {}
    for position, argument in enumerate(arguments):
        if position < len(args):
            if argument.name in kwargs:
                raise UnsupportedPlan("Duplicate native argument")
            bound[argument.name] = args[position]
        elif argument.name in kwargs:
            bound[argument.name] = kwargs[argument.name]
        elif argument.has_default_value():
            bound[argument.name] = argument.default_value
        else:
            raise UnsupportedPlan("Missing native argument: " + argument.name)
    return bound


def _check_options(family, kind, bound):
    if family == "gelu" and bound.get("approximate", "none") != "none":
        raise UnsupportedPlan("Only exact GELU is registered")
    if family == "softmax":
        tensor = bound.get("self") if kind == "forward" else bound.get("output")
        if bound.get("dim") not in (-1, tensor.ndim - 1):
            raise UnsupportedPlan("Only last-axis softmax is registered")
        if kind == "forward" and bound.get("half_to_float") is not False:
            raise UnsupportedPlan("Softmax half conversion is not registered")
        if kind == "backward" and bound.get("input_dtype") != torch.float32:
            raise UnsupportedPlan("Softmax backward input dtype changed")
    if family == "relu" and kind == "backward" and bound.get("threshold") != 0:
        raise UnsupportedPlan("Only ReLU threshold zero is registered")


def _signature_key(signature, metadata):
    with _disable_current_modes():
        words = tuple(int(word) for word in signature.detach().cpu().tolist())
    return (words, tuple(metadata["shape"]), metadata["dtype"], metadata["device"])


def _site_id(region, microbatch, ordinal, schema, path):
    return canonical_json([region, microbatch, ordinal, schema, path])


def compile_plan(profile_records, registry=REGISTRY, declared_envelope=None):
    """Freeze a clean profile, retaining all compatible original value sources.

    Equal values supply a value class, never a storage or alias identity. Every
    paired original/replay field must agree during calibration. A demanded
    class mixing checkpoint and external sites is unqualified; an all-external
    class is explicitly outside scope rather than being silently dropped.
    """
    if not profile_records.get("complete"):
        raise UnsupportedPlan("An incomplete profile cannot compile a plan")
    originals, replays, groups = {}, {}, {}
    for row in profile_records["forward_fields"]:
        table = replays if row["role"] == "recompute" else originals
        if row["site_id"] in table:
            raise UnsupportedPlan("Duplicate logical forward field")
        table[row["site_id"]] = row
    if not originals or not replays:
        raise UnsupportedPlan("No complete checkpoint original/replay profile")
    for site, row in originals.items():
        if row["role"] == "original" and site not in replays:
            raise UnsupportedPlan("Missing replay field during calibration")
        key = _signature_key(row["signature"], row["metadata"])
        groups.setdefault(key, []).append(site)
    for site, row in replays.items():
        if site not in originals:
            raise UnsupportedPlan("Replay lacks an original field")
        original = originals[site]
        if _signature_key(row["signature"], row["metadata"]) != _signature_key(original["signature"], original["metadata"]):
            raise UnsupportedPlan("Clean original/replay values differ")

    classes, bindings, group_ids = {}, [], {}
    for row in profile_records["backward_fields"]:
        key = _signature_key(row["signature"], row["metadata"])
        members = sorted(groups.get(key, []))
        if not members:
            raise UnsupportedPlan("Actual clean backward argument has no forward value class")
        inside = {originals[site]["role"] == "original" for site in members}
        if len(inside) != 1:
            raise UnsupportedPlan("A demanded class mixes checkpoint and external fields")
        if key not in group_ids:
            identity = "class-" + str(len(classes))
            group_ids[key] = identity
            classes[identity] = {"members": members,
                                 "scope": "checkpoint" if True in inside else "outside",
                                 "shape": row["metadata"]["shape"],
                                 "dtype": row["metadata"]["dtype"],
                                 "device": row["metadata"]["device"]}
        bindings.append({"event": row["event"], "argument": row["argument"],
                         "class_id": group_ids[key]})
    if not any(item["scope"] == "checkpoint" for item in classes.values()):
        raise UnsupportedPlan("No admitted checkpoint consumer obligations")
    plan = {"schema": 1, "method": "BB-DEMAND-20261108-v1", "registry": copy.deepcopy(list(registry)),
            "declared_envelope": declared_envelope or {},
            "events": copy.deepcopy(profile_records["events"]),
            "classes": classes, "bindings": bindings,
            "profile_counters": copy.deepcopy(profile_records["counters"]),
            "scope": "Registered checkpoint activation values; no alias, full-model or post-check guarantee"}
    plan["plan_sha256"] = plan_digest(plan)
    return plan


class SignatureStore:
    """Device-resident compact signatures with one fixed CUDA stream per device."""

    def __init__(self):
        self.streams = {}
        self.fingerprints = 0
        self.payload_bytes = 0

    def capture(self, value):
        _metadata(value)
        with torch.no_grad(), _disable_current_modes():
            if value.device.type == "cuda":
                key = str(value.device)
                stream = torch.cuda.current_stream(value.device).cuda_stream
                if key in self.streams and self.streams[key] != stream:
                    raise UnsupportedPlan("CUDA stream changed during one observation")
                self.streams[key] = stream
            # The explicit clone also canonicalizes singleton expanded strides.
            canonical = value.detach().clone(memory_format=torch.contiguous_format)
            digest = fingerprint_tensor(canonical, include_sketch=False, backend="full")
            self.fingerprints += 1
            self.payload_bytes += value.numel() * value.element_size()
            return digest.signatures


class _BaseMode(TorchDispatchMode):
    def __init__(self, registry=REGISTRY, external_fault=None):
        super().__init__()
        self.registry = copy.deepcopy(list(registry))
        self.forward_registry = {item["forward"]: item for item in self.registry}
        self.backward_registry = {item["backward"]: item for item in self.registry}
        self.registered_operator_bases = {name.rsplit(".", 1)[0]
                                          for name in (*self.forward_registry, *self.backward_registry)}
        if len(self.forward_registry) != len(self.registry) or len(self.backward_registry) != len(self.registry):
            raise UnsupportedPlan("Duplicate registered overload")
        self.external_fault = external_fault
        self.store = SignatureStore()
        self.events = []
        self.forward_fields = []
        self.backward_fields = []
        self.forward_counts = Counter()
        self.backward_counts = Counter()
        self.context_counts = Counter()
        self.root_calls = 0
        self.microbatch = -1
        self.root_depth = 0
        self.handles = []
        self._busy = False
        self.gated = False
        self.closed = False
        self.unsupported = None
        self.started = None

    def _signature(self, value):
        previous = self._busy
        self._busy = True
        try:
            return self.store.capture(value)
        finally:
            self._busy = previous

    @contextmanager
    def track_model(self, model):
        """Install metadata-only root hooks and restore them even after failure."""
        if self.handles:
            raise UnsupportedPlan("Root observation hooks are already installed")
        def before(module, args):
            if self.root_depth:
                raise UnsupportedPlan("Nested root forward is unqualified")
            self.root_calls += 1
            self.microbatch = self.root_calls - 1
            self.root_depth = 1

        def after(module, args, output):
            self.root_depth = 0

        first = model.register_forward_pre_hook(before)
        second = model.register_forward_hook(after, always_call=True)
        self.handles.extend((first, second))
        try:
            yield self
        finally:
            second.remove()
            first.remove()
            self.handles.clear()
            self.root_depth = 0

    def _forward_location(self):
        context = REGION.get()
        if context is not None:
            region, role = context
            self.context_counts[(region, role, self.microbatch)] += 0
            return region, role
        if self.root_depth:
            return "__external_forward__", "external_forward"
        raise UnsupportedPlan("Registered forward outside root or checkpoint context")

    def _event(self, func, kind, item, bound):
        _check_options(item["family"], kind, bound)
        description = {"kind": kind, "operator": str(func), "schema": str(func._schema),
                       "family": item["family"], "arguments": {}}
        for name, value in bound.items():
            # Gradients are not fixed activation metadata: their layout can
            # differ across fault arms without changing the observed contract.
            if kind == "backward" and name not in {x["name"] for x in item["backward_activation_args"]}:
                if isinstance(value, torch.Tensor):
                    continue
            description["arguments"][name] = _scalar_metadata(value)
        if kind == "forward":
            region, role = self._forward_location()
            key = (region, role, self.microbatch, item["family"])
            ordinal = self.forward_counts[key]
            self.forward_counts[key] += 1
            description.update(region=region, role=role, microbatch=self.microbatch, ordinal=ordinal)
        else:
            ordinal = self.backward_counts[item["family"]]
            self.backward_counts[item["family"]] += 1
            description.update(ordinal=ordinal, microbatch=self.microbatch)
        return description

    def _field(self, description, path, value, event_index):
        metadata = _metadata(value)
        site = None
        if description["kind"] == "forward":
            site = _site_id(description["region"], description["microbatch"],
                            description["ordinal"], description["schema"], path)
        return {"event": event_index, "site_id": site, "path": path,
                "metadata": metadata, "role": description.get("role"),
                "argument": path.removeprefix("input.")}

    def _capture_field(self, description, path, value, event_index):
        raise NotImplementedError

    def _validate_event(self, description):
        return None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self._busy:
            return func(*args, **kwargs)
        name = str(func)
        kind = "forward" if name in self.forward_registry else "backward" if name in self.backward_registry else None
        if kind is None:
            if name.rsplit(".", 1)[0] in self.registered_operator_bases:
                self.unsupported = "Unregistered overload of a registered native family: " + name
                raise UnsupportedPlan(self.unsupported)
            return func(*args, **kwargs)
        item = self.forward_registry[name] if kind == "forward" else self.backward_registry[name]
        try:
            bound = _bind(func, args, kwargs)
            description = self._event(func, kind, item, bound)
            event_index = len(self.events)
            self._validate_event(description)
            self.events.append(description)
            if kind == "backward" and self.external_fault is not None:
                with torch.no_grad(), _disable_current_modes():
                    self.external_fault(func, bound, description)
            fields = item["forward_activation_inputs"] if kind == "forward" else [x["name"] for x in item["backward_activation_args"]]
            for field in fields:
                if field not in bound:
                    raise UnsupportedPlan("Registered tensor argument is missing")
                self._capture_field(description, "input." + field, bound[field], event_index)
            output = func(*args, **kwargs)
            if kind == "forward":
                output_metadata = {}
                for path in item["forward_outputs"]:
                    value = output if path == "output" else output[int(path[7:-1])]
                    output_metadata[path] = _metadata(value)
                    self._capture_field(description, path, value, event_index)
                description["outputs"] = output_metadata
                self._validate_outputs(description, event_index)
            return output
        except UnsupportedPlan as error:
            self.unsupported = str(error)
            raise

    def _validate_outputs(self, description, event_index):
        return None

    def counters(self):
        return {"root_forwards": self.root_calls,
                "forward": [{"region": k[0], "role": k[1], "microbatch": k[2],
                             "family": k[3], "count": value}
                            for k, value in sorted(self.forward_counts.items())],
                "backward": dict(self.backward_counts),
                "fingerprints": self.store.fingerprints,
                "fingerprint_payload_bytes": self.store.payload_bytes,
                "algorithm": ALGORITHM, "backend": "full", "include_sketch": False,
                "canonical_clone_charged": True}


class ProfileMode(_BaseMode):
    """Capture a clean event/value profile; compile before any fault evaluation."""

    def _capture_field(self, description, path, value, event_index):
        row = self._field(description, path, value, event_index)
        row["signature"] = self._signature(value)
        target = self.forward_fields if description["kind"] == "forward" else self.backward_fields
        target.append(row)

    def finish(self, declared_envelope=None):
        if self.unsupported:
            raise UnsupportedPlan(self.unsupported)
        if self.root_calls < 1 or self.root_depth:
            raise UnsupportedPlan("A complete root model-forward context is required")
        if self.gated:
            return self.plan
        self.plan = compile_plan({"complete": True, "events": self.events,
                                  "forward_fields": self.forward_fields,
                                  "backward_fields": self.backward_fields,
                                  "counters": self.counters()}, self.registry, declared_envelope)
        self.gated = True
        return self.plan


class CheckMode(_BaseMode):
    """Use a frozen clean profile; collect compact comparisons for the final gate."""

    def __init__(self, plan, policy, signature_store=None, external_fault=None):
        if policy not in POLICIES:
            raise ValueError("Unknown BB observation policy")
        if plan.get("schema") != 1 or plan.get("plan_sha256") != plan_digest(plan):
            raise UnsupportedPlan("Frozen plan digest or schema differs")
        super().__init__(plan["registry"], external_fault)
        self.plan = copy.deepcopy(plan)
        self.policy = policy
        if signature_store is not None:
            self.store = signature_store
        self.selected_sites = set()
        for item in plan["classes"].values():
            if item["scope"] == "checkpoint":
                self.selected_sites.update(item["members"])
        self.bindings = {(row["event"], row["argument"]): row["class_id"] for row in plan["bindings"]}
        if len(self.bindings) != len(plan["bindings"]):
            raise UnsupportedPlan("Duplicate frozen backward obligation")
        self.values = {}
        self.consumer_values = {}
        self.result = None

    def _validate_event(self, description):
        index = len(self.events)
        if index >= len(self.plan["events"]):
            raise UnsupportedPlan("Unexpected registered native event")
        expected = {key: value for key, value in self.plan["events"][index].items() if key != "outputs"}
        if description != expected:
            raise UnsupportedPlan("Registered schema, options, metadata or event order changed")

    def _validate_outputs(self, description, event_index):
        if description.get("outputs") != self.plan["events"][event_index].get("outputs"):
            raise UnsupportedPlan("Registered output metadata changed")

    def _capture_field(self, description, path, value, event_index):
        row = self._field(description, path, value, event_index)
        if description["kind"] == "backward":
            key = (event_index, row["argument"])
            if key not in self.bindings:
                raise UnsupportedPlan("Unknown native backward activation slot")
            group = self.plan["classes"][self.bindings[key]]
            if self.policy == "selected_consumer" and group["scope"] == "checkpoint":
                if key in self.consumer_values:
                    raise UnsupportedPlan("Duplicate consumer argument")
                self.consumer_values[key] = self._signature(value)
            return
        if description["role"] == "external_forward":
            return
        site = row["site_id"]
        selected = site in self.selected_sites
        take = (self.policy == "full_forward_io"
                or self.policy == "outputs_only" and path.startswith("output")
                or self.policy == "selected_forward" and selected
                or self.policy == "selected_consumer" and selected and description["role"] == "original")
        if take:
            key = (description["role"], site)
            if key in self.values:
                raise UnsupportedPlan("Duplicate runtime forward field")
            self.values[key] = self._signature(value)

    def finish(self):
        if self.result is not None:
            return self.result
        if self.unsupported or len(self.events) != len(self.plan["events"]):
            raise UnsupportedPlan(self.unsupported or "Registered event count changed")
        if self.root_depth or self.root_calls != self.plan["profile_counters"]["root_forwards"]:
            raise UnsupportedPlan("Root-forward count or completion changed")
        comparisons, class_flags = [], []
        with torch.no_grad(), _disable_current_modes():
            if self.policy == "selected_consumer":
                references = {}
                for identity, group in self.plan["classes"].items():
                    if group["scope"] != "checkpoint":
                        continue
                    members = []
                    for site in group["members"]:
                        if ("original", site) not in self.values:
                            raise UnsupportedPlan("Required original class member is missing")
                        members.append(self.values[("original", site)])
                    references[identity] = members[0]
                    for member in members[1:]:
                        class_flags.append(torch.ne(member, members[0]).any())
                for key, identity in self.bindings.items():
                    if self.plan["classes"][identity]["scope"] != "checkpoint":
                        continue
                    if key not in self.consumer_values:
                        raise UnsupportedPlan("Required actual consumer argument is missing")
                    comparisons.append(torch.ne(self.consumer_values[key], references[identity]).any())
            else:
                originals = {site for role, site in self.values if role == "original"}
                replays = {site for role, site in self.values if role == "recompute"}
                if not originals or originals != replays:
                    raise UnsupportedPlan("Missing or asymmetric original/replay value pairs")
                for site in sorted(originals):
                    comparisons.append(torch.ne(self.values[("original", site)], self.values[("recompute", site)]).any())
            if not comparisons:
                raise UnsupportedPlan("Empty comparison coverage")
            # One compact transfer per flag set; payload tensors never reach CPU.
            split_values = torch.stack(class_flags).detach().cpu().tolist() if class_flags else []
            mismatch_values = torch.stack(comparisons).detach().cpu().tolist()
        if any(split_values):
            self.unsupported = "Original value class split"
            raise UnsupportedPlan(self.unsupported)
        self.gated = True
        self.result = {"status": "VALUE_MISMATCH" if any(mismatch_values) else "PASS",
                       "policy": self.policy, "failed": any(mismatch_values),
                       "mismatches": sum(bool(value) for value in mismatch_values),
                       "comparison_count": len(comparisons),
                       "class_unanimity_comparisons": len(class_flags),
                       "counters": self.counters(), "plan_sha256": self.plan["plan_sha256"],
                       "storage_alias_identity_proved": False,
                       "scientific_acceptance": False}
        return self.result

    def summary(self):
        return self.result or {"status": "UNSUPPORTED_PLAN" if self.unsupported else "INCOMPLETE",
                               "reason": self.unsupported, "policy": self.policy,
                               "counters": self.counters(), "scientific_acceptance": False}


def make_pre_gate_callback(mode, enforce=False, declared_envelope=None):
    """Finalize after backward and gradient capture, before native clipping/step."""
    def callback(engine, mean_loss):
        if isinstance(mode, ProfileMode):
            mode.finish(declared_envelope)
            return
        result = mode.finish()
        if enforce and result["failed"]:
            from ac_integrity.distributed import IntegrityError
            raise IntegrityError("BB selected activation obligation failed before optimizer")
    return callback


def require_fault_target(plan, family, ordinal, argument):
    """Qualify a registered target before a case, without choosing its location."""
    events = [(index, row) for index, row in enumerate(plan["events"])
              if row["kind"] == "backward" and row["family"] == family
              and row["ordinal"] == ordinal]
    if len(events) != 1:
        raise UnsupportedPlan("Independent fault target has no unique native slot")
    matches = [row for row in plan["bindings"] if row["event"] == events[0][0]
               and row["argument"] == argument]
    if len(matches) != 1 or plan["classes"][matches[0]["class_id"]]["scope"] != "checkpoint":
        raise UnsupportedPlan("Requested fault target lies outside admitted coverage")
    return events[0][0]


class ConsumerFaultMode(TorchDispatchMode):
    """Deliver an independent entry fault even with all checking disabled.

    Install this above an observation mode, so delivery precedes every candidate
    fingerprint and the native call. The callback owns its fixed target, counts
    and event journal; no selector plan participates in choosing the mutation.
    """

    def __init__(self, callback, registry=REGISTRY):
        super().__init__()
        self.callback = callback
        self.registry = {item["backward"]: item for item in registry}
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if name in self.registry:
            family = self.registry[name]["family"]
            ordinal = self.counts[family]
            self.counts[family] += 1
            bound = _bind(func, args, kwargs)
            description = {"kind": "backward", "operator": name,
                           "schema": str(func._schema), "family": family,
                           "ordinal": ordinal}
            with torch.no_grad(), _disable_current_modes():
                self.callback(func, bound, description)
        return func(*args, **kwargs)
