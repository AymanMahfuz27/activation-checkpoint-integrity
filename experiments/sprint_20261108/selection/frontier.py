"""Derivative-demand filtering with per-update replay provenance at native use.

The qualified unit is the activation arguments of six explicit ATen backward
families. This does not verify incoming gradients, parameters, unrelated
operators, or a complete training step. It never infers provenance from equal
values and needs no clean value profile. See method.md for the conditional
correctness statement, storage retention cost, and unsupported cases.
"""

from collections import Counter
import copy

import selector as core

import torch
from torch.utils._python_dispatch import _disable_current_modes


POLICIES = ("consumer", "selected_forward", "simple_input", "outputs_only", "full_forward_io")
# These fields are read by the corresponding native derivative, not every
# forward argument. ReLU backward receives the forward result, named `self`.
DEMANDS = {
    "gelu": {"self": "input.self"},
    "native_layer_norm": {"input": "input.input", "mean": "output[1]", "rstd": "output[2]"},
    "softmax": {"output": "output"},
    "relu": {"self": "output"},
    "tanh": {"output": "output"},
    "sigmoid": {"output": "output"},
}


def _storage_view(value):
    """Return an exact materialized storage/view key and its retained storage.

    Retention prevents allocator address reuse from fabricating provenance.
    Empty tensors and copied/untracked saved values are unsupported. Pointer
    identity alone is never described as a logical original-forward identity.
    """
    metadata = core._metadata(value)
    if value.numel() == 0:
        raise core.UnsupportedPlan("Empty demanded activation has no qualified storage identity")
    with _disable_current_modes():
        storage = value.untyped_storage()
        key = (metadata["device"], int(storage.data_ptr()), int(storage.nbytes()),
               metadata["storage_offset"], tuple(metadata["shape"]),
               tuple(metadata["stride"]), metadata["dtype"])
    return key, storage


def _forward_description(description):
    return {key: value for key, value in description.items() if key not in ("role", "outputs")}


class DemandFrontierMode(core._BaseMode):
    """Capture original obligations and check forward replay or actual consumers.

    Inputs are the native events inside explicit BB checkpoint contexts; output
    is a gate result after one or more sequential forward/backward microbatches.
    Original signatures are fresh every mode instance. The consumer policy
    retains demanded replay storage until finish(), binds exact storage/views,
    and compares every compatible original reference to the consumed value.
    """

    def __init__(self, policy="consumer", *, external_fault=None):
        if policy not in POLICIES:
            raise ValueError("Unknown derivative-frontier policy: " + str(policy))
        super().__init__(registry=core.REGISTRY, external_fault=external_fault)
        self.policy = policy
        self.originals = {}
        self.replays = {}
        self.forward_values = {}
        self.sources = {}
        self.storage_refs = {}
        self.original_demand_sites = set()
        self.consumed_sites = set()
        self.comparisons = []
        self.provenance = []
        self.original_descriptions = {}
        self.replay_descriptions = {}
        self.max_retained_storage_bytes = 0
        self.max_retained_storage_count = 0
        self.consumer_calls = Counter()
        self.external_consumer_arguments = 0
        self.result = None

    def _validate_event(self, description):
        if description["kind"] != "forward":
            return
        if description["role"] == "external_forward":
            return
        key = (description["region"], description["microbatch"],
               description["family"], description["ordinal"])
        table = self.original_descriptions if description["role"] == "original" else self.replay_descriptions
        if key in table:
            raise core.UnsupportedPlan("Duplicate logical registered forward event")
        table[key] = copy.deepcopy(_forward_description(description))
        if description["role"] == "recompute":
            if key not in self.original_descriptions or table[key] != _forward_description(self.original_descriptions[key]):
                raise core.UnsupportedPlan("Original/replay schema, options or activation metadata changed")

    def _validate_outputs(self, description, event_index):
        if description["role"] == "external_forward":
            return
        key = (description["region"], description["microbatch"],
               description["family"], description["ordinal"])
        table = self.original_descriptions if description["role"] == "original" else self.replay_descriptions
        table[key]["outputs"] = copy.deepcopy(description["outputs"])
        if description["role"] == "recompute" and table[key] != self.original_descriptions[key]:
            raise core.UnsupportedPlan("Original/replay output metadata changed")

    def _selected(self, family, path):
        return path in DEMANDS[family].values()

    def _observe_forward(self, row, description, path, value):
        role, site = row["role"], row["site_id"]
        demanded = self._selected(description["family"], path)
        if role != "external_forward":
            table = self.originals if role == "original" else self.replays
            if site in table:
                raise core.UnsupportedPlan("Duplicate runtime logical forward field")
            table[site] = copy.deepcopy(row["metadata"])
            if role == "recompute" and (site not in self.originals or self.originals[site] != table[site]):
                raise core.UnsupportedPlan("Original/replay forward-field metadata changed")
            if demanded and role == "original":
                self.original_demand_sites.add(site)
            take = (self.policy == "full_forward_io"
                    or self.policy == "simple_input" and path.startswith("input.")
                    or self.policy == "outputs_only" and path.startswith("output")
                    or self.policy == "selected_forward" and demanded
                    or self.policy == "consumer" and demanded and role == "original")
            if take:
                self.forward_values[(role, site)] = self._signature(value)
        if self.policy == "consumer" and demanded and role in ("recompute", "external_forward"):
            key, storage = _storage_view(value)
            candidate = {"site_id": site, "family": description["family"], "path": path,
                         "scope": "checkpoint" if role == "recompute" else "outside"}
            self.sources.setdefault(key, []).append(candidate)
            # Hold one storage reference per allocation, not one per alias view.
            allocation = key[:3]
            self.storage_refs[allocation] = storage
            retained = sum(int(item.nbytes()) for item in self.storage_refs.values())
            self.max_retained_storage_bytes = max(self.max_retained_storage_bytes, retained)
            self.max_retained_storage_count = max(self.max_retained_storage_count, len(self.storage_refs))

    def _observe_consumer(self, row, description, value):
        if self.policy != "consumer":
            return
        family, argument = description["family"], row["argument"]
        expected_path = DEMANDS[family].get(argument)
        if expected_path is None:
            raise core.UnsupportedPlan("Native backward activation slot lacks a derivative mapping")
        key, _ = _storage_view(value)
        candidates = [item for item in self.sources.get(key, [])
                      if item["family"] == family and item["path"] == expected_path]
        if not candidates:
            raise core.UnsupportedPlan("Native backward argument has no exact replay/external storage-view source")
        scopes = {item["scope"] for item in candidates}
        if len(scopes) != 1:
            raise core.UnsupportedPlan("Native backward source aliases checkpoint and outside-scope fields")
        if scopes == {"outside"}:
            self.external_consumer_arguments += 1
            self.provenance.append({"family": family, "argument": argument, "scope": "outside",
                                    "candidate_sites": [item["site_id"] for item in candidates]})
            return
        signature = self._signature(value)
        sites = sorted({item["site_id"] for item in candidates})
        with torch.no_grad(), _disable_current_modes():
            for site in sites:
                original = self.forward_values.get(("original", site))
                if original is None:
                    raise core.UnsupportedPlan("A storage-bound replay source lacks its original obligation")
                self.comparisons.append(torch.ne(signature, original).any())
                self.consumed_sites.add(site)
        self.consumer_calls[(family, argument)] += 1
        self.provenance.append({"family": family, "argument": argument, "scope": "checkpoint",
                                "candidate_sites": sites, "binding": "exact_retained_storage_and_view"})

    def _capture_field(self, description, path, value, event_index):
        row = self._field(description, path, value, event_index)
        if description["kind"] == "forward":
            self._observe_forward(row, description, path, value)
        else:
            self._observe_consumer(row, description, value)

    def finish(self):
        """Decide after backward, before clipping or optimizer mutation."""
        if self.result is not None:
            return self.result
        try:
            if self.unsupported:
                raise core.UnsupportedPlan(self.unsupported)
            if self.root_calls < 1 or self.root_depth:
                raise core.UnsupportedPlan("A completed root-forward context is required")
            if not self.originals or self.originals != self.replays:
                raise core.UnsupportedPlan("Missing or asymmetric registered original/replay fields")
            if self.original_descriptions != self.replay_descriptions:
                raise core.UnsupportedPlan("Registered original/replay event metadata differs")
            if self.policy == "consumer":
                if self.original_demand_sites != self.consumed_sites:
                    raise core.UnsupportedPlan("A demanded original field has no qualified native consumer")
            else:
                originals = {site for role, site in self.forward_values if role == "original"}
                replays = {site for role, site in self.forward_values if role == "recompute"}
                if not originals or originals != replays:
                    raise core.UnsupportedPlan("Missing or asymmetric filtered forward comparisons")
                with torch.no_grad(), _disable_current_modes():
                    for site in sorted(originals):
                        a = self.forward_values[("original", site)]
                        b = self.forward_values[("recompute", site)]
                        self.comparisons.append(torch.ne(a, b).any())
            if not self.comparisons:
                raise core.UnsupportedPlan("Empty activation comparison coverage")
            with torch.no_grad(), _disable_current_modes():
                flags = torch.stack(self.comparisons).detach().cpu().tolist()
            self.gated = True
            self.result = {"status": "VALUE_MISMATCH" if any(flags) else "PASS", "failed": any(flags),
                           "mismatches": sum(bool(value) for value in flags),
                           "comparison_count": len(flags), "policy": self.policy,
                           "method": "STATIC_DERIVATIVE_FRONTIER_RETAINED_REPLAY_STORAGE_V1",
                           "clean_value_profile_required": False,
                           "value_equality_used_for_provenance": False,
                           "max_retained_storage_bytes": self.max_retained_storage_bytes,
                           "max_retained_storage_allocations": self.max_retained_storage_count,
                           "retention_scope": "selected replay and registered external demanded storage until gate",
                           "external_consumer_arguments_outside_scope": self.external_consumer_arguments,
                           "counters": self.counters(), "scientific_acceptance": False}
            return self.result
        except core.UnsupportedPlan as error:
            self.unsupported = str(error)
            raise
        finally:
            # Release only our extra storage references. No application tensor
            # or autograd saved-tensor hook is replaced or modified.
            self.storage_refs.clear()

    def summary(self):
        return self.result or {"status": "UNSUPPORTED_PLAN" if self.unsupported else "INCOMPLETE",
                               "reason": self.unsupported, "policy": self.policy,
                               "counters": self.counters(), "scientific_acceptance": False}
