"""Move one declared native-result fault before the existing upper output record.

The original module-hook path remains untouched. Early mode replaces only its
installation, calls the native operator once, clones its result and invokes the
same historical mutation implementation while all bookkeeping modes are disabled.
"""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes


TARGET = "gpt_neox.layers.6"
IDENTITY = TARGET + ".mlp.dense_h_to_4h.output"
_BOOKKEEPING = ContextVar("ba_placement_bookkeeping", default=False)


def require(value, message):
    if not value:
        raise RuntimeError(message)


@dataclass(frozen=True)
class Site:
    operator: str = "aten.addmm.default"
    input_shape: tuple = (2048, 768)
    weight_shape: tuple = (768, 3072)
    bias_shape: tuple = (3072,)
    output_shape: tuple = (2048, 3072)
    numel: int = 6291456
    element: int = 6154933
    old_bits: int = 3198802410
    new_bits: int = 3198802411


@contextmanager
def intervention_class(completion, holder, contexts, early):
    """Expose the one actual instance and disable only its old hook for EARLY."""
    original = completion.Intervention

    class PlacementIntervention(original):
        def __init__(self, spec):
            super().__init__(spec)
            require(
                "instance" not in holder,
                "Multiple intervention instances in one placement arm",
            )
            holder["instance"] = self

        def install(self, model):
            if not early:
                return super().install(model)
            # EARLY has no module hook; the lower dispatch handler owns delivery.
            require(not self.handles, "Early intervention unexpectedly owns hooks")

        @contextmanager
        def context(self, region, role):
            with super().context(region, role):
                contexts[(region, role)] += 1
                yield

    completion.Intervention = PlacementIntervention
    try:
        yield
    finally:
        completion.Intervention = original


class EarlyResultFault(TorchDispatchMode):
    """A lower mode that mutates before upper CaptureMode.record sees the output.

    The shared capture._in_operator flag is deliberately not consulted: the upper
    mode sets it around natural func calls. A private guard and disabled modes
    isolate clone/site-selection/error-report bookkeeping instead.
    """

    def __init__(self, holder, site=None):
        super().__init__()
        self.holder = holder
        self.site = site or Site()
        self.candidates = []
        self.deliveries = []
        self.order = []
        self.seen = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        if _BOOKKEEPING.get():
            return output
        fault = self.holder.get("instance")
        if (
            fault is None
            or fault.region != TARGET
            or fault.role not in ("original", "recompute")
        ):
            return output
        if str(func) != self.site.operator or len(args) < 3:
            return output
        # Operator plus exact operand schema identifies this dense producer;
        # attention/other addmm calls have different fixed operands.
        if (tuple(args[0].shape), tuple(args[1].shape), tuple(args[2].shape)) != (
            self.site.bias_shape,
            self.site.input_shape,
            self.site.weight_shape,
        ):
            return output
        require(
            kwargs.get("alpha", 1) == 1 and kwargs.get("beta", 1) == 1,
            "Registered addmm scalar operands differ",
        )
        require(
            isinstance(output, torch.Tensor)
            and tuple(output.shape) == self.site.output_shape
            and output.dtype == torch.float32
            and output.layout == torch.strided
            and output.is_contiguous()
            and output.numel() == self.site.numel,
            "Registered native producer shape/dtype/layout differs",
        )
        role = fault.role
        ordinal = self.seen[role]
        self.seen[role] += 1
        self.candidates.append(
            {
                "operator": str(func),
                "region": fault.region,
                "role": role,
                "ordinal": ordinal,
                "shape": list(output.shape),
                "numel": output.numel(),
                "stride": list(output.stride()),
            }
        )
        if role != "recompute" or not fault.spec.active(fault.update):
            return output
        key = [fault.update, role]
        if any(event.get("event_key") == key for event in fault.events):
            return output
        require(
            ordinal == 0 and not self.deliveries,
            "Registered first producer occurrence was missed or repeated",
        )
        require(
            fault.spec.location == "recompute"
            and fault.spec.layer == 6
            and fault.spec.kind == "bit"
            and fault.spec.bit == 0
            and fault.spec.seed == 1729,
            "Early placement spec differs from registered fault",
        )
        selected = (
            int.from_bytes(
                hashlib.sha256(f"m4m5-site-v1:1729:{IDENTITY}".encode()).digest()[:8],
                "big",
            )
            % output.numel()
        )
        require(
            selected == self.site.element,
            "Registered logical bit coordinate differs; no site search",
        )
        self.order.append("native_func_return")
        token = _BOOKKEEPING.set(True)
        try:
            with _disable_current_modes():
                old_bits = (
                    int(output.detach().reshape(-1).view(torch.int32)[selected].item())
                    & 0xFFFFFFFF
                )
                require(
                    old_bits == self.site.old_bits,
                    "Registered old word does not reproduce; no adaptation",
                )
                before = {
                    "version": output._version,
                    "data_ptr": output.data_ptr(),
                    "storage_offset": output.storage_offset(),
                    "is_view": output._base is not None,
                }
                changed = output.clone()
                fault.mutate(changed, IDENTITY, event_key=key)
                event = fault.events[-1]
                require(
                    event["element"] == selected
                    and event["old_bits_u32"] == self.site.old_bits
                    and event["new_bits_u32"] == self.site.new_bits,
                    "Early event bytes differ",
                )
                after = {
                    "version": changed._version,
                    "data_ptr": changed.data_ptr(),
                    "storage_offset": changed.storage_offset(),
                    "is_view": changed._base is not None,
                }
                require(
                    before["data_ptr"] != after["data_ptr"],
                    "Early strategy must clone native result",
                )
                self.deliveries.append(
                    {
                        "placement": "native addmm return before upper output record",
                        "strategy": "clone_then_original_Intervention.mutate",
                        "native": before,
                        "replacement": after,
                        "event": event,
                    }
                )
        finally:
            _BOOKKEEPING.reset(token)
        self.order.append("lower_mutation")
        return changed

    def validate(self, enabled, expected_per_role=4):
        require(
            dict(self.seen)
            == {"original": expected_per_role, "recompute": expected_per_role},
            "Missing or ambiguous registered native producer census",
        )
        require(
            len(self.deliveries) == (1 if enabled else 0),
            "Early delivery count differs",
        )
        return {
            "candidates": self.candidates,
            "deliveries": self.deliveries,
            "strategy": "clone native result and mutate via unchanged historical helper",
            "shared_capture_recursion_flag_ignored": True,
        }
