"""Capture the outermost visible operator output without observing our own work."""

from contextvars import ContextVar
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from torch.utils._pytree import tree_flatten_with_path, keystr

_in_operator = ContextVar("aci_in_operator", default=False)


class CaptureMode(TorchDispatchMode):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if _in_operator.get():
            return func(*args, **kwargs)
        token = _in_operator.set(True)
        try:
            with _disable_current_modes():
                from torch.utils._pytree import tree_flatten
                for value in tree_flatten((args, kwargs))[0]:
                    if isinstance(value, torch.Tensor) and type(value) not in (torch.Tensor, torch.nn.Parameter):
                        self.runtime.unsupported.append({"operator": str(func), "type": str(type(value)), "position": "input"})
                        raise TypeError("Unsupported tensor subclass input; exhaustive capture aborted")
                inputs = self.runtime.describe_inputs((args, kwargs))
            output = func(*args, **kwargs)
            with _disable_current_modes():
                leaves, spec = tree_flatten_with_path(output)
                self.runtime.record(func, inputs, leaves, str(spec), args, kwargs)
            return output
        finally:
            _in_operator.reset(token)
