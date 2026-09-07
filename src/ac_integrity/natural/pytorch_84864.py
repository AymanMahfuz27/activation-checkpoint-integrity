"""LM transplant of https://github.com/pytorch/pytorch/issues/84864.

The upstream source is preserved separately. This adapter removes its debugger
and supports a device argument, but preserves the rand-to-constant dispatch mode
and its forward-only lifetime. It is a transplant, not the unchanged R0 script.
"""

from contextlib import nullcontext
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class NoRandomnessMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten.rand.default:
            return torch.ones(*args, **kwargs) / 2
        return func(*args, **kwargs)


def forward_context(config):
    if config.adapter.name == "pytorch_84864" and config.adapter.trigger:
        return NoRandomnessMode()
    return nullcontext()
