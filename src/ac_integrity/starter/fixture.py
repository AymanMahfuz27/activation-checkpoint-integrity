"""Fixed tensors, checkpointed math, semantic tags, and analytic references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from ac_integrity.starter.capture import TensorRecorder


TAGS = ("h", "g", "y")
TAG_TOKENS = {"h": 41001, "g": 41002, "y": 41003}
TOKEN_TAGS = {token: tag for tag, token in TAG_TOKENS.items()}


@dataclass
class FixtureTensors:
    """The differentiable inputs and fixed loss target for one fixture run."""

    x: torch.Tensor
    w1: torch.Tensor
    b: torch.Tensor
    w2: torch.Tensor
    target: torch.Tensor

    def differentiable(self) -> dict[str, torch.Tensor]:
        return {"x": self.x, "w1": self.w1, "b": self.b, "w2": self.w2}


def make_fixture(*, device: torch.device, dtype: torch.dtype) -> FixtureTensors:
    """Create fresh fixed tensors; every arm receives the same literal values."""

    def variable(values: list[list[float]] | list[float]) -> torch.Tensor:
        return torch.tensor(values, device=device, dtype=dtype, requires_grad=True)

    return FixtureTensors(
        x=variable([[1.25, -0.75], [0.5, 2.0]]),
        w1=variable([[0.8, -1.1], [1.4, 0.3]]),
        b=variable([0.2, -0.4]),
        w2=variable([[1.3, -0.6], [0.7, 1.1]]),
        target=torch.tensor(
            [[0.9, -1.2], [0.4, 0.8]], device=device, dtype=dtype
        ),
    )


class TaggedSave(torch.autograd.Function):
    """Make one semantic tensor and its integer token explicitly backward-relevant."""

    @staticmethod
    def forward(ctx: object, tensor: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(tensor, token)
        return tensor.clone()

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        saved_tensor, saved_token = ctx.saved_tensors
        # Reading both saved values is the E2 access contract. Multiplication by
        # zero leaves the fixture derivative unchanged while keeping access explicit.
        del saved_token
        return grad_output + saved_tensor * 0.0, None


Transform = Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor]


def fixture_forward(
    tensors: FixtureTensors,
    recorder: TensorRecorder,
    transform: Transform,
    *,
    tagged_save: bool = False,
) -> torch.Tensor:
    """Run the three-stage matrix fixture and record each semantic boundary."""

    phase = recorder.current_phase
    h = transform(tensors.x, tensors.w1, phase) + tensors.b
    h = _tag_if_requested(h, "h", tagged_save)
    recorder.record("h", h)

    g = h.square() + 0.5 * h
    g = _tag_if_requested(g, "g", tagged_save)
    recorder.record("g", g)

    y = g @ tensors.w2
    y = _tag_if_requested(y, "y", tagged_save)
    recorder.record("y", y)
    return y


def _tag_if_requested(tensor: torch.Tensor, tag: str, enabled: bool) -> torch.Tensor:
    if not enabled:
        return tensor
    token = torch.tensor(TAG_TOKENS[tag], device=tensor.device, dtype=torch.int64)
    return TaggedSave.apply(tensor, token)


def independent_reference(
    tensors: FixtureTensors,
    transform_original: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Calculate forward values and all gradients from closed-form derivatives."""

    with torch.no_grad():
        h = transform_original(tensors.x @ tensors.w1) + tensors.b
        g = h.square() + 0.5 * h
        y = g @ tensors.w2
        loss = (y * tensors.target).sum()

        d_y = tensors.target
        d_g = d_y @ tensors.w2.transpose(0, 1)
        d_h = d_g * (2.0 * h + 0.5)
        gradients = {
            "x": d_h @ tensors.w1.transpose(0, 1),
            "w1": tensors.x.transpose(0, 1) @ d_h,
            "b": d_h.sum(dim=0),
            "w2": g.transpose(0, 1) @ d_y,
        }
    return {"h": h, "g": g, "y": y, "loss": loss, "gradients": gradients}
