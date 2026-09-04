"""State controllers for the five correct/broken E1 fault families."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


VALID_CASES = ("counter", "buffer", "rng", "precision", "fp8")
VALID_RNG_VARIANTS = ("python", "numpy")


class CaseController:
    """Provide original and phase-dependent matmul transforms for one isolated arm."""

    def __init__(self, *, mode: str, seed: int, device: torch.device) -> None:
        if mode not in {"correct", "broken", "trigger_off"}:
            raise ValueError(f"invalid arm mode: {mode}")
        self.mode = mode
        self.seed = seed
        self.device = device

    def original_transform(self, base: torch.Tensor) -> torch.Tensor:
        return base

    def original_transform_vjp(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Map an output gradient back through the original transform."""

        return grad_output

    def transform(
        self, left: torch.Tensor, right: torch.Tensor, phase: str
    ) -> torch.Tensor:
        return self.original_transform(left @ right)

    def after_forward(self) -> None:
        return None

    def initial_state(self) -> dict[str, object]:
        return {"mode": self.mode}


class CounterController(CaseController):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.counter = 0

    def transform(
        self, left: torch.Tensor, right: torch.Tensor, phase: str
    ) -> torch.Tensor:
        base = left @ right
        observed = self.counter
        self.counter += 1
        scale = 1.0
        if self.mode == "broken":
            scale = 1.0 if observed == 0 else 1.25
        return base * scale

    def initial_state(self) -> dict[str, object]:
        return {"mode": self.mode, "counter": 0, "original_scale": 1.0, "recompute_scale": 1.25}


class _BufferScale(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0, device=device))


class BufferController(CaseController):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.module = _BufferScale(self.device)

    def transform(
        self, left: torch.Tensor, right: torch.Tensor, phase: str
    ) -> torch.Tensor:
        base = left @ right
        # item() intentionally makes this non-differentiable and avoids an
        # unrelated autograd version-counter failure after buffer mutation.
        scale = float(self.module.scale.item())
        return base * scale

    def after_forward(self) -> None:
        if self.mode == "broken":
            self.module.scale.fill_(1.25)

    def initial_state(self) -> dict[str, object]:
        return {"mode": self.mode, "registered_buffer": 1.0, "advanced_buffer": 1.25}


class RngController(CaseController):
    def __init__(self, *, variant: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if variant not in VALID_RNG_VARIANTS:
            raise ValueError(f"invalid RNG variant: {variant}")
        self.variant = variant
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.fixed_draw = self._reference_draw()

    def _reference_draw(self) -> float:
        if self.variant == "python":
            return random.Random(self.seed).random()
        return float(np.random.RandomState(self.seed).random())

    def _global_draw(self) -> float:
        if self.variant == "python":
            return random.random()
        return float(np.random.random())

    @staticmethod
    def _scale(draw: float) -> float:
        return 1.0 + 0.25 * draw

    def original_transform(self, base: torch.Tensor) -> torch.Tensor:
        return base * self._scale(self.fixed_draw)

    def original_transform_vjp(self, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output * self._scale(self.fixed_draw)

    def transform(
        self, left: torch.Tensor, right: torch.Tensor, phase: str
    ) -> torch.Tensor:
        base = left @ right
        draw = self._global_draw() if self.mode == "broken" else self.fixed_draw
        return base * self._scale(draw)

    def initial_state(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "variant": self.variant,
            "seed": self.seed,
            "original_draw": self.fixed_draw,
        }


class PrecisionController(CaseController):
    def transform(
        self, left: torch.Tensor, right: torch.Tensor, phase: str
    ) -> torch.Tensor:
        fp32 = left.to(torch.float32) @ right.to(torch.float32)
        bf16 = left.to(torch.bfloat16) @ right.to(torch.bfloat16)
        if self.mode == "broken" and phase == "recompute":
            return bf16.to(torch.float32)
        return fp32

    def initial_state(self) -> dict[str, object]:
        return {"mode": self.mode, "original_policy": "fp32", "recompute_policy": "bf16"}


class Fp8Controller(CaseController):
    """Toy delayed scaling with an STE; this is not native FP8 behavior."""

    qmax = 7.0

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.history = 1.0

    def _quantize(self, base: torch.Tensor, history: float) -> torch.Tensor:
        scale = history / self.qmax
        quantized = torch.clamp(torch.round(base / scale), -self.qmax, self.qmax) * scale
        return base + (quantized - base).detach()

    def original_transform(self, base: torch.Tensor) -> torch.Tensor:
        return self._quantize(base, 1.0)

    def transform(
        self, left: torch.Tensor, right: torch.Tensor, phase: str
    ) -> torch.Tensor:
        base = left @ right
        output = self._quantize(base, self.history)
        if self.mode == "broken" and phase == "original":
            self.history = float(base.detach().abs().max().item())
        return output

    def initial_state(self) -> dict[str, object]:
        return {"mode": self.mode, "initial_history": 1.0, "qmax": self.qmax}


def make_controller(
    case: str,
    *,
    mode: str,
    seed: int,
    device: torch.device,
    variant: str | None = None,
) -> CaseController:
    if case == "counter":
        return CounterController(mode=mode, seed=seed, device=device)
    if case == "buffer":
        return BufferController(mode=mode, seed=seed, device=device)
    if case == "rng":
        return RngController(mode=mode, seed=seed, device=device, variant=variant or "python")
    if case == "precision":
        return PrecisionController(mode=mode, seed=seed, device=device)
    if case == "fp8":
        return Fp8Controller(mode=mode, seed=seed, device=device)
    raise ValueError(f"invalid case: {case}")
