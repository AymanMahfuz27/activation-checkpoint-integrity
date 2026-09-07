"""Collective failure agreement before any optimizer-side mutation."""

import torch
import torch.distributed as dist


class IntegrityError(RuntimeError):
    pass


def failure_agreement(failed, device="cpu"):
    """Every rank must call this in identical order, including locally failed ranks."""
    if not dist.is_initialized():
        return bool(failed)
    reduction_device = device if dist.get_backend() == "nccl" else "cpu"
    flag = torch.tensor(int(bool(failed)), dtype=torch.int32, device=reduction_device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def optimizer_gate(failed, device="cpu"):
    if failure_agreement(failed, device):
        raise IntegrityError("All-rank abort before clipping, optimizer and scheduler mutation")
