"""Failure injection at the storage/coordination boundary, not synthetic LM evidence."""

from datetime import timedelta
import json
from pathlib import Path
import time
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ac_integrity.capture.writer import ShardWriter, read_payload
from ac_integrity.capture.runtime import CaptureRuntime
from ac_integrity.capture.dispatch import CaptureMode
from ac_integrity.config import load_config
from ac_integrity.distributed import optimizer_gate, IntegrityError


def test_backpressure_never_drops(tmp_path):
    writer = ShardWriter(tmp_path, queue_size=1, shard_bytes=16)
    actual_write = writer._write
    def slow(item):
        time.sleep(0.01)
        actual_write(item)
    writer._write = slow
    for i in range(12):
        writer.submit({"event_id": str(i)}, torch.tensor([i]))
    writer.close()
    records = [json.loads(line) for line in (tmp_path / "events/rank_0.jsonl").read_text().splitlines()]
    assert [r["event_id"] for r in records] == [str(i) for i in range(12)]
    assert writer.high_water == 1
    assert writer.blocked_seconds > 0.05
    assert not list(tmp_path.rglob("*.partial"))


def test_disk_error_retains_partial_and_propagates(tmp_path):
    writer = ShardWriter(tmp_path)
    def fail(item):
        writer._open_shard()
        writer._file.write(b"partial")
        raise OSError(28, "No space left on device")
    writer._write = fail
    writer.submit({"event_id": "failed"}, torch.ones(2))
    with pytest.raises(RuntimeError):
        writer.close()
    assert list(tmp_path.rglob("*.partial"))
    assert (tmp_path / "events/rank_0.jsonl").read_text() == ""
    assert not writer.thread.is_alive()


def test_partial_shard_read_is_explicit(tmp_path):
    writer = ShardWriter(tmp_path)
    writer.submit({"event_id": "retained"}, torch.ones(2))
    writer.close()
    event = json.loads((tmp_path / "events/rank_0.jsonl").read_text())
    path = tmp_path / event["payload"]["path"]
    path.rename(path.with_suffix(path.suffix + ".partial"))
    with pytest.raises(FileNotFoundError):
        read_payload(tmp_path, event)
    assert len(read_payload(tmp_path, event, allow_partial=True)) == 8


def test_subclass_aborts_and_mode_cleans_up(tmp_path):
    class Unsupported(torch.Tensor):
        pass
    c = load_config(Path(__file__).parents[1] / "configs/smoke.toml")
    c.capture.mode = "census"
    runtime = CaptureRuntime(tmp_path, c)
    value = torch.ones(3).as_subclass(Unsupported)
    with pytest.raises(TypeError):
        with CaptureMode(runtime):
            value + 1
    runtime.close()
    count = runtime.events
    torch.ones(2).sin()
    assert runtime.events == count
    assert len(runtime.unsupported) == 1


def rank_gate(rank, address, output):
    dist.init_process_group("gloo", init_method=address, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    try:
        optimizer_gate(rank == 1)
        optimizer.step()
        result = "updated"
    except IntegrityError:
        result = "aborted"
    Path(output, str(rank)).write_text(json.dumps({"status": result, "parameter": parameter.item(), "state_entries": len(optimizer.state)}))
    dist.destroy_process_group()


def test_two_rank_abort_before_any_optimizer_state(tmp_path):
    address = "file://" + str(tmp_path / "rendezvous")
    mp.spawn(rank_gate, args=(address, str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        assert json.loads((tmp_path / str(rank)).read_text()) == {"status": "aborted", "parameter": 1.0, "state_entries": 0}


def test_nested_checkpoint_replays_each_invocation(tmp_path):
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
    from ac_integrity.capture.compare import compare_events
    c = load_config(Path(__file__).parents[1] / "configs/smoke.toml")
    c.capture.mode = "full"
    runtime = CaptureRuntime(tmp_path, c)
    def inner(x):
        return x.sin().square()
    def outer(x):
        return checkpoint(inner, x, use_reentrant=False,
                          context_fn=runtime.checkpoint_contexts("inner")) + x.cos()
    x = torch.ones(3, requires_grad=True)
    with CaptureMode(runtime), set_checkpoint_early_stop(False):
        y = checkpoint(outer, x, use_reentrant=False,
                       context_fn=runtime.checkpoint_contexts("outer"))
        y.sum().backward()
    runtime.close()
    result = compare_events(tmp_path)
    assert not result["failed"], result
    assert result["pair_coverage"] == 1
    assert result["eligible_pairs"] == 6
