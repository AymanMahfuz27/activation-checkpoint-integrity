"""Storage reductions must preserve every byte and every observation."""

import json
import pytest
import torch
from ac_integrity.capture.writer import ShardWriter, read_tensor, read_payload


@pytest.mark.parametrize("encoding", ["raw", "zlib"])
def test_lossless_reuse_preserves_mutation_views_and_special_values(tmp_path, encoding):
    writer = ShardWriter(tmp_path, encoding=encoding, deduplicate=True, shard_bytes=32)
    values = [torch.tensor([float("nan"), -0.0, float("inf"), 1.0]),
              torch.arange(12).reshape(3, 4).T, torch.empty(0)]
    expected = []
    for tensor in values:
        for _ in range(2):
            event = {"event_id": str(len(expected)), "shape": list(tensor.shape),
                     "dtype": str(tensor.dtype)}
            writer.submit(event, tensor)
            expected.append(tensor.clone())
    tensor = torch.zeros(4)
    for _ in range(2):
        event = {"event_id": str(len(expected)), "shape": [4], "dtype": str(tensor.dtype)}
        writer.submit(event, tensor)
        expected.append(tensor.clone())
        tensor.add_(1)
    writer.close()
    events = [json.loads(line) for line in (tmp_path / "events/rank_0.jsonl").read_text().splitlines()]
    assert len(events) == len(expected) == 8
    assert writer.reused_payloads == 3
    for event, value in zip(events, expected, strict=True):
        actual = read_tensor(tmp_path, event)
        assert torch.equal(actual.contiguous().view(torch.uint8), value.contiguous().view(torch.uint8))
    assert events[0]["payload"] == events[1]["payload"]
    assert events[-1]["payload"] != events[-2]["payload"]


def test_compressed_payload_corruption_fails_closed(tmp_path):
    writer = ShardWriter(tmp_path, encoding="zlib", deduplicate=True)
    writer.submit({"event_id": "zeros"}, torch.zeros(1000))
    writer.close()
    event = json.loads((tmp_path / "events/rank_0.jsonl").read_text())
    assert writer.total_bytes < writer.logical_bytes
    with (tmp_path / event["payload"]["path"]).open("r+b") as target:
        target.write(b"BAD!")
    with pytest.raises(ValueError, match="checksum"):
        read_payload(tmp_path, event)


def test_hash_collision_does_not_authorize_reuse(tmp_path, monkeypatch):
    import ac_integrity.capture.writer as module

    class Collision:
        def hexdigest(self):
            return "collision"

    monkeypatch.setattr(module.hashlib, "sha256", lambda value: Collision())
    writer = ShardWriter(tmp_path, deduplicate=True)
    for i in [1, 2, 1]:
        writer.submit({"event_id": str(i)}, torch.tensor([i]))
    writer.close()
    records = [json.loads(line) for line in (tmp_path / "events/rank_0.jsonl").read_text().splitlines()]
    assert writer.reused_payloads == 1
    assert records[0]["payload"] == records[2]["payload"]
    assert records[0]["payload"] != records[1]["payload"]
