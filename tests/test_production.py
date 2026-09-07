"""Independent contracts for the decoder lifecycle and lossless capture oracle."""

import copy
import json
from pathlib import Path
import pytest
import torch
from ac_integrity.config import Config, load_config, write_config
from ac_integrity.data import prepare, PackedCorpus
from ac_integrity.model import Decoder
from ac_integrity.train import run, compare_state, census
from ac_integrity.capture.runtime import CaptureRuntime
from ac_integrity.capture.dispatch import CaptureMode
from ac_integrity.capture.compare import compare_tensors, events, validate_artifacts
from ac_integrity.capture.writer import read_tensor, ShardWriter


@pytest.fixture
def config(tmp_path):
    c = load_config(Path(__file__).parents[1] / "configs/smoke.toml")
    c.data.path = str(tmp_path / "data")
    c.artifact_root = str(tmp_path / "runs")
    prepare(c)
    return c


def test_exact_parameter_counts():
    for layers, width, heads, intermediate, expected in [(8, 512, 8, 1408, 39985664), (16, 768, 12, 2048, 125264640)]:
        c = Config()
        c.model.layers, c.model.width, c.model.heads, c.model.intermediate = layers, width, heads, intermediate
        with torch.device("meta"):
            model = Decoder(c)
        assert sum(p.numel() for p in model.parameters()) == expected
        assert model.output.weight is model.embedding.weight


def test_config_roundtrip_and_rejection(config, tmp_path):
    path = tmp_path / "config.toml"
    write_config(config, path)
    assert load_config(path) == config
    path.write_text(path.read_text() + "\nunknown = true\n")
    with pytest.raises(TypeError):
        load_config(path)
    config.capture.mode = "full"
    config.model.attention = "sdpa"
    with pytest.raises(ValueError, match="explicit"):
        config.validate()


def test_corpus_checksums_and_split(config):
    corpus = PackedCorpus(config)
    batch, cursor = corpus.batch(0, 2, 8)
    assert cursor == 1
    assert torch.equal(batch["input"][:, 1:], batch["target"][:, :-1])
    assert max(batch["sample_ids"]) < corpus.train_end
    shard = Path(config.data.path) / "tokens_00000.bin"
    with shard.open("r+b") as output:
        output.write(b"bad!")
    with pytest.raises(ValueError, match="Corrupt"):
        PackedCorpus(config)


def test_three_steps_checkpoint_and_resume(config):
    root, summary = run(config)
    assert summary["status"] == "PASS"
    checkpointed = torch.load(root / "outcome.pt", weights_only=False)
    plain = copy.deepcopy(config)
    plain.checkpoint.enabled = False
    other, summary = run(plain)
    assert summary["status"] == "PASS"
    assert compare_state(checkpointed, torch.load(other / "outcome.pt", weights_only=False)) == []
    resumed, summary = run(config, root / "states/resume_before_2.pt")
    assert summary["status"] == "PASS"
    assert compare_state(checkpointed, torch.load(resumed / "outcome.pt", weights_only=False)) == []


def test_bytes_nan_zero_and_noncontiguous():
    x = torch.tensor([float("nan"), 0.0, float("inf"), -float("inf")])
    same = compare_tensors(x, x.clone())
    assert same["raw_equal"] and not same["torch_equal"]
    y = x.clone()
    y[1] = -0.0
    changed = compare_tensors(x, y)
    assert not changed["raw_equal"] and changed["different_elements"] == 1
    assert changed["first_flat_index"] == 1
    assert compare_tensors(torch.arange(6).view(2, 3).T, torch.arange(6).view(2, 3).T)["raw_equal"]
    assert compare_tensors(torch.empty(0), torch.empty(0))["raw_equal"]


def test_immediate_copy_and_alias_metadata(config, tmp_path):
    config.capture.mode = "full"
    root = tmp_path / "capture"
    root.mkdir()
    runtime = CaptureRuntime(root, config)
    with CaptureMode(runtime):
        x = torch.ones(4)
        view = x[1:]
        x.add_(7)
    runtime.close()
    records = list(events(root))
    original = next(e for e in records if e["operator"] == "aten.ones.default")
    sliced = next(e for e in records if e["operator"] == "aten.slice.Tensor")
    assert torch.equal(read_tensor(root, original), torch.ones(4))
    assert torch.equal(read_tensor(root, sliced), torch.ones(3))
    assert sliced["storage_offset"] == 1
    assert original["storage"] == sliced["storage"]


def test_writer_failure_cleanup(config, tmp_path):
    writer = ShardWriter(tmp_path / "writer", max_bytes=1, queue_size=1)
    writer.submit({"event_id": "too-large"}, torch.ones(4))
    with pytest.raises(RuntimeError, match="writer failed"):
        writer.flush()
    with pytest.raises(RuntimeError):
        writer.close()
    assert not writer.thread.is_alive()


def test_clean_full_capture(config):
    root, _ = run(config)
    snapshot = root / "states/pre_step_1.pt"
    config.capture.mode = "full"
    census_path, estimate = census(config, snapshot)
    config.capture.census_path = str(census_path)
    assert estimate["per_step_tensor_count"] > 100
    captured, summary = run(config, snapshot, one_step=True)
    assert summary["status"] == "PASS", summary
    comparison = summary["steps"][0]["comparison"]
    assert comparison["pair_coverage"] == 1
    assert comparison["eligible_pairs"] > 0
    assert not comparison["failed"]
    assert validate_artifacts(captured / "captures/step_1")["valid"]
    plain = copy.deepcopy(config)
    plain.capture.mode = "off"
    other, _ = run(plain, snapshot, one_step=True)
    assert compare_state(torch.load(captured / "outcome.pt", weights_only=False),
                         torch.load(other / "outcome.pt", weights_only=False)) == []


def test_natural_mode_smoke_and_enforcement(config):
    """Integration smoke only: fixture tokens do not satisfy M0.3."""
    config.adapter.name = "pytorch_84864"
    config.adapter.trigger = True
    root, _ = run(config)
    snapshot = root / "states/pre_step_1.pt"
    config.capture.mode = "full"
    config.capture.policy = "observe"
    path, _ = census(config, snapshot)
    config.capture.census_path = str(path)
    observed, summary = run(config, snapshot, one_step=True)
    assert summary["status"] == "OBSERVED_MISMATCH", summary
    first = summary["steps"][0]["comparison"]["first_divergence"]
    assert first["status"] == "value_mismatch", first
    assert first["metadata_equal"] and not first["default_metadata_would_detect"]
    assert summary["steps"][0]["optimizer_updates"] == 1
    assert not summary["steps"][0]["default_check_raised"]
    reference = copy.deepcopy(config)
    reference.checkpoint.enabled = False
    reference.capture.mode = "off"
    clean_root, _ = run(reference, snapshot, one_step=True)
    clean = torch.load(clean_root / "outcome.pt", weights_only=False)
    wrong = torch.load(observed / "outcome.pt", weights_only=False)
    assert compare_state(clean["gradients"], wrong["gradients"])
    assert compare_state(clean["model"], wrong["model"])
    assert clean["losses"] == wrong["losses"]
    disabled = copy.deepcopy(config)
    disabled.capture.mode = "off"
    disabled_root, _ = run(disabled, snapshot, one_step=True)
    assert compare_state(wrong, torch.load(disabled_root / "outcome.pt", weights_only=False)) == []
    config.capture.policy = "enforce"
    enforced, summary = run(config, snapshot, one_step=True)
    assert summary["status"] == "ENFORCED_ABORT"
    after = torch.load(enforced / "outcome.pt", weights_only=False)
    before = torch.load(snapshot, weights_only=False)
    for name in ("model", "optimizer", "scheduler"):
        assert compare_state(before[name], after[name]) == []
    assert not list((enforced / "steps").glob("*/STEP_COMMIT"))
    config.adapter.trigger = False
    config.capture.mode = "off"
    off_candidate, _ = run(config, snapshot, one_step=True)
    config.checkpoint.enabled = False
    off_reference, _ = run(config, snapshot, one_step=True)
    assert compare_state(torch.load(off_candidate / "outcome.pt", weights_only=False),
                         torch.load(off_reference / "outcome.pt", weights_only=False)) == []


def test_pair_identity_and_missing_recompute(config, tmp_path):
    from ac_integrity.capture.compare import compare_events
    all_ids = []
    for attempt in range(2):
        root = tmp_path / f"identity-{attempt}"
        root.mkdir()
        config.capture.mode = "full"
        runtime = CaptureRuntime(root, config)
        factory = runtime.checkpoint_contexts("repeated")
        original, recompute = factory()
        with original:
            torch.ones(2).sin()
        with recompute:
            torch.ones(2).sin()
        runtime.close()
        result = compare_events(root)
        assert result["eligible_pairs"] == 2 and not result["failed"]
        all_ids.append([e["event_id"] for e in events(root)])
    assert all_ids[0] == all_ids[1]
    # Losing the recompute index rows cannot be relabeled a successful comparison.
    index = root / "events/rank_0.jsonl"
    records = [e for e in events(root) if e["checkpoint_role"] == "original"]
    index.write_text("".join(json.dumps(e) + "\n" for e in records))
    assert compare_events(root)["counts"]["missing_recompute"] == 2


def test_multi_output_and_corruption(config, tmp_path):
    config.capture.mode = "full"
    root = tmp_path / "multi"
    root.mkdir()
    runtime = CaptureRuntime(root, config)
    with CaptureMode(runtime):
        torch.max(torch.tensor([[1.0, 3.0], [4.0, 2.0]]), dim=1)
    runtime.close()
    outputs = [e for e in events(root) if e["operator"] == "aten.max.dim"]
    assert {e["output_path"] for e in outputs} == {"[0]", "[1]"}
    assert torch.equal(read_tensor(root, outputs[0]), torch.tensor([3.0, 4.0]))
    location = outputs[0]["payload"]
    with (root / location["path"]).open("r+b") as shard:
        shard.seek(location["offset"])
        shard.write(b"BAD!")
    with pytest.raises(ValueError, match="checksum"):
        read_tensor(root, outputs[0])
