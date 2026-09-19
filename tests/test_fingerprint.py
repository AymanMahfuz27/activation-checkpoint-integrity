"""Correctness and enforcement contracts for the compact fingerprinter."""

import copy
import json
from pathlib import Path

import pytest
import torch

from ac_integrity.capture.fingerprint import (
    FingerprintCapacityError,
    fingerprint_tensor,
)
from ac_integrity.capture.runtime import CaptureRuntime
from ac_integrity.config import load_config
from ac_integrity.data import prepare
from ac_integrity.train import compare_state, run
from ac_integrity.validation import arm_specifications, load_historical_oracle


def signatures(tensor, chunk_bytes=2 * 1024 * 1024):
    return fingerprint_tensor(
        tensor, chunk_bytes=chunk_bytes, include_sketch=False
    ).signatures.cpu()


def test_fingerprint_stage_excludes_expensive_full_capture(tmp_path):
    specifications = arm_specifications("fingerprint")
    names = [row[0] for row in specifications]
    modes = [row[3] for row in specifications]

    assert names == [
        "trigger_reference",
        "trigger_candidate",
        "trigger_off_candidate",
        "fingerprinted_candidate",
        "clean_fingerprinted",
    ]
    assert "full" not in modes
    assert "census" not in modes

    oracle_path = tmp_path / "oracle.json"
    oracle_path.write_text(json.dumps({
        "arms": {
            "trigger_candidate": {"seconds": 5.25},
            "recorded_candidate": {
                "seconds": 1081.77,
                "result": {"comparison": {
                    "eligible_pairs": 3232,
                    "counts": {"value_mismatch": 96},
                    "first_divergence": {
                        "pair_id": "pair-98",
                        "operator": "aten.rand.default",
                    },
                }},
            },
        },
    }))
    oracle = load_historical_oracle(oracle_path)
    assert oracle["eligible_pairs"] == 3232
    assert oracle["first_pair_id"] == "pair-98"
    assert oracle["full_capture_seconds"] == 1081.77


def test_signatures_are_exact_position_sensitive_and_chunk_independent():
    original = torch.tensor([1.0, -0.0, float("nan"), 4.0, 5.0])
    same = original.clone()
    permuted = original[[1, 0, 2, 4, 3]]
    changed_zero = original.clone()
    changed_zero[1] = 0.0

    assert torch.equal(signatures(original), signatures(same))
    assert torch.equal(signatures(original, 8), signatures(original, 4096))
    assert not torch.equal(signatures(original), signatures(permuted))
    assert not torch.equal(signatures(original), signatures(changed_zero))

    matrix = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    noncontiguous = matrix.T
    assert torch.equal(signatures(noncontiguous), signatures(noncontiguous.contiguous()))

    empty = signatures(torch.empty(0))
    assert empty.shape == (2,)
    assert torch.equal(empty, signatures(torch.empty(0)))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32,
                                    torch.float64, torch.int32, torch.int64])
def test_one_element_changes_are_detected_across_supported_dtypes(dtype):
    original = torch.arange(17).to(dtype)
    changed = original.clone()
    changed[8] += 1
    assert not torch.equal(signatures(original), signatures(changed))


def test_one_ulp_float_change_is_detected():
    original = torch.tensor([1.0], dtype=torch.float32)
    changed = torch.nextafter(original, torch.tensor([float("inf")]))
    assert not torch.equal(signatures(original), signatures(changed))


def _fingerprint_config():
    config = load_config(Path(__file__).parents[1] / "configs/smoke.toml")
    config.capture.mode = "fingerprint"
    config.capture.fingerprint_capacity = 32
    config.capture.fingerprint_chunk_bytes = 64
    config.capture.fingerprint_sketches = True
    return config


@pytest.mark.parametrize("different", [False, True])
def test_runtime_joins_exact_pair_identity_and_detects_values(tmp_path, different):
    config = _fingerprint_config()
    runtime = CaptureRuntime(tmp_path, config)
    factory = runtime.checkpoint_contexts("block")
    original, recompute = factory()
    left = torch.tensor([0.25, 0.5, 0.75])
    right = left.clone()
    if different:
        right[1] += 0.125

    with original:
        left.sin()
    with recompute:
        right.sin()

    comparison = runtime.compare()
    runtime.close()
    assert comparison["eligible_pairs"] == 1
    assert comparison["pair_coverage"] == 1.0
    assert comparison["decision_host_checks"] == 1
    assert comparison["failed"] is different
    if different:
        first = comparison["first_divergence"]
        assert first["status"] == "value_mismatch"
        assert first["metadata_equal"]
        assert first["signatures"]["original"] != first["signatures"]["recompute"]
        assert first["numerical_sketch"]["policy"] == "diagnostic_only"
    else:
        assert comparison["exact_pairs"] == 1
        assert comparison["diagnostic_transfers_after_failure"] == 0


def test_missing_recompute_and_capacity_overflow_fail_closed(tmp_path):
    config = _fingerprint_config()
    runtime = CaptureRuntime(tmp_path / "missing", config)
    original, _ = runtime.checkpoint_contexts("block")()
    with original:
        torch.ones(3).sin()
    comparison = runtime.compare()
    runtime.close()
    assert comparison["failed"]
    assert comparison["counts"]["missing_recompute"] == 2

    config.capture.fingerprint_capacity = 1
    runtime = CaptureRuntime(tmp_path / "capacity", config)
    original, _ = runtime.checkpoint_contexts("block")()
    with pytest.raises(FingerprintCapacityError):
        with original:
            value = torch.ones(3)
            value.sin()
            value.cos()
    runtime.close()
    assert runtime.summary()["fingerprint"]["failed"]
    assert runtime.summary()["fingerprint"]["counts"]["capacity_exceeded"] >= 1


def test_checkpointed_audit_with_no_pairs_fails_closed(tmp_path):
    config = _fingerprint_config()
    runtime = CaptureRuntime(tmp_path, config)
    comparison = runtime.compare()
    runtime.close()
    assert comparison["failed"]
    assert comparison["counts"]["no_eligible_pairs"] == 1


def test_metadata_mismatch_fails_even_when_payload_bytes_match(tmp_path):
    config = _fingerprint_config()
    runtime = CaptureRuntime(tmp_path, config)
    original, recompute = runtime.checkpoint_contexts("block")()
    value = torch.tensor([1.0, 2.0])
    with original:
        value.view(2)
    with recompute:
        value.view(1, 2)
    comparison = runtime.compare()
    runtime.close()
    assert comparison["failed"]
    assert comparison["first_divergence"]["status"] == "metadata_mismatch"
    assert comparison["first_divergence"]["default_metadata_would_detect"]


def test_nested_checkpoint_pairs_cleanly(tmp_path):
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
    from ac_integrity.capture.dispatch import CaptureMode

    config = _fingerprint_config()
    runtime = CaptureRuntime(tmp_path, config)

    def inner(x):
        return x.sin().square()

    def outer(x):
        return checkpoint(
            inner,
            x,
            use_reentrant=False,
            context_fn=runtime.checkpoint_contexts("inner"),
        ) + x.cos()

    x = torch.ones(3, requires_grad=True)
    with CaptureMode(runtime), set_checkpoint_early_stop(False):
        output = checkpoint(
            outer,
            x,
            use_reentrant=False,
            context_fn=runtime.checkpoint_contexts("outer"),
        )
        output.sum().backward()
    comparison = runtime.compare()
    runtime.close()
    assert comparison["eligible_pairs"] == 6
    assert comparison["pair_coverage"] == 1.0
    assert not comparison["failed"], comparison


def test_training_fingerprint_blocks_bad_update_and_preserves_clean_outcome(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs/smoke.toml")
    config.data.path = str(tmp_path / "data")
    config.artifact_root = str(tmp_path / "runs")
    config.adapter.name = "pytorch_84864"
    config.adapter.trigger = True
    prepare(config)

    setup_root, setup_summary = run(config)
    assert setup_summary["status"] == "PASS"
    snapshot_path = setup_root / "states/pre_step_1.pt"
    before = torch.load(snapshot_path, weights_only=False)

    guarded = copy.deepcopy(config)
    guarded.capture.mode = "fingerprint"
    guarded.capture.policy = "enforce"
    guarded.capture.audit_steps = [1]
    guarded.capture.fingerprint_capacity = 4096
    failed_root, failed_summary = run(guarded, snapshot_path, one_step=True)
    assert failed_summary["status"] == "ENFORCED_ABORT", failed_summary
    after = torch.load(failed_root / "outcome.pt", weights_only=False)
    for name in ("model", "optimizer", "scheduler"):
        assert compare_state(before[name], after[name]) == []
    failure = json.loads(
        (failed_root / "captures/step_1/failure.json").read_text()
    )
    assert failure["optimizer_updates"] == 0
    assert failure["comparison"]["failed"]
    assert failure["comparison"]["first_divergence"]["status"] == "value_mismatch"

    clean = copy.deepcopy(guarded)
    clean.adapter.trigger = False
    clean_root, clean_summary = run(clean, snapshot_path, one_step=True)
    assert clean_summary["status"] == "PASS", clean_summary
    comparison = clean_summary["steps"][0]["comparison"]
    assert not comparison["failed"]
    assert comparison["pair_coverage"] == 1.0
    assert comparison["decision_host_checks"] == 1

    unobserved = copy.deepcopy(clean)
    unobserved.capture.mode = "off"
    plain_root, plain_summary = run(unobserved, snapshot_path, one_step=True)
    assert plain_summary["status"] == "PASS"
    assert compare_state(
        torch.load(clean_root / "outcome.pt", weights_only=False),
        torch.load(plain_root / "outcome.pt", weights_only=False),
    ) == []
