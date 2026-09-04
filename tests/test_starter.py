from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ac_integrity.starter.capture import (
    CapturedTensor,
    TensorRecorder,
    compare_tensors,
    pair_records,
    tensor_metadata,
)
from ac_integrity.starter.fixture import TAGS, independent_reference, make_fixture
from ac_integrity.starter.runner import run_arm, run_e1, run_e2


def captured(value: torch.Tensor, side: str = "original") -> CapturedTensor:
    return CapturedTensor(
        tensor=value.detach().clone(),
        metadata=tensor_metadata(value),
        pair_id="E/test/checkpoint-0/x/0",
        event_id=f"run/E/test/checkpoint-0/x/0/{side}",
        phase=side,
        tag="x",
        occurrence=0,
    )


def test_independent_forward_and_gradient_formulas() -> None:
    tensors = make_fixture(device=torch.device("cpu"), dtype=torch.float64)
    reference = independent_reference(tensors, lambda value: value)
    h = tensors.x @ tensors.w1 + tensors.b
    g = h.square() + 0.5 * h
    y = g @ tensors.w2
    loss = (y * tensors.target).sum()
    loss.backward()

    assert torch.equal(y, reference["y"])
    assert torch.equal(loss, reference["loss"])
    for name, tensor in tensors.differentiable().items():
        assert torch.equal(tensor.grad, reference["gradients"][name])


def test_phase_labels_stable_pair_ids_and_rejection() -> None:
    recorder = TensorRecorder(run_id="run", experiment="E0", case="clean")
    for phase in ("original", "recompute"):
        with recorder.phase(phase):
            for tag in TAGS:
                recorder.record(tag, torch.ones(2, 2))
    pairs, issues = pair_records(recorder, TAGS)
    assert issues == {"missing_events": [], "duplicate_events": []}
    assert [pair[0].pair_id for pair in pairs] == [
        f"E0/clean/checkpoint-0/{tag}/0" for tag in TAGS
    ]
    assert pairs[0][0].event_id == "run/E0/clean/checkpoint-0/h/0/original"

    with recorder.phase("original"):
        recorder.record("h", torch.ones(2, 2))
    _, duplicate_issues = pair_records(recorder, TAGS)
    assert duplicate_issues["duplicate_events"] == [
        {"phase": "original", "tag": "h", "count": 2}
    ]

    missing = TensorRecorder(run_id="missing", experiment="E0", case="clean")
    with missing.phase("original"):
        missing.record("h", torch.ones(1))
    _, missing_issues = pair_records(missing, TAGS)
    assert {tuple(item.values()) for item in missing_issues["missing_events"]} == {
        ("recompute", "h"),
        ("original", "g"),
        ("recompute", "g"),
        ("original", "y"),
        ("recompute", "y"),
    }


@pytest.mark.parametrize(
    ("left", "right", "exact", "count", "index"),
    [
        (torch.tensor(2.0), torch.tensor(3.0), False, 1, []),
        (torch.empty(0), torch.empty(0), True, 0, None),
        (torch.tensor([float("nan")]), torch.tensor([float("nan")]), True, 0, None),
        (torch.tensor([float("inf")]), torch.tensor([float("inf")]), True, 0, None),
        (torch.tensor([float("inf")]), torch.tensor([-float("inf")]), False, 1, [0]),
        (torch.tensor([0.0]), torch.tensor([-0.0]), False, 1, [0]),
        (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[1.0, 2.0], [5.0, 4.0]]),
            False,
            1,
            [1, 0],
        ),
    ],
)
def test_exact_comparison_edge_semantics(
    left: torch.Tensor,
    right: torch.Tensor,
    exact: bool,
    count: int,
    index: list[int] | None,
) -> None:
    result = compare_tensors(captured(left), captured(right, "recompute"))
    assert result["exact_equal"] is exact
    assert result["different_elements"] == count
    assert result["first_differing_index"] == index


def test_e0_clean_checkpoint_and_artifact_reload(tmp_path: Path) -> None:
    outcome = run_arm(
        experiment="E0",
        case="counter",
        mode="correct",
        variant=None,
        device_name="cpu",
        dtype_name="float64",
        artifact_root=tmp_path,
        command=["aci-starter", "run", "E0"],
    )
    assert outcome.exit_code == 0
    assert outcome.summary["result"] == "PASS"
    assert outcome.summary["phase_counts"] == {"original": 1, "recompute": 1}
    assert outcome.summary["exact_matches"] == 3
    assert outcome.summary["no_checkpoint_matches_independent_formulas"] is True
    assert json.loads((outcome.run_dir / "summary.json").read_text()) == outcome.summary
    assert len((outcome.run_dir / "comparisons.jsonl").read_text().splitlines()) == 3
    for tag in TAGS:
        assert (outcome.run_dir / "tensors" / "counter" / f"{tag}.original.pt").is_file()
        assert (outcome.run_dir / "tensors" / "counter" / f"{tag}.recompute.pt").is_file()


@pytest.mark.parametrize(
    ("case", "variant"),
    [
        ("counter", None),
        ("buffer", None),
        ("rng", "python"),
        ("rng", "numpy"),
        ("precision", None),
        ("fp8", None),
    ],
)
def test_all_e1_arms(case: str, variant: str | None, tmp_path: Path) -> None:
    outcomes = {}
    for mode in ("correct", "broken", "trigger_off"):
        outcomes[mode] = run_arm(
            experiment="E1",
            case=case,
            mode=mode,
            variant=variant,
            device_name="cpu",
            dtype_name="float32",
            artifact_root=tmp_path,
            command=["isolated-test", case, mode],
        )
        assert outcomes[mode].exit_code == 0
        assert outcomes[mode].summary["result"] == "PASS"

    assert outcomes["correct"].summary["exact_matches"] == 3
    assert outcomes["broken"].summary["first_mismatch"]["tag"] == "h"
    assert outcomes["broken"].summary["first_mismatch"]["metadata_equal"] is True
    assert outcomes["broken"].summary["gradient_comparison"]["different_names"]
    assert outcomes["trigger_off"].summary["trigger_removal_restored_equality"] is True
    for tag in TAGS:
        correct = torch.load(
            outcomes["correct"].run_dir / "tensors" / case / f"{tag}.original.pt",
            weights_only=True,
        )
        broken = torch.load(
            outcomes["broken"].run_dir / "tensors" / case / f"{tag}.original.pt",
            weights_only=True,
        )
        assert torch.equal(correct, broken)


def test_e1_fresh_process_reset_and_repeat(tmp_path: Path) -> None:
    outcome = run_e1(
        selected_case="rng",
        variant="python",
        repetitions=2,
        device="cpu",
        artifact_root=tmp_path,
        command=["aci-starter", "run", "E1", "--case", "rng", "--variant", "python"],
    )
    assert outcome.exit_code == 0
    scenario = outcome.summary["scenarios"][0]
    assert scenario["broken_original_matches_correct"] is True
    assert scenario["repeat_scientific_and_tensor_contents_match"] is True
    assert scenario["trigger_removal_restored_equality"] is True


def test_e2_reports_complete_coverage_or_honest_insufficiency(tmp_path: Path) -> None:
    outcome = run_e2(
        device="cpu",
        dtype="float64",
        artifact_root=tmp_path,
        command=["aci-starter", "run", "E2"],
    )
    assert outcome.exit_code in {0, 1}
    assert outcome.summary["result"] in {"PASS", "PUBLIC_HOOKS_INSUFFICIENT"}
    candidate = json.loads(
        (Path(outcome.summary["candidate_run_dir"]) / "summary.json").read_text()
    )
    assert candidate["hook_coverage"] is not None
    assert candidate["result"] in {"PASS", "PUBLIC_HOOKS_INSUFFICIENT"}


def test_cli_exit_codes(tmp_path: Path) -> None:
    success = subprocess.run(
        [
            sys.executable,
            "-m",
            "ac_integrity.cli",
            "run",
            "E0",
            "--artifact-root",
            str(tmp_path / "success"),
        ],
        capture_output=True,
        text=True,
    )
    invalid = subprocess.run(
        [
            sys.executable,
            "-m",
            "ac_integrity.cli",
            "run",
            "E0",
            "--dtype",
            "float32",
            "--artifact-root",
            str(tmp_path / "invalid"),
        ],
        capture_output=True,
        text=True,
    )
    assert success.returncode == 0
    assert invalid.returncode == 2
