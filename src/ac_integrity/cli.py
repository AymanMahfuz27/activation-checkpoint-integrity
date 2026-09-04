"""Command-line interface for the bounded E0-E2 starter experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ac_integrity.starter.cases import VALID_CASES, VALID_RNG_VARIANTS
from ac_integrity.starter.runner import run_arm, run_e1, run_e2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aci-starter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("experiment", choices=("E0", "E1", "E2"))
    run.add_argument("--case", choices=("all", *VALID_CASES), default="all")
    run.add_argument("--variant", choices=VALID_RNG_VARIANTS)
    run.add_argument("--repetitions", type=int, default=3)
    run.add_argument("--device", default="cpu")
    run.add_argument("--dtype", choices=("float64", "float32"))
    run.add_argument("--artifact-root", type=Path, default=Path("artifacts/starter"))

    arm = subparsers.add_parser("_arm", help=argparse.SUPPRESS)
    arm.add_argument("experiment", choices=("E0", "E1", "E2"))
    arm.add_argument("--case", choices=VALID_CASES, required=True)
    arm.add_argument("--mode", choices=("correct", "broken", "trigger_off"), required=True)
    arm.add_argument("--variant", choices=VALID_RNG_VARIANTS)
    arm.add_argument("--device", required=True)
    arm.add_argument("--dtype", choices=("float64", "float32"), required=True)
    arm.add_argument("--artifact-root", type=Path, required=True)
    arm.add_argument("--hooks", action="store_true")
    arm.add_argument("--tagged-save", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    command = ["aci-starter", *(argv if argv is not None else sys.argv[1:])]
    if arguments.command == "_arm":
        outcome = run_arm(
            experiment=arguments.experiment,
            case=arguments.case,
            mode=arguments.mode,
            variant=arguments.variant,
            device_name=arguments.device,
            dtype_name=arguments.dtype,
            artifact_root=arguments.artifact_root,
            hooks=arguments.hooks,
            tagged_save=arguments.tagged_save,
            command=command,
        )
    elif arguments.experiment == "E0":
        if arguments.case != "all" or arguments.variant is not None:
            return 2
        dtype = arguments.dtype or ("float64" if arguments.device == "cpu" else "float32")
        outcome = run_arm(
            experiment="E0",
            case="counter",
            mode="correct",
            variant=None,
            device_name=arguments.device,
            dtype_name=dtype,
            artifact_root=arguments.artifact_root,
            command=command,
        )
    elif arguments.experiment == "E1":
        if arguments.repetitions < 1 or arguments.dtype not in (None, "float32"):
            return 2
        if arguments.variant is not None and arguments.case not in {"all", "rng"}:
            return 2
        outcome = run_e1(
            selected_case=arguments.case,
            variant=arguments.variant,
            repetitions=arguments.repetitions,
            device=arguments.device,
            artifact_root=arguments.artifact_root,
            command=command,
        )
    else:
        if arguments.case != "all" or arguments.variant is not None:
            return 2
        dtype = arguments.dtype or ("float64" if arguments.device == "cpu" else "float32")
        outcome = run_e2(
            device=arguments.device,
            dtype=dtype,
            artifact_root=arguments.artifact_root,
            command=command,
        )
    payload = {
        "run_id": outcome.run_id,
        "run_dir": str(outcome.run_dir),
        "result": outcome.summary["result"],
    }
    for key in ("environment_error", "artifacts_persisted"):
        if key in outcome.summary:
            payload[key] = outcome.summary[key]
    print(json.dumps(payload))
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
