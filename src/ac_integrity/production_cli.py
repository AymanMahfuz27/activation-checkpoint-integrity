"""Public aci commands for reproducible training and exhaustive forensic capture."""

import argparse
import json
from pathlib import Path
import sys
import tomllib
from ac_integrity.config import load_config
from ac_integrity import state


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aci")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data")
    prepare = data.add_subparsers(dest="action", required=True).add_parser("prepare")
    prepare.add_argument("--config", required=True)
    for command in ("train", "census", "pair", "_arm"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True)
        sub.add_argument("--snapshot", required=command in {"census", "pair", "_arm"})
        if command == "_arm":
            sub.add_argument("--destination", required=True)
    for command in ("replay", "validate-artifacts", "report", "inspect"):
        sub = commands.add_parser(command)
        sub.add_argument("artifact")
        if command == "inspect":
            sub.add_argument("--event-id", required=True)
        if command == "validate-artifacts":
            sub.add_argument("--recover-partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "data":
            from ac_integrity.data import prepare
            result = prepare(load_config(args.config))
        elif args.command in {"train", "_arm", "pair", "census"}:
            from ac_integrity.train import run, pair, census
            config = load_config(args.config)
            if args.command == "pair":
                root, result = pair(config, args.snapshot)
            elif args.command == "census":
                root, result = census(config, args.snapshot)
            else:
                root, result = run(config, args.snapshot, one_step=args.command == "_arm",
                                   destination=getattr(args, "destination", None))
            result = {"artifact": str(root.resolve()), **result}
        elif args.command == "replay":
            from ac_integrity.train import run
            spec = tomllib.loads(Path(args.artifact).read_text())
            if state.sha256(spec["config"]) != spec["config_sha256"]:
                raise ValueError("Replay config changed")
            if state.environment()["code_sha256"] != spec["code_sha256"]:
                raise ValueError("Replay code changed; restore the recorded source revision/tree")
            snapshot = spec["snapshot"] or None
            if snapshot and state.sha256(snapshot) != spec["snapshot_sha256"]:
                raise ValueError("Replay snapshot changed")
            root, result = run(load_config(spec["config"]), snapshot, one_step=spec["one_step"])
            result["artifact"] = str(root.resolve())
        elif args.command == "validate-artifacts":
            from ac_integrity.capture.compare import validate_artifacts
            root = Path(args.artifact)
            captures = sorted((root / "captures").glob("step_*"))
            if captures:
                checks = {str(p): validate_artifacts(p, args.recover_partial) for p in captures}
                result = {"valid": all(c["valid"] for c in checks.values()), "captures": checks}
            else:
                result = validate_artifacts(root, args.recover_partial)
        elif args.command == "report":
            from ac_integrity.capture.report import report
            result = {"report": report(args.artifact)}
        else:
            from ac_integrity.capture.report import inspect_event
            result = inspect_event(args.artifact, args.event_id)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result.get("status") in {"FAILED", "ENFORCED_ABORT"} or result.get("valid") is False else 0
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(error).__name__}: {error}"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
