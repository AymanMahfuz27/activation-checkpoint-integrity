"""Persist bounded fresh-process evidence. This script never accepts M0.3."""

import copy
import json
from pathlib import Path
import sys
import torch
from ac_integrity.config import load_config
from ac_integrity.train import run, pair, compare_state
from ac_integrity.state import write_json


def main():
    config = load_config("configs/smoke.toml")
    config.run_id = "verified-clean-smoke"
    root, result = run(config)
    assert result["status"] == "PASS", result
    snapshot = root / "states/pre_step_1.pt"
    config.capture.mode = "full"
    clean_pair, result = pair(config, snapshot)
    assert result["status"] == "EXACT_MATCH", result
    record = {"scope": "miniature text-fixture smoke; no 40M M0.3 acceptance", "clean": str(clean_pair), "natural": []}
    for seed in (17, 23, 47):
        c = load_config("configs/smoke.toml")
        c.run_id = f"verified-natural-smoke-{seed}"
        c.seed = seed
        c.adapter.name = "pytorch_84864"
        c.adapter.trigger = True
        c.training.stop_after = 1
        initial, _ = run(c)
        snapshot = initial / "states/pre_step_1.pt"
        arms = {}
        for name, capture, trigger, policy in [
            ("capture_off", "off", True, "observe"),
            ("observe", "full", True, "observe"),
            ("trigger_off", "full", False, "observe"),
            ("enforce", "full", True, "enforce"),
        ]:
            arm = copy.deepcopy(c)
            arm.capture.mode, arm.adapter.trigger, arm.capture.policy = capture, trigger, policy
            folder, result = pair(arm, snapshot)
            arms[name] = str(folder)
            if name == "trigger_off":
                assert result["status"] == "EXACT_MATCH", result
            else:
                assert result["status"] == "DIVERGENCE_OR_ABORT", result
        observed = torch.load(Path(arms["observe"]) / "candidate/outcome.pt", weights_only=False)
        disabled = torch.load(Path(arms["capture_off"]) / "candidate/outcome.pt", weights_only=False)
        assert not compare_state(observed, disabled)
        before = torch.load(snapshot, weights_only=False)
        aborted = torch.load(Path(arms["enforce"]) / "candidate/outcome.pt", weights_only=False)
        for name in ("model", "optimizer", "scheduler"):
            assert not compare_state(before[name], aborted[name])
        record["natural"].append({"seed": seed, "snapshot": str(snapshot), "arms": arms})
        print(json.dumps({"seed": seed, "status": "SMOKE_PASS"}), flush=True)
    write_json("artifacts/production-bootstrap/verified-smoke.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
