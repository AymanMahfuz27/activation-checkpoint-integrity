"""Run bounded real-LM causal arms; retain all evidence and fail closed on quota.

First run --stage off. Then pass its snapshot to --stage census or --stage full.
Full mode captures the candidate while using the retained fresh-process
no-checkpoint reference. It never accepts a long-control or TorchTitan gate.
"""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import uuid
import torch
from ac_integrity.config import load_config, write_config
from ac_integrity.data import PackedCorpus
from ac_integrity.state import save_snapshot, write_json, sha256
from ac_integrity.train import build, collect_batches, snapshot_identity, pair, census, compare_state


def execute_arm(config, snapshot, root):
    config_path = root.with_suffix(".toml")
    write_config(config, config_path)
    command = [sys.executable, "-m", "ac_integrity.production_cli", "_arm",
               "--config", str(config_path), "--snapshot", str(snapshot),
               "--destination", str(root)]
    with root.with_suffix(".stdout").open("w") as out, root.with_suffix(".stderr").open("w") as err:
        completed = subprocess.run(command, stdout=out, stderr=err, timeout=1800)
    summary = json.loads((root / "summary.json").read_text())
    return {"command": command, "returncode": completed.returncode, "root": str(root), "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("off", "census", "full", "enforce", "clean-full", "trigger-off-full"), required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--budget-bytes", type=int, default=0)
    args = parser.parse_args()
    config = load_config("configs/lm40m.toml")
    config.run_id = f"real-natural-{args.seed}"
    config.seed = args.seed
    config.device = args.device
    config.expected_torch = "2.13.0" if args.device == "cpu" else "2.13.0+cu126"
    config.threads = 4
    config.training.stop_after = 1
    config.training.snapshot_steps = []
    config.training.checkpoint_every = 0
    config.capture.audit_steps = [1]
    config.capture.policy = "observe"
    config.adapter.name = "pytorch_84864"
    config.adapter.trigger = True
    root = Path(config.artifact_root) / f"real-natural-{args.stage}-{uuid.uuid4().hex[:12]}"
    root.mkdir(parents=True)
    if args.stage == "off":
        if args.snapshot:
            parser.error("off creates its own immutable snapshot")
        model, optimizer, scheduler = build(config)
        corpus = PackedCorpus(config)
        batches, _ = collect_batches(corpus, 0, config)
        snapshot = (root / "pre_step_1.pt").resolve()
        save_snapshot(snapshot, model, optimizer, scheduler, 0, 1, batches,
                      snapshot_identity(config, corpus.digest), trigger_state={"name": config.adapter.name, "trigger": True})
        del model, optimizer, scheduler, batches, corpus
        trigger_on, on = pair(config, snapshot)
        assert on["status"] == "DIVERGENCE_OR_ABORT", on["status"]
        assert all(a["returncode"] == 0 for a in on["arms"].values()), "An abort is not silent corruption"
        assert not on["differences"]["losses"], "Forward loss must match"
        assert on["differences"]["gradients"] and on["differences"]["model"], "Need gradient and update divergence"
        config.adapter.trigger = False
        trigger_off, off = pair(config, snapshot)
        assert off["status"] == "EXACT_MATCH", off["status"]
        record = {"status": "REAL_LM_CAPTURE_OFF_FAILURE_REPRODUCED", "seed": args.seed,
                  "snapshot": str(snapshot), "snapshot_sha256": sha256(snapshot),
                  "trigger_on": str(trigger_on.resolve()), "trigger_off": str(trigger_off.resolve()),
                  "first_gradient": on["differences"]["gradients"][0],
                  "differing_named_gradients": len(on["differences"]["gradients"]),
                  "differing_model_entries": len(on["differences"]["model"]),
                  "limitation": "Full activation capture and enforcement remain separate gates"}
    else:
        if not args.snapshot:
            parser.error("Pass the immutable snapshot returned by --stage off")
        snapshot = args.snapshot.resolve()
        config.capture.mode = "full"
        config.capture.max_bytes = args.budget_bytes or 10_000_000_000
        config.capture.verified_quota_bytes = args.budget_bytes
        config.capture.policy = "enforce" if args.stage == "enforce" else "observe"
        if args.stage in {"clean-full", "trigger-off-full"}:
            config.adapter.trigger = False
        census_path, estimate = census(config, snapshot)
        config.capture.census_path = str(census_path.resolve())
        if args.stage == "census":
            record = estimate
        else:
            record = execute_arm(config, snapshot, root / "candidate")
            record["snapshot"] = str(snapshot)
            record["census"] = estimate
    write_json(root / "evidence.json", record)
    print(json.dumps({"evidence": str((root / "evidence.json").resolve()), **record}, indent=2), flush=True)


if __name__ == "__main__":
    main()
