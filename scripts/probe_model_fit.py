"""One allocated GPU fit probe; generated tokens are explicitly non-research data."""

import argparse
import json
import torch
from ac_integrity.config import load_config
from ac_integrity.train import build, execute_step

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
config = load_config(args.config)
if config.device != "cuda" or not torch.cuda.is_available():
    raise SystemExit("GPU fit probe requires configured CUDA hardware")
model, optimizer, scheduler = build(config)
t = config.training
batch = {"input": torch.arange(t.microbatch_size * t.sequence_length).reshape(t.microbatch_size, t.sequence_length) % config.model.vocab_size}
batch["target"] = (batch["input"] + 1) % config.model.vocab_size
torch.cuda.reset_peak_memory_stats()
result = execute_step(model, optimizer, scheduler, [batch] * t.accumulation, config, collect_evidence=False)
torch.cuda.synchronize()
print(json.dumps({"status": "FIT_PROBE_PASS", "evidence_class": "generated-token-memory-probe",
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "peak_allocated": torch.cuda.max_memory_allocated(),
                  "peak_reserved": torch.cuda.max_memory_reserved(), "losses": result["losses"]}))
