# Activation Checkpoint Integrity

This repository contains a bounded calibration harness for activation-checkpoint
recomputation. It records complete original and recomputed tensors under stable
semantic identities, compares their metadata and values exactly, and checks
whether controlled value drift reaches gradients without being caught by
PyTorch's default metadata check.

The starter phase is complete for one fixed tiny fixture on an Apple Silicon
CPU and one GTX 1080 Ti Pascal CUDA environment. It is a test bench for future
detector work, not a production detector.

## Reproduce locally

The project is pinned to Python 3.13, PyTorch 2.13.0, NumPy 2.5.2, and pytest
9.1.1 through `pyproject.toml` and `uv.lock`.

```bash
uv sync --locked
uv run pytest

uv run aci-starter run E0
uv run aci-starter run E1 --case all
uv run aci-starter run E1 --case rng --variant python
uv run aci-starter run E1 --case rng --variant numpy
uv run aci-starter run E2
```

E0 and E2 default to CPU `float64`; E1 uses `float32`. Each E1 arm executes in
a fresh process. E2 returns exit code 1 when the exact public-hook candidate is
insufficient, which is the observed and archived result. `uv` installs the
package into the project environment, so no external `PYTHONPATH` is required.

## What the experiments measure

- **E0 — clean oracle:** compares no-checkpoint and non-reentrant-checkpoint
  output, loss, gradients, and the tagged `h`, `g`, and `y` tensors against
  separately implemented forward and gradient formulas.
- **E1 — controlled faults:** evaluates counter drift, mutable-buffer drift,
  Python and NumPy randomness, changed precision policy, and toy FP8-style
  delayed scaling. Every family has correct, broken, and trigger-disabled
  controls; the broken case must first differ at same-metadata `h`, alter at
  least one gradient, and return to equality when its trigger is removed.
- **E2 — exact capture candidate:** compares a no-hook baseline with one exact
  composition of checkpoint `context_fn`, `TaggedSave`, and inner
  `saved_tensors_hooks`. Direct tags are the coverage ground truth.

## Current bounded results

On the Apple Silicon CPU, three E0 `float64` runs passed, and all 54 E1 child
arms passed across six scenarios, three modes, and three repetitions. The E2
baseline executed original/recompute phases 1/1, while the exact hook candidate
executed phases 1/0 and missed all recompute tags. The candidate was therefore
rejected.

Condor job **1553510.0** repeated the unchanged gates on one GTX 1080 Ti using
CUDA FP32 and PyTorch 2.12.1+cu126. Three E0 runs passed; their checkpoint
comparisons were bit-exact and their independently ordered formulas agreed
within the configured `1e-6` tolerance. All 54 E1 child arms passed, and E2
reproduced the same baseline 1/1 versus candidate 1/0 result. This is bounded
portability evidence for the complete CPU-to-Pascal environment change; it
does not isolate hardware, dtype, operating system, or PyTorch version as a
cause.

Historical aggregate manifests are immutable. The GPU E1/E2 aggregate records
contain a documented device-label defect that is corrected for future runs;
their child manifests are the authoritative historical device evidence. E2
rejects only the tested hook placement and composition, not every possible
public PyTorch observation method.

## Code and artifacts

- `fixture.py` defines fixed tensors, matrix math, semantic tags, `TaggedSave`,
  and independent derivatives.
- `cases.py` owns the correct and broken state controllers.
- `capture.py` labels phases, records and pairs tensors, observes hooks,
  compares exact values, and writes artifacts.
- `runner.py` isolates arms, records manifests, and evaluates experiment gates.
- `cli.py` exposes the `aci-starter` commands.
- `tests/test_starter.py` covers formulas, pairing, comparisons, fault controls,
  aggregate provenance, artifacts, and CLI exit codes.
- `scripts/` and `condor/` contain the pinned Pascal job bootstrap and submit
  path. `RESEARCH_LOG.md` is the chronological source for commands, raw IDs,
  environment facts, failures, and Analyst verdicts.

Each run writes ignored evidence under `artifacts/starter/<run_id>/`: a
manifest, event and comparison JSONL, complete tagged tensors, and a summary.
The retained GPU evidence is under
`/u/ayman27/activation-checkpoint-integrity/artifacts/starter`; scheduler files
for job 1553510.0 are under the adjacent `artifacts/condor/` directory.

## Explicit exclusions

The current results do not establish a naturally occurring pretraining failure,
arbitrary-program capture coverage, a production training integration, blocked
optimizer mutation, a GPU fingerprint implementation, detector overhead,
TorchTitan or compiler compatibility, distributed behavior, BF16 or native FP8
support, modern-GPU behavior, or acceptable production performance. Those
questions remain in the separate deferred production-shaped plan.
