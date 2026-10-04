# Production implementation and evidence status

The published implementation completed the bounded GPU follow-through reported
in [GPU results](followthrough-results.md). The current tree contains the repaired [BLAKE3-256 fingerprinter](fingerprinter.md).
The previous weighted-sum algorithm is invalidated by a deterministic sign-flip
collision. Its 0.88% timing result is historical evidence, not replacement
performance. The corrected repair passes local 117 and scheduled GPU 82 tests and all eight
40M workload gates; see the [reliability report](fingerprinter-reliability.md).
Observed 1.65% harness overhead includes evidence export and does not establish
stable sub-2% training-step cost. No general detector promotion is implied. Boundary-only observation still cannot meet the interior-only
coverage requirement.
The full production-shaped
milestone is **not complete**. The attached contract is preserved verbatim in
[production-plan.md](production-plan.md); the chronological evidence is in
[RESEARCH_LOG.md](../RESEARCH_LOG.md).

## Implemented interfaces

`aci data prepare`, `aci train`, `aci census`, `aci pair`, `aci replay`,
`aci validate-artifacts`, `aci report`, and `aci inspect` are implemented.
The historical `aci-starter` command remains available for archived work.

The research decoder has tied embeddings, RMSNorm, RoPE, grouped-query causal
attention and SwiGLU. Both planned architectures have the exact specified
parameter counts. Audit attention is explicit; SDPA is available with capture off.
The initial supported precision cell is deterministic FP32 on one rank.

Training includes accumulation, AdamW, clipping, warmup/stable/cosine scheduling,
immutable pre-step snapshots, resume snapshots, packed data cursors and fresh
interpreter reference/candidate arms. Snapshot checks include model/data/software
identity, precision/backend state and exact input/target batches.

Capture records dense tensor outputs in all training phases. Original/recompute
pairs require identical region/call/operator position and output structure.
Framework-generated checkpoint pack-hook detaches are retained as intentionally
unpaired bookkeeping; user detaches remain eligible. Unsupported subclasses
abort. Queue pressure blocks execution; no sampling/drop path exists. Checksums
and committed index hashes detect corruption and truncation.

Fingerprint mode reuses exact pair identities and reduces each observed tensor
to one standard 256-bit BLAKE3 digest in four 64-bit storage slots on its device. The production
default observes checkpointed block returns; exhaustive operator observation is
reserved for diagnostic replay. Fixed-capacity buffers fail closed. The normal
clean single-device path makes one post-backward host decision; failure
diagnostics are transferred only after an update has been rejected. Optional
numerical sketches are calibration evidence and never override an exact
mismatch.

## Run locally

```bash
uv sync --locked --extra data --no-editable
aci data prepare --config configs/smoke.toml
aci train --config configs/smoke.toml
aci pair --config configs/smoke.toml --snapshot <pre-step-snapshot>
```

Use the environment's `aci` executable or `uv run --no-sync aci`. If developing
source changes after a non-editable install, use
`PYTHONPATH=src .venv/bin/python -m ac_integrity.production_cli ...`.
A local process repeatedly marks editable `.pth` files hidden, and Python 3.13
then ignores them. Non-editable installation avoids relying on those files.

For full capture, set `capture.mode = "full"`, run `aci census` against the exact
snapshot/configuration, and put its result path in `capture.census_path`.
Set an actually verified quota/budget and free-space reserve. Changes to source,
snapshot or capture-relevant configuration invalidate the census. `aci pair`
performs each arm's census before its fresh-process full run.

For compact fingerprint enforcement, set `capture.mode = "fingerprint"` and
`capture.policy = "enforce"`. The default `fingerprint_scope =
"checkpoint_boundaries"` is the low-frequency normal path; use
`fingerprint_scope = "all_operators"` for diagnostic replay and operator-level
localization. No census or payload-storage budget is required. The configured
pair capacity remains a hard bound, and exceeding it aborts.

The smoke configuration intentionally uses fixture text and small dimensions.
It cannot pass real-corpus gates. `configs/lm40m.toml` and `lm125m.toml` contain
production dimensions and immutable source/tokenizer revisions. The packed
100M-token corpus is complete and its manifest hash is frozen in both configs.
Its versioned manifest is `reports/corpus-manifest.json`.

## Remaining contract gaps

- Execute the 2,000-step 40M stop/resume control, all-step 10-step full capture,
  and snapshots/audits at steps 1/1,000/2,000 on suitable allocated compute.
  The preliminary real-corpus three-step 40M checkpoint/no-checkpoint control
  passes exact equality for the complete final training state.
- Accept R0 and real-corpus 40M M0.1/M0.3 with complete causal reports. The controlled 40M GPU causal suite now passes ten gates; complete archive
  retention is pending. This does not imply acceptance of every longer-plan gate.
- Implement the full controlled mechanism suite after M0.3; no substitute
  synthetic result is allowed to satisfy that gate.
- Extend the passing nested-checkpoint and multi-output tests to deeper nesting
  and arbitrary custom operator pytree schemas.
- Add a reusable preallocated pinned-host buffer pool. Current CUDA copies are
  immediate, stream ordered and queued, but allocate their host buffers.
- Implement automatic crash-attempt recovery and all-rank STEP_COMMIT protocol.
  Current partial validation is read-only recovery inspection; invalid steps
  stay abandoned. Fresh replay uses a new artifact directory.
- Complete mismatch capsules with standalone gradient/update delta exports,
  comprehensive grouped tallies and proven-cause gate evaluation.
- Add long-control validation-loss logging and three-repeat pristine overhead
  measurement. The controlled 40M CUDA run establishes exact-mode correctness
  for one eager FP32 cell, not representative production overhead.
- Repeat the passing checkpoint-boundary timing and correctness contract on
  modern GPUs, mixed precision, distributed execution, and long training.
  Retain the fused all-operator path for diagnostic replay and run the
  perturbation ladder before broadening the supported execution cells.
- Implement and validate the pinned TorchTitan extension. Candidate source was
  inspected at `d263ca0a1b569ed198b9943b6e8c2117a61d8843`; this is **not** a
  compatible-stack pin or an implemented production integration.
- Replay additional upstream issues and native Transformer Engine FP8 on their
  faithful, compatible stacks. No native FP8 claim is made.

## GPU execution boundary

`condor/production-cu126.lock` is separate from historical starter requirements.
The official torch 2.13.0+cu126 wheel was resolved with its published SHA256.
The scheduler-only runner checks exact build, CUDA runtime, Pascal capability
and a real CUDA matrix operation before the 125M fit probe.

`condor/production-probe.submit` passed scheduler dry-run and job 1553517.0
completed successfully on eldar-44 with exit 0. The exact CUDA12.6/Pascal
kernel check and 125M generated-token memory-fit update passed. The repository's AGENTS.md requires commit/push, remote
fast-forward synchronization and recorded revision before remote execution;
it explicitly forbids pushing without user authorization. Complete local review
and verification before that boundary. Shared filesystem free space must never
be substituted for the user's quota.


## September 7 real-error reproduction

The real 40M model reproduces the transplanted #84864 silent error with capture
 disabled: identical forward losses, 73 differing named gradients, different
parameter updates, no default checkpoint exception, and exact equality restored
by removing only the dispatch-mode trigger. See
[seed17 evidence](../reports/lm40m-natural-seed17.json).

The complete candidate census measured 15,329 tensors and 23,977,519,895 payload
bytes. Required two-arm retention with snapshots/indexes and 25% reserve is
61,949,406,248 bytes. This exceeds available local storage, so exhaustive
capture was not started. M0.3 remains unaccepted until first-activation,
capture-effect and full-model enforcement evidence can be retained and reviewed.


## September 12 follow-through correction

The September 7 storage blocker above is historical. Scheduled scratch,
lossless compression/deduplication and acknowledged archive transfer enabled
full 40M GPU capture and all ten causal/localization/enforcement gates. Full
local archive recovery is in progress. See the current [GPU report](followthrough-results.md)
for the upstream clean cells, compiled numerical discrepancy, setup failure,
recorder cost and remaining limits.
