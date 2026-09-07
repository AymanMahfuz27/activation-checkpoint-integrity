# Production implementation and evidence status

The new implementation is local. All 39 tests pass; wheel build and dependency
lock validation pass. The full production-shaped
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
- Accept R0 and real-corpus 40M M0.1/M0.3 with complete causal reports. Current
  natural mechanism tests are miniature integration checks only.
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
  measurement. No runtime/memory overhead claim is established.
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

`condor/production-probe.submit` is prepared but has not been submitted or
scheduler-validated. The repository's AGENTS.md requires commit/push, remote
fast-forward synchronization and recorded revision before remote execution;
it explicitly forbids pushing without user authorization. Complete local review
and verification before that boundary. Shared filesystem free space must never
be substituted for the user's quota.
