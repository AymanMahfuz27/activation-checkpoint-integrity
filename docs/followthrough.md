# GPU reproduction and upstream compatibility follow-through

This extension tests the four requested next steps. It does not claim the full
2,000-step research milestone, saved-tensor-complete coverage, native FP8,
TorchTitan integration or a low-overhead production detector.

## Causal proof

`python -m ac_integrity.validation suite --config configs/lm40m.toml
--output <retained-run-root> --budget-bytes <verified-budget>` starts with one
clean update, then freezes the model, populated Adam state, scheduler, RNG and
all four real-corpus microbatches for the next step. Each arm runs in a fresh
Python interpreter:

| Arm | Checkpointing | Known mode trigger | Recording | Update policy |
|---|---|---|---|---|
| Reference | Off | On | Off | Observe |
| Candidate and repeat | On | On | Off | Observe |
| Trigger-off reference/candidate | Off/On | Off | Off | Observe |
| Recorded reference/candidate | Off/On | On | Full | Observe |
| Enforced candidate | On | On | Full | Enforce |
| Clean enforced candidate | On | Off | Full | Enforce |

The trigger-off arms retain the random multiplication but disable only the
forward-only dispatch mode. They isolate the proposed cause. A separate upstream
screen removes the entire reproduction adapter.

The suite requires matching initial provenance and batches, equal forward losses,
differing gradients and model updates in the triggered pair, exact repeated and
trigger-off outcomes, and recording-on/off equality. It compares pre-step and
post-abort model, existing optimizer moments, scheduler and data cursor, and counts
actual calls to `optimizer.step()`. Clean enforcement must permit one update.

Full recording keeps every supported visible dense operator output in every
training phase. Compression and payload reuse are lossless. Hashes select reuse
candidates; full byte equality is required before two events share storage.
The conservative storage preflight does not assume compression will save space.

Reports distinguish the first mismatch in original execution order from the
first mismatch encountered during backward, which visits blocks in reverse order.
`CAPTURE_COMMIT` confirms a complete trace. `STEP_COMMIT` confirms an optimizer
update. A detected failure may have the former and must not have the latter.

## Upstream screen

`python -m ac_integrity.upstream matrix --config <config> --output <new-root>
--cells eager_fp32,sdpa_fp32,dropout_fp32,eager_fp16,compile_fp32 --steps 1 --record`
uses Transformers **5.16.1** `LlamaForCausalLM`, with no reproduction adapter and
no modifications to the upstream model source. The project provides the training
loop and optional recording integration at the native checkpoint callback.

Every cell uses independent reference, repeated-reference and checkpoint runs.
Eager cells also use a recorded checkpoint arm. Starting state and each update's
losses, gradients, model, optimizer, scaler, RNG and cursor are compared exactly.
Different precision/backend cells are not compared against each other as if they
were required to be bit-identical. Non-repeatable baselines, runtime failures,
recording interference and checkpoint differences receive distinct verdicts.

`aot_eager` tests compilation through Dynamo/AOTAutograd; it is not Inductor kernel
fusion or GPU performance evidence. Compiled internals are not covered by the
current eager recorder. Native BF16 is marked unsupported on GPUs that lack it.
The screen is a bounded compatibility result, not a claim of converged training
or a newly discovered naturally occurring bug.

## Scheduled GPU execution and artifact retention

`condor/followthrough.submit` prepares one core causal job and one upstream job.
The runner requires a clean synchronized commit and scheduled scratch, clones an
immutable copy of source, installs a hash-locked CUDA12.6 environment inside the
allocation, verifies an actual CUDA operation, and uses the same verified corpus.
The two jobs never edit the shared source checkout.

The UT home quota cannot hold full traces. Full evidence stays in scheduled
scratch while a lossless archive is transferred through an acknowledged 512MiB
inbox. Repeated whole files use exact-verified tar hard links. The source stays
intact until a receiver verifies the entire archive's SHA256. Only temporary
transfer chunks are retired after their own acknowledgement.

Start one collector per Condor process:

```bash
PYTHONPATH=src .venv/bin/python -m ac_integrity.archive_transfer receive \
  --inbox /u/ayman27/activation-checkpoint-integrity/artifacts/followthrough/JOB.PROCESS/transfer \
  --output artifacts/followthrough/gpu-JOB.PROCESS-archive \
  --max-bytes VERIFIED_LOCAL_BUDGET
```

A failed or interrupted transfer leaves the scheduled source waiting for up to
12 hours. Do not count a run as archived until `verified.json` exists locally and
its complete-archive hash matches. Artifacts include source/config/environment,
CUDA/driver/hardware, scheduler identity, immutable initial states, per-arm
outcomes, all recorded tensors, comparisons and the complete file manifest.
