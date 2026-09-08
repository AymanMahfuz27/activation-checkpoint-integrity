## 1. What we are building and what counts as proof

Yes—this should be a real decoder-only pretraining pipeline with an exhaustive diagnostic harness around it. The project succeeds only if we can show the failure occurring during actual next-token training, identify the first incorrect recomputation, trace the damage into gradients and the optimizer update, and replay it from an identical saved state.

There are two required training stages:

1. **Project-owned research trainer:** approximately 40M parameters, designed for exhaustive inspection while retaining a real pretraining lifecycle.
2. **TorchTitan integration:** approximately 125M parameters, using a pinned upstream TorchTitan/PyTorch stack to demonstrate that the harness transfers to a production-oriented trainer. TorchTitan already supports Llama-family training, activation checkpointing, FSDP2, compilation, mixed precision, and Float8/MXFP8 workflows. TorchTitan

The central comparison is:

```
original execution of checkpointed block during forward
                            versus
recomputation of that same block during backward
```

We do not compare an arbitrary forward activation directly to a gradient. Instead:

- Original-forward and recomputation tensors are paired and compared exactly.
- Gradients, parameter changes, and optimizer state are compared between clean reference and checkpointed candidate runs started from the same pre-step snapshot.
- Forward, backward, gradient-clipping, and optimizer tensors are all captured, even when they are not eligible for an original/recompute equality pair.

“Capture every matrix” will mean every tensor output visible at PyTorch’s operator-dispatch boundary during the audited phases, including scalars, vectors, matrices, and higher-rank tensors. PyTorch’s `TorchDispatchMode` provides this operator-level interception point. Extending PyTorch

There is one unavoidable boundary: fused CUDA kernels, compiled kernels, and device libraries can contain internal temporary matrices that PyTorch never exposes as tensors. Therefore:

- Exhaustive audit runs use eager execution and an explicit, unfused attention implementation so Q, K, V, attention scores, masks, probabilities, context matrices, projections, and MLP intermediates are individually visible.
- Production controls use the fastest supported SDPA/fusion/compile settings with capture disabled.
- A measurement-effect experiment proves whether enabling the harness creates, removes, or changes the observed failure.
- Reports will say “every visible tensor at the supported PyTorch operator boundary,” never “every hidden register or fused-kernel temporary.”

The evidence gates are:

- **R0:** reproduce an unchanged, documented upstream activation-checkpoint failure in its pinned environment.
- **M0.1:** show that the clean LM produces complete original/recompute pairing with zero unexplained mismatches and identical reference gradients/updates.
- **M0.3:** transplant the documented natural mechanism into the real LM and demonstrate:
    1. same tensor metadata;
    2. different tensor values;
    3. no default metadata error;
    4. gradient divergence;
    5. a wrong optimizer update in an isolated evidence arm;
    6. exact restoration when only the natural trigger is removed.
- **M0.2:** exercise controlled fault families in the same LM to establish mechanism coverage. Controlled faults are test cases, not evidence that the problem naturally occurs.
- No detector or repair work begins until R0 and M0.3 pass. If the natural mechanism cannot be preserved in the LM, the milestone is recorded as blocked instead of replacing it with a synthetic success.

This distinction matters because PyTorch’s default checkpoint determinism check compares shape, dtype, and device rather than activation values, and its RNG preservation has documented limits. PyTorch checkpoint documentation

## 2. Trainer, capture harness, interfaces, and artifacts

### Training pipeline

The initial exhaustive model is a Llama-style 39,985,664-parameter decoder:

- 8 transformer blocks.
- Width 512.
- 8 query heads and 4 key/value heads.
- Head dimension 64.
- SwiGLU intermediate width 1,408.
- RMSNorm, RoPE, grouped-query causal attention, tied embeddings, and no linear biases.
- Vocabulary size 32,000.
- Sequence length 256.
- Two sequences per microbatch and four accumulation groups: 2,048 tokens per optimizer step.
- All blocks use non-reentrant activation checkpointing with `early_stop=false`, ensuring the recomputation completes the entire checkpointed block.
- AdamW with learning rate `3e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`, and gradient clipping at `1.0`.
- A 100-step warmup followed by stable training and a final cosine-decay interval.
- Deterministic FP32 is the first exact-equality cell. Dropout is zero except in the dedicated RNG experiments.

The Condor scale model is 125,264,640 parameters:

- 16 blocks.
- Width 768.
- 12 query heads and 4 key/value heads.
- SwiGLU width 2,048.
- Sequence length 512.
- One sequence per microbatch and four accumulation groups.
- Full activation checkpointing and FP32 on a GTX 1080 Ti.
- Memory fit must pass an actual scheduled GPU probe before the run is accepted.

The corpus will be a deterministically materialized, immutable 100M-token subset of FineWeb-Edu with a public 32K tokenizer. Dataset and tokenizer references are resolved to exact immutable revisions during bootstrap. The resulting artifact records:

- Source revision and record identifiers.
- Tokenizer revision and files.
- Preprocessing implementation version.
- Packed token order.
- Train/validation boundaries.
- SHA-256 hashes for every shard and manifest.
- Exact batch order used by every run.

The same packed corpus and tokenizer are exposed through the TorchTitan extension so the two trainers are not confounded by different data.

### New package and command interfaces

The repository receives one installable Python package:

```
src/ac_integrity/
  cli.py, config.py, train.py, model.py, data.py, state.py, distributed.py
  capture/{runtime,dispatch,writer,compare,report}.py
  faults/{base,rng,mutable_state,precision,compiler,fp8_style}.py
  natural/{base,pytorch_84864,pytorch_166926,transformer_engine_1190}.py
  torchtitan_ext/{config,trainer,activation_checkpoint,optimizer}.py
configs/
tests/
RESEARCH_LOG.md
```

Dependencies are managed with `uv`; each platform receives a resolved environment manifest. Dataset revisions, PyTorch builds, and TorchTitan commits may be discovered during bootstrap, but every actual experiment must use exact versions and wheel/commit hashes.

Public commands:

```bash
uv run aci data prepare --config CONFIG
uv run aci train --config CONFIG
uv run aci census --config CONFIG --snapshot SNAPSHOT
uv run aci pair --config CONFIG --snapshot SNAPSHOT
uv run aci replay ARTIFACT_DIR/replay.toml
uv run aci validate-artifacts ARTIFACT_DIR
uv run aci report ARTIFACT_DIR
uv run aci inspect ARTIFACT_DIR --event-id EVENT_ID
```

`aci pair` always launches the reference and candidate as separate fresh processes from the same immutable snapshot. Reusing one Python process is forbidden because library caches, counters, dispatch modes, or other hidden state could survive between arms.

The versioned TOML configuration contains:

- Run ID, evidence class, seeds, steps, audited steps, resume policy, and artifact limits.
- Complete model architecture.
- Dataset/tokenizer revisions and packed-corpus hashes.
- Batch, sequence, accumulation, optimizer, scheduler, and precision settings.
- Checkpointed blocks, reentrant setting, RNG policy, early-stop policy, and determinism setting.
- Capture mode: `off`, `census`, or `full`.
- Fault or natural-reproduction adapter.
- Distributed topology and timeouts.
- Expected software and hardware constraints.

### Event capture and pairing

A `TorchDispatchMode` surrounds each audited training step. Every checkpointed block receives a stable module name, and its checkpoint `context_fn` creates two contexts that share a checkpoint-call token:

- `checkpoint_role=original`
- `checkpoint_role=recompute`

Every captured output receives:

```
event_id =
  run/attempt/rank/step/microbatch/phase/region/call/op_ordinal/output_path

pair_id =
  attempt/rank/step/microbatch/region/call/op_ordinal/output_path
```

The pair ID omits original/recompute role so the two executions meet at the same identifier. Pairing also requires the exact operator overload and output schema to agree. Different operator sequences are recorded as structural mismatches; the harness will not heuristically realign them and accidentally hide a failure.

Each event records:

- Complete tensor payload.
- Shape, dtype, device, layout, stride, storage offset, element count, byte count, and `requires_grad`.
- Raw-payload SHA-256.
- Operator overload, output position, scalar arguments, sanitized keyword arguments, and deduplicated stack trace.
- Input provenance, parameter/buffer names, version counters, and view/alias relationships.
- Deepest known module path and complete module-parent set.
- Step, microbatch, rank, thread, stream, checkpoint region, and original/recompute role.
- Grad mode, autocast state, matmul precision, deterministic-algorithm setting, TF32/backend settings, and compilation state.
- Whether PyTorch’s default metadata check would detect the mismatch and whether an actual default-check run raised.

The harness disables its own dispatch interception while cloning, hashing, comparing, and writing data so it does not recursively capture itself.

### Lossless files and comparisons

Each rank writes append-only tensor shards rather than creating millions of tiny files:

```
artifacts/<experiment>/<run_id>/
  manifest.json
  config.resolved.toml
  environment.json
  data_manifest.json
  states/pre_step_<step>/
  events/rank_<rank>.jsonl
  comparisons/rank_<rank>.jsonl
  tensors/rank_<rank>/shard_<n>.bin
  mismatches/<mismatch_id>/
  summary.json
  report.html
  replay.toml
```

Tensor bytes are stored without dtype conversion. Original stride, layout, aliasing, and storage-offset information remains in metadata even when the payload is serialized in logical contiguous order.

On CUDA:

- The output is copied immediately into a preallocated pinned-host buffer on the producing stream, before a later in-place mutation can change it.
- A CUDA event marks copy completion.
- A bounded writer queue transfers completed buffers into append-only shards.
- If the queue fills, training blocks. Full mode never samples or silently drops tensors.
- The current step cannot reach the optimizer gate until all required payloads and comparisons are durable.

CPU and MPS use synchronous clones.

Each original/recompute comparison records:

- Metadata equality.
- Raw-byte equality.
- `torch.equal` separately, preserving the distinction around NaNs and signed zero.
- Number of differing elements.
- First differing flat and multidimensional index.
- Maximum and mean absolute error.
- Maximum relative error and relative L1/L2 error.
- ULP distance for supported floating-point types.
- NaN, positive/negative infinity, and signed-zero counts.
- Both complete tensor locations and hashes.

A mismatch capsule exports the original tensor, recomputed tensor, local difference data, operator trace, state difference, trigger manifest, relevant gradients, and optimizer deltas in directly inspectable files.

The summary tallies:

- All tensors captured by phase.
- Original/recompute-eligible tensors.
- Exact matches.
- Same-metadata value mismatches.
- Metadata mismatches.
- Structural sequence mismatches.
- Intentionally unpaired tensors.
- Missing originals or recomputes.
- Unsupported tensor types.
- Bytes written, queue high-water marks, and time blocked on capture.

Missing or unsupported data is never counted as a match.

### State, replay, and optimizer safety

Before every paired or audited step, save atomically:

- Model parameters and buffers.
- Optimizer, scheduler, scaler, and accumulated gradients.
- Python, NumPy, Torch CPU, MPS, every CUDA-device RNG, and registered custom-generator states.
- Exact token batch, data cursor, and sample IDs.
- Precision, autocast, backend, compiler, and deterministic settings.
- Fault/natural-trigger state.
- Distributed topology and rank-local state.
- Code, environment, model, data, and tokenizer hashes.

The harness has two mismatch policies:

- `observe`: finish the sacrificial step and permit exactly one incorrect optimizer update so it can be compared with the clean update. Save the pre/post parameters and optimizer state, then terminate.
- `enforce`: finish backward and all comparisons, combine rank-local failure flags, and stop every rank before gradient clipping, optimizer, or scheduler mutation.

In distributed runs, every rank writes its own artifacts and participates in an all-rank pre-optimizer failure reduction. No rank may update while another rank has detected a mismatch, incomplete capture, disk error, or writer failure.

Shards remain `.partial` until committed. A committed index line is written only after its payload and checksum are durable. A step becomes valid only after every rank writes `STEP_COMMIT`. An interrupted step is marked abandoned and replayed from its saved pre-step state using a new attempt ID.

Before full capture, `aci census` measures event counts and projected bytes without writing payloads. The estimate includes both arms, all microbatches, ranks, state snapshots, indexes, and a 25% safety margin. The harness refuses to start if artifact limits, free-space reserves, or the verified user quota would be exceeded; it never silently reduces coverage.

## 3. How we will make the real and controlled failures occur

The mechanisms are deliberately separated because they do not all represent the same failure.

### Natural evidence

The first natural reproduction is PyTorch issue #84864. It reports a silent checkpoint error in which a `TorchDispatchMode` active during the original forward is not automatically restored during recomputation.

The sequence is:

1. Run the issue’s original reproducer unchanged on the current pinned PyTorch environment.
2. If it no longer reproduces, run it unchanged in its issue-era pinned environment.
3. Record whether it is current, historical, fixed, or not reproduced. An invented substitute cannot pass R0.
4. Transplant the same legitimate dispatch-mode/checkpoint interaction into a stochastic operation inside the real transformer block.
5. Compare no-checkpoint and checkpoint arms under the same mode from identical state.
6. Prove that the first activation mismatch has equal shape/dtype/device, that the default metadata check does not raise, and that downstream gradients and the update differ.
7. Remove only the mode/checkpoint interaction and prove exact equality returns.
8. Reproduce the symptom both with capture disabled and with full capture enabled.

Additional natural cases broaden the evidence:

- PyTorch #166926: compilation/cache state selects different recomputation behavior. The reported manifestation is structural and caught, so it cannot by itself satisfy the silent-value M0.3 gate.
- PyTorch #159359: FSDP2 casting behavior differs during recomputation. The reported dtype mismatch is metadata-visible.
- PyTorch #193338: selective checkpointing, shared weights, and autocast-cache state produce an invocation mismatch.
- Transformer Engine #1190: FP8 scaling state produces a later gradient discrepancy despite matching forward outputs. Native replay waits for appropriate modern hardware because Transformer Engine’s FP8 support requires newer NVIDIA architectures. Transformer Engine documentation

Each case remains labeled according to what it actually demonstrates: silent value corruption, metadata mismatch, structural mismatch, false-positive error, or delayed state/gradient divergence.

### Controlled mechanism suite

After the clean oracle passes, run one controlled mechanism per fresh-process experiment:

1. **Normal dropout control:** standard Torch dropout with checkpoint RNG preservation enabled; this must match.
2. **Unpreserved Torch RNG:** dropout with RNG preservation disabled.
3. **Custom Torch generator:** randomness from a generator checkpointing does not restore.
4. **Python RNG:** `random` used inside the checkpointed block.
5. **NumPy RNG:** NumPy randomness used inside the checkpointed block.
6. **Call counter:** a counter changes the recomputation branch or value.
7. **Mutable registered buffer:** a buffer is updated during original forward and read differently during recomputation.
8. **Precision-context asymmetry:** an internal recomputation matmul uses a different precision but casts back so external metadata remains identical.
9. **FP8-style state emulator:** mutable amax/scaling history changes recomputation. This is explicitly an emulator, not native FP8 evidence.
10. **Compiler/cache selection:** graph warmth or cache ordering changes which compiled path recomputation selects.
11. **Selective-checkpoint/autocast cache:** sibling checkpoint regions share a weight whose cast/cache behavior changes invocation order.
12. **Aliasing/in-place mutation:** a view or cached tensor is mutated between original use and recomputation.
13. **Nondeterministic kernel/collective exploration:** evaluate separately on suitable hardware, without assuming that ordinary numerical nondeterminism is checkpoint-specific.

Every experiment declares its expected first mismatch surface:

- Value-only.
- Metadata.
- Operator/event structure.
- Hidden state or side-effect.

For a controlled experiment to pass:

- The expected first divergence must occur.
- The changed tensor must influence backward.
- At least one named parameter gradient and optimizer delta must differ.
- Removing only the injector must restore equality.
- Capture-off and capture-on runs must agree on whether the symptom exists.
- The report must state whether default checkpoint checking caught it.

The report may label a cause `proven` only when it contains the first divergent operation, the relevant state difference, and a trigger-off replay restoring equality. Otherwise it must say `consistent_with` or `unknown`.

## 4. Implementation and experiment order

### Phase A — Freeze truth and reproduce a real upstream case

- Keep E0–E2 absent; they are not part of the program.
- Make `RESEARCH_LOG.md` the versioned canonical project memory.
- Record current repository state, this contract, evidence gates, model/data decisions, hardware limits, and outstanding blockers.
- Every later command, failure, success, job ID, artifact hash, decision, and action item receives a dated append-only record with `PLANNED`, `RUNNING`, `PASS`, `FAIL`, or `BLOCKED`.
- Implement the exact #84864 upstream reproducer first and run it before building synthetic injectors.
- Preserve the unmodified upstream code, environment lock, trigger-on output, trigger-off output, and current-versus-historical result.

### Phase B — Build the clean production-shaped trainer

- Implement immutable data preparation, explicit audit attention, the 40M model, loss, backward, accumulation, clipping, optimizer, scheduler, checkpoint/resume, and structured run manifests.
- Run three clean steps without checkpointing and three with checkpointing.
- Verify loss, gradient, parameter, optimizer, RNG, and data-cursor behavior before adding the exhaustive observer.
- Run a 2,000-step capture-off 40M control with a deliberate stop/resume cycle.
- Save immutable snapshots before steps 1, 1,000, and 2,000.

### Phase C — Build and validate exhaustive capture

- Implement event identity, checkpoint contexts, operator interception, state snapshots, append-only shards, bounded writer queues, exact comparison, crash recovery, reports, and replay.
- Run a miniature two-layer census and full-capture smoke test on the Mac.
- Run a complete bounded 10-step 40M training job with `audit_steps=all`, capturing every visible tensor in every step.
- From the long control’s steps 1, 1,000, and 2,000, run paired no-checkpoint and full-checkpoint audits.
- M0.1 requires 100% eligible pair coverage, zero unexplained mismatches, and exact named-gradient/update equality.
- Run the same snapshot once with capture disabled and once with capture enabled; the resulting loss, gradients, optimizer state, and post-step parameters must agree exactly in deterministic FP32.

The harness supports `audit_steps=all` for longer jobs, but such a run proceeds only if census and quota checks prove the complete output can be retained. Long production controls are never silently presented as exhaustively captured when only selected snapshots were audited.

### Phase D — Natural LM reproduction, then controlled coverage

- Transplant #84864’s actual mechanism into the 40M LM.
- Execute trigger-on, trigger-off, capture-off, full-capture, observe, and enforce arms from the same snapshot.
- Repeat the successful natural case at seeds 17, 23, and 47.
- Only after M0.3 passes, execute the controlled mechanism suite from the step-1,000 snapshot.
- Generate one causal report per mechanism and one aggregate coverage report.
- Replay #166926 and the other supporting upstream cases on compatible hardware without changing their triggers.

### Phase E — Condor scale and TorchTitan transfer

- Pin the latest PyTorch 2.13 CUDA 12.6 build that still includes Pascal support, then record its exact version and wheel hash. CUDA 13 builds cannot be used for the GTX 1080/1080 Ti path; the compatibility boundary is documented in the PyTorch release matrix.
- Run the 125M Condor model for 500 capture-off steps, with snapshots before steps 1 and 500.
- Perform exhaustive clean and natural-trigger audits at those snapshots.
- Pin an exact TorchTitan commit and its compatible PyTorch revision.
- Integrate through project-owned extension classes and a custom checkpoint function; do not patch TorchTitan core.
- TorchTitan v1 audited support is eager full checkpointing, pipeline parallelism 1, tensor parallelism 1, and either one GPU or FSDP2 data parallelism.
- On A100/H100 hardware, run:
    - 125M BF16 production control.
    - Eager full-capture clean audits.
    - Natural trigger audit.
    - Two-rank FSDP2 pre-optimizer abort test.
- On H100-class hardware, faithfully replay the native Transformer Engine FP8 case before making any native FP8 claim.

## 5. Tests, reports, and final acceptance

Required automated tests include:

- Stable IDs across reruns.
- Nested checkpoint regions and repeated module calls.
- Multi-output and nested-pytree operators.
- Scalars, vectors, matrices, noncontiguous tensors, views, aliases, and in-place mutation.
- NaN, infinity, signed zero, dtype differences, stride differences, and structural trace changes.
- Immediate-copy correctness before later mutation.
- Writer backpressure with no drops.
- Checksum corruption and partial-shard recovery.
- Disk-full and writer-failure behavior.
- Fresh-process snapshot replay.
- Missing original/recompute detection.
- Unsupported tensor-subclass abort.
- Optimizer non-mutation in enforce mode.
- One-update evidence in observe mode.
- Multi-rank mismatch propagation and all-rank abort.
- Capture-disabled equivalence and cleanup of modes, threads, and buffers.
- Checkpoint/resume equivalence in the long training control.

Every report must show:

- Exact code, environment, hardware, data, tokenizer, config, snapshot, seed, rank, and command hashes.
- Total captured tensors and complete/missing/unsupported counts.
- Match/mismatch tally by step, phase, module, operator, dtype, shape, and rank.
- Timeline to the first divergence.
- Exact metadata and difference metrics.
- A small view around the first differing element and links to both complete tensor payloads.
- Relevant state change explaining the suspected mechanism.
- First differing named gradient.
- Clean versus incorrect optimizer delta and resulting parameter.
- Whether PyTorch’s default mechanism detected the problem.
- Trigger-off result and replay command.
- Capture time, bytes, queue pressure, and blocked time.
- Explicit limitations around fused kernels, compilation, unsupported tensor types, and hardware.

The milestone is accepted only when:

- The full pipeline trains, checkpoints, resumes, and produces reproducible snapshots.
- The bounded all-step audit captures every supported visible tensor with no missing records.
- M0.1 achieves complete original/recompute pairing and zero unexplained clean mismatches.
- A documented upstream bug is faithfully reproduced.
- M0.3 shows a naturally triggered same-metadata value mismatch in real next-token training, followed by gradient divergence and an incorrect update.
- The trigger-off arm restores equality.
- Capture-off and capture-on runs preserve the same failure status.
- Enforce mode blocks every rank before optimizer mutation.
- Controlled faults behave according to their preregistered mechanism class.
- All successes, failures, blocks, commands, artifacts, and next actions are recorded in `RESEARCH_LOG.md`.
- The TorchTitan integration passes on its pinned stack before the project claims production-trainer applicability.

Assumptions and boundaries:

- The Mac’s limited free storage makes it a unit-test, smoke-test, and census machine unless a particular audit passes preflight.
- Condor’s shared free space is not treated as the user’s quota; quota must be measured before full capture.
- GTX 1080/1080 Ti results support FP32/FP16 correctness only, not BF16, TF32, FP8, or modern Tensor Core claims.
- A 40M or 125M experiment validates the failure mechanism and end-to-end training lifecycle; it does not establish billion-parameter generalization or meaningful language-model convergence.
- Full capture is an offline forensic oracle with no speed target. Production capture-off integration must remain within 2% median post-warmup step time and 1% peak-memory difference from the corresponding pristine control over three repetitions.
- Generated datasets and tensor artifacts remain outside Git; manifests, configurations, reports, experiment records, and replay instructions are versioned.
