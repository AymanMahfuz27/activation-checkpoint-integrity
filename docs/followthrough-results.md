# GPU follow-through results

The four requested steps now have implementations and measured GPU runs. The
controlled failure is reproduced, localized, and blocked before the optimizer.
The initial upstream Llama screen has four exact eager results and one small
compiled numerical discrepancy. This does not establish a new hidden-state bug.
Full archive recovery is in progress; the raw remote evidence remains retained.

## Controlled 40M model

Condor **1553917.0**, commit `9b922d74168728081d2b8c79c3f9607904d4514d`,
GTX 1080 Ti, PyTorch 2.13.0+cu126, CUDA 12.6, seed 17. Eight decoder blocks,
width 512, sequence length 256, microbatch size 2, four accumulated microbatches:
2,048 tokens in the audited update. The packed source is FineWeb-Edu, with the
source revision, tokenizer revision and corpus checksums frozen in the config.
This is one audited second update after one clean warm-up update, not a
2,000-step training result. Every comparison arm starts from the same immutable
model, populated Adam state, scheduler, input batches, RNG and cursor snapshot.

| Check | Measured outcome |
|---|---|
| Triggered checkpoint versus no checkpoint | Identical four forward losses; 73 gradient tensors differ; model updates differ; no default checkpoint exception. |
| Repeat the triggered checkpoint arm | Complete recorded outcome agrees exactly. |
| Remove only the dispatch-mode trigger | Checkpoint and reference outcomes agree exactly. |
| Turn full recording on | Reference, failing candidate and clean candidate outcomes remain exactly equal to their recording-off controls. |
| Pair original and recomputed operator outputs | 3,232 eligible pairs, all paired; 96 contain different values in the failing arm. |
| Locate the first difference in forward execution order | Microbatch 0, block 0, operator 98, `aten.rand.default`. |
| Locate the first difference encountered during backward | Microbatch 0, block 7, the same operator; backward visits later blocks first. |
| Enforce the guard | Zero optimizer calls; model, populated optimizer state, scheduler and data cursor remain unchanged. |
| Enforce on the clean control | One optimizer call; complete outcome agrees with the clean recording-off run. |

All ten suite gates passed. At the first differing operation, the original values
begin `[0.5, 0.5, 0.5, 0.5]`; recomputation begins
`[0.5395032167, 0.4814732969, 0.1513968855, 0.8988719583]`.
Both tensors have the same shape `[2, 256, 512]`, FP32 dtype and CUDA device.
The whole tensor has 262,144 differing elements. Thus a metadata-only check does
not expose this value discrepancy.

The mode is still a deliberate transplant of the documented #84864 mechanism.
This GPU run validates detection of that controlled failure; it does not show
that an untouched training job naturally installed or lost that mode.

## What the guard actually does

`execute_step` finishes backward, flushes recorded tensors, compares paired
values, and evaluates the integrity gate before clipping, AdamW or scheduler
mutation. The failing arm raises `IntegrityError` there. This protects the
persistent update state checked by the suite. It does not roll back every
possible Python side effect, RNG advance or accumulated gradient buffer.

The failing enforced trace has 14,292 captured tensors and a completed trace
record, with no optimizer commit. The observed failing run has 15,115 captured
tensors because it also executes clipping and the optimizer. A complete trace
and an allowed optimizer update are separate facts.

This recorder is a diagnostic reference implementation. The observed failing
step took about 1,082 seconds with recording versus 5.25 seconds without it.
These individual timings include recording, compression and comparison work;
they are not a controlled performance benchmark. Low overhead is unestablished.

## Upstream Llama without the reproduction adapter

Condor **1553918.0**, commit `2233d31f5b2bae6b639e05cc5f568e00330daaa2`,
Quadro RTX 6000, PyTorch 2.13.0+cu126 and Transformers 5.16.1. The upstream
`LlamaForCausalLM` source is unmodified and has 39,985,664 parameters. The project
supplies the training loop and an optional observer at its native checkpoint
callback. Each cell uses one 64-token update, independent initial states,
a repeated reference, and a checkpoint arm. Eager cells also have a recorded arm.

| Execution setting | GPU result |
|---|---|
| Eager FP32 | Exact initial and final outcome agreement; observer preserves outcome. |
| SDPA FP32 | Exact agreement; observer preserves outcome. |
| Attention dropout, FP32 | Exact agreement; observer preserves outcome. |
| FP16 autocast with gradient scaling | Exact agreement; observer preserves outcome. |
| Inductor compilation, FP32 | Reference repeat exact; checkpoint arm has small numerical differences. |
| Native BF16 | Unsupported on this GPU; no result claimed. |

In the compiled cell, initial states agree, and both loss
`10.36438274383545` and gradient norm `12.381913185119629` agree exactly.
The following entries differ after the update:

| State | Differing tensors | Largest absolute element difference | Largest per-tensor relative L2 difference |
|---|---:|---:|---:|
| Gradients | 64 | 1.3783574e-7 | 6.4703524e-7 |
| Model parameters | 52 | 5.6046993e-6 | 5.6073028e-7 |
| Optimizer state | 128 | 1.1059456e-9 | 7.5543947e-7 |

**Verdict: bounded numerical discrepancy, mechanism unresolved.** Compiler
arithmetic changes are a plausible explanation, but we have not isolated that
cause. Floating-point evaluation order can change results even for mathematically
equivalent computations; exact equality alone is not an appropriate universal
correctness criterion. See [PyTorch numerical accuracy guidance](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).
The compiled recorder cannot inspect fused internal values, and this screen has
not localized a differing recomputed intermediate or established harmful training
behavior. No production-bug or convergence claim follows from this result.

## Failed attempt and repairs

The first upstream attempt, **1553917.1** at `9b922d7`, failed in all five runnable
cells: GCC could start, but its linker lookup failed with `cannot find ld`.
The installed PyTorch eager rotary-embedding path also invoked Triton, so this
setup failure affected eager as well as explicitly compiled cells. BF16 was
separately unsupported. The failed archive is retained and verified.

The rerun exported system tool directories, verified a compiled shared-library
probe, recorded GCC/linker paths, and propagated matrix execution failures to
the process status. The scientific model, corpus, cells and seed were preserved.

A provenance correction moved environment collection after backend setup for
future runs. The original core run's startup environment flags are not its
execution flags; its immutable snapshot stores the configured backend state,
and the pinned loader checks it at execution. The original run is preserved.

## Remaining limits

The requested bounded GPU follow-through is distinct from the longer research
contract. Still unestablished: long training controls, TorchTitan/FSDP or
multi-GPU integration, native BF16/FP8, complete visibility inside fused/custom
operators, low-overhead fingerprints, and a naturally arising semantic failure
in unmodified production training. The next research decision should focus on
compiled numerical baselines and useful tolerance criteria before treating every
bit difference as corruption.

## Evidence locations

Local `artifacts/followthrough/` contains:

- `core-gpu-summary.json`: full ten-gate summary and first-divergence measurements.
- `core-archive-manifest.json`: raw file sizes and SHA256 identities.
- `upstream-gpu-retry-summary.json`: all six rerun cell verdicts and arm metadata.
- `compile-differences.json`: all 244 differing compiled state entries.
- `gpu-1553917.1-archive/`: verified original setup-failure archive.
- `gpu-1553917.0-recovered-archive/`: core archive recovery and verified chunk progress.
- `gpu-1553918.0-archive/`: upstream rerun archive recovery.

Raw evidence, model states and datasets remain excluded from Git. The append-only
research log records scheduler identities, failed attempts, repairs and retention.
