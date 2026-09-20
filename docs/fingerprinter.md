# Position-sensitive fingerprinter

## What is implemented

The fingerprinter is the compact exact-mode path between capture-off training
and the full forensic recorder. Its normal `checkpoint_boundaries` scope
fingerprints each checkpointed block's returned tensor in the original forward
and backward recomputation. The `all_operators` diagnostic scope instead
observes every dense tensor output inside checkpoint regions, assigns it the
same stable `pair_id` used by full capture, and localizes the first changed
operator. Both scopes compare before gradient clipping or optimizer mutation.

Each tensor becomes one 128-bit signature stored as two `int64` values. The
readable reference implementation groups the tensor's exact contiguous byte
representation into 32-bit words. For each word, it nonlinearly mixes the
word's absolute position twice, derives four odd weights, multiplies the word by
those weights, and adds the products modulo 2^32. The four lanes are packed into
the two stored values. Payload length is mixed in separately, so appending zero
bytes changes the result. Position therefore participates in the signature: a
permutation is not treated like the original ordering. This remains a
probabilistic equality check rather than a mathematical proof or a
cryptographic integrity primitive.

On CPU, [`fingerprint.py`](../src/ac_integrity/capture/fingerprint.py) executes
that definition directly in Torch so the algorithm is easy to inspect and test.
On CUDA, [`fingerprint_cuda.cu`](../src/ac_integrity/capture/csrc/fingerprint_cuda.cu)
computes the same four lanes in one fused kernel. The small loader in
[`cuda_fingerprint.py`](../src/ac_integrity/capture/cuda_fingerprint.py) compiles
the header-free kernel with CUDA's runtime compiler, loads device-native CUBIN
through the CUDA Driver API, and launches it on PyTorch's current stream. Using
native machine code avoids depending on the installed driver to JIT a newer PTX
version. It needs neither a system `nvcc` installation nor a C++ extension
build. Kernel compilation occurs when the fingerprint runtime is created,
before the measured validation step.

Recomputed CUDA signatures are compared by a second one-thread native kernel
on PyTorch's current stream. It checks all four 32-bit lanes and writes the
existing mismatch flag directly, replacing separate PyTorch inequality,
reduction, conversion and copy operations. Stream ordering makes the signature
write visible to comparison without a per-pair host synchronization.

The normal path keeps signatures, comparison flags and optional numerical
sketches in fixed-capacity device buffers. Original and recomputed outputs are
joined by exact execution identity; traces are never heuristically realigned.
One scalar mismatch decision is transferred to the host after backward on the
current single-device trainer. Full mismatch rows move to the host only after a
failure, when the update has already been marked unsafe.

## Decision policy

The implemented enforcement policy is deliberately narrow:

1. Matching metadata and both signatures means exact-mode agreement, so the
   optimizer may proceed.
2. A signature mismatch, metadata or structure mismatch, missing/duplicate
   pair, unsupported tensor, or exhausted buffer aborts before clipping and
   optimizer/scheduler mutation.
3. The failed attempt is not repaired in place. The safe recovery unit remains
   a fresh-process replay from the immutable pre-step snapshot.

There is no global floating-point threshold. Hash distance does not measure
numeric distance, so a one-bit change and a large corruption are both simply
`different`. Optional numerical sketches record scale, norms, special-value
counts and deterministic projections for calibration. They are disabled on the
fast exact path by default and cannot authorize an update.

A future tolerant policy requires a measured healthy envelope for each hardware,
dtype, compiler/backend and operator family. Calibration must compare clean and
injected distributions using exact capture, gradients, parameter updates and
multi-step impact. If healthy and harmful distributions overlap, that execution
cell remains unsupported or escalates to exact diagnostic replay.

## Configuration

```toml
[capture]
mode = "fingerprint"
policy = "enforce"
audit_steps = "all"
fingerprint_capacity = 16384
fingerprint_chunk_bytes = 262144
fingerprint_sketches = false
fingerprint_backend = "full"
fingerprint_scope = "checkpoint_boundaries"
```

`fingerprint_capacity` bounds original and comparison rows per device. Overflow
fails closed instead of reallocating silently. `fingerprint_chunk_bytes` bounds
temporary work in the readable CPU reference; the fused CUDA kernel does not
materialize per-chunk weights or products. Signatures are independent of chunk
size.
`fingerprint_sketches = true` enables calibration evidence at additional cost.
`fingerprint_backend = "full"` is the only enforcement-capable backend. The
`bookkeeping` and `launch` values exist only for the preregistered performance
ablation: configuration validation requires observe policy and disables
sketches so neither diagnostic path can authorize an optimizer update.
`fingerprint_scope = "checkpoint_boundaries"` is the low-frequency production
default. It detects divergence that is visible when a checkpointed block
returns. Set `fingerprint_scope = "all_operators"` for expensive diagnostic
localization after an aborted step. The boundary scope cannot detect an
internal error that exactly cancels before the block returns.

## Evidence and limits

The local validation covers exact repetition, permutations, every short tail
length, explicit length separation, direct writes into bounded buffer rows,
one-element and one-ULP changes, FP16/BF16/FP32/FP64 and integer tensors, signed
zero, NaN, non-contiguous inputs, metadata changes, missing recomputation,
buffer overflow, nested checkpoints, clean training noninterference and
controlled-failure optimizer blocking. CUDA environments additionally compare
the fused kernel bit-for-bit against the CPU reference over varied lengths,
dtypes and layouts.

The bounded CPU smoke suite compares 404 exact pairs. The clean fingerprint arm
preserves the capture-off outcome, the failing arm identifies the same first
pair as full capture (`aten.rand.default`), and the enforced failing arm makes
zero optimizer calls. In the latest single-run smoke, the compact arm took about
0.11 seconds versus about 1.21 seconds for full capture and about 0.03 seconds
with capture off. These are tiny-model CPU measurements, not a production
overhead result.

Condor job `1553948.0` applied the compact path to the seed-17 40M FP32
controlled workload on a GTX 1080 Ti. It covered all 3,232 pairs, reproduced the
historical 96 mismatches and first `aten.rand.default` divergence, blocked the
failing update with no persistent-state mutation, and preserved the clean
capture-off outcome. The failing arm took 14.61 seconds versus the historical
full recorder's 1081.77 seconds, a 74.06x speedup and 98.65% reduction. That
historical comparison crosses revisions and regenerated snapshots, so it is an
order-of-magnitude comparison rather than a precise paired timing benchmark.

The pre-optimization same-job production comparison is the clean arm: 15.16 seconds with
fingerprints versus 5.13 seconds with capture off. That is 2.95x total time, or
195.3% overhead, far above the at-most-2% target. Device buffers added 1.125 MiB
and peak CUDA allocation increased about 0.119%, so the memory objective passed
for this cell while the runtime objective failed. The fused kernel removes the
composed per-tensor weight, product and reduction operations that caused that
measurement. Condor job `1553955.0` compiled and loaded the exact production
kernel as native `sm_61` code and matched an independent scalar CPU reference in
all 29 checks across tail lengths, dtypes, layouts and adversarial permutations.
Condor job `1553956.0` then measured three alternating clean pairs. The fused
fingerprint times were 9.17, 9.50 and 9.33 seconds; capture-off times were 4.86,
4.87 and 4.96 seconds. The 9.33-second fingerprint median is 38.5% lower than
the old 15.16-second result, but it still has 91.4% overhead over the paired
4.87-second baseline. It therefore fails both the preregistered 1.25x
intermediate gate and the project's 1.02x production target. The failing arm
still found all 96 mismatches at the historical first operator, blocked the
optimizer, and ran 115.4x faster than the historical full recorder. Peak-memory
overhead fell to 0.066%. The evidence localizes the next performance problem to
per-output Python interception and CUDA launch amplification rather than the
signature arithmetic alone.

The F008 ablation then separated that remaining surcharge: about 68% came from
Python observation and event bookkeeping, about 32% from per-output launches
and comparisons, and no resolvable increment came from scanning payload bytes.
This motivated the production-default checkpoint-boundary scope. Condor job
`1553958.0` reduced observation count from 6,464 to 64 per step and preserved
all safety gates. Its paired clean median was 5.1622 seconds versus 5.0505
seconds off, or 2.21% overhead. That passed the preregistered 5% architecture
gate but narrowly missed the strict 2% production target.

The final native comparison change removed separate PyTorch inequality,
reduction, conversion and flag-copy operations after each recomputed boundary.
Condor job `1553959.0` passed all correctness gates and measured clean boundary
times of 5.2851, 5.2192 and 5.1804 seconds against paired capture-off times of
5.1996, 5.1736 and 5.1139 seconds. Every pair stayed below 2%; the median ratio
of arm medians was 1.00881x, or 0.88% overhead, while the median of paired
ratios was 1.01302x, or 1.30%. The injected arm detected all 32
boundary-visible mismatches, first at block 0, made zero optimizer calls, and
preserved model, Adam, scheduler and cursor state. It took 4.968 seconds, a
217.75x speedup and 99.54% time reduction from the historical 1081.77-second
full recorder. Peak CUDA allocation still rose only 0.0663%.

Modern-hardware, mixed-precision, perturbation-ladder and tolerant-mode
validation remain required after the GTX 1080 Ti gate.
