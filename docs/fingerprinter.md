# BLAKE3 checkpoint fingerprinting

The current algorithm is `blake3-256-v1`: standard unkeyed BLAKE3 over exact
logical contiguous tensor bytes, retaining its full 32-byte digest in four
little-endian int64 slots. It replaces `mixsum128-positioned-u32-v2`, which
accepted pairs of FP32 sign flips. Historical measurements below belong to the
rejected algorithm and do not validate replacement performance.

## Algorithm and implementation

CPU hashes bounded byte chunks with pinned `blake3==1.0.10`, an independently
published implementation. CUDA hashes standard 1024-byte chunks and combines
chaining values in the specified ordered binary tree. ROOT is applied only to
the final compression. Shared compression primitives are independently tested
on CPU; GPU synchronization and execution require separate scheduled tests.

CUDA blocks handle up to 256 chunks each. Larger payloads use extra subtree
reduction launches, retaining one 32-byte chaining value per 256 KiB subtree
at the first level. Each block uses 8 KiB shared memory. Temporary device storage
shrinks by a factor of 256 per level and stays on the current CUDA stream.
CPU and CUDA are the supported digest devices; other devices fail explicitly.
Each session owns one current CUDA stream per device; observing or finalizing
on another stream raises `FingerprintStreamError` and permanently rejects the
session. This prevents finalization from reading an unfinished comparison flag.
Callers must resolve lazy conjugate/negative views before capture. Raw CUDA
helpers require producer ordering and storage lifetime on their current stream.
Metadata and exact pair identity are checked separately. Signed zero and NaN
payload bits remain distinct; there is no rounding or numerical tolerance.

BLAKE3 has a published cryptographic design, rather than custom weighted sums.
Its design assurance does not automatically validate our CUDA implementation.
Official known-answer vectors, the independent library, targeted cancellation
families and chunk/tree boundary tests establish implementation conformance.
Finite testing cannot prove collision freedom or estimate extremely small
collision probabilities. A 256-bit digest has a generic 128-bit collision
security target; output width, collision search complexity and fixed-pair
probability are different quantities. See the
[specification](https://github.com/BLAKE3-team/BLAKE3-specs) and
[reference implementation](https://github.com/BLAKE3-team/BLAKE3/tree/master/reference_impl).

## Policy and coverage

A mismatch rejects the update. A matching digest is probabilistic equality
evidence, not proof. The legacy `exact_pairs` field counts matching digests and
metadata; it does not certify collision-free equality. Numerical sketches are
diagnostic only and never override a mismatch. Missing pairs, structural
errors and buffer overflow fail closed.

The default boundary scope watches returned block tensors. Even an ideal hash
cannot detect an internal or derivative-only change that leaves those returns
unchanged. The all-operator replay observes supported eager outputs, with
fused/compiled interiors unverified. The plan's interior-only criterion remains
a separate gate. The host decision occurs after backward and before clipping,
optimizer or scheduler mutation. It does not repair gradients or hidden state.

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
temporary work in the readable CPU reference; the CUDA tree does not
materialize per-element weights or products. Signatures are independent of chunk
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

## Replacement validation

Corrected revision `8234462` passes 117 local tests (10 CUDA skips), 82 scheduled
preflight tests without skips, and all eight 40M workload gates in Condor
`1553978.0`. Clean training outcomes match; all 32 controlled-failure pairs are
rejected before an optimizer call. The observed 1.6523% overhead belongs to the
existing evidence-exporting harness interval, whose noisy pairs do not establish
stable sub-2% training-step cost. See [reliability evidence](fingerprinter-reliability.md)
for the failed first attempt, independent review, provenance and limitations.

## Historical weighted-sum evidence (not replacement validation)

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
