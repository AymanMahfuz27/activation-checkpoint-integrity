# Position-sensitive fingerprinter

## What is implemented

The fingerprinter is the compact exact-mode path between capture-off training
and the full forensic recorder. It observes dense ordinary tensor outputs inside
checkpoint regions, assigns them the same stable `pair_id` used by full capture,
and compares original-forward and recomputation signatures before gradient
clipping or optimizer mutation.

Each tensor becomes two 64-bit polynomial signatures. The input is the tensor's
exact contiguous byte representation, grouped into 64-bit words. Each word is
multiplied by a position-specific power of a different odd base, and the words
are reduced modulo 2^64. Position therefore participates in the signature: a
permutation is not treated like the original ordering. Two lanes reduce the
chance of an accidental collision, but this remains a probabilistic equality
check rather than a mathematical proof or a cryptographic integrity primitive.

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
```

`fingerprint_capacity` bounds original and comparison rows per device. Overflow
fails closed instead of reallocating silently. `fingerprint_chunk_bytes` bounds
temporary hashing work; signatures are independent of chunk size.
`fingerprint_sketches = true` enables calibration evidence at additional cost.

## Evidence and limits

The local validation covers exact repetition, permutations, one-element and
one-ULP changes, FP16/BF16/FP32/FP64 and integer tensors, signed zero, NaN,
non-contiguous inputs, metadata changes, missing recomputation, buffer overflow,
nested checkpoints, clean training noninterference and controlled-failure
optimizer blocking. The full repository suite passes 60 tests.

The bounded CPU smoke suite compares 404 exact pairs. The clean fingerprint arm
preserves the capture-off outcome, the failing arm identifies the same first
pair as full capture (`aten.rand.default`), and the enforced failing arm makes
zero optimizer calls. In the latest single-run smoke, the compact arm took about
0.11 seconds versus about 1.21 seconds for full capture and about 0.03 seconds
with capture off. These are tiny-model CPU measurements, not a production
overhead result. CUDA correctness, peak GPU memory, repeated timing, the 40M
perturbation ladder and modern-hardware validation remain required.
