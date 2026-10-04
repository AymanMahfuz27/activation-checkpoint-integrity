# BLAKE3 repair: reliability evidence

The deterministic fingerprint collision is repaired. The replacement passes
local and scheduled CUDA conformance, rejects the historical collision through
the actual session, and preserves clean training while blocking the tested
faulty update. This establishes correctness within the declared observation and
execution scope. Stable overhead below 2% for a training step remains unmeasured.

## What changed

`blake3-256-v1` hashes exact logical contiguous tensor bytes with standard unkeyed
BLAKE3, retaining all 32 digest bytes in four int64 slots. CPU uses pinned
`blake3==1.0.10`; CUDA implements the standard chunks, flags and ordered tree.
Pair identity and metadata checks remain separate from the digest. Numerical
sketches cannot override a mismatch. See [implementation](fingerprinter.md) and
[machine-readable evidence](../reports/f011-blake3-reliability.json).

The old odd-weighted sums allowed two FP32 sign changes to cancel modulo 2^32
in every lane. The pair `[1, 2]` versus `[-1, -2]` was accepted by the real
session. The new digest and session both reject that pair on CPU and CUDA.
BLAKE3's reviewed construction replaces the custom linear arithmetic; finite
conformance tests still do not prove collision freedom.

Each session now owns one current CUDA stream per device. Observing or finalizing
on a different stream raises and permanently marks the session failed. This
prevents reading zero-initialized flags before another stream finishes their
comparison writes. Callers must resolve lazy conjugate/negative views. Raw CUDA
helpers require producer ordering and storage lifetime on the current stream.

## Validation and the failed first attempt

The final local suite reports 117 passed, with 10 CUDA-only skips, in 56.89s.
Condor job `1553978.0` reports 82 preflight tests passed, with zero skips or
failures, in 178.06s. The tests exercise all 35 pinned official unkeyed vectors,
28 native tree sizes, CUDA boundaries through 67,108,865 bytes, 120 paired sign
changes and 896 seeded byte-mutation cases per device. They also check tails,
special float bits, dtypes/layouts, full digest comparison, actual-session
collision rejection, wrong-stream rejection and same-stream finalization.

The first attempt, `1553977.0` at `a596b96`, reports 79 passed and three failed.
Recompute accessed existing buffers directly and bypassed the stream guard;
two CPU training test configurations also expected the wheel version without
its CUDA suffix. Preflight stopped training. The correction checks the stream
before recompute reservation or launch and adjusts only the test's expected
version. Production preflight retains its exact build pin. The failed attempt
is retained as negative evidence.

## End-to-end correctness

The scheduled scientific revision is clean
`82344622d899c348741d74f288125dfca55a9958`, running on `slot2@eldar-28`, GTX
1080 Ti, driver 535.247.01, Python 3.13.5, PyTorch 2.13.0+cu126 and CUDA 12.6.
The workload is the 39,985,664-parameter model, deterministic eager FP32,
seed 17, sequence 256, microbatch 2 and accumulation 4, from one shared warmed
Adam snapshot. Eight blocks produce 32 original/recompute pairs per step.

All eight workload gates pass. Each of three clean arms matches all 32 pairs,
uses one host decision and one optimizer call, and has no differences from its
capture-off outcome in model, gradients, losses, optimizer, scheduler, RNG or
cursor. The controlled silent error changes 73 named gradients. The guard
catches all 32 differing boundary pairs, first at `root/blocks.0:0`, makes zero
optimizer calls and preserves model, optimizer, scheduler and cursor. Peak CUDA
allocation rises by 1,179,648 bytes (0.11934%), the fixed session buffers.

## What the timing establishes

The recorded interval includes `execute_step`, CPU gradient evidence copies,
other outcome transfers and writing `outcome.pt`. It is not an isolated training
step. Kernel compilation occurs before this interval.

| Alternating pair | Capture off | BLAKE3 | Paired overhead |
|---|---:|---:|---:|
| 1 | 17.78074 s | 18.07454 s | +1.65235% |
| 2 | 20.01212 s | 17.49083 s | -12.59883% |
| 3 | 17.67092 s | 19.86516 s | +12.41725% |

The ratio of medians is 1.01652347, or 1.65235% observed overhead for this
harness interval. Its separate 2% target field passes; the suite's timing gate
is 5%. The wide paired spread and evidence export prevent a stable sub-2%
training-step claim. Common export cost can dilute observer overhead, while
host/storage variability can dominate the ratios. Neither the negative pair
nor the ratio of mean times establishes acceleration.

The independent Analyst gives HIGH validity to the executed correctness gates
and LOW validity to a stable sub-2% step-cost claim. The formal combined verdict
is REPEAT for performance; the correctness repair can be retained and closed.
F013 should use an optional timing-only path with `collect_evidence=False`,
synchronize immediately before/after `execute_step`, stop the timer before
cleanup/export, and repeat alternating controls enough to resolve the noise.

## Evidence retention and limits

Compact evidence is retained remotely under
`/u/ayman27/activation-checkpoint-integrity/artifacts/followthrough/1553978.0`
and locally under `artifacts/fingerprint/f011-blake3/condor-1553978.0`.
All 79 remote files match SHA256 (761,200 bytes), and all 39 archived code hashes
match their environment record. Summary SHA256:
`912a143726a85437edd003b327315278a1640f1d3f301b6f417d556e6a20eed9`.
The frozen source archive, exact configurations, scheduler records, test XML,
comparison reports and state-preservation decisions are retained. Large
snapshot/outcome tensors were compared in scheduled scratch and were not
retained by the compact runner. The Analyst checked the comparison records and
their generating source; it did not independently rerun those tensor comparisons.

A matching digest remains probabilistic evidence. Boundary scope cannot detect
interior-only or derivative-only divergence when returned tensors remain the
same; F012 is still open. This single-rank eager FP32 Pascal cell does not
establish long-training, modern-GPU, mixed-precision, distributed, fused or
compiled coverage. BF16 storage hashing in conformance tests does not validate
BF16 training on Pascal. The historical 0.88% weighted-sum result does not
validate this replacement.
