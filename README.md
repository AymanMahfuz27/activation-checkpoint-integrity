# Activation Checkpoint Integrity

The 40M-parameter research model now has a GPU proof of controlled failure
reproduction, first-intermediate localization, recording noninterference and
pre-optimizer enforcement. All ten causal checks passed on a GTX 1080 Ti.
An adapter-free upstream Llama GPU screen produced four exact eager results;
Inductor compilation produced small numerical differences requiring analysis.
See [GPU results and limitations](docs/followthrough-results.md),
[implementation status](docs/production-status.md) and
[reproduction commands](docs/FOLLOWTHROUGH.md).

The recorder retains full supported eager operator outputs and is expensive.
The production-default position-sensitive fingerprinter now observes only
checkpointed block returns and keeps its signatures and comparison flags on the
GPU. On the controlled 40M FP32 GTX 1080 Ti cell, three paired clean runs
measured 5.2192 seconds with the detector versus 5.1736 seconds with capture
off: 0.88% median overhead, below the 2% target. The injected fault was detected
at the first block boundary, the optimizer was blocked, and persistent state
was preserved. The failing arm took 4.968 seconds versus the historical full
recorder's 1081.77 seconds, a 217.75x speedup. The exhaustive all-operator
fingerprinter remains available for localization after a boundary abort.
Modern-GPU, mixed-precision, distributed, internally cancelling-fault, and
tolerant-policy coverage are not yet validated. The long training milestone
and TorchTitan integration remain incomplete. See
[fingerprinter design and use](docs/fingerprinter.md).

The sections below describe the completed historical starter work.

So far, we have built a small test bench for checking whether activation
checkpointing recomputes the same values during backward. It uses a tiny matrix
calculation whose correct output and gradients we can calculate independently.

## What we did

1. **Checked the clean case (E0).** We ran the calculation with and without
   checkpointing, saved three intermediate tensors during the original forward
   and backward recomputation, and compared their values. The tensors matched,
   and the outputs and gradients agreed with the independent formulas.

2. **Created controlled failures (E1).** We deliberately changed state before
   recomputation: a counter, a model buffer, Python/NumPy randomness, precision
   selection, or a toy quantization scale. The original output stayed the same,
   but recomputed values and gradients changed. PyTorch's default checkpoint
   check did not flag these value changes because tensor metadata still matched.
   Our comparisons located the first changed element. Removing each trigger
   restored the correct result.

3. **Tested a possible capture method (E2).** The earlier tests explicitly
   recorded tensors at known points in our calculation. We then tried PyTorch's
   saved-tensor hooks as a way to observe them. The tested hook arrangement
   prevented checkpoint recomputation, so it could not capture the recomputed
   tensors. We rejected that arrangement.

We repeated these experiments on the Mac CPU and a GTX 1080 Ti GPU. Both gave
consistent outcomes: the clean and controlled-failure tests passed, and the
hook arrangement failed its coverage test.

## Why this matters

These examples give the future detector known correct and incorrect cases to
check against. Saving complete tensors provides a reference for evaluating a
faster detector later. The hook experiment also exposes a capture problem we
need to resolve before integrating with real training.

The starter results below describe deliberately created failures in a tiny
calculation. The linked GPU report records subsequent real-model validation;
an efficient production detector remains future work.

## Run it

```bash
uv sync --locked
uv run pytest
uv run aci-starter run E0
uv run aci-starter run E1 --case all
uv run aci-starter run E2
```

E2 currently reports `PUBLIC_HOOKS_INSUFFICIENT` and exits with code 1. That is
the recorded experimental result.

Runs save tensors and reports under `artifacts/starter/`. Exact configurations,
run records, and detailed findings are in [RESEARCH_LOG.md](RESEARCH_LOG.md).

The production trainer enables the compact exact detector with
`capture.mode = "fingerprint"` and `capture.policy = "enforce"`. On a clean
single-device step its default checkpoint-boundary scope keeps signatures
on-device and performs one detector host check after backward, before clipping
or optimizer mutation. A mismatch or incomplete observation aborts the update
and writes a compact mismatch record. The more expensive `all_operators` scope
is retained for operator-level diagnostic replay.
