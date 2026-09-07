# Activation Checkpoint Integrity

The production-shaped program is now implemented in part and under validation.
It includes the 40M/125M decoder definitions, real packed corpus, training and
replay lifecycle, exhaustive eager capture, and optimizer enforcement. Three
real-corpus 40M updates agree exactly with and without checkpointing. The full
research milestone and TorchTitan integration remain incomplete. See
[implementation status](docs/production-status.md) and
[40M validation evidence](reports/lm40m-three-step.json).

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

The current code tests deliberately created failures in a tiny calculation.
Real-model validation and an efficient production detector are still ahead.

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
