## 1. Why we are doing this first

E0–E2 are a calibration exercise for the future production harness.

Imagine building a smoke detector. Before installing it throughout a skyscraper, you test it with five tiny, controlled sources of smoke. Because you intentionally created each source, you already know:

- Where the smoke should appear.
- When it should appear.
- What the detector should report.
- What the detector should report when there is no smoke.

That is what these examples do for activation checkpointing.

They do **not** prove that a production training failure naturally occurs. The later production-shaped plan does that. These examples answer an earlier engineering question:

> If an original forward tensor and its backward recomputation differ, can our code reliably capture both tensors, compare them, locate the difference, and avoid false alarms on the correct version?
> 

They also answer:

> Are PyTorch’s public hooks sufficient to observe every backward-relevant tensor we need, or will the later system require a deeper PyTorch integration point?
> 

That is exactly the purpose of Weeks 1–2 in the original Notion plan.

To make the naming clearer, the starter work will be grouped as:

- **E0:** establish a trustworthy clean baseline and exact comparator.
- **E1:** run five broken examples against five correct controls.
- **E2:** verify whether the proposed public capture hooks actually see every required tensor.

This consolidates the older, more confusing E00–E04 numbering into the three milestones you originally expected.

## 2. The three experiments

### E0 — Clean checkpoint baseline and exact comparator

Build one tiny matrix calculation using fixed literal tensors on CPU in `float64`:

```
h = x @ W1 + b
g = h² + 0.5h
y = g @ W2
loss = sum(y * target)
```

The tensors are deliberately tiny—mostly 2×2—and use fixed values. This gives us two independent answers:

1. A normal run without activation checkpointing.
2. A checkpointed run whose forward is recomputed during backward.

The math is simple enough to calculate the correct gradients directly. This prevents the checkpointed run from grading itself.

Inside the checkpointed function, explicitly tag three matrices:

- `h`: first matrix produced by the layer.
- `g`: nonlinear intermediate.
- `y`: final checkpoint output.

Use non-reentrant PyTorch checkpointing with:

- `use_reentrant=False`
- `early_stop=False`, so the entire function is replayed.
- `preserve_rng_state=True`
- `determinism_check="default"`
- A `context_fn` that labels execution as `original` or `recompute`.

PyTorch documents that `context_fn` provides separate contexts for the original execution and recomputation, and that its default determinism check compares shape, dtype, and device rather than values. PyTorch checkpoint documentation

For every tag, save both complete tensors and report:

- Shape, dtype, device, layout, and stride.
- Exact equality.
- Number of different elements.
- First differing index.
- Maximum absolute difference.
- Relative L2 difference.
- Paths to the two tensor files.

E0 passes only if:

- The checkpointed function runs once as `original` and once as `recompute`.
- Each of `h`, `g`, and `y` has exactly one original and one recomputed tensor.
- All three pairs match exactly.
- The checkpointed and no-checkpoint outputs, losses, and gradients match.
- The gradients also match the independently calculated formulas.
- No tensor is missing, duplicated, or incorrectly paired.

Any E0 failure means the fixture or comparator is unreliable. E1 and E2 remain blocked until it is fixed.

### E1 — Five broken examples and their correct controls

Every family has two versions:

- **Correct:** original and recomputation use the same state.
- **Broken:** the original calculation is unchanged, but hidden state changes before recomputation.

Both versions begin from identical tensors, seeds, and state. The broken version’s original forward, loss, and output must match the correct version. The first difference must appear only during backward recomputation.

#### 1. Changing counter

Correct behavior:

- A call counter may increment for logging, but the calculation does not depend on it.

Broken behavior:

- Original forward sees counter value `0` and uses scale `1.0`.
- Recomputation sees counter value `1` and uses scale `1.25`.
- The scale changes `h` without changing its shape, dtype, or device.

Expected result:

- Correct original and recomputation match.
- Broken original and recomputation first differ at `h`.
- At least one gradient differs.

#### 2. Mutable model buffer

Correct behavior:

- A registered model buffer remains frozen until backward is complete.

Broken behavior:

- The original execution reads buffer value `1.0`.
- The buffer is advanced to `1.25`.
- Recomputation reads the new value and changes `h`.

The buffer is read as a non-differentiable scalar before mutation so the example tests recomputation state rather than accidentally triggering an unrelated autograd version-counter error.

Expected result:

- Same metadata.
- Different `h` values.
- Different downstream gradient.
- Freezing the buffer restores equality.

#### 3. Python/NumPy randomness

This is one fault family with two required variants:

- Python’s `random.random()`.
- NumPy’s `numpy.random.random()`.

Correct behavior:

- Generate the random value once before entering the checkpointed function and pass it in as fixed data.

Broken behavior:

- Generate the value inside the checkpointed function.
- Original forward consumes one value.
- Recomputation consumes the next value.

This demonstrates why ordinary PyTorch RNG preservation does not automatically handle unrelated Python or NumPy generators.

Expected result:

- Both variants produce a same-metadata mismatch.
- Fixed external randomness restores equality.
- Repeating the run with the same seed reproduces the same artifacts.

#### 4. Changed precision policy

Correct behavior:

- Original and recomputation select the same FP32 calculation.

Broken behavior:

- Both a high-precision and BF16-rounded version of the matmul are computed in both phases, keeping the operator structure stable.
- Original forward selects the FP32 result.
- Recomputation selects the BF16-rounded result converted back to FP32.

The tagged boundary remains FP32 in both phases. Therefore, shape, dtype, and device remain identical even though values may differ.

Expected result:

- The default metadata check does not reject the pair.
- Exact comparison finds the changed values.
- Using the FP32 selection in both phases restores equality.

If the current CPU/PyTorch build cannot execute the BF16 path correctly, this case is recorded as an environment limitation rather than silently replaced with a different experiment.

#### 5. FP8-style delayed scaling

This is explicitly a toy emulation, not evidence about native FP8 hardware.

Correct behavior:

- A small quantize/dequantize operation uses one frozen scale during original forward and recomputation.

Broken behavior:

- The original execution updates a stored maximum-value history.
- That history changes the scale.
- Recomputation quantizes with the advanced scale.

Use a straight-through estimator so the quantized value affects the downstream nonlinear derivative while gradients can still flow through the tiny fixture.

Expected result:

- Original forward matches the correct control.
- Recomputation first differs at the tagged `h`.
- The gradient differs.
- Freezing the scaling history restores equality.

E1 passes only if, for all five families:

- The correct arm has zero tensor mismatches.
- The broken original forward exactly matches the correct original forward.
- The broken arm completes without the default metadata check raising.
- The first mismatch occurs at the preregistered `h` tag.
- Shape, dtype, device, layout, and stride still match.
- At least one input or parameter gradient differs.
- Disabling only the relevant trigger restores tensor and gradient equality.
- Both full tensors and the exact first differing index are saved.
- Three repeated runs produce the same result and artifacts.

The Python and NumPy variants must both pass for the randomness family to pass.

### E2 — Does the public capture method see everything required?

E0 and E1 use explicit tagging as the trusted ground truth. E2 tests the candidate public PyTorch capture method:

- `context_fn` to distinguish original from recomputation.
- `torch.autograd.graph.saved_tensors_hooks` to observe tensors saved for backward.

PyTorch says a saved-tensor pack hook is called whenever an operation saves a tensor for backward, but that does not by itself prove that nesting these hooks with checkpointing exposes every tensor needed by our design. PyTorch autograd documentation

For E2, route `h`, `g`, and `y` through a tiny `TaggedSave` autograd function. It saves:

- The actual tensor.
- A unique integer token identifying the semantic tag.

This makes all three tensors explicitly backward-relevant and gives the hook stream a stable identity that does not depend on tensor addresses, storage pointers, shapes, or hashes.

Use two independent records:

1. **Direct tag ledger:** the known ground truth from inside the fixture.
2. **Public hook ledger:** what `saved_tensors_hooks` actually observed.

Run E2 twice:

- Baseline checkpoint run without observational saved-tensor hooks.
- Candidate run with the public hooks enabled.

E2 passes only if:

- Original and recomputation each execute exactly once.
- The public hook observes `h`, `g`, and `y` in both phases.
- Every packed tagged tensor has the expected unpack/access event.
- Every direct tag has exactly one unambiguous public-hook match.
- There are no missing or duplicate tagged events.
- Values remain exact in the clean fixture.
- Output and gradients remain identical to the no-hook E0 result.
- Installing the hooks does not suppress recomputation or change program behavior.

If a tag is missed, duplicated, ambiguously ordered, or the hooks change the result, E2 fails with:

```
PUBLIC_HOOKS_INSUFFICIENT
```

That is a useful result. It tells the later production system to use PyTorch’s internal original/recomputation pairing point instead of building on an incomplete public-hook assumption.

## 3. Minimal code and artifacts

Create only the starter infrastructure:

```
pyproject.toml
uv.lock
src/ac_integrity/
  cli.py
  starter/
    fixture.py
    capture.py
    cases.py
    runner.py
tests/
  test_starter.py
```

Responsibilities:

- `fixture.py`: fixed tensors, matrix function, semantic tags, `TaggedSave`, and independent gradient formulas.
- `capture.py`: original/recompute contexts, exact tensor recorder, saved-tensor observer, pairing, comparison, and artifacts.
- `cases.py`: the five correct/broken families.
- `runner.py`: executes isolated arms from reset state and evaluates gates.
- `cli.py`: exposes the commands.
- `test_starter.py`: focused automated test suite.

Commands:

```bash
uv run aci-starter run E0
uv run aci-starter run E1 --case all
uv run aci-starter run E1 --case rng --variant python
uv run aci-starter run E1 --case rng --variant numpy
uv run aci-starter run E2
uv run pytest
```

Each correct and broken arm runs in a fresh process so Python state, NumPy state, counters, or buffers cannot leak between comparisons.

Use stable logical identities:

```
pair_id  = experiment/case/checkpoint_call/tag/occurrence
event_id = run_id/pair_id/phase
```

Never pair tensors using their memory address, storage address, shape alone, or value hash.

Artifacts stay intentionally simple:

```
artifacts/starter/<run_id>/
  manifest.json
  events.jsonl
  comparisons.jsonl
  tensors/<case>/<tag>.original.pt
  tensors/<case>/<tag>.recompute.pt
  summary.json
```

The manifest records:

- Exact command.
- Git commit and dirty state.
- Python, PyTorch, and NumPy versions.
- Device, dtype, and platform.
- Seed and initial state.
- Checkpoint arguments.
- Selected case and trigger parameters.
- Start/end time and exit status.

The summary includes:

- Expected and observed tags.
- Pair count.
- Exact matches.
- Same-metadata value mismatches.
- Metadata mismatches.
- Missing and duplicate events.
- First mismatch.
- Gradient comparison.
- Whether the default PyTorch check raised.
- Whether trigger removal restored equality.
- Final `PASS`, `FAIL`, or `BLOCKED` result.

The CLI exits:

- `0` when every selected gate passes.
- `1` for a scientific or coverage failure.
- `2` for invalid configuration or environment setup failure.

## 4. Verification and completion gates

Automated tests cover:

- Clean checkpoint/no-checkpoint equality.
- Independent forward and gradient formulas.
- Original/recompute phase labeling.
- Stable pair identities.
- Missing and duplicate tag rejection.
- Exact match and same-metadata value mismatch reporting.
- Scalar, empty, NaN, infinity, and signed-zero comparisons.
- Correct first-differing-index reporting.
- All five correct arms.
- All five broken arms and both RNG variants.
- Proof that each broken original matches its correct original.
- Trigger removal and state reset.
- E2 tag, pack, unpack, and recomputation counts.
- Proof that enabling hooks does not change E0.
- Artifact reload and summary consistency.
- CLI exit codes.

Required run sequence:

1. Pin the local `uv` environment and exact stable PyTorch version.
2. Run all unit tests.
3. Run E0 three times on the Apple Silicon Mac CPU.
4. Run every E1 correct/broken pair three times.
5. Run E2 and record either complete public-hook coverage or an honest insufficiency result.
6. After the CPU suite is stable, repeat E0–E2 in FP32 inside one scheduled Condor GPU job without changing experiment logic.
7. Have completed results analyzed and then append the verdict, failures, artifacts, and next action to the research log.

When implementation begins, RESEARCH_LOG.md will receive a new decision entry stating:

- E0–E2 have been restored as the active starter phase.
- The production-shaped plan remains saved in Notion and deferred.
- The simplified E0–E2 definitions above replace the confusing former numbering.
- Every run and failure will be appended rather than overwritten.

E0–E2 are complete when:

- The clean baseline is independently correct.
- All five correct examples remain silent.
- All five broken families produce the intended same-metadata mismatch and gradient difference.
- Exact tensor artifacts show precisely where each difference starts.
- Removing each trigger restores equality.
- E2 gives a defensible yes/no answer about public-hook coverage.
- The entire suite can be repeated from documented commands.

Passing E0–E2 means:

> We have a trustworthy small test bench and know whether our first capture approach works within its declared scope.
> 

It does not yet mean:

- The issue naturally occurred in real pretraining.
- The hooks cover arbitrary PyTorch programs.
- A production training pipeline was validated.
- The GPU fingerprint exists.
- TorchTitan, compilation, selective checkpointing, distributed training, BF16, or native FP8 are supported.
- The detector has acceptable production overhead.

Those remain part of the separate production-shaped plan already saved in Notion.
