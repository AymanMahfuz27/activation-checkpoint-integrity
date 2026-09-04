# Research Log

This is the canonical chronological memory for Activation Checkpoint Integrity.
It records every research-relevant action, experiment, failure, success,
decision, blocker, and planned action. It must never contain credentials or
other secrets.

## Operating contract

- Read this file before planning or doing material project work.
- Consult the exact Notion page titled `MOOTAZ PROJECT` and the relevant child
  pages before material research decisions.
- Record planned work before it starts.
- Record the outcome immediately after each material action and before starting
  the next material action.
- Record failures, inconclusive results, abandoned attempts, and setup problems
  with the same care as successes.
- Never rewrite executed experiment history. Remove abandoned plan-only entries
  only when the user explicitly directs it to keep the active context accurate.
- Record observed facts separately from interpretations. Scientific
  interpretations and experiment verdicts require an Analyst review.
- Routine read-only checks may be grouped into one evidence line. Every
  research-relevant action and state-changing remote action must appear.
- Never record passwords, tokens, private keys, secret values, or
  credential-bearing commands. Use `[REDACTED_SECRET]` if a security-relevant
  event must be noted.

## Current state

- **Last updated**: 2026-09-03 23:15 CDT
- **Research phase**: Active starter phase — E0–E2 exact-capture calibration
  on controlled tiny checkpoint fixtures
- **Repository commit**: `b4afab63918d1a41dfff3e9e8a22397b28b10bc5`
  on `main`; `AGENTS.md` and `RESEARCH_LOG.md` are currently untracked
- **Research objective**: Detect activation-checkpoint recomputation value
  changes before model or optimizer state is mutated
- **Current baseline**: E0 implementation smoke-tested once on CPU at ignored
  artifact `artifacts/starter/e0-counter-correct-20260904T041106.752118Z-20a3c56b`;
  the required clean-commit three-run sequence remains pending
- **Current experiment**: E0, E1, and E2 are implemented and the 19-test suite
  passes; required clean-commit CPU experiment execution is next
- **Best validated result**: None; no scientific experiment has run
- **Compute status**: UTCS Condor access and GPU inventory verified; scheduled
  CUDA/PyTorch execution not yet verified; TACC not configured
- **Active blockers**: None for the CPU sequence; the Condor environment and
  Pascal compatibility remain unverified
- **Immediate next action**: Commit and push the verified starter
  implementation, then run the required clean-commit E0, E1, and E2 CPU
  sequence before preparing the scheduled GPU job
- **Canonical Notion context**: `MOOTAZ PROJECT`, page ID
  `3c65d66f-2c23-80c6-879f-e0aff5e92706`
- **Last synchronized with Notion**: The user explicitly selected `Starter
  E0–E2 Plan: Five Tiny Checkpoint Failures and Capture Validation`, page ID
  `3d05d66f-2c23-8068-b52a-ca774781771e`, last edited
  `2026-09-03T04:15:45.303Z`, as this week's active scope. The separately
  preserved production-shaped plan, page ID
  `3d05d66f-2c23-80d1-882f-ea6c45c89a7e`, is deferred.

## Planned actions

| ID | Priority | Action | Reason | Dependencies | Status |
|---|---:|---|---|---|---|
| A001 | 1 | Commit and push `AGENTS.md` and `RESEARCH_LOG.md` when authorized | Makes the operating rules available to GitHub and remote agents | User authorization to commit/push | Ready |
| A002 | 2 | Pull the documentation commit into the `darmok` checkout | Keeps remote work under the same instructions | A001 and stable CPU suite | Deferred until Condor phase |
| A003 | 2 | Pin compatible Python, PyTorch, and CUDA versions and create a dedicated remote environment | Existing base Conda environment is not reproducible | A016 and stable CPU suite | Deferred until Condor phase |
| A004 | 2 | Add a metadata-printing Condor runner and submit file | Required for reproducible scheduled work | A003 | Deferred until Condor phase |
| A005 | 2 | Run a one-GPU Condor probe | Verify driver, CUDA, GPU, memory, compute capability, and PyTorch CUDA availability | A004 | Deferred until Condor phase |
| A006 | 2 | Define the first falsifiable hypothesis, baseline, metrics, and falsification criterion | No scientific run should start without a preregistered contract | Notion plan and source audit | Completed |
| A007 | 3 | Verify and document TACC access, allocation, paths, scheduler, and environment | Required for final modern-GPU evaluation | Live TACC access/allocation | Blocked |
| A008 | — | Select and exactly replay R0 in its documented pinned environment | Preserved production-shaped evidence track | A003 and compatibility review | Deferred by L0007 |
| A009 | — | Build the genuine small decoder-only LM pipeline and run M0.1 | Preserved production-shaped clean oracle | A003 plus frozen data/model contract | Deferred by L0007 |
| A010 | — | Add one controlled mutable-state fault to the same LM and run M0.2 | Preserved production-shaped causal chain | M0.1 passes | Deferred by L0007 |
| A011 | — | Preserve R0's natural trigger inside the genuine LM and run M0.3 | Preserved production-shaped natural-trigger proof | R0 and M0.1 | Deferred by L0007 |
| A012 | — | Add the remaining fault families one at a time | Preserved production-shaped breadth work | Production-shaped Milestone 0 | Deferred by L0007 |
| A013 | — | Measure production capture and saved-tensor coverage | Defines the honest supported production scope | Production-shaped Milestone 0 and A012 | Deferred by L0007 |
| A014 | — | Implement and validate position-sensitive GPU fingerprints | Tests the proposed low-memory detector against the exact oracle | A013 | Deferred by L0007 |
| A015 | — | Integrate one pinned TorchTitan workload | Tests external validity after detector work | Modern GPU access and later detector milestones | Deferred by L0007 |
| A016 | 1 | Pin the local `uv` environment and exact stable PyTorch and NumPy versions | Makes starter results reproducible | None | Completed |
| A017 | 1 | Implement only the E0–E2 starter package, CLI, artifact writer, and focused tests specified in L0007 | Builds the selected weekly scope without production-shaped expansion | A016 | Completed |
| A018 | 1 | Run the full automated test suite, then make E0 pass | E0 is the trust gate for the fixture and comparator | A017 | Completed |
| A019 | 1 | Run E0 three times, every E1 control/fault pair three times including both RNG variants, and E2 on the Apple Silicon CPU | Produces the required repeatable calibration and coverage evidence | A018 | Blocked on E0 pass |
| A020 | 1 | Analyze and append every CPU result, failure, artifact path, and verdict | Results cannot drive later work until reviewed and archived | A019 | Blocked on completed runs |
| A021 | 2 | Repeat unchanged E0–E2 logic in FP32 inside one scheduled Condor GPU job | Checks portability after the CPU suite is stable | A020 plus synchronized commit and pinned remote environment | Planned after CPU suite |

## Experiment index

| ID | Date | Hypothesis | Baseline | Status | Verdict | Record |
|---|---|---|---|---|---|---|
| E0 | 2026-09-03 | The fixed clean fixture, exact comparator, and independent formulas agree across no-checkpoint, original, and recompute execution | Fixed literal CPU `float64` no-checkpoint run and independent formulas | Implementation smoke PASS; required clean-commit repetitions pending | Pending Analyst | L0007, L0009 |
| E1 | 2026-09-03 | Each of five hidden-state fault families causes a same-metadata value mismatch first at `h` and a gradient difference, while its control and trigger-disabled arm remain exact | E0 plus a fresh-process correct arm with identical tensors, seeds, and state | Automated coverage passes; required clean-commit repetitions pending | Pending Analyst | L0007, L0009 |
| E2 | 2026-09-03 | Public `context_fn` and `saved_tensors_hooks` capture each explicitly backward-relevant `h`, `g`, and `y` exactly once per phase without changing behavior | E0 no-hook result and checkpoint baseline without observational hooks | Automated coverage path exercised; required clean-commit run pending | Pending Analyst | L0007, L0009 |
| R0 | 2026-09-02 | A documented natural activation-checkpoint bug reproduces in its pinned original environment without a project-invented fault | Exact upstream safe/failing comparison | Preserved; deferred | — | L0006, L0007 |
| M0.1 | 2026-09-02 | Clean no-checkpoint and checkpointed genuine LM pretraining agree | Identical data/model/optimizer state | Preserved; deferred | — | L0006, L0007 |
| M0.2 | 2026-09-02 | A controlled mutable-state fault changes a recomputed LM activation, gradient, and optimizer update while metadata remain equal | M0.1 clean oracle | Preserved; deferred | — | L0006, L0007 |
| M0.3 | 2026-09-02 | R0's documented natural trigger produces the same failure class inside the genuine LM pipeline | R0 and M0.1 | Preserved; deferred | — | L0006, L0007 |
| M1 | 2026-09-02 | Remaining fault families reproduce one at a time in the LM or bounded fixtures | Production-shaped Milestone 0 | Preserved; deferred | — | L0006, L0007 |
| M2 | 2026-09-02 | Capture path covers every site in each claimed checkpoint pattern | Production-shaped Milestone 0 evidence | Preserved; deferred | — | L0006, L0007 |
| M3 | 2026-09-02 | Exact debug checking and position-sensitive GPU fingerprints detect oracle-confirmed mismatches | M1 and M2 | Preserved; deferred | — | L0006, L0007 |
| M4 | 2026-09-02 | Longer LM runs establish downstream consequence and reproducibility | M3 | Preserved; deferred | — | L0006, L0007 |
| M5 | 2026-09-02 | TorchTitan and modern-GPU validation establish external compatibility and overhead | M3 and modern GPU access | Preserved; deferred | — | L0006, L0007 |

Infrastructure probes and environment checks are log entries, not scientific
experiments, unless they test a preregistered research hypothesis.

## Append-only ledger

### L0001 — 2026-08-31 11:17 CDT — Repository initialized

- **Type**: setup / context
- **Status**: completed
- **Objective**: Establish a minimal public repository for Activation
  Checkpoint Integrity.
- **Context consulted**: Project README and initial repository configuration.
- **Repository state**: Commit
  `b4afab63918d1a41dfff3e9e8a22397b28b10bc5`, branch `main`, subject
  `Initialize project`.
- **Action**: Created the public GitHub repository and initial local checkout.
- **Evidence**:
  - GitHub: `https://github.com/AymanMahfuz27/activation-checkpoint-integrity.git`
  - Local checkout:
    `/Users/ayman/Documents/ChatGPT/Mootaz Project/activation-checkpoint-integrity`
  - Tracked files at the commit: `.gitignore` and `README.md`
- **Outcome**: Repository initialized with a truthfully planning-stage README.
  No implementation, experiment configuration, runner, or result exists.
- **Failure or caveat**: None observed in the tracked repository. Research and
  implementation claims remain unmeasured.
- **Decision / lesson**: Keep public claims limited to planned scope until
  reproducible measurements exist.
- **Next actions**: Establish compute access and reproducible execution rules.
- **Secrets**: No secret material recorded.

### L0002 — 2026-09-02 CDT — UTCS Condor access and remote checkout

- **Type**: setup / failure / decision
- **Status**: completed with open environment work
- **Objective**: Determine whether project development can begin on UTCS Condor
  before TACC is configured.
- **Context consulted**: UTCS Condor documentation, local SSH configuration,
  local Git repository, remote submit host, and live HTCondor ClassAds.
- **Repository state**:
  - Local tracked commit:
    `b4afab63918d1a41dfff3e9e8a22397b28b10bc5`
  - Remote checkout: `/u/ayman27/activation-checkpoint-integrity`
  - Remote branch: clean `main` tracking `origin/main`
  - Remote commit matched the local tracked commit
- **Action and evidence**:
  - Verified passwordless SSH to `ayman27@darmok.cs.utexas.edu`.
  - Cloned the repository from GitHub into the remote checkout.
  - Verified HTCondor 9.0.8 at `/lusr/opt/condor/bin`.
  - Live pool snapshot: 1,024 total compute slots, 7 claimed and 1,017
    unclaimed at query time.
  - Live `eldar` snapshot: 42 nodes and 83 advertised usable GPU slots, all
    `Unclaimed/Idle` at query time; 26 GTX 1080 GPUs and 57 GTX 1080 Ti GPUs.
  - GPU ClassAds reported compute capability 6.1 and approximately 8 GB on GTX
    1080 or 11 GB on GTX 1080 Ti.
  - Remote tools: Git at `/usr/bin/git`, tmux at `/usr/bin/tmux`, and Python
    3.13.5 at `/u/ayman27/miniconda3/bin/python3`; `uv` was absent.
- **Outcome**: Condor is usable now for setup, correctness work, controlled
  fault injection, and preliminary FP32/FP16 experiments.
- **Failures and caveats**:
  - Initial SSH attempts used the wrong automatically selected username and
    failed. The verified UTCS username is `ayman27`.
  - `aries.cs.utexas.edu` timed out. `darmok.cs.utexas.edu` is the verified
    submit host.
  - The public SSH key was initially added to the Mac's local
    `authorized_keys`, which did not configure UTCS. It was then added to the
    remote account and passwordless access was verified.
  - The base Conda environment is not a pinned project environment.
  - CUDA and PyTorch have not been tested inside a scheduled GPU job.
  - Pascal GPUs are not suitable as the sole evidence for BF16, FP8, Tensor
    Core, large-model, distributed-scaling, or final modern-GPU overhead
    claims. GPU availability is dynamic.
- **Decision / lesson**: Start development on Condor now, but reserve final
  A100/H100-class evaluation for verified TACC access.
- **Next actions**: Pin dependencies, add a scheduled probe, and verify CUDA and
  PyTorch on an `eldar` GPU.
- **Secrets**: No secret material recorded.

### L0003 — 2026-09-02 CDT — Remote-execution runbook added

- **Type**: setup / decision / success
- **Status**: completed locally; not committed or pushed
- **Objective**: Make Condor execution traceable and leave a safe placeholder
  for future TACC configuration.
- **Context consulted**: Live SSH/Condor checks, repository README, and global
  remote-execution rules.
- **Repository state**: Tracked commit remained
  `b4afab63918d1a41dfff3e9e8a22397b28b10bc5`; `AGENTS.md` was added as an
  untracked local file.
- **Action**: Added `AGENTS.md` with repository synchronization rules,
  passwordless Condor access, `remote-run` usage, pool and queue inspection,
  environment status, a one-GPU GTX 1080 Ti submit template, reproducibility
  metadata, failure handling, and a reserved TACC section.
- **Evidence**: The exact submit template extracted from `AGENTS.md` was parsed
  successfully by `/lusr/opt/condor/bin/condor_submit -dry-run` on `darmok`.
  The generated ClassAd contained the intended CPU, memory, disk, GPU,
  `GPUSlot`, `Eldar`, `GTX1080Ti`, group, project, and project-description
  requirements.
- **Outcome**: A verified runbook exists locally. No job was submitted and no
  scientific result was produced.
- **Failure or caveat**: The runbook is not available in GitHub or the remote
  checkout until it is committed, pushed, and pulled.
- **Decision / lesson**: Never guess TACC details; add them only after live
  verification.
- **Next actions**: Add canonical research-memory rules and version the
  documentation when authorized.
- **Secrets**: No secret material recorded.

### L0004 — 2026-09-02 16:33 CDT — Canonical context and research memory initialized

- **Type**: context / setup / plan
- **Status**: completed locally; not committed or pushed
- **Objective**: Establish a permanent, append-only record of every material
  research action, experiment, failure, success, blocker, decision, and planned
  action.
- **Context consulted**:
  - `AGENTS.md`, `README.md`, current Git state, and Git history
  - Notion workspace search for the exact title `MOOTAZ PROJECT`
  - Research-orchestrator archival protocol
- **Repository state**: Tracked commit remained
  `b4afab63918d1a41dfff3e9e8a22397b28b10bc5`; `AGENTS.md` and
  `RESEARCH_LOG.md` are local documentation additions.
- **Action**:
  - Resolved one exact Notion page titled `MOOTAZ PROJECT` with ID
    `3c65d66f-2c23-80c6-879f-e0aff5e92706` and URL
    `https://app.notion.com/p/3c65d66f2c2380c6879fe0aff5e92706?pvs=204`.
  - Notion search reported the root page last edited at
    `2026-08-31T16:07:00.000Z`; its contents were not fetched during this
    logging setup.
  - Added mandatory Notion/context and research-log rules to `AGENTS.md`.
  - Created this log with current state, planned actions, an empty experiment
    index, the history known so far, and a reusable ledger template.
- **Outcome**: Future agents have one stable project-context page and one
  versioned research memory. The scientific experiment record remains empty.
- **Failure or caveat**: The new documentation is local until versioned and
  synchronized. Notion content can change and must be re-read before material
  decisions.
- **Decision / lesson**: Notion governs high-level context and user decisions;
  exact commits, code, scheduler records, raw artifacts, and measurements
  govern implementation and experimental facts. Conflicts must be logged and
  surfaced.
- **Next actions**: A001 through A007 in the planned-actions table.
- **Secrets**: No secret material recorded.

### L0005 — 2026-09-02 16:34 CDT — Exact project plan and primary evidence reviewed

- **Type**: context / research / decision
- **Status**: completed
- **Objective**: Ground the first milestone and long-term architecture in the
  exact Notion plan, current repository, and current primary-source evidence.
- **Context consulted**:
  - Notion page `Activation Checkpoint Project Plan (working name valueguard)`,
    ID `3c95d66f-2c23-8125-9018-cb0b03aac718`, last edited 2026-08-31
  - Notion child page `About the Project`, ID
    `3c85d66f-2c23-80b4-9235-c9fd3d0c2031`, last edited 2026-08-27
  - Current `README.md`, `AGENTS.md`, Git state, and prior project history
  - PyTorch main `torch/utils/checkpoint.py`, inspected 2026-09-02
  - PyTorch issue `#166926` and NVIDIA Transformer Engine issue `#1190`,
    inspected 2026-09-02
- **Repository state**: Tracked commit remained
  `b4afab63918d1a41dfff3e9e8a22397b28b10bc5`; local documentation remained
  untracked; no implementation or experiment was run.
- **Action**:
  - Confirmed the Notion Semester 1 boundary: detect mismatches and stop the
    unsafe step; do not repair state.
  - Confirmed the planned five fault families, exact-comparison reference,
    saved-tensor coverage gate, GPU fingerprint work, overhead measurements,
    and later TorchTitan evaluation.
  - Verified that current PyTorch source warns that different forward and
    recomputation behavior can cause silently incorrect gradients and that the
    default determinism check compares shape, dtype, and device, not values.
  - Noted that PyTorch main currently has a `respect_saved_tensors_hooks`
    option relevant to selective activation checkpointing. This is an API
    observation, not evidence of complete coverage.
  - Reviewed current external cases: PyTorch issue `#166926` describes a
    compile-cache-dependent recomputation path; Transformer Engine issue
    `#1190` reports a later gradient difference between checkpointed and
    uncheckpointed FP8 runs.
- **Evidence**:
  - `https://app.notion.com/p/3c95d66f2c2381259018cb0b03aac718`
  - `https://app.notion.com/p/3c85d66f2c2380b49235c9fd3d0c2031`
  - `https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py`
  - `https://github.com/pytorch/pytorch/issues/166926`
  - `https://github.com/NVIDIA/TransformerEngine/issues/1190`
- **Outcome**: The problem class is externally supported, but this repository
  has not reproduced it. Coverage, detection, compatibility, and overhead
  remain unmeasured.
- **Failure or caveat**: External reports establish relevance but do not prove
  this project's causal chain or proposed checker. The historical TACC request
  was incomplete and does not establish current access.
- **Decision / lesson**: The first milestone must reproduce a documented,
  non-project-invented activation-checkpoint failure during genuine
  decoder-only LM pretraining and trace it from the recomputed activation
  matrix through the gradient and optimizer update. Public hook coverage must
  be measured before any complete-coverage claim.
- **Next actions**: R0 and M0.1–M0.3 as preregistered in L0006.
- **Secrets**: No secret material recorded.

### L0006 — 2026-09-02 16:53 CDT — Genuine LM reproduction preregistered as Milestone 0

- **Type**: plan / decision / experiment preregistration
- **Status**: completed planning; implementation not started
- **Objective**: Make the first research milestone a genuine small decoder-only
  language-model pretraining run that reproduces a documented,
  non-project-invented activation-checkpoint failure before detector or repair
  work begins.
- **Context consulted**: L0005, the user's Milestone 0 clarification, the
  revised R0/M0 contract, PyTorch issue `#166926`, Transformer Engine issue
  `#1190`, and current NVIDIA Transformer Engine hardware documentation.
- **Repository state**: Tracked commit
  `b4afab63918d1a41dfff3e9e8a22397b28b10bc5` on `main`; `AGENTS.md` and
  `RESEARCH_LOG.md` were untracked. No scientific implementation or experiment
  had run.

- **Milestone 0 eligibility contract**:
  - Use a real, provenance-recorded tokenized text corpus, not literal tensor
    fixtures or synthetic classifier inputs.
  - Record corpus source and usage basis, immutable hash, preprocessing,
    tokenizer type/configuration/hash, vocabulary size, and packed sample order.
  - Use a decoder-only transformer with causal self-attention, token and
    positional representations, transformer blocks, normalization, and an LM
    output head.
  - Train with next-token cross-entropy over real token sequences.
  - Execute forward, LM loss, backward, optimizer update, and activation
    checkpoint recomputation inside actual transformer blocks.
  - Run a preregistered multi-step optimizer-driven pretraining interval. A
    single isolated toy step does not satisfy the milestone.
  - Record model/config manifest, initialization, RNG states, batch construction,
    sequence length, optimizer and schedule, precision, accumulation, clipping,
    checkpoint pattern, processed tokens, exact run length, and all artifacts.
  - Paired arms must start from identical model, optimizer, RNG, data-order, and
    scheduler state.

- **Evidence track R — documented natural bug**:
  - **Experiment ID**: R0
  - **Objective**: Reproduce one documented natural activation-checkpoint
    recomputation bug without replacing its trigger with a project-invented
    fault.
  - **Required preregistration**: Name the exact upstream issue/report,
    repository and commit or release, dependency lock, Python/PyTorch/CUDA
    versions, hardware constraints, seed, command, input/configuration,
    documented trigger, expected symptom, and comparison baseline.
  - **Method**: Replay the documented case in its own pinned environment as
    faithfully as available artifacts allow. Preserve the original trigger and
    distinguish faithful replay from any later adaptation.
  - **Pass evidence**: The recorded symptom appears under the documented trigger
    and disappears under the documented safe, disabled-trigger, or corrected
    comparison when one exists.
  - **Failure handling**: Dependency loss, unavailable hardware, incomplete
    upstream artifacts, or failure to reproduce is recorded as setup failure or
    inconclusive evidence, not silently converted into a synthetic reproduction.
  - **Current candidate order**: Check PyTorch issue `#166926` compatibility and
    replay feasibility first. Transformer Engine issue `#1190` remains a later
    candidate because official documentation limits FP8 support to Hopper, Ada,
    and Blackwell GPUs; the currently verified GTX 1080 Ti cannot supply that
    evidence.
  - **Boundary**: R0 establishes that a natural documented failure occurs in its
    pinned setting. It does not by itself validate this project's future guard.

- **Evidence track M — genuine LM pretraining**:
  - **Candidate smallest credible harness**: a pinned public text corpus;
    recorded tokenizer; roughly four decoder blocks, width 256, four attention
    heads, MLP width 1024, context 128, and approximately 5–8M parameters;
    next-token cross-entropy; fixed token batches; FP32; dropout disabled for the
    first study; one middle block checkpointed; and at least 20–50 optimizer
    steps. These are proposal ranges and must be frozen before execution.
  - **Diagnostic activation**: capture the pre-GELU MLP matrix
    `A = fc1(layer_norm(residual))` in the checkpointed middle block, shaped
    `[batch, sequence, 4 * model_width]`. GELU backward depends on this matrix.
  - **Diagnostic capture only**: use `context_fn` solely to label original versus
    recomputation and temporarily store the full tagged matrix. This is
    instrumentation, not the proposed fingerprint/checker design.
  - **Exact evidence**: record `torch.equal`, complete metadata, differing
    element count, first differing index, maximum absolute difference, relative
    L2 difference, and a saved tensor artifact or matrix slice.

  - **Experiment ID**: M0.1 — clean checkpoint oracle
  - **Hypothesis**: With hidden state controlled, clean no-checkpoint and clean
    activation-checkpointed LM pretraining agree within preregistered tolerances.
  - **Comparison**: Run both arms from identical initial training state and token
    order.
  - **Required evidence**: At preregistered steps, compare LM loss, selected
    original/recomputed activation matrices, parameter gradients, optimizer
    deltas, optimizer-state transitions, and final model-state hashes.
  - **Falsification**: Any unexplained clean mismatch invalidates the LM oracle
    and blocks M0.2.

  - **Experiment ID**: M0.2 — controlled mutable-state fault in the LM
  - **Hypothesis**: A realistic call-counter or mutable-buffer dependency inside
    a checkpointed transformer path can leave the original LM forward and tensor
    metadata unchanged while changing a backward-relevant recomputed activation
    matrix, producing a wrong gradient and wrong optimizer update.
  - **Controlled change**: Introduce one call-counter or mutable-state dependency.
    The original invocation uses clean state and recomputation observes changed
    state. All other paired-run state remains fixed.
  - **Causal evidence**:
    1. The faulty original forward matrix, logits, and LM loss match M0.1 before
       backward.
    2. Checked metadata match and the default metadata check does not report the
       value-only difference.
    3. The exact recomputed activation matrix differs.
    4. A backward-relevant parameter gradient differs from M0.1.
    5. Starting from the same pre-step snapshot, the unguarded faulty run applies
       an optimizer update different from the clean expected update.
    6. Disabling only the controlled fault restores activation, gradient, and
       update equality.
  - **Boundary**: M0.2 proves deterministic occurrence and consequence in real LM
    pretraining, but does not establish natural prevalence in an unmodified
    training stack.

  - **Experiment ID**: M0.3 — natural-trigger LM transplant
  - **Status**: required after R0 and M0.1.
  - **Hypothesis**: R0's documented trigger can be preserved without changing
    its essential semantics and produces the same class of recomputation
    mismatch inside the genuine LM pipeline.
  - **Pass evidence**: Preserve the natural trigger; show matching metadata and
    an exact original-versus-recomputed activation-matrix mismatch; show the
    resulting gradient divergence and wrong optimizer update; and show that
    removing only the natural trigger restores equality.
  - **Blocked outcome**: If the trigger cannot be transplanted because of
    hardware, dependencies, missing upstream artifacts, or incompatible
    semantics, record the exact incompatibility and mark Milestone 0 blocked.
    Do not relabel M0.2's controlled fault as natural evidence, and do not begin
    detector or repair work unless the user explicitly approves a narrower
    research claim.

- **Milestone 0 hard completion gate**:
  - R0, M0.1, M0.2, and M0.3 have reproducible records and Analyst verdicts.
  - M0.1 proves the genuine LM checkpoint harness is clean.
  - M0.2 proves the complete activation-to-gradient-to-update causal chain in
    the LM.
  - M0.3 preserves the natural trigger in the genuine LM and proves matching
    metadata, activation-matrix mismatch, gradient divergence, and a wrong
    optimizer update. Removing only that trigger restores equality.
  - All claims cite immutable code, data/tokenizer manifests, environments,
    commands, seeds, raw artifacts, and paired-state evidence.
  - Loss curves may support downstream impact but cannot substitute for exact
    activation, gradient, update, and state-mutation evidence.

- **Outcome**: Planning only. No implementation or scientific result was
  produced.
- **Failure or caveat**: The exact R0 bug, replay environment, corpus, tokenizer,
  LM configuration, optimizer, run length, tolerances, and artifact layout must
  be frozen before execution. Proposed model and token ranges remain planning
  ranges, not commitments.
- **Decision / lesson**: R0 establishes that a documented natural bug exists in
  its original setting. M0.1 proves the LM harness is clean. M0.2 provides a
  controlled causal benchmark. Only required M0.3 proves that a preserved
  natural trigger causes the full failure during genuine LM pretraining.
- **Next actions**: A003, A008, A009, A010, and required A011. Do not begin
  M1–M5 detector or repair work before the hard gate passes.
- **Secrets**: No secret material recorded.

### L0007 — 2026-09-03 22:56 CDT — E0–E2 starter calibration restored as active weekly scope

- **Type**: context / plan / decision / experiment preregistration
- **Status**: in progress; implementation and experiments not started
- **Objective**: Execute the selected starter plan end to end this week: build
  a trustworthy exact-comparison fixture, validate five controlled checkpoint
  fault families against clean controls, and determine whether the proposed
  public PyTorch hook path observes every explicitly required tensor.
- **Context consulted**:
  - User-selected Notion page `Starter E0–E2 Plan: Five Tiny Checkpoint
    Failures and Capture Validation`, page ID
    `3d05d66f-2c23-8068-b52a-ca774781771e`, last edited
    `2026-09-03T04:15:45.303Z`; its full attached plan text was supplied for
    this work.
  - Separately preserved production-shaped plan, page ID
    `3d05d66f-2c23-80d1-882f-ea6c45c89a7e`.
  - `AGENTS.md`, `README.md`, this log through L0006, current Git state, and
    tracked commit history.
- **Repository state**: Branch `main` at tracked commit
  `b4afab63918d1a41dfff3e9e8a22397b28b10bc5`, matching `origin/main` at the
  time of this entry. `AGENTS.md` and `RESEARCH_LOG.md` were untracked. No
  starter implementation, environment lock, generated artifact, or scientific
  result existed.
- **User decision and scope reconciliation**:
  - The attached E0–E2 starter plan is this week's active implementation and
    experiment scope.
  - The simplified E0–E2 names replace the older E00–E04 starter numbering.
  - L0006 and its R0/M0.1–M0.3 production-shaped sequence remain intact as
    historical planning. L0006 is superseded only as the current execution
    order; its plan is preserved and deferred to a separate later production
    phase.
  - E0–E2 calibrate capture, comparison, fault localization, false-positive
    behavior, and public-hook coverage. Passing them does not establish a
    naturally occurring pretraining failure, arbitrary-program coverage,
    production integration, GPU fingerprints, acceptable overhead,
    TorchTitan/compile/distributed support, BF16 support, or native FP8 support.
  - Every run, failure, setup limitation, artifact, analysis, and correction
    will be appended to this ledger. No later run may overwrite an earlier run
    or inconvenient result.

- **Shared execution contract**:
  - Use non-reentrant activation checkpointing with `use_reentrant=False`,
    `early_stop=False`, `preserve_rng_state=True`,
    `determinism_check="default"`, and a `context_fn` that labels `original`
    and `recompute` execution.
  - Give every pair the stable logical identity
    `experiment/case/checkpoint_call/tag/occurrence` and every event the identity
    `run_id/pair_id/phase`. Never pair by object or storage address, shape
    alone, or a value hash.
  - Save complete original and recomputed tensors. For each pair, report shape,
    dtype, device, layout, stride, exact equality, differing-element count,
    first differing index, maximum absolute difference, relative L2 difference,
    and both tensor paths.
  - Run each correct and broken arm in a fresh process. Record exact command,
    Git commit and dirty state, Python/PyTorch/NumPy versions, platform, device,
    dtype, seed and initial state, checkpoint arguments, selected case and
    trigger parameters, timestamps, exit status, events, comparisons,
    gradients, and final gate result.
  - Use exit code `0` only when every selected gate passes, `1` for a scientific
    or coverage failure, and `2` for invalid configuration or an environment
    setup failure.

- **Experiment E0 — clean checkpoint baseline and exact comparator**:
  - **Falsifiable hypothesis**: On fixed literal CPU `float64` tensors, the
    checkpointed function executes exactly once as `original` and once as
    `recompute`; the tagged `h`, `g`, and `y` tensors pair one-to-one and match
    exactly; and checkpointed and no-checkpoint outputs, losses, and gradients
    equal independently calculated formulas.
  - **Baseline**: A no-checkpoint evaluation of
    `h = x @ W1 + b`, `g = h**2 + 0.5*h`, `y = g @ W2`, and
    `loss = sum(y * target)` using the same fixed inputs and parameters, plus
    separately implemented forward and gradient formulas that do not use the
    checkpointed path as their oracle.
  - **Isolated change**: Wrap the fixed matrix function in the shared
    non-reentrant checkpoint configuration and directly tag complete `h`, `g`,
    and `y` tensors in both phases.
  - **Metrics**: Phase execution counts; expected, observed, missing, and
    duplicate tags; pair count; complete metadata; exact matches and
    same-metadata value mismatches; differing-element count and first differing
    index; maximum absolute and relative L2 differences; output and loss
    equality; every input and parameter gradient comparison; independent
    formula agreement; tensor and summary artifact paths.
  - **Falsification gate**: Any phase count other than one, missing/duplicate or
    ambiguous tag, non-exact `h`/`g`/`y` pair, checkpoint/no-checkpoint output,
    loss, or gradient mismatch, or disagreement with an independent formula
    falsifies E0 and blocks E1 and E2 until the fixture or comparator is fixed.

- **Experiment E1 — five controlled faults and correct controls**:
  - **Falsifiable hypothesis**: For each preregistered family, a correct arm has
    zero tensor mismatches, while changing only hidden state after the original
    forward makes the broken recomputation first differ at `h` without changing
    shape, dtype, device, layout, or stride; at least one gradient differs; and
    disabling only that trigger restores exact tensor and gradient equality.
  - **Baseline**: E0 plus a fresh-process correct arm initialized with tensors,
    seeds, and state identical to its broken arm. The broken arm's original
    output, loss, and tagged tensors must exactly equal the correct original
    before backward begins.
  - **Isolated changes and expected trigger values**:
    1. Changing counter: correct computation ignores a logging counter; broken
       original uses counter `0` and scale `1.0`, while recomputation uses
       counter `1` and scale `1.25`.
    2. Mutable registered buffer: correct buffer remains `1.0` through
       backward; broken state advances from `1.0` to `1.25` after the original
       read. Read it as a non-differentiable scalar before mutation to avoid an
       unrelated autograd version-counter failure.
    3. Python/NumPy randomness: run both required variants. The correct arm
       samples once outside the checkpointed function and passes fixed data;
       the broken arm samples inside so original and recompute consume
       consecutive `random.random()` or `numpy.random.random()` values.
    4. Precision policy: both phases compute an FP32 and a BF16-rounded matmul;
       the correct arm selects FP32 in both phases, while the broken original
       selects FP32 and recompute selects the BF16-rounded result converted to
       FP32, preserving operator structure and tagged-boundary metadata.
    5. FP8-style delayed scaling: a toy emulation, not native-FP8 evidence. The
       correct arm freezes one quantize/dequantize scale; the broken original
       updates maximum-value history and recompute uses the advanced scale. A
       straight-through estimator preserves gradient flow.
  - **Metrics**: All E0 tensor-comparison fields; correct-versus-broken original
    equality; first mismatch tag and index; metadata equality; gradient
    equality/difference; whether PyTorch's default metadata check raised;
    trigger-disabled restoration; state reset; repeat-stable scientific results
    and saved tensor/comparison contents across three runs, excluding run IDs
    and timestamps; and family/variant gate status.
  - **Falsification gate**: E1 fails unless all five families pass, including
    both Python and NumPy RNG variants. Each correct arm must have zero
    mismatches; each broken original must match its correct original; the broken
    arm must finish without a default metadata-check rejection; its first
    mismatch must be `h` with unchanged metadata; at least one input or
    parameter gradient must differ; removing only the trigger must restore
    equality; complete tensors and the exact first differing index must be
    saved; and three seeded repeats must reproduce the same scientific result
    and saved tensor/comparison contents, excluding run IDs and timestamps. If
    the pinned CPU/PyTorch environment cannot execute the BF16 path correctly,
    record E1 as blocked by that environment limitation and do not substitute a
    different experiment.

- **Experiment E2 — public capture coverage**:
  - **Falsifiable hypothesis**: When `h`, `g`, and `y` pass through a
    `TaggedSave` autograd function that saves the actual tensor plus a unique
    semantic integer token, public `context_fn` and
    `torch.autograd.graph.saved_tensors_hooks` observe every tag exactly once in
    both original and recompute phases, with the expected unpack/access events,
    without changing values, gradients, phase counts, or recomputation.
  - **Baselines**: The direct in-fixture tag ledger is the coverage ground truth.
    Behavior baselines are an otherwise identical checkpoint run without
    observational saved-tensor hooks and the no-hook E0 output and gradients.
  - **Isolated change**: Enable only the public saved-tensor pack/unpack observer
    around the checkpointed `TaggedSave` fixture. Match hook events to direct
    tags by semantic token and stable logical identity.
  - **Metrics**: Original/recompute counts; direct tag counts; tagged pack,
    unpack/access, missing, duplicate, and ambiguously matched events; exact
    values; output and gradient equality to E0; hook/no-hook behavioral
    equality; whether hooks suppress or alter recomputation; and final coverage
    classification.
  - **Falsification gate**: A missing, duplicate, ambiguously ordered, or
    unmatched `h`/`g`/`y` event, missing expected unpack/access, changed value,
    output, gradient, phase count, or recomputation behavior falsifies E2. The
    required failure classification is `PUBLIC_HOOKS_INSUFFICIENT`; this is a
    useful coverage result that directs the later production design to an
    internal PyTorch pairing point rather than an incomplete public-hook path.

- **Implementation and artifact boundary**:
  - Create only `pyproject.toml`, `uv.lock`, `src/ac_integrity/cli.py`,
    `src/ac_integrity/starter/{fixture,capture,cases,runner}.py`, and
    `tests/test_starter.py`, apart from required package markers and existing
    research documentation.
  - Write each run under `artifacts/starter/<run_id>/` with `manifest.json`,
    `events.jsonl`, `comparisons.jsonl`, original/recompute tensor files, and
    `summary.json`. Generated artifacts remain ignored and must not be committed.
  - Required CLI surface: `uv run aci-starter run E0`,
    `uv run aci-starter run E1 --case all`, individual Python and NumPy RNG
    variants, `uv run aci-starter run E2`, and `uv run pytest`.
- **Required execution order**: Pin the local environment; run automated tests;
  run E0 three times on the Apple Silicon CPU; run every E1 correct/broken pair
  three times including both RNG variants; run E2; archive an Analyst verdict;
  then, only after the CPU suite is stable, repeat unchanged E0–E2 logic in FP32
  inside one scheduled Condor GPU job. E0–E2 completion requires reproducible
  documented commands and all gates above; it does not activate the deferred
  production-shaped plan.
- **Expected evidence**: Pinned dependency files, the bounded source and test
  files, exact CLI commands, passing/failing test output, one immutable artifact
  directory per attempted run, appended run entries including all failures, and
  an Analyst verdict for each completed experiment.
- **Outcome**: Planning and scope reconciliation only. No code was implemented,
  no environment was installed, no command was run, no artifact was generated,
  and no scientific result was observed in this entry.
- **Failure or caveat**: PyTorch API behavior and CPU BF16 support remain
  unverified in the pinned implementation environment. E2 may validly falsify
  the public-hook approach. Production-shaped evidence remains intentionally
  deferred.
- **Decision / lesson**: Build and validate the small exact oracle before using
  it to judge broader instrumentation. Treat an honest E2 insufficiency as a
  routing result, not as a failed research program.
- **Next actions**: A016, A017, and A018. Do not run E1 or E2 before E0 passes;
  do not begin the deferred production-shaped plan in this phase.
- **Secrets**: No secret material recorded.

## Ledger entry template

### L0008 — 2026-09-03 23:03 CDT — Starter implementation and execution begun

- **Type**: setup / implementation / experiment preparation
- **Status**: in progress
- **Objective**: Implement and execute only the L0007 E0–E2 starter
  calibration, first on the Apple Silicon CPU and then in one scheduled UTCS
  Condor GPU job after the CPU gates are stable.
- **Context consulted**: `AGENTS.md`, `README.md`, this log through L0007, the
  complete attached text of the user-selected Notion starter plan (page ID
  `3d05d66f-2c23-8068-b52a-ca774781771e`, last-edited time recorded in L0007),
  the `research-orchestrator` workflow, current Git state, and relevant prior
  repository memory.
- **Repository state**: `main` at
  `b4afab63918d1a41dfff3e9e8a22397b28b10bc5`, matching `origin/main`;
  `AGENTS.md` and `RESEARCH_LOG.md` are untracked pre-existing documentation.
- **Action**: Began the bounded package, exact comparator, isolated-arm runner,
  tests, dependency lock, and starter-only Condor job support specified in
  L0007. The production-shaped R0/M0 plan remains deferred and will not be
  implemented or run in this phase.
- **Dependency evidence**: Official PyTorch release/PyPI records confirm
  `torch==2.13.0` is published with CPython 3.13 macOS ARM64 and Linux x86-64
  wheels; current PyPI stable releases selected for the other direct
  dependencies are `numpy==2.5.2` and `pytest==9.1.1`. Availability and the
  resolved transitive graph will be proven by `uv lock` and `uv sync`.
- **Expected evidence**: A minimal config-driven implementation, exact lock,
  green tests, immutable ignored run artifacts, three deterministic CPU
  repetitions per required arm, an honest E2 coverage classification, exact
  Git commits and remote-main verification, and one scheduler-isolated GPU
  result or an exact scheduler/environment blocker.
- **Failure or caveat**: PyTorch 2.13 API behavior, Apple CPU BF16 execution,
  public-hook coverage, and Pascal compatibility with the chosen CUDA 12.6
  wheel remain unverified at this point.
- **Decision / lesson**: Pending implementation and measurements; scientific
  interpretation remains pending Analyst review.
- **Next actions**: Create and lock the implementation, run the smallest test
  loop until E0 passes, then execute E1, E2, and the separate Condor job in the
  L0007 order.
- **Secrets**: No secret material recorded.

### L0009 — 2026-09-03 23:15 CDT — Starter package locked, implemented, and test-green

- **Type**: setup / implementation / failure / success
- **Status**: completed
- **Objective**: Establish the smallest reproducible E0–E2 implementation and
  make the clean E0 trust gate pass before executing the required experiment
  sequence.
- **Context consulted**: L0007–L0008 and the complete attached starter plan.
- **Repository state**: Worktree based on
  `b4afab63918d1a41dfff3e9e8a22397b28b10bc5`; implementation and documentation
  are not yet committed, so the smoke artifact correctly records `dirty=true`.
- **Action**:
  - Added the bounded package/config files from L0007: exact fixed fixture and
    closed-form gradients; bitwise comparator with stable semantic pair/event
    identities; original/recompute contexts; full-tensor artifacts; all five
    E1 controllers; `TaggedSave` and public saved-tensor observation; isolated
    subprocess orchestration; CLI exit codes; and focused tests.
  - Pinned CPython `>=3.13,<3.14`, `torch==2.13.0`, `numpy==2.5.2`, and
    `pytest==9.1.1`; `uv lock` resolved 37 packages and `uv sync --frozen`
    installed the exact graph using CPython 3.13.15.
  - Ran one E0 CPU `float64` smoke and the focused/full automated test loops.
- **Evidence**:
  - `uv lock`; `uv sync --frozen`
  - `uv run aci-starter run E0` -> run ID
    `e0-counter-correct-20260904T041106.752118Z-20a3c56b`, result `PASS`, phase
    counts `1/1`, three exact pairs, and exact checkpoint/no-checkpoint/formula
    agreement; artifact under `artifacts/starter/`
  - Initial `uv run pytest -q` -> 12 passed and 7 failed. The failure was
    localized to requiring bit-exact equality between FP32 autograd gradients
    and separately ordered closed-form arithmetic in all E1 arms; this was a
    test/gate error because L0007 requires the independent exact formula gate
    for E0's `float64` fixture, while E1 correctness is defined against its
    same-state no-checkpoint baseline.
  - Corrected the E1 reference gate to compare checkpoint gradients exactly to
    the no-checkpoint autograd baseline and use a `1e-6` numerical check only
    for the separately ordered diagnostic formula; targeted E1 test -> 6
    passed; final `uv run pytest -q` -> 19 passed in 40.81 seconds.
- **Outcome**: E0's fixture/comparator trust gate passes in the smoke run; all
  required automated behavior is covered by 19 passing tests. Generated
  artifacts are ignored. Parameter count is N/A because the literal fixture
  has tensors but no model architecture or parameter-budget constraint.
- **Failure or caveat**: The first E0 artifact and test artifacts are
  implementation smoke evidence from a dirty worktree, not the preregistered
  clean-commit experiment sequence. E2's scientific coverage classification
  and all repeatability gates remain pending that sequence.
- **Decision / lesson**: The implementation is ready to version; scientific
  interpretation remains pending Analyst review.
- **Next actions**: Commit and push the tested implementation, verify remote
  `main`, then execute A019 without changing experiment logic.
- **Secrets**: No secret material recorded.

### LNNNN — YYYY-MM-DD HH:MM TZ — Short title

- **Type**: context / plan / setup / implementation / experiment / analysis /
  decision / failure / success / blocker
- **Status**: planned / in progress / completed / failed / blocked /
  inconclusive / superseded
- **Objective**:
- **Context consulted**: Repository files, prior log IDs, exact Notion page and
  last-edited time when available
- **Repository state**: Commit, branch, and clean/dirty state
- **Action**: Exact research-relevant action taken
- **Evidence**: Commands, files, commits, job IDs, logs, artifacts, or sources
- **Outcome**: Observed facts only
- **Failure or caveat**: `None observed` or the exact failure/uncertainty
- **Decision / lesson**: Decision made, or `Pending analysis`
- **Next actions**:
- **Secrets**: `No secret material recorded`

For an experiment entry, also include:

- **Experiment ID**:
- **Parent idea / branch**:
- **Falsifiable hypothesis**:
- **Falsification criterion**:
- **Baseline and metrics**:
- **Change introduced and configuration diff**:
- **Exact command**:
- **Environment and hardware**:
- **Seed and deterministic settings**:
- **Start/end timestamps and scheduler job ID**:
- **Raw artifacts**:
- **Baseline integrity**:
- **Validity**:
- **Result classification**:
- **Analyst verdict**:
- **Analyst interpretation**:
- **Lesson and follow-up**:
