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

- **Last updated**: 2026-09-04 13:07 CDT
- **Research phase**: Active starter phase — E0–E2 exact-capture calibration
  on controlled tiny checkpoint fixtures; CPU results analyzed and archived,
  Condor FP32 portability confirmation blocked during setup
- **Repository state**: Local `main`, `origin/main`, and the clean `darmok`
  checkout match implementation commit
  `05b477a9b3246b56b4be6a64c04280ebbfad1150`; only this L0017 remediation
  record is now modified locally
- **Research objective**: Detect activation-checkpoint recomputation value
  changes before model or optimizer state is mutated
- **Current baseline**: E0 is promoted as a bounded oracle for this exact tiny
  CPU `float64` fixture and comparator after three clean repetitions passed all
  preregistered gates
- **Current experiment**: CPU verdicts remain E0/E1 `PROMOTE` and exact E2
  candidate `KILL`. Condor job 1553506.0 remains `BLOCKED —
  REMOTE_DISK_QUOTA`. A022 is synchronized and A023's exact authorized removal
  is complete; the one authorized retry has not been submitted
- **Best validated result**: The exact E0 fixture/comparator is a bounded clean
  oracle, and E1 is a bounded controlled regression suite. Neither is
  production or natural-failure evidence
- **Compute status**: Apple Silicon CPU suite complete. Condor job 1553506.0
  recognized PyTorch 2.12.1+cu126, CUDA 12.6, and a GTX 1080 Ti on
  `eldar-44`; its 19 passing tests exercised CPU fixtures only. The job stopped
  before E0 artifact creation, so no GPU experiment or CUDA-kernel validation
  occurred; TACC is not configured
- **Active blockers**: No Condor E0/E1/E2 result exists. The scratch rebuild,
  live preflights, and one scheduler retry remain pending. Removing the 6.1 GiB
  environment freed enough submit-side storage for a durable write probe and
  Git synchronization, but does not establish it as the sole quota cause
- **Immediate next action**: Commit and synchronize L0017, re-query the live
  pool and queue, dry-run the submit description, and submit exactly one
  unchanged-science retry
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
| A001 | 1 | Commit and push `AGENTS.md` and `RESEARCH_LOG.md` when authorized | Makes the operating rules available to GitHub and remote agents | User authorization to commit/push | Completed |
| A002 | 2 | Pull the current operational commit into the `darmok` checkout | Keeps remote work under the same instructions and code | A001 and stable CPU suite | Completed at `bb95225` |
| A003 | 2 | Pin compatible Python, PyTorch, and CUDA versions and create a dedicated remote environment | Existing base Conda environment is not reproducible | A016 and stable CPU suite | Completed; 6.1 GiB environment exposed quota blocker |
| A004 | 2 | Add a metadata-printing Condor runner and submit file | Required for reproducible scheduled work | A003 | Completed; scheduler dry-run PASS |
| A005 | 2 | Run a one-GPU Condor probe | Verify driver, CUDA, GPU, memory, compute capability, and PyTorch CUDA availability | A004 | Metadata discovery PASS in 1553506.0; CUDA kernels untested |
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
| A019 | 1 | Run E0 three times, every E1 control/fault pair three times including both RNG variants, and E2 on the Apple Silicon CPU | Produces the required repeatable calibration and coverage evidence | A018 | Completed and analyzed |
| A020 | 1 | Analyze and append every CPU result, failure, artifact path, and verdict | Results cannot drive later work until reviewed and archived | A019 | Completed in L0012 |
| A021 | 2 | Repeat unchanged E0–E2 logic in FP32 inside one scheduled Condor GPU job | Portability confirmation only; it does not broaden CPU verdicts | A020, A022, A023, and L0015 authorization | `REPEAT` after A023 remediation is separately recorded; one new job authorized |
| A022 | 1 | Move `ArtifactWriter` setup into classified exception handling and add writable-artifact and CUDA-kernel preflights | Ensures setup failures return exit 2 and kernel incompatibility is detected before scientific execution | L0014 diagnosis | Completed in L0016 at `05b477a`; synchronized locally, on origin, and remotely |
| A023 | 1 | Preserve prior logs/spec, remove only `.condor-venv-968f64415c1731fa`, and rebuild the same locked environment in per-job scratch | Frees the likely 6.1 GiB home-storage contributor without changing scientific logic or destroying prior evidence | A022 local verification and L0015 authorization | Authorized persistent-environment removal completed in L0017; per-job scratch rebuild pending in retry |

## Experiment index

| ID | Date | Hypothesis | Baseline | Status | Verdict | Record |
|---|---|---|---|---|---|---|
| E0 | 2026-09-03 | The fixed clean fixture, exact comparator, and independent formulas agree across no-checkpoint, original, and recompute execution | Fixed literal CPU `float64` no-checkpoint run and independent formulas | Three clean-commit CPU repetitions PASS; bounded oracle | PROMOTE | L0007, L0009, L0010, L0012 |
| E1 | 2026-09-03 | Each of five hidden-state fault families causes a same-metadata value mismatch first at `h` and a gradient difference, while its control and trigger-disabled arm remain exact | E0 plus a fresh-process correct arm with identical tensors, seeds, and state | Six CPU scenarios, 54 fresh processes PASS; bounded regression suite | PROMOTE | L0007, L0009, L0010, L0012 |
| E2 | 2026-09-03 | Public `context_fn` and inner `saved_tensors_hooks` capture each explicitly backward-relevant `h`, `g`, and `y` once per phase without changing behavior | E0 no-hook result and checkpoint baseline without observational hooks | `PUBLIC_HOOKS_INSUFFICIENT`: hook candidate suppressed recompute | KILL exact candidate | L0007, L0009, L0010, L0012 |
| Condor 1553506.0 | 2026-09-03 | Unchanged E0–E2 gates remain valid under the preregistered Pascal CUDA FP32 environment change | Archived CPU E0/E1/E2 results | `BLOCKED — REMOTE_DISK_QUOTA`; no GPU experiment ran | REPEAT only after separately recorded storage remediation; one retry authorized by L0015 | L0011, L0013, L0014, L0015 |
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

### L0010 — 2026-09-03 23:22 CDT — Required Apple Silicon CPU sequence completed

- **Type**: experiment / success / failure
- **Status**: completed; pending Analyst review
- **Objective**: Execute the complete L0007 CPU sequence from one clean,
  pushed commit and preserve each arm's exact tensor and structured evidence.
- **Repository state**: `main` at clean commit
  `14bce330cd1259192c2d8209ecb0a556346a2534`; local and `origin/main` were
  verified equal before execution. Ignored artifacts do not dirty Git state.
- **Shared environment and hardware**: Apple Silicon ARM64 CPU on macOS
  26.4.1; CPython 3.13.15; PyTorch 2.13.0; NumPy 2.5.2; pytest 9.1.1; no CUDA;
  seed 20260903; `use_reentrant=False`; early stop disabled through
  `set_checkpoint_early_stop(False)`; `preserve_rng_state=True`;
  `determinism_check="default"`; phase-labeling `context_fn`.
- **Test evidence**: `uv run pytest -q` -> 19 passed in 42.56 seconds from the
  clean commit.

- **Experiment ID**: E0
- **Falsifiable hypothesis and criterion**: As preregistered in L0007; any
  phase-count, pairing, tensor, output, loss, autograd-gradient, or independent
  formula mismatch fails E0.
- **Baseline and isolated change**: Fixed literal CPU `float64` no-checkpoint
  evaluation plus independent formulas; isolated change is the configured
  non-reentrant checkpoint wrapper and direct tags.
- **Exact command**: `uv run aci-starter run E0`, invoked three times.
- **Start/end timestamps**: 2026-09-04T04:16:21.911697Z through
  2026-09-04T04:16:35.565536Z; no scheduler job.
- **Raw artifacts and results**:
  - `e0-counter-correct-20260904T041621.910016Z-cbce59d8` -> PASS, 1.287 s
  - `e0-counter-correct-20260904T041625.574015Z-64a78dbe` -> PASS, 1.249 s
  - `e0-counter-correct-20260904T041634.248155Z-349df9f0` -> PASS, 1.317 s
  - All are under ignored `artifacts/starter/<run_id>/`; each has one original
    and one recompute phase, three unambiguous exact tensor pairs, no missing,
    duplicate, value, or metadata mismatch, and exact output/loss/all-gradient
    equality to the no-checkpoint and independent formula references.
- **Baseline integrity / validity**: All three manifests record the exact
  commit with `dirty=false`; repeated scientific and tensor results agree.
- **Result classification**: `PASS` for all three runs.
- **Analyst verdict / interpretation**: Pending Analyst.

- **Experiment ID**: E1
- **Falsifiable hypothesis and criterion**: As preregistered in L0007; every
  clean/trigger-disabled arm must remain exact, and every broken arm must first
  differ at same-metadata `h`, change at least one gradient, match its correct
  original, and repeat exactly three times.
- **Baseline and isolated changes**: E0-style same-state no-checkpoint/correct
  controls versus one counter, registered-buffer, Python RNG, NumPy RNG,
  precision-policy, or toy delayed-scaling trigger at a time.
- **Exact command**: `uv run aci-starter run E1 --case all`.
- **Start/end timestamps**: 2026-09-04T04:16:47.876596Z through
  2026-09-04T04:20:38.132250Z; duration 230.411 s; no scheduler job.
- **Raw artifact**:
  `artifacts/starter/e1-all-20260904T041647.718907Z-4f759aa3/summary.json` links
  all 54 immutable child runs: six scenarios times three repetitions times
  correct/broken/trigger-off fresh processes.
- **Observed result**: Aggregate `PASS`. All six scenarios pass; every broken
  original equals its correct original, every broken first mismatch is `h`
  with unchanged metadata and at least one different gradient, trigger removal
  restores equality, PyTorch's default metadata check does not raise, and
  saved tensor/scientific contents repeat across all three seeded repetitions.
- **Baseline integrity / validity**: Every child manifest records commit
  `14bce33` and `dirty=false`; fresh-process isolation is recorded by the
  aggregate.
- **Result classification**: `PASS` for the intended controlled calibration.
- **Analyst verdict / interpretation**: Pending Analyst.

- **Experiment ID**: E2
- **Falsifiable hypothesis and criterion**: As preregistered in L0007; public
  hooks must capture each semantic tag once per phase with unpack/access and
  must not change recomputation or numerical behavior.
- **Baseline and isolated change**: Isolated no-observer `TaggedSave`
  checkpoint baseline versus the same fixture with public
  `saved_tensors_hooks` installed inside the phase `context_fn`.
- **Exact command**: `uv run aci-starter run E2`.
- **Start/end timestamps**: 2026-09-04T04:20:47.114152Z through
  2026-09-04T04:20:56.926239Z; duration 9.971 s; no scheduler job.
- **Raw artifacts**:
  - Aggregate:
    `artifacts/starter/e2-coverage-20260904T042046.953728Z-aab0a6b4/`
  - No-hook baseline:
    `e2-counter-correct-20260904T042049.181490Z-8359e6dc` -> PASS, phases 1/1,
    three exact pairs
  - Hook candidate:
    `e2-counter-correct-hooks-20260904T042054.099751Z-ba76592a`
- **Observed result**: `PUBLIC_HOOKS_INSUFFICIENT`, CLI exit 1. The candidate
  observed one token-identified pack and two unpack/access events for each of
  `h`, `g`, and `y` in the original phase, but installing the hooks suppressed
  checkpoint recomputation: phase counts were original=1/recompute=0, with all
  three recompute tags and hook events missing. Output and gradients remained
  exact, but the preregistered behavioral and coverage gate failed.
- **Baseline integrity / validity**: Baseline passed, only observation was
  toggled, both manifests record the same clean commit, and the failure is
  explicit rather than replaced with a different capture mechanism.
- **Result classification**: `PUBLIC_HOOKS_INSUFFICIENT`.
- **Analyst verdict / interpretation**: Pending Analyst.

- **Failure or caveat**: E2 is a preregistered coverage failure and does not
  invalidate E0/E1. CPU results do not establish CUDA/Pascal behavior.
- **Decision / lesson**: No scientific decision here; pending Analyst review.
- **Next actions**: Version this factual record, execute the unchanged starter
  logic in one scheduled Condor GPU job, and route all results to Analyst before
  follow-up planning.
- **Secrets**: No secret material recorded.

### L0011 — 2026-09-03 23:27 CDT — Starter-only Condor execution prepared

- **Type**: setup / implementation / environment / cluster probe
- **Status**: in progress; submission paused pending the CPU Analyst verdict
- **Objective**: Prepare one reproducible scheduled Pascal GPU job that repeats
  unchanged E0–E2 logic in FP32 without installing persistent account-wide
  tooling or running compute on the submit host.
- **Context consulted**: AGENTS remote policy, L0007 and L0010, official
  PyTorch previous-version wheel matrix, official PyTorch package indexes, and
  the current UTCS pool/queue.
- **Repository state**: Local `main` at
  `86c0781eb45544b41524947a03c3ee2c5591ea0c`, equal to `origin/main` before
  the operational-file changes. Remote checkout remained clean at historical
  commit `b4afab63918d1a41dfff3e9e8a22397b28b10bc5` and has not yet been pulled.
- **Environment decision**:
  - PyTorch 2.13.0 is the stable local CPU pin but its official Linux wheel
    matrix provides CUDA 13.0 and 13.2, not CUDA 12.6. CUDA 13 removes Pascal
    support, so using the 2.13 CUDA wheel would violate the cluster constraint.
  - PyTorch 2.12.1 is the newest official build with a CUDA 12.6 wheel. The
    GPU-only environment therefore pins `torch==2.12.1+cu126`, retains
    `numpy==2.5.2` and `pytest==9.1.1`, and records this required environment
    config difference without changing fixture, fault, capture, gate, or
    runner logic.
  - Generated `condor/requirements-cu126.lock` with hashes for the complete
    35-package CPython 3.13 / manylinux 2.28 graph. The selected x86-64 torch
    wheel is published with SHA-256
    `b30ef03ebb87d6b7f5d8b1982bb08cf6a42bde552c9e6acf6a9c097b2700d0f1`.
  - The bootstrap downloads `uv` 0.12.3 into ignored `.condor-tools/`, verifies
    its archive SHA-256
    `600cf9a742aca00d292673b16b5acffaa7b8c269a364ad0c2e79498dcb1fe101`,
    and syncs a content-addressed ignored `.condor-venv-<lock-hash>/`. No
    account-wide installation or base-Conda mutation is performed.
- **Operational files**: Added the hash-locked Condor input/lock, project-local
  environment bootstrap, one-job E0/E1/E2 runner with fail-fast CUDA/Pascal
  checks and complete metadata output, submit description, and ignore rules.
  `bash -n` and `git diff --check` pass locally.
- **Live cluster evidence**: At 2026-09-03 23:25 CDT, bounded SSH checks found
  `darmok` on glibc 2.31, 1024 total advertised slots with 954 unclaimed,
  numerous idle `eldar` GTX 1080 and GTX 1080 Ti slots at compute capability
  6.1, and no jobs owned by `ayman27`. The global queue had 385 jobs: 65
  running, 318 idle, and 2 held. Availability is dynamic and will be queried
  again immediately before submission.
- **Failure or caveat**: PyTorch 2.12.1+cu126 import and kernel compatibility
  on the `eldar` execution image remain unverified until a scheduled job runs.
  No scientific GPU job has been submitted.
- **Decision / lesson**: Submission is explicitly paused until the Orchestrator
  relays the CPU Analyst verdict. Environment synchronization and submit-file
  dry-run may proceed without executing scientific compute.
- **Next actions**: Commit/push the operational files, pull the exact commit on
  `darmok`, build the repository-local environment via named `remote-run`, and
  syntax-check the submit description. Do not call `condor_submit` for the live
  job until the Analyst gate is cleared.
- **Secrets**: No secret material recorded.

### L0012 — 2026-09-03 23:30 CDT — CPU E0–E2 Analyst verdicts archived

- **Type**: analysis / decision / success / failure / archival
- **Status**: completed; Condor portability confirmation authorized
- **Objective**: Archive the supplied Analyst verdict for each completed CPU
  experiment, bound each resulting claim to its evidence, and decide whether
  the preregistered starter-only Condor repeat may proceed.
- **Context consulted**: L0007 preregistration, L0009 implementation record,
  L0010 factual CPU results and immutable artifact identifiers, L0011 Condor
  preparation, and the supplied Analyst review.
- **Repository state**: Current local `main` was clean at
  `d5940c75abf7528bd4f322e2e9dc45e9a07a974d`, equal to `origin/main`. The CPU
  artifacts analyzed below were generated from clean implementation commit
  `14bce330cd1259192c2d8209ecb0a556346a2534`; L0010 and the Condor preparation
  were subsequently committed and pushed in `86c0781` and `d5940c7`.

- **Experiment E0 — clean checkpoint baseline and exact comparator**:
  - **Parent idea / branch**: L0007 active starter calibration.
  - **Hypothesis and baseline**: The preregistered fixed CPU `float64` fixture
    should agree across no-checkpoint execution, original/recompute checkpoint
    execution, and independently implemented forward and gradient formulas.
  - **Change introduced / config diff**: No-checkpoint fixed calculation to the
    same calculation under the preregistered non-reentrant checkpoint wrapper;
    no unregistered change from L0007.
  - **Parameter count**: Not applicable; this is the fixed tiny matrix fixture,
    not a parameter-budget model experiment.
  - **Command and artifacts**: `uv run aci-starter run E0`, repeated three
    times. Run IDs:
    - `e0-counter-correct-20260904T041621.910016Z-cbce59d8`
    - `e0-counter-correct-20260904T041625.574015Z-64a78dbe`
    - `e0-counter-correct-20260904T041634.248155Z-349df9f0`
  - **Metrics**: Three of three runs passed. Each executed one original and one
    recompute phase; all three expected `h`/`g`/`y` pairs were present and exact
    (`3/3` per run); there were no missing, duplicate, value, or metadata
    mismatches; and output, loss, all autograd gradients, and independent
    formula/gradient references were exact.
  - **Baseline integrity**: Confirmed for the recorded CPU runs; all manifests
    identify clean commit `14bce33`, and the three repetitions agree.
  - **Validity**: The Analyst supplied no separate categorical validity grade;
    the verdict is explicitly bounded to this exact tiny CPU `float64` fixture,
    comparator, and environment.
  - **Result classification**: Clean calibration success.
  - **Analyst verdict**: `PROMOTE`.
  - **Analyst interpretation / lesson**: The evidence authorizes this exact tiny
    fixture and comparator as a bounded oracle. It does not establish a
    production baseline, natural pretraining failure, or broader capture
    coverage.
  - **Status / follow-up**: Promoted as the oracle for the controlled starter
    regression suite and unchanged Condor portability confirmation.

- **Experiment E1 — five fault families, six required scenarios**:
  - **Parent idea / branch**: E0 bounded oracle and L0007 controlled-fault
    calibration. The randomness family contains separate Python and NumPy
    scenarios, yielding six required scenarios across five families.
  - **Hypothesis and baseline**: Each correct and trigger-off arm should remain
    exact; every broken original should equal its correct original; and every
    broken recomputation should first differ at same-metadata `h`, alter a
    gradient, and repeat deterministically.
  - **Change introduced / config diff**: Exactly one preregistered counter,
    buffer, Python RNG, NumPy RNG, precision-policy, or delayed-scaling trigger
    at a time against identical fresh-process control and trigger-off state.
  - **Parameter count**: Not applicable; these reuse the fixed tiny fixture.
  - **Command and artifacts**: `uv run aci-starter run E1 --case all`; aggregate
    run ID `e1-all-20260904T041647.718907Z-4f759aa3`, whose summary links all 54
    fresh child processes: six scenarios times three repetitions times
    correct/broken/trigger-off arms.
  - **Metrics**: All six scenarios passed. Every correct and trigger-off arm was
    exact; every broken original equaled the corresponding correct original;
    every first mismatch was `h` with unchanged shape, dtype, device, layout,
    and stride; at least one gradient differed; the default PyTorch metadata
    check did not raise; and scientific and saved-tensor contents repeated
    deterministically across all three seeded repetitions.
  - **Baseline integrity**: Confirmed for the recorded CPU suite; all child
    manifests identify clean commit `14bce33`, and fresh-process isolation is
    recorded in the aggregate.
  - **Validity**: The Analyst supplied no separate categorical validity grade;
    the verdict is bounded to these six controlled scenarios, implementation,
    CPU environment, and preregistered gates.
  - **Result classification**: Controlled fault-regression success.
  - **Analyst verdict**: `PROMOTE`.
  - **Analyst interpretation / lesson**: The evidence authorizes E1 as a
    controlled regression suite only. It does not show that any fault occurs
    naturally or that the mechanism generalizes beyond the declared fixtures.
  - **Status / follow-up**: Promoted for regression and unchanged Condor
    portability confirmation.

- **Experiment E2 — exact public-hook candidate**:
  - **Parent idea / branch**: E0 bounded oracle plus L0007 public capture
    coverage question.
  - **Hypothesis and baseline**: The exact composition of checkpoint
    `context_fn` with inner `torch.autograd.graph.saved_tensors_hooks` should
    observe `h`, `g`, and `y` in original and recompute phases without changing
    checkpoint behavior. Its no-hook baseline should execute both phases once.
  - **Change introduced / config diff**: Only the inner public pack/unpack
    observer was enabled around the otherwise identical `TaggedSave` fixture.
  - **Parameter count**: Not applicable; this is the fixed tiny fixture.
  - **Command and artifacts**: `uv run aci-starter run E2`; aggregate run ID
    `e2-coverage-20260904T042046.953728Z-aab0a6b4`, no-hook baseline
    `e2-counter-correct-20260904T042049.181490Z-8359e6dc`, and hook candidate
    `e2-counter-correct-hooks-20260904T042054.099751Z-ba76592a`.
  - **Metrics**: The no-hook baseline executed original/recompute phases `1/1`
    with three exact pairs. The hook candidate executed phases `1/0`:
    recomputation was suppressed, and recompute `h`, `g`, and `y` were all
    missing. Output and gradients remained unchanged, which does not repair the
    failed behavioral and coverage gate.
  - **Baseline integrity**: Confirmed for this comparison; both arms identify
    the same clean implementation commit and toggle only the observer.
  - **Validity**: The Analyst supplied no separate categorical validity grade;
    the verdict is bounded to this exact composition under PyTorch 2.13.0 on
    the recorded Apple Silicon CPU environment.
  - **Result classification**: `PUBLIC_HOOKS_INSUFFICIENT`.
  - **Analyst verdict**: `KILL` the exact `context_fn` plus inner
    `saved_tensors_hooks` candidate.
  - **Analyst interpretation / lesson**: The likely mechanism is that the
    innermost user hook displaced checkpoint's internal saved-tensor hook, so
    recomputation was no longer triggered. This mechanism is an Analyst
    inference from the observed phase suppression, not an independently proven
    PyTorch implementation fact.
  - **Claim boundary**: This verdict does not kill every public PyTorch API or
    every possible public-hook composition. It does not prove that an internal
    integration point is sufficient. The observer's retention of direct tensor
    objects is diagnostic instrumentation and is not production code.
  - **Status / follow-up**: Exact candidate killed and retained as negative
    evidence. No replacement capture design or internal integration has been
    validated by these results.

- **Cross-experiment decision**: The Analyst authorizes the unchanged
  preregistered E0–E2 experiment logic and gates to run in FP32 in one scheduled
  Condor job as portability confirmation. The required PyTorch 2.12.1+cu126
  environment difference is already disclosed in L0011. The Condor result may
  confirm or contradict portability on Pascal, but cannot broaden the bounded
  CPU claims or revive the killed E2 composition without a new Analyst review.
- **Failure or caveat**: CPU evidence does not establish CUDA/Pascal behavior,
  production overhead, natural-failure prevalence, general public-hook
  insufficiency, or internal-hook sufficiency.
- **Next actions**: Complete A002–A005 and A021 from exact operational commit
  `d5940c75abf7528bd4f322e2e9dc45e9a07a974d`: re-query the live pool and queue,
  synchronize the remote checkout, build the repository-local locked
  environment, dry-run the submit description, and submit the one authorized
  scheduler job. Append every setup failure and run result without overwrite,
  then obtain another Analyst verdict before follow-up planning.
- **Secrets**: No secret material recorded.

### L0013 — 2026-09-03 23:40 CDT — Condor starter job blocked by remote disk quota

- **Type**: setup / experiment / failure / blocker
- **Status**: blocked; pending Analyst review
- **Objective**: Repeat the unchanged E0–E2 starter logic in FP32 in exactly
  one scheduled UTCS Condor GPU job after the CPU verdicts were archived.
- **Context consulted**: L0011 environment/submit preparation, L0012 CPU
  verdicts and authorization, current Git state, live Condor pool/queue, and
  scheduler/stdout/stderr records for cluster 1553506.
- **Repository state**: Local, `origin/main`, and the clean remote checkout all
  matched `bb952254ebe1fc2466cb2b00f2635455d616ae3d` before submission.

- **Environment setup**:
  - Named durable setup command: `remote-run --name aci-condor-env-cu126 --cd
    /u/ayman27/activation-checkpoint-integrity ayman27@darmok.cs.utexas.edu
    './scripts/bootstrap_condor_env.sh'`.
  - Setup completed with exit 0. Inspection commands remain:
    `remote-run --check ayman27@darmok.cs.utexas.edu aci-condor-env-cu126`,
    `remote-run --log ayman27@darmok.cs.utexas.edu aci-condor-env-cu126`, and
    `remote-run --attach ayman27@darmok.cs.utexas.edu aci-condor-env-cu126`.
  - Repository-local `uv` 0.12.3 and content-addressed environment
    `.condor-venv-968f64415c1731fa` were installed without changing the user or
    base-Conda environment. The lock SHA-256 is
    `968f64415c1731fa2729ecb519169d74389d661ce1c78c9aac0a7d4c72c6e72e`.
  - CPython 3.13.5, PyTorch 2.12.1+cu126, CUDA runtime 12.6, NumPy 2.5.2, and
    the complete 35-package environment passed `uv pip check`. The environment
    occupies 6.1 GiB in the remote repository; the uv cache was only 20 KiB
    after setup.
  - `/lusr/opt/condor/bin/condor_submit -dry-run /dev/stdout
    condor/starter.submit` passed before submission.

- **Experiment ID**: E0/E1/E2 Condor FP32 portability confirmation
- **Parent idea / branch**: L0007 starter plan and L0012 Analyst authorization.
- **Falsifiable hypothesis / criterion**: Retain the exact E0/E1/E2 gates from
  L0007 while changing only the recorded environment from Apple CPU/PyTorch
  2.13 to Pascal CUDA FP32/PyTorch 2.12.1+cu126. Any environment failure before
  scientific execution blocks the portability result and may not be
  substituted with a different experiment.
- **Baseline and metrics**: CPU E0/E1/E2 results in L0010/L0012; planned GPU
  metrics were the same tensor, phase, gradient, control, repeat, and coverage
  fields plus exact scheduler, driver, CUDA, and GPU metadata.
- **Config diff**: device CPU -> CUDA; E0/E2 dtype float64 -> float32; E1 dtype
  remains float32; PyTorch 2.13.0 macOS build -> 2.12.1+cu126 because 2.13 has
  no official CUDA 12.6 wheel; all experiment logic and seed 20260903 unchanged.
- **Pre-submit live state**: 58 eligible unclaimed `eldar` GTX 1080 Ti slots,
  zero jobs owned by `ayman27`, 368 global jobs (65 running, 301 idle, 2 held),
  and a clean exact remote commit.
- **Exact submission**: `/lusr/opt/condor/bin/condor_submit
  condor/starter.submit`; exactly one job was submitted as 1553506.0.
- **Scheduler execution**: Submitted 2026-09-04T04:34:59Z, began
  2026-09-04T04:35:01Z on `slot1@eldar-44.cs.utexas.edu`, and terminated
  2026-09-04T04:37:02Z with normal return value 1 and no restart. Scheduler
  usage was 1:13 user CPU, 0:10 system CPU, 190 MiB memory, one assigned GPU,
  and 0 transferred bytes.
- **Environment and hardware observed in the job**: NVIDIA GeForce GTX 1080 Ti,
  compute capability 6.1, 11,714,887,680 bytes reported by PyTorch, driver
  535.247.01, PyTorch 2.12.1+cu126, CUDA runtime 12.6, cuDNN 9.10.2, CPython
  3.13.5, and NumPy 2.5.2. `torch.cuda.is_available()` was true because the
  fail-fast hardware record completed.
- **Operational validation**: The scheduled job's unchanged automated suite
  passed 19 tests in 97.40 seconds. Pytest emitted a warning that it could not
  create `.pytest_cache` because of error 122, `Disk quota exceeded`.
- **Failure**: Before E0 repetition 1 could create its artifact directory,
  `ArtifactWriter` raised `OSError: [Errno 122] Disk quota exceeded` for
  `/u/ayman27/activation-checkpoint-integrity/artifacts/starter/
  e0-counter-correct-20260904T043701.655665Z-d7349d18`. The directory was not
  created. E0, E1, and E2 did not execute and produced no GPU run IDs,
  summaries, tensor pairs, gradients, or scientific classifications.
- **Raw artifacts**: Remote ignored files
  `/u/ayman27/activation-checkpoint-integrity/artifacts/condor/1553506.log`,
  `1553506.0.out`, and `1553506.0.err` (approximately 1.5 KiB, 4.0 KiB, and
  1.2 KiB respectively). No starter scientific artifact exists for this job.
- **Baseline integrity**: CPU baseline remains intact and unaffected. The GPU
  comparison is absent, not negative.
- **Validity / result classification**: `BLOCKED — REMOTE_DISK_QUOTA`; no
  scientific result and no E0/E1/E2 portability verdict.
- **Analyst verdict / interpretation**: Pending Analyst. The only established
  mechanism is artifact creation failing after the environment consumed the
  available quota; Pascal numerical or hook behavior was not tested.
- **Potential remediation boundary**: Do not delete the environment, redesign
  dependency selection, move artifacts, or submit a second job without explicit
  direction. Any retry must preserve the failed attempt and use a newly
  preregistered storage/environment action rather than relabeling 1553506.0.
- **Next actions**: Route this blocker to Analyst, preserve the remote records,
  and obtain explicit authorization for any cleanup or retry.
- **Secrets**: No secret material recorded.

### L0014 — 2026-09-04 00:20 CDT — Condor quota blocker analyzed and repeat gated

- **Type**: analysis / decision / failure / blocker / archival
- **Status**: `BLOCKED — REMOTE_DISK_QUOTA`; conditional `REPEAT`, not yet
  authorized for remediation or resubmission
- **Objective**: Archive the supplied Analyst verdict for Condor job
  1553506.0, distinguish setup evidence from absent scientific evidence, and
  state the exact conditions required before another scheduler attempt.
- **Context consulted**: L0011 Condor preregistration and preparation, L0012
  CPU verdicts, L0013 factual scheduler record, the retained scheduler
  log/stdout/stderr, and the supplied Analyst review.
- **Repository state**: Local `main` and `origin/main` were clean and equal at
  `2231b2981e686cf626c4ed04983396b6d09a9e40` before this archival edit. The
  failed remote attempt ran clean experiment commit
  `bb952254ebe1fc2466cb2b00f2635455d616ae3d` as recorded in L0013.
- **Experiment / job**: Preregistered E0–E2 Condor FP32 portability
  confirmation, scheduler job 1553506.0, executed once on
  `slot1@eldar-44.cs.utexas.edu`.
- **Baseline and isolated change**: CPU E0/E1/E2 records in L0010/L0012;
  planned isolated environment changes were CPU to CUDA, E0/E2 `float64` to
  FP32, and PyTorch 2.13.0 macOS to 2.12.1+cu126. No scientific GPU arm reached
  execution.
- **Observed setup evidence**:
  - The scheduled process recognized NVIDIA GeForce GTX 1080 Ti hardware,
    compute capability 6.1, driver 535.247.01, PyTorch 2.12.1+cu126, CUDA
    runtime 12.6, cuDNN 9.10.2, CPython 3.13.5, and NumPy 2.5.2.
  - The unchanged automated suite reported `19 passed`; these tests used CPU
    fixtures and did not launch the preregistered GPU E0/E1/E2 arms or
    establish CUDA-kernel compatibility.
  - Pytest could not create `.pytest_cache` and reported
    `OSError: [Errno 122] Disk quota exceeded`.
  - The first E0 attempt then failed on `mkdir` for
    `artifacts/starter/e0-counter-correct-20260904T043701.655665Z-d7349d18`
    with the same `EDQUOT` error.
  - The job produced no E0, E1, or E2 run summary, tensor pair, comparison,
    gradient, mismatch classification, or portability result.
- **Raw evidence**: Remote ignored files
  `artifacts/condor/1553506.log`, `artifacts/condor/1553506.0.out`, and
  `artifacts/condor/1553506.0.err`, as fully located in L0013. There is no
  starter scientific artifact for this job.
- **Baseline integrity**: The archived CPU results and E0/E1/E2 Analyst
  verdicts are unchanged. Job 1553506.0 neither confirms nor contradicts them.
- **Scientific validity**: `LOW`; no GPU experiment ran, so there is no
  scientific portability evidence to interpret.
- **Setup-diagnosis confidence**: `HIGH`; two independent write attempts in the
  same job failed with `EDQUOT` before E0 execution.
- **Result classification**: `BLOCKED — REMOTE_DISK_QUOTA`.
- **Analyst verdict**: `REPEAT` only after setup-code correction/preflight, a
  separately preregistered and recorded storage remediation, and explicit user
  authorization for the storage mutation and second scheduler submission.
- **Analyst interpretation / failure mechanism**:
  - Remote home-directory quota exhaustion prevented artifact creation. The
    6.1 GiB repository-local PyTorch/CUDA environment is a likely contributor,
    but exact quota and preexisting usage were not measured, so it is not
    established as the sole cause. This supersedes L0013's stronger inference
    that the environment itself consumed the available quota.
  - HTCondor `request_disk` was not the limiting resource. It schedules execute
    storage and does not remove the submit-side home-filesystem quota observed
    here; increasing it is not evidence-based remediation for this failure.
  - Driver/runtime metadata discovery succeeded, but no preregistered CUDA
    fixture executed. CUDA kernel compatibility on the GTX 1080 Ti remains
    untested.
- **CLI contract defect**: `ArtifactWriter` is constructed outside
  `run_arm`'s setup-exception handling. The `EDQUOT` exception therefore
  escaped as process exit 1 instead of the CLI contract's required setup exit
  2. The correction must classify artifact-directory creation and other setup
  writes before any scientific arm begins, and a preflight must distinguish
  unwritable/quota-exhausted storage from a scientific failure.
- **Claim boundary**: The 19 passing tests are CPU-fixture evidence only. They
  do not count as a GPU E0 pass or as evidence for E1, E2, Pascal numerics,
  public-hook behavior on CUDA, or production compatibility. The CPU `PROMOTE`,
  `PROMOTE`, and exact-candidate `KILL` verdicts remain unaffected.
- **Decision / authorization boundary**: `REPEAT` is a conditional Analyst
  verdict, not permission to delete or move remote files, rebuild or relocate
  the environment, or submit another job now. Preserve job 1553506.0 and its
  records. Record quota/preusage measurements and the exact storage action in a
  new ledger entry, obtain explicit user authorization, and use a new job ID
  for any repeat.
- **Next actions**:
  1. Implement and test A022 so setup `EDQUOT` returns exit 2 before scientific
     execution and the storage preflight is explicit.
  2. Perform only read-only quota/preusage measurement, then preregister A023's
     minimal storage remediation without executing it.
  3. Obtain explicit user authorization for that exact remediation and one new
     submission.
  4. If authorized, append the remediation outcome before submitting, repeat
     the unchanged E0–E2 FP32 logic once, preserve both attempts, and route the
     new result to Analyst.
- **Secrets**: No secret material recorded.

### L0015 — 2026-09-04 10:32 CDT — Scoped Condor remediation and one retry authorized

- **Type**: decision / authorization / plan
- **Status**: authorized; remediation and retry not yet executed
- **Objective**: Preregister the exact bounded response to L0014 so the quota
  blocker can be remediated without changing the E0–E2 scientific logic or
  erasing evidence from failed job 1553506.0.
- **Context consulted**: L0013 factual scheduler record, L0014 Analyst verdict,
  current repository state, and the user's instruction to keep going.
- **Repository state**: Local `main` and `origin/main` are equal at
  `2231b2981e686cf626c4ed04983396b6d09a9e40`; `RESEARCH_LOG.md` contains the
  uncommitted L0014/L0015 archival update. No implementation or remote mutation
  is recorded by this entry.
- **Authorized implementation scope**:
  - Preserve the submit specification and all retained scheduler log, stdout,
    and stderr records for job 1553506.0.
  - Correct `ArtifactWriter` setup classification so storage-setup failures use
    contract exit 2, and add a writable-artifact preflight before any
    scientific arm begins.
  - Add a minimal CUDA-kernel preflight after metadata discovery and before E0,
    so GTX 1080 Ti/PyTorch 2.12.1+cu126/CUDA 12.6 kernel compatibility is tested
    separately from E0–E2.
  - Remove only the generated reproducible remote environment
    `.condor-venv-968f64415c1731fa`; do not remove prior logs, the submit
    specification, source, lock files, or any other remote data.
  - Rebuild the same lock-derived environment inside Condor per-job scratch
    rather than the quota-limited remote home checkout.
  - After local verification and a separately appended remediation outcome,
    submit exactly one newly identified retry with unchanged E0–E2 scientific
    logic and preserve both job attempts.
- **Scientific invariants**: Device remains Pascal CUDA, E0/E2 remain FP32, E1
  remains FP32, seed remains 20260903, and all E0/E1/E2 gates, repeat counts,
  fault definitions, controls, and capture candidate remain unchanged. Setup
  classification, preflights, and environment placement are operational
  changes, not scientific-variable changes.
- **Authorization boundary**: This authorizes only the exact environment
  removal, scratch rebuild, preflight work, and one new scheduler submission
  above. It does not authorize deletion of any other file, extra retries,
  dependency changes, experiment-logic changes, or broader production work.
- **Evidence boundary**: The selected remediation does not prove that the 6.1
  GiB environment was the sole quota cause; exact quota and preexisting usage
  remain unmeasured. Job 1553506.0 remains `BLOCKED — REMOTE_DISK_QUOTA`, with
  no scientific GPU result. Its Analyst verdict remains `REPEAT` only after the
  storage remediation is executed and separately recorded.
- **Execution order**:
  1. Implement and locally verify A022 without changing scientific logic.
  2. Execute only A023's named environment removal and record the observed
     outcome in a new ledger entry.
  3. Build the locked environment in per-job scratch and submit one new job ID.
  4. Preserve all outputs and route the retry to Analyst before changing any
     experiment verdict.
- **Secrets**: No secret material recorded.

### L0016 — 2026-09-04 13:03 CDT — Condor remediation implementation verified locally

- **Type**: context / implementation / setup / verification
- **Status**: completed locally; commit, synchronization, authorized remote
  removal, and retry pending
- **Objective**: Implement only L0015's setup-exit correction, scratch-local
  Condor environment build, writable-artifact preflight, and real CUDA-kernel
  preflight without changing E0–E2 scientific logic.
- **Context consulted**:
  - `AGENTS.md`, `README.md`, L0011–L0015, current source, tests, dependency
    lock, Condor scripts, submit description, and retained live records for job
    1553506.0
  - Exact Notion root `MOOTAZ PROJECT`, page ID
    `3c65d66f-2c23-80c6-879f-e0aff5e92706`, and selected child `Starter E0–E2
    Plan: Five Tiny Checkpoint Failures and Capture Validation`, page ID
    `3d05d66f-2c23-8068-b52a-ca774781771e`; both were fetched on 2026-09-04,
    and the child reported last edit `2026-09-03T04:15:45.303Z`
- **Repository state**: Work began from local `main` commit
  `2231b2981e686cf626c4ed04983396b6d09a9e40`, equal to `origin/main` before
  the preserved L0014–L0015 log edits and this implementation. The dependency
  lock remains unchanged at SHA-256
  `968f64415c1731fa2729ecb519169d74389d661ce1c78c9aac0a7d4c72c6e72e`.
- **Files changed**:
  - `src/ac_integrity/starter/runner.py`: catches only `OSError` from initial
    `ArtifactWriter` construction and returns setup exit 2 with `BLOCKED`, the
    exact error, intended run path, and `artifacts_persisted=false`.
  - `src/ac_integrity/cli.py`: includes setup-error and artifact-persistence
    fields in its one-line result when present; ordinary successful and
    scientific-failure output remains unchanged.
  - `tests/test_starter.py`: extends the CLI contract test to verify successful
    exit 0, scientific/coverage exit 1, and a deterministic artifact-directory
    setup failure exiting 2 with honest structured output and no false artifact.
  - `scripts/bootstrap_condor_env.sh`: retains project-local checksum-verified
    `uv` 0.12.3 and the unchanged hash-locked dependency graph, but requires a
    supplied Condor scratch directory and creates the content-addressed
    environment there with caching disabled.
  - `scripts/run_condor_starter.sh`: builds that environment inside
    `$_CONDOR_SCRATCH_DIR`, checks artifact storage by create/write/flush/fsync/
    read/remove, then launches and synchronizes a deterministic CUDA FP32 2x2
    matmul-plus-bias and checks its exact expected device and value before
    pytest or any E0–E2 arm. Pytest cache writes are disabled.
- **Config diff**:
  - Environment location:
    `/u/ayman27/activation-checkpoint-integrity/.condor-venv-968f64415c1731fa`
    -> `$_CONDOR_SCRATCH_DIR/aci-condor-venv-968f64415c1731fa`.
  - Added operational artifact and CUDA-kernel preflights before tests/science.
  - Dependency versions, lock, Condor submit specification, device, dtypes,
    seed 20260903, cases, fault triggers, gates, E0 count 3, E1 scenario count 6,
    E1 repetition count 3, fresh-process isolation, and E2 logic are unchanged.
- **Parameter count**: N/A — the fixed tiny fixtures and scientific code did
  not change.
- **Verification evidence**:
  - `git diff --check` -> PASS.
  - `bash -n scripts/bootstrap_condor_env.sh scripts/run_condor_starter.sh` ->
    PASS.
  - `PYTHONPATH=src uv run pytest -q` -> 19 passed in 22.29 seconds.
  - A deterministic local CLI probe using an artifact root that was a regular
    file returned exit 2 and one JSON record with `result=BLOCKED`,
    `NotADirectoryError`, the intended run path, and
    `artifacts_persisted=false`; no traceback escaped.
  - The expanded test checks the unchanged exit-0 and exit-1 paths in the same
    suite.
- **Environment caveat**: Bare `uv run pytest -q` in the existing ignored
  laptop `.venv` did not expose the editable `src` package to child Python
  processes. The explicit `PYTHONPATH=src` invocation passed, and the Condor
  runner already exports the same path before pytest and science. This local
  editable-environment condition did not change tracked files or the locked
  Condor environment.
- **Outcome**: A022 is locally verified. No remote file was removed, no Condor
  job was submitted, and no scientific result was produced by this entry.
- **Baseline integrity**: Focused and full tests passed with unchanged
  scientific sources, configuration, and dependency lock.
- **Analyst verdict / interpretation**: Pending. This entry records operational
  implementation evidence only.
- **Expected runtime**: Prior scheduled test phase took 97.40 seconds after the
  persistent environment already existed. The retry adds a fresh scratch
  dependency build of unknown live duration; no wall-clock estimate is claimed
  until observed.
- **Next actions**: Commit and push this bounded implementation, pull the exact
  commit into the remote checkout, revalidate and remove only the authorized
  generated environment, record the removal outcome and remaining sizes, then
  re-query Condor, dry-run the unchanged submit description, and submit exactly
  one newly identified retry.
- **Secrets**: No secret material recorded.

### L0017 — 2026-09-04 13:07 CDT — Authorized remote environment removed

- **Type**: setup / remediation / failure / success
- **Status**: completed; retry not submitted
- **Objective**: Execute only L0015's authorized removal of the generated,
  reproducible persistent Condor environment, prove the retained evidence and
  specification remained unchanged, and restore enough home storage for the
  scratch-based retry workflow.
- **Context consulted**: L0013–L0016, exact remote job 1553506.0 records,
  implementation commit `05b477a9b3246b56b4be6a64c04280ebbfad1150`,
  remote Git state, dependency-lock hash, target type/path, and directory sizes.
- **Repository state before removal**: Local and `origin/main` were clean and
  equal at `05b477a9b3246b56b4be6a64c04280ebbfad1150`. The first remote
  `git pull --ff-only` from clean commit
  `bb952254ebe1fc2466cb2b00f2635455d616ae3d` failed while closing a loose Git
  object with `Disk quota exceeded`; `git fsck --no-dangling` then passed, the
  remote HEAD stayed unchanged, and the worktree stayed clean.
- **Exact target validation**:
  - Target:
    `/u/ayman27/activation-checkpoint-integrity/.condor-venv-968f64415c1731fa`.
  - The resolved parent was exactly
    `/u/ayman27/activation-checkpoint-integrity`; the target existed as a
    directory, was not a symbolic link, was ignored by Git, and its basename
    exactly matched `.condor-venv-` plus the first 16 characters of the current
    dependency-lock SHA-256.
  - Lock SHA-256 remained
    `968f64415c1731fa2729ecb519169d74389d661ce1c78c9aac0a7d4c72c6e72e`.
  - Measured target size before removal: 6,297,092 KiB, reported by `du -sh` as
    6.1 GiB.
- **Action**: Removed only that exact generated environment. No cache, tool,
  source, lock, submit description, job log, artifact, or other data was
  removed. The environment is reproducible from
  `condor/requirements-cu126.lock` with project-local `uv` 0.12.3.
- **Post-removal evidence**:
  - Confirmed the exact target no longer existed.
  - Created a 28-byte file under `artifacts/condor`, wrote, flushed, fsynced,
    read back the exact content, and removed it; the probe passed.
  - Remaining remote sizes: `.git` 572 KiB, `.condor-tools` 54 MiB,
    `artifacts` 32 KiB, `condor` 32 KiB, `src` 188 KiB, and `tests` 56 KiB.
  - The system has no `quota` command, so an assigned numeric quota and total
    account usage remain unknown.
  - After storage was freed, `git pull --ff-only` succeeded. Remote HEAD,
    remote `origin/main`, local HEAD, and local `origin/main` all matched
    `05b477a9b3246b56b4be6a64c04280ebbfad1150`, and both worktrees were clean.
  - Retained job hashes remained: `1553506.log`
    `00db2a17266c47119a4d98330222ba4339a796e292edc67606c465d89f66ff16`,
    `1553506.0.out`
    `27351240b313c7dc47442bf2e32fd0f15a0d24345225b035f35b3f836e313c65`,
    and `1553506.0.err`
    `e0334b16ed96b9f8531b4ed38571c592f694a9b036c5b73e7d6be27d305cbda9`.
  - Retained submit-description hash remained
    `bca9cd94ef40b02378aff70064c8130e0f23164089c5f6bc17672ad277ca31de`.
- **Outcome**: The authorized persistent-environment removal and post-removal
  storage probe completed. Prior failed-job evidence, hash-locked requirements,
  project-local tooling, source, and submit specification remain intact.
- **Failure or caveat**: The failed first pull is additional direct evidence of
  submit-side quota exhaustion before removal. The successful post-removal
  write and pull show usable space was restored, but do not quantify the quota
  or prove the environment was the sole cause. The scratch environment has not
  yet been built because that action belongs inside the one scheduled retry.
- **Scientific result**: None. E0–E2 logic did not execute and job 1553506.0
  remains `BLOCKED — REMOTE_DISK_QUOTA`.
- **Analyst verdict / interpretation**: Pending for the remediation evidence;
  no scientific interpretation is made here.
- **Next actions**: Commit and synchronize this record, re-query live Condor
  state, run the scheduler dry-run, and submit exactly one newly identified
  retry. Preserve its scratch-bootstrap, preflight, tests, E0/E1/E2, scheduler,
  and artifact evidence without another substitute if it blocks.
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
