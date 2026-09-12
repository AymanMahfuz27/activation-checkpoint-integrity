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

- **Last updated**: 2026-09-11 CDT.
- **Active contract**: User requests the four follow-through steps: GPU reproduction,
  first-intermediate localization, recording noninterference and optimizer guard,
  plus initial adapter-free upstream LLM screening. The older full research plan
  remains context; the exact Notion project and production child were read today.
- **Repository**: Local and origin/main at `d63d326`; follow-through implementation
  is local and tested. Publication permission requested and pending under AGENTS.md.
  Earlier uncommitted research-log entries are preserved.
- **Implemented**: Existing research trainer plus lossless compressed/deduplicated
  full capture, warmed-state fresh-process causal suite, upstream Transformers
  Llama integration/screen, complete-trace versus optimizer-commit distinction,
  isolated scheduled source clones and checksum-verified bounded archive transfer.
- **Verified now**: 44 tests pass; two causal miniature suite runs pass (final run
  10 gates); five upstream CPU smoke cells EXACT_MATCH over two steps, including
  dropout, SDPA, BF16 autocast and aot_eager compilation. No natural failure found
  in those small cells. Final GPU submit profiles pass scheduler dry-run.
- **Existing real-model evidence**: Historical 40M CPU capture-off trigger gives
  73 differing gradients and different updates with identical forward losses;
  trigger-off restores equality. Previous 125M GPU fit probe passed. These are
  not the new GPU causal/capture results, which have not run yet.
- **Storage/access**: UT Condor access works; home quota15GiB, used9GiB. Cluster
  scratch quota exists but no writable user directory. Scheduled scratch plus
  acknowledged local archive transfer is prepared. TACC BatchMode login fails;
  password/token authentication and allocation verification remain outstanding.
- **Pending**: Commit/push authorization, corpus transfer and synchronized GPU
  runs; retain and analyze full evidence, then explain actual results. TorchTitan,
  native GPU BF16/FP8, saved-tensor-complete coverage, optimized fingerprints,
  2,000-step controls and low-overhead performance claims remain unestablished.

## Planned actions

| ID | Priority | Action | Reason | Dependencies | Status |
|---|---:|---|---|---|---|
| A001 | 1 | Commit and push `AGENTS.md` and `RESEARCH_LOG.md` when authorized | Makes the operating rules available to GitHub and remote agents | User authorization to commit/push | Completed |
| A002 | 2 | Pull the current operational commit into the `darmok` checkout | Keeps remote work under the same instructions and code | A001 and stable CPU suite | Completed at `bb95225` |
| A003 | 2 | Pin compatible Python, PyTorch, and CUDA versions and create a dedicated remote environment | Existing base Conda environment is not reproducible | A016 and stable CPU suite | Completed; 6.1 GiB environment exposed quota blocker |
| A004 | 2 | Add a metadata-printing Condor runner and submit file | Required for reproducible scheduled work | A003 | Completed; scheduler dry-run PASS |
| A005 | 2 | Run a one-GPU Condor probe | Verify driver, CUDA, GPU, memory, compute capability, and PyTorch CUDA availability | A004 | Completed in 1553510.0; deterministic CUDA tensor kernel PASS |
| A006 | 2 | Define the first falsifiable hypothesis, baseline, metrics, and falsification criterion | No scientific run should start without a preregistered contract | Notion plan and source audit | Completed |
| A007 | 3 | Verify and document TACC access, allocation, paths, scheduler, and environment | Required for final modern-GPU evaluation | Live TACC access/allocation | Blocked |
| A008 | — | Select and exactly replay R0 in its documented pinned environment | Preserved production-shaped evidence track | A003 and compatibility review | Superseded by active production contract L0023 |
| A009 | — | Build the genuine small decoder-only LM pipeline and run M0.1 | Preserved production-shaped clean oracle | A003 plus frozen data/model contract | Superseded by active production contract L0023 |
| A010 | — | Add one controlled mutable-state fault to the same LM and run M0.2 | Preserved production-shaped causal chain | M0.1 passes | Superseded by active production contract L0023 |
| A011 | — | Preserve R0's natural trigger inside the genuine LM and run M0.3 | Preserved production-shaped natural-trigger proof | R0 and M0.1 | Superseded by active production contract L0023 |
| A012 | — | Add the remaining fault families one at a time | Preserved production-shaped breadth work | Production-shaped Milestone 0 | Superseded by active production contract L0023 |
| A013 | — | Measure production capture and saved-tensor coverage | Defines the honest supported production scope | Production-shaped Milestone 0 and A012 | Superseded by active production contract L0023 |
| A014 | — | Implement and validate position-sensitive GPU fingerprints | Tests the proposed low-memory detector against the exact oracle | A013 | Superseded by active production contract L0023 |
| A015 | — | Integrate one pinned TorchTitan workload | Tests external validity after detector work | Modern GPU access and later detector milestones | Superseded by active production contract L0023 |
| A016 | 1 | Pin the local `uv` environment and exact stable PyTorch and NumPy versions | Makes starter results reproducible | None | Completed |
| A017 | 1 | Implement only the E0–E2 starter package, CLI, artifact writer, and focused tests specified in L0007 | Builds the selected weekly scope without production-shaped expansion | A016 | Completed |
| A018 | 1 | Run the full automated test suite, then make E0 pass | E0 is the trust gate for the fixture and comparator | A017 | Completed |
| A019 | 1 | Run E0 three times, every E1 control/fault pair three times including both RNG variants, and E2 on the Apple Silicon CPU | Produces the required repeatable calibration and coverage evidence | A018 | Completed and analyzed |
| A020 | 1 | Analyze and append every CPU result, failure, artifact path, and verdict | Results cannot drive later work until reviewed and archived | A019 | Completed in L0012 |
| A021 | 2 | Repeat unchanged E0–E2 logic in FP32 inside one scheduled Condor GPU job | Portability confirmation only; it does not broaden CPU verdicts | A020, A022, A023, and L0015 authorization | Completed once as 1553510.0; E0/E1 `PROMOTE`, exact E2 candidate `KILL` in L0019 |
| A022 | 1 | Move `ArtifactWriter` setup into classified exception handling and add writable-artifact and CUDA-kernel preflights | Ensures setup failures return exit 2 and kernel incompatibility is detected before scientific execution | L0014 diagnosis | Completed in L0016 at `05b477a`; synchronized locally, on origin, and remotely |
| A023 | 1 | Preserve prior logs/spec, remove only `.condor-venv-968f64415c1731fa`, and rebuild the same locked environment in per-job scratch | Frees the likely 6.1 GiB home-storage contributor without changing scientific logic or destroying prior evidence | A022 local verification and L0015 authorization | Completed: removal in L0017 and scratch rebuild in retry 1553510.0 |
| A024 | 1 | Repair aggregate provenance to derive the actual child device and add a regression test | Historical E1/E2 aggregates hardcode CPU while authoritative child evidence records CUDA | L0019; preserve historical artifacts | Completed in L0020; future aggregate manifests record selected device/dtype/environment, no scientific rerun |
| A025 | 1 | Correct transform-scale derivatives in `independent_reference` for Python/NumPy RNG and add focused tests | The forward reference applies scale but its `x`/`w1` derivatives omit it, making the diagnostic false on CPU and GPU | L0019; same-state E1 gates remain authoritative | Completed in L0020; Python and NumPy focused references pass, no scientific rerun |
| P001 | 1 | Reproduce exact #84864 and preserve source/control evidence | R0 | None | Implemented and current symptom reproduced; review pending |
| P002 | 1 | Build decoder, corpus, lifecycle, snapshot and fresh replay | Phase B | P001 | Implemented; corpus frozen; miniature and three-step 40M checks pass |
| P003 | 1 | Validate exhaustive eager capture and pre-update gate | Phase C | P002 | Implemented; miniature/adversarial validation in progress |
| P004 | 2 | Execute full 40M controls and M0.1/M0.3 | Real-model evidence | Corpus, quota, scheduled compute | Pending |
| P005 | 2 | Execute controlled mechanism suite | M0.2 | Accepted M0.3 | Gated; not executed |
| P006 | 2 | Run 125M CUDA 12.6 fit probe/control | Phase E | Explicit synchronization, scheduler and quota | GPU fit passed in 1553517.0; long control pending |
| P007 | 3 | Implement pinned TorchTitan extension and modern GPU proof | Production transfer | Verified compatible stack and hardware | Pending |

| F001 | 1 | Lossless capture and warmed-state proof suite | Close full-model evidence gates | Local tests | Implemented; 10 miniature gates pass |
| F002 | 1 | Execute and archive full-size GPU causal suite | Establish CUDA localization/noninterference/enforcement | Publication permission and scheduled scratch | Ready; permission pending |
| F003 | 1 | Adapter-free upstream Llama screen | Begin production-feature compatibility | Pinned upstream environment | Five CPU smoke cells pass; GPU screen ready |
| F004 | 1 | Analyze measured GPU results and teach mechanisms with code | User requested explanation and conclusions | F002 and F003 | Pending actual runs |


## Experiment index

| ID | Date | Hypothesis | Baseline | Status | Verdict | Record |
|---|---|---|---|---|---|---|
| Follow-through smoke | 2026-09-11 | Recording preserves a warmed-state controlled failure and blocks only unsafe updates | Fresh no-capture/reference/repeat/trigger-off arms | Two runs PASS, final 10 gates | PROMOTE infrastructure only | L0039-L0041 |
| Upstream CPU smoke | 2026-09-11 | Unmodified upstream Llama checkpoint arms agree within each execution setting | Fresh repeated reference and checkpoint arms | Five cells EXACT_MATCH, two steps each | Bounded clean result | L0041 |

| E0 | 2026-09-03/04 | The fixed clean fixture, exact comparator, and independent formulas agree across no-checkpoint, original, and recompute execution | Fixed literal CPU `float64`/CUDA FP32 no-checkpoint runs and independent formulas | Three CPU and three GPU repetitions pass; bounded oracle; identity formula reconfirmed after diagnostic fix | PROMOTE CPU and GPU | L0007, L0009, L0010, L0012, L0018, L0019, L0020 |
| E1 | 2026-09-03/04 | Each of five hidden-state fault families causes a same-metadata value mismatch first at `h` and a gradient difference, while its control and trigger-disabled arm remain exact | E0 plus a fresh-process correct arm with identical tensors, seeds, and state | CPU and GPU suites each pass all 54 fresh arms; future Python/NumPy formula diagnostic corrected | PROMOTE CPU and GPU | L0007, L0009, L0010, L0012, L0018, L0019, L0020 |
| E2 | 2026-09-03/04 | Public `context_fn` and inner `saved_tensors_hooks` capture each explicitly backward-relevant `h`, `g`, and `y` once per phase without changing behavior | E0 no-hook result and checkpoint baseline without observational hooks | CPU and bundled CUDA runs both show baseline 1/1, candidate 1/0 | KILL exact candidate | L0007, L0009, L0010, L0012, L0018, L0019 |
| Condor 1553506.0 | 2026-09-03 | Unchanged E0–E2 gates remain valid under the preregistered Pascal CUDA FP32 environment change | Archived CPU E0/E1/E2 results | `BLOCKED — REMOTE_DISK_QUOTA`; no GPU experiment ran | REPEAT only after separately recorded storage remediation; one retry authorized by L0015 | L0011, L0013, L0014, L0015 |
| Condor 1553510.0 | 2026-09-04 | Unchanged E0–E2 gates remain valid under the preregistered Pascal CUDA FP32 environment change | Archived CPU E0/E1/E2 results and preserved blocked job 1553506.0 | Completed and analyzed; bundled portability evidence only | E0 `PROMOTE`; E1 `PROMOTE`; exact E2 candidate `KILL` | L0015, L0016, L0017, L0018, L0019 |
| R0 | 2026-09-02 | A documented natural activation-checkpoint bug reproduces in its pinned original environment without a project-invented fault | Exact upstream safe/failing comparison | Reactivated by L0023; acceptance pending | — | L0006, L0007 |
| M0.1 | 2026-09-02 | Clean no-checkpoint and checkpointed genuine LM pretraining agree | Identical data/model/optimizer state | Reactivated by L0023; acceptance pending | — | L0006, L0007 |
| M0.2 | 2026-09-02 | A controlled mutable-state fault changes a recomputed LM activation, gradient, and optimizer update while metadata remain equal | M0.1 clean oracle | Reactivated by L0023; acceptance pending | — | L0006, L0007 |
| M0.3 | 2026-09-02 | R0's documented natural trigger produces the same failure class inside the genuine LM pipeline | R0 and M0.1 | Reactivated by L0023; acceptance pending | — | L0006, L0007 |
| M1 | 2026-09-02 | Remaining fault families reproduce one at a time in the LM or bounded fixtures | Production-shaped Milestone 0 | Reactivated by L0023; acceptance pending | — | L0006, L0007 |
| M2 | 2026-09-02 | Capture path covers every site in each claimed checkpoint pattern | Production-shaped Milestone 0 evidence | Reactivated by L0023; acceptance pending | — | L0006, L0007 |
| M3 | 2026-09-02 | Exact debug checking and position-sensitive GPU fingerprints detect oracle-confirmed mismatches | M1 and M2 | Reactivated by L0023; acceptance pending | — | L0006, L0007 |
| M4 | 2026-09-02 | Longer LM runs establish downstream consequence and reproducibility | M3 | Reactivated by L0023; acceptance pending | — | L0006, L0007 |
| M5 | 2026-09-02 | TorchTitan and modern-GPU validation establish external compatibility and overhead | M3 and modern GPU access | Reactivated by L0023; acceptance pending | — | L0006, L0007 |

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

### L0018 — 2026-09-04 13:20 CDT — Condor retry completed with E0–E2 results

- **Type**: experiment / setup / success / result
- **Status**: completed once; pending Analyst review
- **Objective**: Execute the one L0015-authorized retry of unchanged E0–E2
  scientific logic in the preregistered Pascal CUDA FP32 environment after the
  separately recorded storage remediation.
- **Context consulted**: L0012 CPU verdicts, blocked attempt L0013–L0014,
  authorization L0015, implementation L0016, remediation L0017, exact commit,
  live pool/queue, scheduler dry-run, job history, stdout/stderr, and every
  aggregate and child manifest/summary from retry 1553510.0.
- **Repository state**: Before submission, local `main`, local `origin/main`,
  remote `main`, and remote `origin/main` were clean and equal at
  `7cdce891c8ef4d9418f1ece0a8dbcb88e5accc67`. The scheduled job and all 59
  scientific child manifests record that commit with `dirty=false`.
- **Experiment ID**: E0/E1/E2 Condor FP32 portability retry, scheduler job
  1553510.0.
- **Parent idea / branch**: L0007 starter plan, L0012 CPU results, L0014
  conditional `REPEAT`, and L0015 exact authorization.
- **Falsifiable hypothesis**: The unchanged E0–E2 gates remain valid when the
  recorded environment changes from Apple CPU/PyTorch 2.13.0 to Pascal CUDA
  FP32/PyTorch 2.12.1+cu126.
- **Falsification criterion**: Any setup/preflight failure blocks the result.
  Otherwise, any E0 clean-oracle gate failure, E1 case/repetition gate failure,
  or change from E2's declared coverage behavior contradicts the corresponding
  bounded portability expectation and requires Analyst review.
- **Baseline and metrics**: Archived CPU results in L0010/L0012 and blocked job
  1553506.0. Recorded metrics include setup and CUDA preflights, automated
  tests, E0 tensor/phase/formula/gradient gates, E1 all case/repetition/control/
  mismatch/gradient/trigger-restoration gates, E2 baseline/candidate phase and
  hook counts, exact environment, scheduler facts, and full tensor artifacts.
- **Change introduced and configuration diff**:
  - CPU -> NVIDIA GeForce GTX 1080 Ti CUDA device.
  - E0/E2 `float64` -> `float32`; E1 remains `float32`.
  - PyTorch 2.13.0 macOS -> 2.12.1+cu126 Linux because the locked CUDA 12.6
    environment is required for Pascal.
  - Operationally, the same lock-derived environment was built under Condor
    scratch and artifact/CUDA-kernel preflights ran before tests or science.
  - Seed 20260903, deterministic settings, fixtures, cases, triggers, controls,
    capture candidate, gates, three E0 runs, six E1 scenarios, three E1
    repetitions, 54 fresh E1 processes, and E2 logic were unchanged.
- **Parameter count**: N/A — fixed tiny fixtures; no model or scientific source
  changed.
- **Pre-submit evidence**:
  - Local and remote repositories were clean at the exact commit.
  - Live eligible pool contained 57 unclaimed/idle GTX 1080 Ti slots.
  - `ayman27` owned zero queued jobs. The global queue had 3,586 jobs: 3,519
    idle, 65 running, and 2 held.
  - `/lusr/opt/condor/bin/condor_submit -dry-run /dev/stdout
    condor/starter.submit` emitted exactly one valid job with the unchanged
    requests and requirements.
- **Exact submission and scheduler record**:
  - Exact command: `/lusr/opt/condor/bin/condor_submit condor/starter.submit`.
  - Exactly one job was submitted as 1553510.0 at
    `2026-09-04T18:09:27Z`; it began at `2026-09-04T18:09:28Z` on
    `slot1@eldar-44.cs.utexas.edu` and ended normally at
    `2026-09-04T18:17:56Z` with scheduler return value 0.
  - One start, no restart, 508 seconds remote wall time, 471 seconds user CPU,
    97 seconds system CPU, 489 MiB recorded memory, and 288 MiB GPU memory in
    the terminal scheduler log.
- **Environment and hardware**:
  - Scratch environment:
    `/var/condor/execute/dir_3377999/aci-condor-venv-968f64415c1731fa`.
  - Lock SHA-256:
    `968f64415c1731fa2729ecb519169d74389d661ce1c78c9aac0a7d4c72c6e72e`;
    project-local `uv` 0.12.3; CPython 3.13.5; PyTorch 2.12.1+cu126; NumPy
    2.5.2; CUDA runtime 12.6; cuDNN 9.10.2; driver 535.247.01.
  - NVIDIA GeForce GTX 1080 Ti, compute capability 6.1, and 11,714,887,680
    bytes reported by PyTorch.
- **Preflight and test results**:
  - Artifact preflight `PASS`: created the artifact root, wrote 51 bytes,
    flushed/fsynced, read the exact content, and removed the probe.
  - CUDA tensor-kernel preflight `PASS`: synchronized one FP32 2x2
    matmul-plus-bias on `cuda:0` and matched exact expected result
    `[[0.25, 6.0], [2.25, 13.0]]`.
  - Automated suite: 19 passed in 66.16 seconds with cache writes disabled.
- **E0 observed results**:
  - Three runs all reported `PASS`:
    `e0-counter-correct-20260904T181213.251421Z-563c04d9`,
    `e0-counter-correct-20260904T181219.054351Z-ade699c2`, and
    `e0-counter-correct-20260904T181224.856296Z-296db303`.
  - Each recorded original/recompute phase counts 1/1, three exact pairs,
    zero value or metadata mismatches, exact independent-formula agreement,
    no gradient differences, exit 0, CUDA FP32, seed 20260903, and job
    1553510.0.
- **E1 observed results**:
  - Aggregate run `e1-all-20260904T181230.773563Z-27b4793b` reported `PASS`
    for six scenarios: counter, buffer, Python RNG, NumPy RNG, precision, and
    toy FP8 scaling.
  - All 54 unique fresh-process child arms reported `PASS`, phase counts 1/1,
    three pairs, zero metadata mismatches, and no default PyTorch check.
  - The 18 correct arms had three exact pairs and no gradient difference. The
    18 broken arms had three same-metadata value mismatches, first mismatch at
    `h`, and at least one gradient difference. The 18 trigger-off arms had
    three exact pairs, no gradient difference, and trigger restoration true.
  - Each scenario recorded three repetitions, identical scientific signatures
    and tensor contents across repeats, broken-original equality with the
    correct original, and trigger restoration.
- **E2 observed results**:
  - Aggregate run `e2-coverage-20260904T181743.638925Z-bd490760` reported
    `PUBLIC_HOOKS_INSUFFICIENT`, which the preregistered runner accepted while
    returning overall scheduler exit 0.
  - No-hook baseline
    `e2-counter-correct-20260904T181746.258627Z-82f42e43` reported `PASS`,
    phases 1/1, and child exit 0.
  - Public-hook candidate
    `e2-counter-correct-hooks-20260904T181752.258091Z-e9164fcb` reported
    `PUBLIC_HOOKS_INSUFFICIENT`, phases 1/0, missing recompute `h`, `g`, and
    `y`, hooks-preserve-behavior false, and child exit 1. Original-phase direct
    matches were one per tag, while recompute-phase direct matches and all hook
    counts were zero.
- **Raw artifacts**:
  - Scientific root:
    `/u/ayman27/activation-checkpoint-integrity/artifacts/starter` — 3.2 MiB
    after the run, with 595 files newer than submission.
  - Scheduler log/stdout/stderr:
    `/u/ayman27/activation-checkpoint-integrity/artifacts/condor/1553510.log`,
    `1553510.0.out`, and `1553510.0.err`.
  - SHA-256 values: log
    `c2ac854c7373cb8510e3346e02549fb9d0efc716b8eec8d67f626f8158093128`,
    stdout
    `98bc8098dc7f082496fba7168ef92c9f41289502929373efa7541d9c69620fb0`,
    and stderr
    `607cf488c4383602446cd818fc3e780cad2127655e7466ae4645a856c0c3bea0`.
- **Baseline integrity**: Every E0 run, all 54 E1 child runs, and both E2 child
  runs identify clean commit `7cdce89`, CUDA FP32, seed 20260903, Condor job
  1553510.0, PyTorch 2.12.1+cu126, and compute capability 6.1. The archived CPU
  records and blocked-job evidence were not modified.
- **Validity caveat**: `_aggregate_writer` has a pre-existing metadata defect:
  the E1 and E2 aggregate manifests report `environment.device=cpu` because the
  aggregate writer hardcodes a CPU device for environment capture. Their exact
  commands and timestamps are intact, and all 54 E1 plus both E2 scientific
  child manifests record CUDA correctly. This must be considered by Analyst;
  it was not changed after execution and no substitute run is authorized.
- **Result classification**: `RESULTS_AVAILABLE — PENDING_ANALYST`. The strings
  `PASS` and `PUBLIC_HOOKS_INSUFFICIENT` above are runner outputs, not a new
  scientific verdict.
- **Analyst verdict / interpretation**: Pending. No scientific claim is made by
  this entry.
- **Next actions**: Commit and push this factual result record, verify local,
  origin, and remote state, then route the raw evidence to Analyst. Do not run
  another scheduler substitute or begin the deferred production-shaped plan.
- **Secrets**: No secret material recorded.

### L0019 — 2026-09-04 13:30 CDT — GPU E0–E2 Analyst verdicts archived

- **Type**: analysis / decision / success / failure / archival
- **Status**: GPU verdicts complete; two non-gating instrumentation corrections
  required before starter-scope completion
- **Objective**: Archive the Analyst's separate verdicts for the E0, E1, and E2
  results from Condor retry 1553510.0, preserve the exact claim boundaries, and
  record two discovered instrumentation defects without changing historical
  evidence.
- **Context consulted**: L0012 CPU verdicts, L0014 conditional repeat, L0015
  authorization, L0016–L0017 remediation, L0018 factual GPU results, every
  aggregate and child record, raw tensors and comparisons, exact commands, and
  the supplied Analyst review.
- **Repository state**: Local `main`, `origin/main`, and the clean `darmok`
  checkout matched factual-result commit
  `e05660c35100ebae9986789ba6ce2d2dc333bd56` before this archival edit.
- **Environment bundle**: NVIDIA GeForce GTX 1080 Ti, CUDA FP32, Linux,
  PyTorch 2.12.1+cu126, CUDA runtime 12.6, driver 535.247.01, CPython 3.13.5,
  and NumPy 2.5.2. Compared with the CPU baseline, device, dtype, operating
  system, and PyTorch build changed together; the result tests bundled
  portability and does not isolate a hardware, dtype, OS, or framework cause.
- **Baseline integrity**: `CONFIRMED`. All 59 scientific child manifests identify
  clean commit `7cdce891c8ef4d9418f1ece0a8dbcb88e5accc67`, CUDA FP32, seed 20260903,
  job 1553510.0, PyTorch 2.12.1+cu126, and compute capability 6.1. The prior CPU
  artifacts and blocked-job records remain intact.

- **Experiment E0 GPU — clean checkpoint baseline and exact comparator**:
  - **Parent idea / branch**: E0 bounded CPU oracle and L0014 conditional
    portability repeat.
  - **Hypothesis**: The fixed clean fixture preserves original/recompute tensor
    values and metadata and checkpoint/no-checkpoint behavior in the bundled
    Pascal CUDA FP32 environment.
  - **Baseline**: Archived E0 CPU `float64` oracle plus each CUDA run's
    no-checkpoint execution and separately implemented closed-form reference.
  - **Change introduced / config diff**: Apple CPU `float64`, macOS, and PyTorch
    2.13.0 -> GTX 1080 Ti CUDA FP32, Linux, and PyTorch 2.12.1+cu126; fixture,
    seed, comparator, formulas, phase gates, and three-run count unchanged.
  - **Parameter count**: N/A — fixed tiny fixture; no scientific source or
    parameterization changed.
  - **Metrics and artifacts**: All three runs recorded original/recompute phases
    1/1. Saved `h`, `g`, and `y` pairs were bit-exact. Checkpoint and
    no-checkpoint output, loss, and gradients were bit-exact. Independent
    closed-form gradients agreed within the configured `1e-6` tolerance but
    were not all bit-exact: maximum absolute differences were
    `b=0`, `w1=1.1920929e-7`, `w2=0`, and `x=4.7683716e-7`.
  - **Validity**: High for the exact bounded fixture and environment bundle;
    attribution to any one changed environment dimension is not established.
  - **Result classification**: improvement/neutral are not applicable; bounded
    clean-oracle and portability gates passed.
  - **Analyst verdict**: `PROMOTE` as a bounded GPU clean oracle.
  - **Analyst interpretation / lesson**: Exact checkpoint equivalence and
    comparator behavior survive this bundled CUDA FP32 change. Tolerance-level
    closed-form differences are expected FP32 arithmetic-order effects and do
    not contradict the preregistered gate.
  - **Status / follow-up**: Promoted for the bounded regression harness. No
    additional GPU run is required for this verdict.

- **Experiment E1 GPU — five controlled fault families**:
  - **Parent idea / branch**: Promoted E0 bounded oracle and the CPU E1
    controlled-fault suite.
  - **Hypothesis**: Each fault family produces the preregistered same-metadata
    value and gradient divergence while correct and trigger-disabled controls
    remain exact in the bundled CUDA FP32 environment.
  - **Baseline**: Same-state no-checkpoint and correct checkpoint arms with
    identical fixture tensors, seeds, and process isolation.
  - **Change introduced / config diff**: CPU-to-CUDA environment bundle only;
    FP32 dtype, seed, six scenarios, three repetitions, 54 fresh processes,
    fault triggers, controls, and causal gates unchanged.
  - **Parameter count**: N/A — fixed tiny fixtures; no scientific source or
    parameterization changed.
  - **Metrics and artifacts**: Authoritative aggregate
    `e1-all-20260904T181230.773563Z-27b4793b`. All 54 fresh child arms passed:
    18 correct arms were exact, 18 broken arms first mismatched at `h` with
    same metadata and a gradient difference, and 18 trigger-disabled arms were
    exact with restoration true. All scenario and repeatability gates passed.
  - **Validity**: High for the preregistered causal gates across these six
    bounded fixtures; it does not estimate natural-failure prevalence.
  - **Result classification**: bounded controlled-fault regression suite passed.
  - **Analyst verdict**: `PROMOTE` as a bounded GPU regression suite only.
  - **Analyst interpretation / lesson**: The same-state autograd controls and
    broken-arm divergences preserve the intended causal chain across the
    bundled CUDA environment change.
  - **Status / follow-up**: Promoted for bounded regression use. The
    independent-formula instrumentation defect below is excluded from the E1
    pass gate and does not change this verdict.

- **Experiment E2 GPU — exact public-hook capture candidate**:
  - **Parent idea / branch**: E0 bounded oracle plus the exact public
    `context_fn` and inner `saved_tensors_hooks` composition killed on CPU.
  - **Hypothesis**: The candidate observes backward-relevant `h`, `g`, and `y`
    once in original and recompute without changing checkpoint behavior.
  - **Baseline**: No-hook CUDA FP32 checkpoint execution.
  - **Change introduced / config diff**: No-hook baseline -> exact public-hook
    candidate within the same GPU environment; relative to CPU, the complete
    environment bundle changed.
  - **Parameter count**: N/A — fixed tiny fixture; no model parameters.
  - **Metrics and artifacts**: Authoritative aggregate
    `e2-coverage-20260904T181743.638925Z-bd490760`. The no-hook baseline
    recorded phases 1/1; the exact candidate recorded phases 1/0, missed
    recompute `h`, `g`, and `y`, and did not preserve behavior.
  - **Validity**: High for this exact candidate on the tested CPU and bundled
    CUDA environments; it does not cover all possible public PyTorch APIs or
    compositions.
  - **Result classification**: replicated exact-candidate coverage failure.
  - **Analyst verdict**: `KILL` the exact candidate.
  - **Analyst interpretation / lesson**: The candidate suppresses recomputation
    in both tested environment bundles. Replication strengthens the negative
    result for this composition but is not a universal public-API insufficiency
    claim.
  - **Status / follow-up**: Preserve as negative evidence. Do not revive this
    exact candidate without a new mechanism or new evidence.

- **Artifact identity correction**: The aggregate IDs included earlier in the
  Analyst handoff prompt did not exist. The authoritative aggregate artifacts
  are `e1-all-20260904T181230.773563Z-27b4793b` and
  `e2-coverage-20260904T181743.638925Z-bd490760`; these match L0018 and the
  retained remote artifacts.
- **Non-gating defect A — aggregate provenance**:
  - `_aggregate_writer` hardcodes a CPU device only in aggregate provenance.
    Every E1/E2 child manifest, comparison, exact command, raw tensor, and
    scheduler record proves CUDA execution, so child records are authoritative
    for the historical GPU result.
  - Preserve the historical aggregate files unchanged and append this
    correction rather than rewriting executed evidence. Repair the writer to
    derive truthful provenance and add a regression test. No scientific rerun
    is required solely for this metadata correction.
- **Non-gating defect B — independent reference derivatives**:
  - `independent_reference` applies the transform scale in its forward formula
    but omits that scale from the `x` and `w1` derivatives for the Python RNG
    and NumPy RNG cases. Consequently,
    `no_checkpoint_matches_independent_formulas=false` for those cases on both
    CPU and GPU.
  - E1 intentionally excludes this diagnostic from its causal pass gate, and
    its same-state autograd baseline remains valid. Correct the reference
    derivatives and add focused tests before starter-scope completion. This
    defect does not retroactively alter the E1 verdict or require a scientific
    GPU rerun by itself.
- **Claim boundary**: These verdicts establish bounded fixture behavior across
  one bundled CPU-to-Pascal-CUDA environment change. They do not establish
  hardware-isolated causality, production capture coverage, natural-failure
  prevalence, training consequence, detector overhead, TorchTitan integration,
  modern-GPU behavior, or general public-API insufficiency.
- **Next actions**: Complete A024 and A025, preserve all raw and historical
  aggregate artifacts, rerun focused/full regression tests, and archive the
  correction evidence before declaring the starter scope complete. Do not run
  another scientific GPU job solely for these non-gating corrections.
- **Secrets**: No secret material recorded.

### L0020 — 2026-09-04 13:46 CDT — Starter diagnostics corrected and scope completed

- **Type**: setup / implementation / verification / documentation / success
- **Status**: completed; E0–E2 starter scope complete
- **Objective**: Complete A024 and A025 without changing E1 causal gates,
  historical artifacts, archived verdicts, or the deferred production-shaped
  plan; restore the documented local commands and replace the planning-only
  README with a bounded implementation and result description.
- **Context consulted**:
  - `AGENTS.md`, `README.md`, the complete user-attached starter plan, source,
    tests, scripts, and `RESEARCH_LOG.md` through uncommitted L0019.
  - Exact Notion page `MOOTAZ PROJECT`, page ID
    `3c65d66f-2c23-80c6-879f-e0aff5e92706`, last edited
    `2026-09-03T04:15:43.249Z`, and its child `Starter E0–E2 Plan: Five Tiny
    Checkpoint Failures and Capture Validation`, page ID
    `3d05d66f-2c23-8068-b52a-ca774781771e`, last edited
    `2026-09-03T04:15:45.303Z`.
- **Repository state**: Work began from local `main` at
  `e05660c35100ebae9986789ba6ce2d2dc333bd56`, matching `origin/main` and the
  clean `darmok` checkout. L0019 was preserved as an uncommitted append-only
  change and extended rather than rewritten or discarded.
- **A024 implementation — aggregate provenance**:
  - `_aggregate_writer` now receives the requested device and dtype for both E1
    and E2, records normalized `device` and `dtype` fields, and builds the
    aggregate environment record from that selected device instead of CPU.
  - Aggregate provenance capture does not require local CUDA availability;
    child arms still validate whether the selected execution device exists.
  - A regression uses a mocked unavailable-CUDA state with selected `cuda:0`
    and proves that `device=cuda:0`, `dtype=torch.float32`, and
    `environment.device=cuda:0` survive into the aggregate manifest.
- **A025 implementation — independent derivative formulas**:
  - For `A = x @ W1`, `h = T(A) + b`, the reference now computes
    `dA = VJP_T(dh)`, then `dx = dA @ W1.T` and `dW1 = x.T @ dA`. The bias and
    second-matrix derivatives remain `db = sum(dh)` and `dW2 = g.T @ dy`.
  - The base controller's original transform has the identity vector-Jacobian
    product. The Python and NumPy RNG controllers multiply that gradient by
    the same fixed scale used by their original forward transform. The toy FP8
    straight-through transform correctly retains the identity derivative.
  - Focused Python and NumPy tests compare all four closed-form gradients with
    autograd at `1e-12` tolerance. The pre-existing E0 identity-transform test
    still requires bit-exact forward, loss, and gradient formulas and passed.
- **Local environment repair**:
  - Initial exact `uv run pytest` failed during collection with
    `ModuleNotFoundError: No module named 'ac_integrity'` even though the
    distribution metadata existed. CPython 3.13.15 verbose startup showed that
    it skipped the editable-install path file because the migrated local
    `.venv` and its `.pth` descendants carried the macOS `UF_HIDDEN` flag; the
    environment also contained a duplicate editable path file with ` 2` in its
    name.
  - The generated, ignored environment was rebuilt as a normal directory with
    `uv sync --locked`, which recreated one non-hidden editable path file. No
    tracked dependency changed, no external `PYTHONPATH` is required, and bare
    `uv run pytest` now imports the source package from `src/ac_integrity`.
- **Documentation**: `README.md` now explains the file responsibilities, exact
  local commands, E0/E1/E2 measurements, bounded CPU and GTX 1080 Ti results,
  raw Condor job 1553510.0, E2's exact-candidate limitation, ignored artifact
  locations, the historical aggregate provenance disposition, and explicit
  production exclusions.
- **Parameter count**: N/A — fixed literal fixtures and diagnostics only; no
  model architecture or parameter set changed.
- **Configuration diff**: No scientific experiment configuration changed.
  Future aggregate provenance changed from an unconditional CPU label to the
  selected device and dtype. The independent formula changed only from using
  `dh` directly in the `x` and `W1` derivatives to applying the original
  transform's vector-Jacobian product first.
- **Verification evidence**:
  - Focused provenance, E0 identity-formula, and two RNG derivative tests:
    `4 passed`.
  - Exact required full command `uv run pytest -q`: `22 passed in 21.02s`.
  - From the freshly rebuilt normal `.venv`, documented bare
    `uv run pytest`: `22 passed in 37.27s`.
  - `uv lock --check` found the lock current; `uv sync --check` found all 17
    installed packages synchronized and would make no changes.
  - CLI smoke root `/tmp/aci-cli-smoke.PLWFdV`: E0 `PASS`; Python RNG E1
    `PASS`; NumPy RNG E1 `PASS`; both future correct-arm formula diagnostics
    true; E2 returned the expected exit 1 and `PUBLIC_HOOKS_INSUFFICIENT`.
    The two E1 aggregate smoke manifests recorded CPU and `torch.float32`.
  - `bash -n scripts/bootstrap_condor_env.sh scripts/run_condor_starter.sh` and
    `git diff --check` passed.
  - SHA-256 inventories of all 625 files under the retained local `artifacts/`
    tree matched exactly before and after the work. No remote artifact was
    written, rewritten, or deleted, and no Condor job was run.
- **Baseline preservation**: Confirmed for the implementation. All 22 tests
  pass, E0's original exact formula test passes, the E1 scientific gate code is
  unchanged, and the E2 candidate and gate are unchanged. A024/A025 affect
  future provenance and formula diagnostics only.
- **Outcome**: A024 and A025 are complete. The starter package is reproducible
  from the documented local commands, and the README and current state now
  describe the completed bounded phase rather than planned work.
- **Scientific disposition**: No new scientific experiment or verdict. Per the
  L0019 Analyst review, neither correction requires a scientific rerun and
  neither changes the archived E0/E1 `PROMOTE` or exact E2 candidate `KILL`.
- **Failure or caveat**: Historical E1/E2 aggregate manifests intentionally
  retain their incorrect CPU device label, with child manifests authoritative.
  E2 rejects only the exact tested public-hook composition. Production-shaped,
  overhead, TorchTitan, distributed, native-FP8, BF16, and modern-GPU work
  remain deferred.
- **Next actions**: Completed after this entry and recorded in L0021: the
  coherent implementation/results/documentation change was committed as
  `3ae4aa9429a168a2c416a6bb905c6f1878aa58c3`, pushed, fast-forwarded on
  `darmok`, and verified clean and equal. Keep the production-shaped plan
  deferred until separately activated.
- **Secrets**: No secret material recorded.

### L0021 — 2026-09-04 13:51 CDT — Starter record synchronized

- **Type**: archival / synchronization / verification
- **Status**: completed through verified implementation commit; final
  archival-only commit follows this record
- **Objective**: Correct the stale pre-commit repository description and record
  the final synchronized state of the completed E0–E2 starter phase without
  changing code, artifacts, or scientific conclusions.
- **Base and verified implementation commit**:
  `3ae4aa9429a168a2c416a6bb905c6f1878aa58c3`.
- **Verification evidence**:
  - Local `main`, `origin/main`, and the clean `darmok` checkout matched
    `3ae4aa9429a168a2c416a6bb905c6f1878aa58c3` before this archival-only edit.
  - L0020 already records exact `uv run pytest -q` as 22 passed and bare
    `uv run pytest` as 22 passed. No test was rerun for this archival entry.
  - A live read-only queue query showed no Condor job owned by `ayman27`.
- **Change and evidence boundary**: This entry changes only the research log.
  No scientific experiment ran, no Condor job was submitted, and no raw or
  historical artifact was written, rewritten, or deleted. The final archival
  commit cannot record its own SHA and must be reported externally after push.
- **Scientific disposition**: Unchanged from L0019–L0020: E0/E1 remain
  `PROMOTE` for bounded CPU/GPU use, the exact E2 candidate remains `KILL`, and
  no production, overhead, hardware-isolation, or universal public-API claim is
  added.
- **Next actions**: Keep the production-shaped plan deferred until it is
  separately activated; do not infer production, overhead, modern-GPU, or
  universal public-hook claims from E0–E2.
- **Secrets**: No secret material recorded.

### L0022 — 2026-09-04 19:21 CDT — Interrupted starter closure recovered

- **Type**: context / archival / verification / synchronization
- **Initial status**: in progress; documentation-only recovery.
- **Objective**: Finish the interrupted archival commit and synchronization
  after verifying that the selected E0–E2 completion gates are already met.
- **Context consulted**: `AGENTS.md`, `README.md`, current state and L0020–L0021;
  interrupted task `01a06a8c-56af-7621-bc2e-449d1f11a8b4`; exact Notion parent
  `MOOTAZ PROJECT` (`3c65d66f-2c23-80c6-879f-e0aff5e92706`, last edited
  `2026-09-03T04:15:43.249Z`) and its selected starter child
  `3d05d66f-2c23-8068-b52a-ca774781771e` (last edited
  `2026-09-03T04:15:45.303Z`), both successfully re-read.
- **Base commit**: `3ae4aa9429a168a2c416a6bb905c6f1878aa58c3` on `main`;
  the interrupted L0021 documentation edit is preserved.
- **Planned action and expected evidence**: Read-only audit of retained CPU
  artifacts, Condor output and queue, and local/GitHub/remote revisions; retain
  existing Analyst verdicts, then commit and synchronize the archival record
  under the prior push authorization. No new scientific run is needed.
- **Verification outcome**: Local HEAD, GitHub `refs/heads/main`, and the clean
  `darmok` checkout all match the base commit. Local modifications are confined
  to `RESEARCH_LOG.md`; `git diff --check` passes. The live
  `condor_q ayman27` query returns no jobs.
- **Retained evidence audit**: All 62 local run directories contain manifest,
  events, comparisons, and summary files. The three clean CPU E0 runs pass;
  all 54 E1 child arms and their aggregate pass (six scenarios, three
  repetitions, fresh-process isolation). Every retained E0/E1 child contains
  full original/recompute `h`, `g`, and `y` tensor files, with zero missing.
  E2's baseline passes and its candidate/aggregate record
  `PUBLIC_HOOKS_INSUFFICIENT`. The clean CPU run commit is
  `14bce330cd1259192c2d8209ecb0a556346a2534`; earlier dirty smoke evidence remains
  preserved. Exact raw run IDs and paths remain in L0010–L0012.
- **Retained GPU evidence**: `artifacts/condor/1553510.0.out` on `darmok`
  confirms CUDA kernel PASS, 19 tests passed, three E0 passes, E1 all PASS,
  E2 `PUBLIC_HOOKS_INSUFFICIENT`, and completion at
  `2026-09-04T18:17:56Z`. L0018–L0019 retain the exact GPU run provenance and
  analysis. L0020 records 22 tests passed on the final implementation; no
  tests or experiments were rerun for this documentation-only recovery.
- **Completion-gate disposition**: Existing L0012/L0019 Analyst verdicts are
  unchanged: bounded E0/E1 `PROMOTE`, exact E2 candidate `KILL`. The selected
  starter plan explicitly permits an honest E2 insufficiency result. All
  scientific starter work is complete and archived; this recovery adds no
  scientific interpretation or broader coverage claim.
- **Caveats and scope**: Historical aggregate-device and RNG-formula diagnostic
  corrections remain as documented in L0019–L0020. Production-shaped work is
  still deferred. No code or historical artifacts changed.
- **Final status**: Recovery verification complete; archival commit, push, and
  remote synchronization follow this entry. Report their resulting SHA
  externally to avoid a self-referential archival commit.
- **Next actions**: Synchronize this final archival record under the existing
  authorization, then stop at the completed starter scope.
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

### L0023 — 2026-09-06 CDT — Production plan activated

- **Status**: RUNNING.
- **Objective**: Implement the user-supplied Production-Shaped Activation Checkpoint Reproduction and Capture Plan, archived verbatim at `docs/production-plan.md`.
- **Base**: `0241f1bda17f1a7044382c0086fdf4b03e0764b7`; README already modified by user, preserved. Starter history and implementation are retained as historical evidence; E0–E2 are excluded from the new program.
- **Context**: AGENTS.md, README.md, current state and latest ledger entries read. No Notion connector available; exact parent URL web retrieval failed; browser retrieval in progress. The attached complete plan is sufficient to implement without substituting another page.
- **Preregistered R0**: Execute the exact #84864 issue code on locked local PyTorch 2.13.0. Hypothesis: unpreserved dispatch mode changes backward gradient without a metadata exception. Failure criterion: assertion passes or unrelated runtime failure. Preserve source hash, command, environment and output; pdb input is external, never remove the source debugger line.
- **Planned implementation**: strict TOML configuration, genuine decoder and immutable packed data, snapshots and fresh-process arms, dispatch capture and durable shards, exact pairing and optimizer gate, reports/replay and adversarial tests. Long controls and modern GPU claims require actual hardware/data/quota evidence. No detector/repair work before R0 and M0.3.
- **Evidence limits**: Local CPU/MPS PyTorch 2.13.0; 56 GiB free at inspection. Existing Condor environment is historical 2.12.1+cu126; do not silently relabel as the requested 2.13 build. Remote synchronization requires explicit push authorization under AGENTS.md.

### L0024 — 2026-09-06 CDT — Upstream extraction and context access

- **Status**: PASS source retrieval; BLOCKED live Notion access (sign-in required). Proceed from exact user attachment.
- **Action/outcome**: Initial source extraction failed with StopIteration because upstream uses a `py` fence and CRLF. Corrected extraction to accept that exact fence and preserved original code bytes, including debugger statement. No source semantics changed.
- **Source/evidence**: `upstream/pytorch_84864/{reproducer.py,source.json}`; complete issue response and run output in `artifacts/production-bootstrap/`. Command: local locked Python executing reproducer with 100 `c` debugger commands on stdin.
- **Next**: Inspect R0 outcome, then implement clean trainer. No controlled injectors executed.

### L0025 — 2026-09-07 CDT — R0 result and first implementation increment

- **Status**: PASS observed current R0 reproduction; implementation RUNNING; scientific Analyst review pending.
- **R0 evidence**: `artifacts/production-bootstrap/r0.json`, `r0.stdout`, `r0.stderr`; exact upstream source fails its final assertion on Python 3.13.15, Torch 2.13.0, macOS arm64. No metadata exception. Classification: current reproduction. Historical fallback is unnecessary for this cell; trigger-off evidence remains to be added.
- **Implementation**: Added strict config, explicit Llama-style decoder, immutable packed corpus, atomic pre-step state and RNG replay, training/accumulation/AdamW/scheduler, operator capture, bounded shard writer, exact comparator, pre-optimizer failure agreement, fresh-process pair orchestration and CLI.
- **Preregistered clean checks**: Tiny two-layer text-fixture smoke only. Compare three no-checkpoint/checkpoint steps; snapshot resume and fresh one-step pair must match loss, named gradients, parameters, optimizer, scheduler, RNG and cursor bitwise. Capture must preserve these outcomes; any missing/structural pair fails clean coverage.
- **Commands**: `uv run aci data prepare --config configs/smoke.toml`; `uv run aci train --config configs/smoke.toml`. Outcomes pending below.
- **User steering**: User explicitly says Notion is unnecessary; continue from attachment, no further Notion access.
- **Limits**: No 40M/125M milestone accepted. No controlled suite or detector implemented/executed yet. Pinned-host CUDA copies implemented, reusable preallocated buffer pool still pending. Hardware proof and TorchTitan remain pending.

### L0026 — 2026-09-07 CDT — Clean lifecycle and census probes

- **Status**: PASS three-step checkpoint smoke and first fresh-process pair; full-capture test RUNNING.
- **Evidence**: `artifacts/production/smoke-312e6cd15cc9` runs three updates; `artifacts/production/pair-smoke-529cd1e68a68` reports EXACT_MATCH for model/optimizer/scheduler/gradients/losses/cursor/RNG. Both arms are separate interpreter subprocesses.
- **Architecture**: Meta-device parameter count independently inspected: 39,985,664 (8x512) and 125,264,640 (16x768), including tied embeddings.
- **Census**: `artifacts/production/smoke-ac7709bef363/census.json`: 2,155 visible tensor outputs, 5,850,727 payload bytes for one miniature step. Conservative two-arm/three-step estimate including indexes and snapshots: 179,518,298 bytes. These are smoke measurements, not 40M storage estimates.
- **Environment failure**: Editable import intermittently failed. Python verbose startup and filesystem flags establish that a macOS hidden flag on `.pth` files makes Python 3.13 skip them. Clearing it was transient; use `PYTHONPATH=src .venv/bin/python` for local development tests. The package source itself is intact. Cause of recurrent flag setting is unknown.
- **Verification command**: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_production.py -q`; output pending.
- **Next**: Resolve any capture completeness defect before natural LM smoke; preserve first failed artifacts.

### L0027 — 2026-09-07 CDT — Capture defect isolated and natural integration smoke

- **Status**: PASS local implementation tests (11 tests); scientific milestone acceptance pending.
- **Failed clean capture**: First attempt reported 396 original versus 532 recompute outputs and aborted before update. First differing event was original `aten.pow.Tensor_Scalar` versus recompute `aten.detach.default`.
- **Cause and scoped change**: Inspection of exact operator stacks identifies PyTorch checkpoint.py `pack_hook`'s internal `x.detach()`. Those outputs remain fully captured under `checkpoint_bookkeeping`, intentionally unpaired with an explicit reason. Only detach calls originating in that exact upstream hook receive this classification; user detach operations remain pair eligible. No heuristic trace realignment is used.
- **Verification**: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_production.py -q`: 11 passed in 9.63s. Tests cover parameter counts, config rejection, corpus corruption, three-step clean equality, resume equality, NaN/signed-zero bytes, mutation-safe copying/alias metadata, writer failure cleanup, clean full capture and measurement equivalence, natural mechanism smoke plus trigger-off/observe/enforce, IDs/missing pairs, multi-output/checksum corruption.
- **Natural smoke bounds**: The miniature text fixture exhibits value-only first mismatch, exact forward loss but different gradients/updates, identical capture-on/off outcomes, zero optimizer/scheduler/model mutation in enforce, and equality after removing the dispatch trigger. This is integration validation only; it does not pass the real-corpus 40M M0.3 gate or authorize controlled-suite execution.
- **Pin discovery**: FineWeb-Edu `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`; open_llama_3b tokenizer `141067009124b9c0aea62c76b3eb952174864057`; TorchTitan candidate `d263ca0a1b569ed198b9943b6e8c2117a61d8843` (compatibility pending). Official CUDA index contains torch 2.13.0+cu126 cp313 manylinux_2_28 x86_64 wheel, SHA256 `4198c8d7478ab47ad2569309387d88b21fb553a1cf8ab06260fbd5a6ab9b9712`. Availability is not proof of Pascal kernel compatibility; scheduled probe remains required.
- **Next**: Pin data extra, materialize exact corpus, adversarial tests, durable smoke evidence, prepare Condor bootstrap/probe. TorchTitan compatibility and modern hardware remain separate gates.

### L0028 — 2026-09-07 CDT — Corpus frozen and bounded 40M control preregistered

- **Status**: PASS corpus preparation, 17 production/adversarial tests, all three fresh-process natural smoke matrices.
- **Corpus**: Exactly 100,000,000 tokens, 99M train/1M validation. `data/fineweb-edu-100m/manifest.json` SHA256 `314862232c3e15eb21b7830b30d237d02cb793d07fd22864713237b03e59e0fb`. Every shard and source record ledger is hashed. Both production configs now freeze that manifest hash.
- **Dependency failures**: AutoTokenizer first required SentencePiece, then protobuf through its conversion path. Used the pinned tokenizer's native SentencePiece model directly, preserving the declared tokenization algorithm (native IDs plus EOS per record); no tiktoken fallback. Both failed outputs retained in `artifacts/production-bootstrap/data-prepare*.stderr`.
- **R0 control**: `upstream/pytorch_84864/trigger_off.py` changes only the checkpoint call to direct `f(x)`, is separately labeled and exits 0. Original source remains unchanged. `r0-trigger-off.json` records control source hash and change.
- **Natural smoke**: `artifacts/production-bootstrap/verified-smoke.json` indexes all fresh-process arms at seeds 17/23/47: capture off, full observe, full trigger off, full enforce. Each seed passes same failure status with/without capture, trigger-off equality, and optimizer/model/scheduler non-mutation in enforce. Clean full pair: `pair-verified-clean-smoke-cba066acf601`. These do not accept M0.3.
- **Nested capture**: A further nested-checkpoint probe found inner original replay during parent recomputation. Added explicit `recompute_parent` role and compare each replay to its original; every visible payload retained. Nested test now passes six eligible comparisons, full coverage. No heuristic realignment.
- **Verification**: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_capture_failures.py tests/test_production.py -q`: 17 passed in 17.51s. Prior whole suite 33 passed; historical starter code untouched.
- **Preregistered next run**: 40M model, immutable real corpus, deterministic CPU FP32, seed17, 4 CPU threads, full planned optimizer/batch/sequence configuration, stop after three updates. Fresh processes with checkpoint off/on must match losses, gradients, parameters, optimizer/scheduler, RNG and data cursor exactly. This is Phase B bounded correctness, not a long-control or exhaustive milestone. Failure criterion is any mismatch or runtime failure. About 51 GiB free; capture remains off and snapshots/results remain bounded to several GiB.
- **Commands**: `PYTHONPATH=src .venv/bin/python -m ac_integrity.production_cli train --config artifacts/production-bootstrap/lm40m-three-{plain,checkpoint}.toml` in separate processes; stdout/stderr retained beside configs.

### L0029 — 2026-09-07 CDT — Failure-test follow-up and durable evidence index

- **Status**: PASS all 17 production tests. Prior 33-test complete suite passed before adding the final six adversarial tests; final whole-suite check remains pending.
- **Failure correction**: Tensor-subclass test initially expected the internal error message, but Torch's Python operator translated it to an unsupported-operand TypeError. Relaxed only the message assertion; retained assertions that capture records the unsupported input, aborts and cleans up. No implementation behavior changed to satisfy that test.
- **Durable records**: `reports/production-smoke.json` contains seed17/23/47 fresh-process artifact paths/statuses/summary hashes; `reports/corpus-manifest.json` is the versioned corpus manifest. Raw tensors, source data and checkpoints remain ignored.
- **Remaining boundary**: CPU collective test proves two-rank failure agreement before AdamW state creation, not FSDP2/TorchTitan behavior. Full-model controls and modern hardware acceptance remain pending.

### L0030 — 2026-09-07 CDT — Real 40M clean control passes

- **Status**: PASS bounded exact equality; no long-control/M0.1/M0.3 acceptance.
- **Environment**: Deterministic PyTorch 2.13.0 CPU FP32, Apple Silicon,
  seed17, four threads, 39,985,664 parameters, 2x256 tokens per microbatch,
  four accumulation groups, original planned AdamW and LR schedule.
- **Runs**: `artifacts/production/lm40m-clean-three-plain-d2428c9c9912`
  and `artifacts/production/lm40m-clean-three-checkpoint-d7c627e6212b`.
  Each completed three real-corpus optimizer updates in a separate process.
- **Result**: Exact equality of complete final model, named gradients,
  optimizer, scheduler, losses, CPU/MPS/Python/NumPy RNG state and data cursor.
  `reports/lm40m-three-step.json` contains zero differences and both run records.
- **Source provenance**: Both runs retain `source.zip` verified against every
  code-file digest in their recorded environment. Future runs now archive their
  source automatically, including dirty local changes and dependency locks.
- **Timing limit**: These three steps were correctness runs with snapshots and
  no pristine three-repeat benchmark. Their times establish no overhead claim.
- **Next**: Scheduled long controls and exhaustive
  40M audits after exact remote synchronization and quota validation. Controlled
  faults remain gated on accepted M0.3. TorchTitan/native FP8 remain unimplemented
  or hardware-blocked, as itemized in `docs/production-status.md`.

### L0031 — 2026-09-07 CDT — Local verification and execution boundary

- **Status**: PASS local verification; full production contract INCOMPLETE.
- **Checks**: `PYTHONPATH=src .venv/bin/python -m pytest -q`: 39 passed in
  34.26s. `uv lock --check`, wheel build, shell syntax, Python compileall and
  `git diff --check` pass. Outputs: `artifacts/production-bootstrap/final-all-tests.*`
  and `build.*`; wheel in adjacent `dist/`.
- **Preservation**: Existing user README text retained with a scoped status
  addition. Starter code, tests, historical artifacts and CUDA lock retained.
  All new source and reports remain local/uncommitted. No remote submission,
  publication, commit, push or Notion mutation was performed.
- **Execution boundary**: AGENTS.md requires commit/push and remote fast-forward
  synchronization before remote runs, and forbids pushing without explicit user
  authorization. New Condor fit probe is prepared for that next stage; not yet
  scheduler-validated or submitted. User quota must be rechecked before capture.
- **Unfinished scope**: Long 40M controls, full-model exhaustive M0.1/M0.3,
  controlled-suite implementation/execution after M0.3, complete CUDA buffer pool
  and crash/distributed commit protocol, full causal report coverage, TorchTitan
  extension/compatible-stack validation, native modern-GPU cases and overhead
  measurements. See `docs/production-status.md`; none is silently accepted.

### L0032 — 2026-09-07 CDT — Publication and real-error reproduction authorized

- **Status**: RUNNING.
- **Authorization**: User explicitly authorizes commit/push, Condor synchronization and the GPU fit probe, and asks to reproduce the target error in the real model. Existing Notion waiver remains in force.
- **Analysis of clean control**: Exact clean equality is the expected positive control, not evidence against the hypothesis. The falsifiable failure hypothesis concerns forward-only dispatch-mode state across checkpoint recomputation. Prior miniature observations support testing that unchanged mechanism in the full architecture; they do not establish 40M M0.3.
- **Preregistration**: Fresh 40M FP32 seed17 pre-step snapshot, full immutable FineWeb-Edu corpus, planned batch/sequence/optimizer. Trigger-on reference versus candidate must have equal forward loss, divergent named gradients and model/optimizer update; trigger-off must restore exact equality. Full-capture census must pass before tensor payload retention, and the first eligible mismatch must have matching shape/dtype/device with no default checkpoint exception. Capture-on/off outcome must agree; enforce must preserve pre-step parameters/optimizer/scheduler. Failure or storage blocks are recorded, not replaced with synthetic success.
- **Planned actions**: Review/stage the already verified implementation, add a reusable bounded real-model evidence runner, commit/push, remote fast-forward and scheduled probe. Run real-model CPU error arms while scheduler access is prepared. Raw SSH limited to short status/quota/queue checks; remote setup/submission through named remote-run.

### L0033 — 2026-09-07 CDT — Implementation published and probe dispatched

- **Status**: PASS publication; GPU dispatch and local real-error arms RUNNING.
- **Commit**: `6218f32` pushed to origin/main under explicit user authorization. Existing README changes included as previewed. Raw corpus/tensors/checkpoints remain ignored.
- **Whitespace caveat**: Staged diff check flags the intentionally byte-preserved upstream reproducer's CRLF endings; source was preserved rather than normalized. Prior whitespace checks omitted untracked files; no claim that the initial staged check passed. Add a scoped CRLF attribute in follow-up documentation, leaving source bytes unchanged.
- **Live remote check**: darmok checkout was clean; no current user jobs. `quota` is absent from PATH, so user quota remains unverified. GPU fit probe writes its environment into allocated scratch and does not require full tensor capture or corpus transfer.
- **Dispatch**: `remote-run --name aci-production-probe-0907 --cd /u/ayman27/activation-checkpoint-integrity ayman27@darmok.cs.utexas.edu` runs fast-forward pull, records HEAD, scheduler dry-run, then submits `condor/production-probe.submit`. Scheduler result pending.
- **Local real-error command**: `PYTHONPATH=src .venv/bin/python scripts/verify_real_natural.py --stage off`; outputs in `artifacts/production-bootstrap/real-natural-off.{stdout,stderr}`. Fresh snapshot, trigger-on pair and trigger-off pair; assertions require no arm exception and actual gradient/update divergence.

### L0034 — 2026-09-07 CDT — Real 40M capture-off error reproduced

- **Status**: PASS preregistered capture-off causal arms; exhaustive activation gate pending.
- **Evidence**: `artifacts/production/real-natural-off-c958c44bc3ca/evidence.json`; stdout in `artifacts/production-bootstrap/real-natural-off.stdout`. Seed17, full 39,985,664-parameter LM and immutable FineWeb-Edu batch.
- **Observed**: Trigger-on fresh-process reference and checkpointed candidate both finish without an exception, produce exactly equal forward losses, but differ in 73 named gradients and 74 model state entries (tied embedding/output counted twice in state_dict). Trigger-off fresh-process pair restores exact equality. This reproduces the requested silent downstream error in real next-token training; first activation evidence, capture measurement-effect and enforce gates remain to be measured at this scale.
- **Next command**: `PYTHONPATH=src .venv/bin/python scripts/verify_real_natural.py --stage census --snapshot artifacts/production/real-natural-off-c958c44bc3ca/pre_step_1.pt`; estimate both arms with the complete configured microbatches before considering full capture. No payload coverage will be reduced to fit.
- **Remote outcome**: Named dispatch fast-forwarded to `6218f329848eddfec6537a55a8f0719756618e2f`, scheduler dry-run passed, and job `1553517.0` was submitted successfully. This is submission evidence, not GPU-fit completion.

### L0035 — 2026-09-07 CDT — Full 40M capture exceeds local retention budget

- **Status**: PASS census; BLOCKED full capture on local storage.
- **Measurement**: 15,329 visible tensor outputs and 23,977,519,895 payload bytes for the candidate's complete 40M step. The prescribed two-arm retention estimate including indexes/snapshots and 25% safety margin is 61,949,406,248 bytes. Live APFS capacity not allocated is about 45.4GB before reserve (df about 42GiB).
- **Decision**: Do not start exhaustive capture, reduce audited microbatches, sample tensors, remove old artifacts, or call M0.3 complete. A retained capture location with sufficient verified quota is required. Condor home quota is still unverified; shared free space is not substituted.
- **Causal evidence retained**: `reports/lm40m-natural-seed17.json`. All four microbatch forward losses are identical in trigger-on reference/candidate; default_check_raised is false in both. First differing named gradient is embedding.weight: 474,112 differing elements, relative L2 error 0.09843773970885245 for that gradient tensor. Removing only the mode trigger restores exact equality. This is real-model downstream silent corruption, not yet full first-activation or production-trainer acceptance.
- **Analyst assessment**: PROMOTE the bounded capture-off reproduction for follow-up forensic capture. Baseline equality and trigger-off reversal support the transplanted #84864 mechanism; no arbitrary synthetic fault or convergence claim. Remaining confounder checks are full capture measurement-effect and full-model enforcement, explicitly blocked on storage.
- **Next**: Finish scheduled GPU fit probe, preserve its raw evidence and publish the bounded findings. Full-capture storage remains a separate action item.

### L0036 — 2026-09-07 CDT — GPU fit probe completes and results archived

- **Status**: PASS scheduled fit probe and bounded capture-off reproduction.
- **Scheduler**: `condor_history 1553517` reports JobStatus 4, ExitCode 0,
  113.0 seconds remote wall time, `slot1@eldar-44.cs.utexas.edu`.
- **GPU evidence**: `reports/condor-1553517-fit.json`; retained raw stdout/stderr
  copied to `artifacts/production-bootstrap/condor-1553517.{stdout,stderr}`.
  Exact PyTorch 2.13.0+cu126, CUDA12.6 and Pascal (6,1) checks and a CUDA
  matrix kernel passed, then the 125,264,640-parameter generated-token probe
  completed one optimizer update. Peak allocated 2,348,692,992 bytes;
  peak reserved 2,503,999,488 bytes. This is memory-fit evidence only.
- **Real-error assessment**: The requested error is recreated in the actual
  40M model on CPU with capture off. The GPU probe is not mislabeled as a
  natural-error reproduction. Trigger-on/off reference/candidate evidence and
  full capture storage block are in L0034–L0035.
- **Archival**: Add scoped `.gitattributes` preserving upstream CRLF bytes and
  allowing its source-verbatim whitespace convention. Publish documentation
  and bounded result manifests under existing commit/push authorization.
- **Remaining**: No full capture, full-model enforcement, M0.3 acceptance,
  controlled-suite execution, 2,000-step control or TorchTitan claim.

### L0037 — 2026-09-07 CDT — Repository synchronization and evidence walkthrough

- **Status**: Complete read-only evidence review and user-requested local pull.
- **Synchronization**: Fast-forwarded local main from `9c78d88` to user commit
  `d63d326`. The checkout was clean after the pull. User-added root
  `activation_checkpoint_reproduction.md` matches `docs/production-plan.md`
  except for a final newline; `starter_test_cases.md` describes E0–E2.
- **Review**: Rechecked real-model trigger-on/off summaries, saved outcomes,
  reports, storage census, starter history and GPU-fit evidence. Both trigger-on
  arms completed successfully; pair divergence means differing training state,
  not a process crash. The 61.95GB figure is a projected full-capture budget,
  not an existing archive. Current writer stores uncompressed payloads.
- **Clarification**: The older miniature-only sentence in the status document's
  remaining-gaps section is superseded by its September 7 section and L0034–L0036.
  Real-model capture-off failure is reproduced; full activation localization,
  measurement-effect checks and enforcement remain unaccepted at that scale.
- **Scope**: No new experiment, archive, upload, deletion or remote run in this
  walkthrough. Notion remains waived by the user's session instruction. This
  log addition is local and uncommitted.

### L0038 — 2026-09-08 CDT — Code walkthrough and professor-update preparation

- **Status**: Read-only source/evidence review; explanatory response and unsent
  email draft prepared. No new training or benchmark was run.
- **Sources**: User-attached twelve-week Semester 1 master plan; starter fixture,
  controllers and hook observer; real-model adapter, model and training loop;
  causal-arm assertions, stored gradient report and upstream issue #84864.
- **Clarification**: Reproduction of a known bug is a valid result, distinct from
  discovery of a new bug. The LM experiment deliberately enables an adapter;
  it is not an unchanged production-workload failure. The planned contribution
  is position-sensitive fingerprints plus pre-optimizer enforcement, neither
  novelty nor performance yet established. Exact full capture is the reference
  for that checker, not the intended low-overhead implementation.
- **Plan alignment**: Next proposed work is retained full 40M capture,
  capture-effect comparison and real-model enforcement validation, followed by
  the master plan's two fingerprint baselines and interior-only test. Finding
  the first differing recorded value supports experiment validation; automatic
  root-cause diagnosis and repair remain outside Semester 1. Current observer
  coverage does not yet close the master plan's saved-tensor coverage gate.
- **Boundary**: External storage/quota, full 40M enforcement, native BF16/FP8,
  TorchTitan integration and runtime-overhead evidence remain outstanding.
  Email is a draft only. Research log changes remain local and uncommitted.

### L0038 — 2026-09-08 CDT — Plan-grounded code teaching and professor draft

- **In progress**: Explain PyTorch fundamentals, exact gradient failure, E0–E2
  implementation and motivation, reproduction versus discovery, and next-week
  priorities. User supplied the twelve-week Semester 1 meeting plan directly;
  Notion remains waived. Read current source, archived evidence and primary
  PyTorch issue #84864 and checkpoint documentation. Starting HEAD `d63d326`,
  dirty only from L0037 in this log.
- **Scope**: Run a tiny instructional example using the existing mode adapter,
  inspect saved real-model gradients and E2 summaries, then draft (do not send)
  the professor email. No training-scale run or production implementation change.
- **Expected evidence**: Toy forward/gradient disagreement only with checkpointing
  and forward-only mode; agreement when the mode is supplied for both phases.
  This illustrates the known mechanism and is not a new milestone experiment.
- **Outcome**: Instructional CPU example on torch 2.13.0, seed 17, three
  differentiable ones and loss `sum(x * torch.rand(x.shape))`: direct forward
  under NoRandomnessMode gives loss 1.5 and gradients [0.5,0.5,0.5]; checkpoint
  under a forward-only mode gives loss 1.5 and gradients
  [0.43424123525619507,0.5351095795631409,0.8302087187767029]; supplying separate
  NoRandomnessMode instances through checkpoint context_fn for both phases
  restores [0.5,0.5,0.5]. No production source was changed.
- **Raw evidence review**: Reloaded trusted local reference/candidate outcome.pt
  from `artifacts/production/pair-real-natural-17-5bae2d4fc9b7`; embedding gradient
  shape is (32000,512), and row 2 columns 0–3 reproduce the report values.
  E2 aggregate `artifacts/starter/e2-coverage-20260904T042046.953728Z-aab0a6b4/summary.json`
  reports hooks_preserve_behavior=false and zero recompute direct/hook events;
  explain this as behavior-changing observation, not simply missing callbacks.
- **Plan mapping**: Attached Semester 1 plan targets probabilistic position-sensitive
  GPU fingerprints and pre-update enforcement. Current exact full capture is
  reference infrastructure, not that optimized detector. Next-week draft focuses
  on storage, full real-model localization, capture-on/off equivalence, and
  pre-update enforcement; no promise of new bugs in an untouched trainer,
  completed coverage, native FP8, TorchTitan, or measured low overhead.
- **Status**: Explanation and unsent email draft prepared from source and saved
  evidence. No training-scale experiment, production edit, send, commit or push.

### L0039 — 2026-09-11 CDT — GPU, forensic capture, enforcement and upstream follow-through

- **Status**: In progress, preregistered before implementation or experiment.
- **Request**: Complete the four next steps from the referenced Codex and ChatGPT
  conversations: GPU failure reproduction, first intermediate mismatch, recorder
  noninterference and optimizer enforcement, and initial adapter-free upstream
  LLM compatibility experiments. Explain mechanisms and measured results plainly.
- **Context**: Read both referenced tasks, current README/source/status and ledger.
  Starting HEAD `d63d326`; RESEARCH_LOG.md already dirty from prior user work and
  preserved. Actual local checkout is now `/Users/ayman/ChatGPT/Mootaz Project/activation-checkpoint-integrity`.
  Notion exact root verified: `3c65d66f-2c23-80c6-879f-e0aff5e92706`, edited
  2026-09-10T21:00:36.980Z; one unknown alias block, relevant production-plan child
  `3d05d66f-2c23-80d1-882f-ea6c45c89a7e` fetched, edited 2026-09-03T04:04:10.547Z.
  Current user request supplies the active bounded scope; no claim to complete
  the older 2,000-step or optimized-fingerprint milestones in this session.
- **Live access**: Condor SSH key access works; remote clean at `6218f329`;
  no current user jobs. Pool advertises idle GTX1080Ti and one Quadro RTX6000.
  `quota` absent; official UTCS storage docs identify `chkquota`, to check next.
  Remote corpus absent. Local free space about 24 GiB, prior artifacts 16 GiB.
- **Implementation plan**: Lossless payload compression/deduplication with every
  event retained and exact reconstruction checked; bounded fresh-process GPU
  experiment runner with common snapshots; preserve trigger-on/off controls;
  compare capture-on/off losses, gradients, model/optimizer/scheduler/RNG; prove
  enforcement preserves pre-step persistent state including warmed Adam moments.
  Initial upstream compatibility runs separately label supported precision/compiler
  cells, runtime errors, clean controls and any actual discrepancies.
- **Hypotheses**: (H1) Trigger-on CUDA arms finish with equal forward losses but
  differing gradients/updates, trigger-off restores exact equality. (H2) Full
  recording preserves capture-off outcomes and finds same-metadata divergence
  at the adapter rand output. (H3) Enforce aborts before model/optimizer/scheduler
  mutation, while clean enforcement allows an update. (H4) Adapter-free upstream
  checkpoint/no-checkpoint arms agree within identical execution settings; a
  negative result is valid and no injected trigger may count as natural discovery.
- **Falsification**: Any failed clean/repeatability/noninterference gate invalidates
  dependent corruption claims. Missing/ambiguous pairs fail closed. Ordinary
  compiler numerical drift is classified separately from inconsistent recompute.
- **Evidence**: Exact source/config/environment/seed/job ID, immutable snapshots,
  full lossless payloads and indexes, outcomes, comparisons and machine-readable
  gate summaries. No performance or saved-tensor-complete claim without evidence.

### L0040 — 2026-09-11 CDT — Lossless storage and causal runner validation

- **Outcome**: Added optional zlib payload encoding and exact byte-verified
  deduplication; defaults preserve existing raw artifacts. Every event and full
  logical payload remain recoverable. SHA256 only indexes reuse candidates;
  actual bytes must also match, including an adversarial hash-collision test.
- **Tests**: 5 targeted tests pass in 6.58s (lossless views/mutation/special values,
  corruption rejection, forced hash collision, existing miniature enforcement).
- **Experiment**: `PYTHONPATH=src .venv/bin/python -m ac_integrity.validation suite
  --config configs/smoke.toml --device cpu --output artifacts/followthrough/smoke-v1
  --budget-bytes 1000000000` passed all 11 gates. Nine fresh causal arms plus
  complete census arms; warmed Adam state; full capture on reference/candidate;
  repeatability, trigger reversal, recording equivalence, bad-update prevention
  and clean update allowed. Source archive/environment in that run, CPU torch2.13.
  This is fixture validation only, not the requested GPU or 40M result.
- **Analysis**: PROMOTE infrastructure for the unchanged-size real model, subject
  to full storage/hardware preflight. Distinguish first mismatch in original
  execution order from first mismatch encountered while backward visits blocks
  in reverse order; report both identities. Completed trace gets CAPTURE_COMMIT
  separately from optimizer STEP_COMMIT, so blocked updates remain auditable.
- **Storage finding**: `/lusr/bin/chkquota` verifies home quota 15GiB and usage9GiB.
  Cluster scratch reports a 200GiB quota but no user directory and root isn't
  writable. Quota without a writable allocation is not usable retention space.
  Scheduled local scratch is available and can hold the conservative census.
- **Upstream decision**: Begin the user-requested adapter-free screen with the
  installed, version-pinned Hugging Face Transformers5.16.1 Llama implementation
  (real corpus, checkpoint off/on, SDPA/dropout/autocast/compiler cells). This is
  an upstream-model integration, not completed TorchTitan/FSDP/distributed work.
  The older TorchTitan candidate now has a different source layout and additional
  dependencies; its milestone remains separate and will not be falsely accepted.

### L0041 — 2026-09-11 CDT — Local gates complete; GPU launch prepared

- **Validation**: Full suite 44 passed in 53.17s, including lossless archive
  round trip across acknowledged chunks and duplicate-file hard links. Shell
  syntax, Python compilation and whitespace checks pass. Final causal smoke-v2
  passes all **10** gates (correcting L0040's prose count of 11). First original
  mismatch: blocks.0 aten.rand; first seen in backward: blocks.1 aten.rand.
- **Upstream experiment**: `PYTHONPATH=src .venv/bin/python -m ac_integrity.upstream
  matrix --config configs/smoke.toml --output artifacts/followthrough/upstream-smoke-v1
  --cells eager_fp32,sdpa_fp32,dropout_fp32,eager_bf16,compile_fp32 --steps 2 --record`.
  Five cells EXACT_MATCH; no differing entries across baseline repeats,
  checkpoint/no-checkpoint or recording-on/off where supported. All eager
  recorded comparisons pass. Compile used aot_eager; no kernel-fusion claim.
- **Analysis**: PROMOTE these bounded implementation checks. CPU BF16 is not
  evidence for native GPU BF16. No naturally occurring inconsistency observed in
  these small upstream runs; this does not prove the absence of other failures.
- **GPU plan**: Two scheduler profiles: full unchanged 40M/2048-token accumulated
  causal step on Pascal, and 40M upstream Llama one-step cells using 64-token
  microbatches on a Quadro RTX6000-class capability7.5 GPU. The latter permits
  an actual Inductor compile attempt. Unsupported native BF16 is recorded.
  Live idle node nandor-5 has a Quadro RTX6000 and about75GB local disk; choose
  50GB upstream scratch reservation,110GB core. No experiment on login nodes.
- **Packaging**: CUDA2.13.0+cu126 lock preserved; separate fully hashed 56-package
  upstream lock pins Transformers5.16.1. Scheduled source clone is immutable.
  Archive sender uses scheduled scratch, exact-verified file reuse and bounded
  512MiB temporary chunks; original evidence stays until full local verification.
- **Pending boundary**: Local AGENTS.md requires GitHub synchronization before
  remote runs and explicitly forbids pushing without user authorization. Prepare
  reviewable docs/diff, request only that publication permission, then carry out
  the already requested UT runs without additional confirmation.

- **Scheduler preflight (L0041 continuation)**: Condor accepted both submit
  profiles in a read-only dry run, including core RequestDisk115343360KiB and
  upstream52428800KiB. No jobs submitted yet. Local HEAD and origin/main both
  `d63d326aa37d1f4fadccfcc78faaa27dd361185c`; dependency lock check passes.
  Publication permission requested with exact AGENTS.md requirement; pending.

- **TACC check (L0041 continuation)**: Existing `ls6` alias reached the login
  service, which requires password/token authentication; BatchMode returned
  Permission denied. No credentials entered and no settings changed. This does
  not block the prepared Condor experiments.

### L0042 — 2026-09-11 CDT — Resume and synchronized execution

- **Authorization**: After the concrete implementation review and publication
  permission request, user instructed to keep going following a usage-limit
  interruption. Proceed with the proposed commit/push, synchronization, and
  already requested scheduled UT experiments. Scope remains the reviewed work.
- **Recheck**: Worktree still contains only the recorded implementation and prior
  log changes. Latest archive test passed in23.45s. Local free storage about21GiB;
  collectors enforce budget and preserve remote sources until acknowledged.
- **Planned action**: Commit/push source/config/tests/docs/log, transfer the frozen
  398MiB public corpus to UT, fast-forward remote through GitHub, submit two
  named profiles, collect evidence with bounded storage and analyze all failures.
