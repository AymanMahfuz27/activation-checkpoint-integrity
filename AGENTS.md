# Activation Checkpoint Integrity: Agent Instructions

These instructions apply to the entire repository. Keep experiment execution
reproducible, keep credentials out of the repository, and treat measured claims
as valid only when the exact code revision, environment, hardware, seed, and
command are recorded.

## Repository and synchronization

- Local checkout: `/Users/ayman/Documents/ChatGPT/Mootaz Project/activation-checkpoint-integrity`
- GitHub repository: `https://github.com/AymanMahfuz27/activation-checkpoint-integrity.git`
- Default branch: `main`
- Condor checkout: `/u/ayman27/activation-checkpoint-integrity`
- GitHub is the synchronization point between the laptop and remote systems.
- Before a remote run, commit and push the required local work, then run
  `git pull --ff-only` in the remote checkout and record `git rev-parse HEAD`.
- Do not edit the same branch concurrently on the laptop and a remote system.
  Do not push, force-push, or rewrite history unless the user explicitly asks.
- Never commit passwords, private keys, tokens, datasets, checkpoints, Condor
  logs, or generated results. Use the ignored `artifacts/`, `checkpoints/`,
  `data/`, and `results/` directories.

## Project context and research memory

### Canonical context

- The designated Notion context is the page whose title is exactly
  `MOOTAZ PROJECT`.
- Stable Notion page ID: `3c65d66f-2c23-80c6-879f-e0aff5e92706`.
- Stable Notion URL:
  `https://app.notion.com/p/3c65d66f2c2380c6879fe0aff5e92706?pvs=204`.
- At the start of every research session, and again before a material planning,
  implementation, or experiment decision, read:
  1. this `AGENTS.md`;
  2. `README.md`;
  3. the current state and latest entries in `RESEARCH_LOG.md`; and
  4. the exact Notion page titled `MOOTAZ PROJECT` and the relevant child pages
     beneath it.
- Do not silently substitute a similarly named Notion page. If the exact page
  cannot be found or accessed, record that fact in `RESEARCH_LOG.md`. Continue
  only when the task can be grounded safely in repository evidence; ask the
  user when the missing context could materially change the work.
- Notion is the canonical source for high-level project context, goals, and
  user decisions. The checked-out code, exact Git commit, raw run artifacts,
  scheduler records, and measured outputs are authoritative for implementation
  and experiment facts. Record and surface material conflicts rather than
  silently choosing one.
- When available, record the Notion page URL or ID and last-edited time in the
  session's log entry. Do not copy credentials, private integration data, or
  irrelevant personal information into the repository.

### Mandatory research log

- `RESEARCH_LOG.md` is the canonical chronological memory for this project. If
  it does not exist, create it before material project work.
- Record every research-relevant action, including context retrieval, plans,
  code or configuration changes, environment setup, repository
  synchronization, cluster probes, job submissions or cancellations,
  experiments, measurements, analyses, failures, successes, blockers,
  decisions, lessons, and next actions. Routine read-only checks may be grouped
  into one evidence line so the log remains queryable.
- Before material work starts, add a planned or in-progress entry with the
  objective, relevant prior context, repository commit, intended action, and
  expected evidence. Before an experiment, also preregister its falsifiable
  hypothesis, baseline, isolated change, metrics, and falsification criterion.
- Immediately after each material action, and before starting the next one,
  append its exact outcome, evidence and artifact locations, success/failure
  status, uncertainty, decision or pending review, and next actions.
- Log failed, killed, aborted, preempted, and inconclusive attempts as carefully
  as successful ones. Never delete or rewrite an inconvenient result. Correct
  an earlier record with a new entry that cites it.
- Keep observed facts separate from interpretation. A completed experiment
  must record its commit, clean/dirty state, exact command and configuration,
  environment, hardware, seed, timestamps, scheduler job ID, raw artifacts,
  metrics, validity, Analyst verdict, lesson, and follow-up. Use `unknown`,
  `not measured`, or `pending analysis` instead of guessing.
- No result is ready for follow-up planning until it has been analyzed and
  archived. Before proposing a new research direction, review the log for prior
  experiments, failures, and unresolved action items and summarize the current
  state.
- Never record passwords, tokens, API keys, private keys, secret values, or
  credential-bearing commands. Use `[REDACTED_SECRET]` and record only safe,
  masked metadata when a security-relevant event must be documented.
- Keep the `Current state`, `Planned actions`, and `Experiment index` sections
  synchronized with the append-only ledger. The ledger is the historical
  source of truth.
- No research turn is complete until the log is updated, or the agent records
  explicitly that no material project state changed.

## Remote execution policy

- Use raw SSH only for short, bounded checks such as `hostname`, `git status`,
  `condor_q`, or `condor_status`.
- Use `remote-run` for setup, monitoring, or other commands that may need to
  survive a dropped laptop connection. Actual computation belongs in the
  remote scheduler, not in a long-running shell on the submit host.
- Name every `remote-run` job. After launching it, report its host and window
  name plus the exact `--check`, `--log`, and `--attach` commands.
- Never run training or benchmarks directly on a login or submit node.
- Do not store or embed account passwords. SSH key authentication is already
  configured for the Condor account.

## UTCS Condor cluster

### Verified access

The following details were verified live on 2026-09-02:

- Submit host: `darmok.cs.utexas.edu`
- Username: `ayman27`
- SSH: `ssh ayman27@darmok.cs.utexas.edu`
- Remote repository: `/u/ayman27/activation-checkpoint-integrity`
- Scheduler binaries: `/lusr/opt/condor/bin`
- Scheduler: HTCondor 9.0.8 on the submit host
- GPU pool: `eldar` nodes with NVIDIA GTX 1080 8 GB and GTX 1080 Ti
  11 GB GPUs, compute capability 6.1
- Live snapshot: 42 `eldar` nodes and 83 advertised GPUs. Availability is
  dynamic; query it again immediately before submitting.

The GPUs are Pascal-generation devices. Use this cluster for implementation,
fault-injection tests, correctness sweeps, and preliminary FP32/FP16 timing.
Do not use it as the sole evidence for BF16, FP8, Tensor Core, large-model,
distributed-scaling, or final modern-GPU overhead claims. Pascal support ends
with CUDA 12.x, so pin and record the CUDA and PyTorch versions used.

### Connect and synchronize

For a short interactive check:

```bash
ssh ayman27@darmok.cs.utexas.edu
cd /u/ayman27/activation-checkpoint-integrity
git status --short --branch
git pull --ff-only
git rev-parse HEAD
```

For work that may outlive the connection:

```bash
remote-run \
  --name <descriptive-job-name> \
  --cd /u/ayman27/activation-checkpoint-integrity \
  ayman27@darmok.cs.utexas.edu \
  '<command>'

remote-run --check ayman27@darmok.cs.utexas.edu <descriptive-job-name>
remote-run --log ayman27@darmok.cs.utexas.edu <descriptive-job-name>
remote-run --attach ayman27@darmok.cs.utexas.edu <descriptive-job-name>
```

### Inspect the pool and queue

Use the full scheduler path so the commands do not depend on shell startup
files:

```bash
/lusr/opt/condor/bin/condor_status -total

/lusr/opt/condor/bin/condor_status \
  -constraint 'GPUSlot && GPUs > 0' \
  -af Name State Activity GPUs CUDADeviceName CUDACapability CUDAGlobalMemoryMb

/lusr/opt/condor/bin/condor_q
/lusr/opt/condor/bin/condor_q -g -run
/lusr/opt/condor/bin/condor_q -g -better-analyze <cluster.process>
```

### Environment status

- `git`, `tmux`, and a user Miniconda installation are present.
- `python3` currently resolves to `/u/ayman27/miniconda3/bin/python3` and was
  Python 3.13.5 when checked.
- `uv` is not currently installed on the Condor account.
- Do not treat the existing base Conda environment as the experiment
  environment. Once the repository pins its Python and PyTorch dependencies,
  create a dedicated reproducible environment. Ask before installing `uv` or
  other persistent user-wide tools.
- The submit host has no experiment GPU. CUDA availability must be tested from
  a scheduled `eldar` job.

### GPU job template

Create an executable repository script such as
`scripts/run_condor_experiment.sh` that activates the pinned environment,
prints the reproducibility record, and runs exactly one experiment. Then use a
submit file based on this template:

```text
universe = vanilla
initialdir = /u/ayman27/activation-checkpoint-integrity

executable = /bin/bash
arguments = /u/ayman27/activation-checkpoint-integrity/scripts/run_condor_experiment.sh

output = artifacts/condor/$(Cluster).$(Process).out
error = artifacts/condor/$(Cluster).$(Process).err
log = artifacts/condor/$(Cluster).log
notification = Error

should_transfer_files = NO
request_cpus = 4
request_memory = 16GB
request_disk = 10GB
request_gpus = 1

requirements = TARGET.GPUSlot && TARGET.Eldar && TARGET.GTX1080Ti
+GPUJob = true
+Group = "UNDER"
+Project = "OPERATING_DISTRIBUTED_SYSTEMS"
+ProjectDescription = "activation checkpoint integrity research"

queue 1
```

This template was syntax-checked with `condor_submit -dry-run` on `darmok`.
Use `TARGET.GTX1080 == true` only when an 8 GB GTX 1080 is sufficient. Prefer
one GPU per job; do not assume the cluster interconnect is suitable for final
multi-GPU scaling measurements.

Before submitting:

```bash
cd /u/ayman27/activation-checkpoint-integrity
mkdir -p artifacts/condor
chmod +x scripts/run_condor_experiment.sh
/lusr/opt/condor/bin/condor_submit -dry-run /dev/stdout <job.submit>
/lusr/opt/condor/bin/condor_submit <job.submit>
```

The runner and each result record must capture at least:

- Git commit and whether the worktree is clean
- Exact command and configuration
- Python, PyTorch, CUDA runtime, and NVIDIA driver versions
- GPU model, memory, and compute capability
- Seed and deterministic settings
- Start/end timestamps and Condor cluster/process ID
- Exit status, stdout/stderr paths, and output artifact paths

Python jobs run in the vanilla universe and are not transparently checkpointed
by Condor. Long jobs must save application-level checkpoints and resume safely
because a preempted job may restart from the beginning.

If a job remains idle, first verify that the submit file contains valid
`+Group`, `+Project`, and `+ProjectDescription` fields, then run
`condor_q -g -better-analyze <cluster.process>`. Never bypass the scheduler by
running experiments directly on `darmok` or an `eldar` machine.

## TACC cluster

Status: reserved for future configuration. Do not guess or copy old allocation
details into commands. Before the first TACC experiment, verify and fill in:

- Login host and username
- Active allocation/project ID and expiration date
- GPU system, partition/queue, node type, and charging unit
- Remote repository and scratch/work paths
- Scheduler commands and tested batch template
- Module, CUDA, Python, PyTorch, and `uv` setup
- Job limits, storage quotas, and data-retention policy
- Exact commands for submit, queue, cancel, logs, and interactive development

TACC should be used for the final modern-GPU evaluation: BF16/FP8, Tensor Core
behavior, larger TorchTitan workloads, multi-GPU compatibility, and performance
claims intended to generalize to A100/H100-class training. Add concrete details
here only after the allocation and live system configuration are verified.
