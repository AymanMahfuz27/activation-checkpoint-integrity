# November 7 B+A experiments

This directory is an isolated research harness, not a replacement for the
historical training/checking implementation. It loads the exact repaired M4
source and corrected update-210 state declared by each contract. Generated
outputs belong under `artifacts/revision3/ba-20261107/`, never in Git.

The four packages answer different questions:

| Package | Question | Execution unit |
| --- | --- | --- |
| `closeout/` | Which retained E1–E5 findings survive strict validation? | Allocated CPU analysis; 90 fixtures before retained-data reduction. |
| `provenance/` | Does native GELU backward actually consume the changed replay input? | Matched trace off/on calls from one exact prestate; six CUDA fixtures and eight primary arms. |
| `placement/` | Does changing fault arrival from after to before the producer record explain the all-output miss? | Seven CUDA fixtures and eight arms; full gradient/state equivalence before interpreting checker decisions. |
| `opguard_baseline/` | Does the pinned official runtime expose the changed activation input without changing execution? | Official native build/qualification, eight fixtures, four normally exiting arm processes and drained raw summaries. |

## How execution is bound to evidence

Each `*-contract.json` contains preregistered methods, exact source/input hashes,
resource limits, fixture identities and runtime settings. An independent source
review binds the contract's bytes. A separate release binds that review and the
actual Git revision; a source precheck is never an actual-result acceptance.

`dispatch_once.py` creates a local exclusive intent, then a new remote checkout
and control namespace, verifies account/quota/owned queue, and invokes `sbatch`
once. It retains the raw response. If the response is lost or cannot be parsed,
reconcile the saved response, queue and accounting before taking any action.
Do not rerun the dispatcher against another directory to bypass that intent.

The batch file establishes the pinned environment and delegates to its
`*_job.py` wrapper. `allocated_common.py` verifies the actual owned Slurm grant,
executed batch bytes, clean tracked revision, environment, time and live quota
before model execution. Fixtures run before the study. The 30-minute DEV
allocations reserve the final 600 seconds for retention. Failures remain
evidence; a numerical hypothesis does not become true because a job exits zero.

`retrieve_results.py JOB STUDY NEW_DESTINATION` requires terminal owned
accounting and matching Scratch/WORK custody receipts, verifies every declared
member locally, and retains only the fixed whitelist of shell control files.
`STUDY` is `ba-closeout`, `ba-provenance`, `ba-placement` or `ba-opguard`.
An admission failure before the normal custody path requires explicit recovery
of its partial evidence; it is not an empty successful experiment.

## Interpretation boundaries

All numerical execution belongs on allocated compute. Follow the repository
`AGENTS.md` for current TACC authentication, quota, DEV-first and release rules.
Do not run the fixture modules locally as a shortcut: they import the numerical
engine or require real CUDA/native execution. Local syntax and byte checks do
not qualify their behavior.

The provenance tracer makes diagnostic copies and observes actual backward
arguments. Its timing is not detector overhead. Its full-byte consumer join
does not prove storage alias identity. Placement equivalence is limited to the
registered event, full gradients and accepted transition. The OpGuard baseline
configures activation I/O at one target, not its full automatic system. Its
regional XOR signatures have a qualified one-bit positive control, not a
collision-free general equality guarantee.

These studies use one selected fault and retained state. They do not estimate
physical fault frequency, population coverage, learning impact of the tiny
fault, distributed behavior or publication novelty. See the independently
reviewed results and retained failure records before planning expansion.
