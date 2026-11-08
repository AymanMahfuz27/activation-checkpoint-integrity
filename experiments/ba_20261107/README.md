# November 7 B+A experiments

This directory is an isolated research harness, not a replacement for the
historical training/checking implementation. It loads the exact repaired M4
source and corrected update-210 state declared by each contract. Generated
outputs belong under `artifacts/revision3/ba-20261107/`, never in Git.

The packages answer different questions:

| Package | Question | Execution unit |
| --- | --- | --- |
| `closeout/` | Which retained E1–E5 findings survive strict validation? | Allocated CPU analysis; 90 fixtures before retained-data reduction. |
| `setup_preflight/` | Does a fresh same-filesystem hardlink environment preserve the locked runtime? | CPU development qualification of all installed versions, a representative cached inode, and CPU-visible imports. |
| `provenance/` | Does native GELU backward actually consume the changed replay input? | Matched trace off/on calls from one exact prestate; six CUDA fixtures, eight primary arms or the frozen six-case/28-arm matrix. |
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

The current conditional `*-copy40-dev` GPU batches first run the standard-library
installer in `copy_runtime.py`, then delegate to their `*_job.py` wrapper.
The installer admits the job before uv and leaves at least 1,260 seconds for
fresh scientific admission. `allocated_common.py` verifies the actual owned Slurm grant,
executed batch bytes, clean tracked revision, environment, time and live quota
before model execution. The wrapper repeats this admission after installation,
then checks the complete installed version map and private environment prefix
before importing Torch. Both stages require sealed reviews of the actual setup outcomes;
dependent studies additionally require accepted actual primary provenance.
A source precheck hash cannot substitute for them. Dispatch these jobs sequentially:
the bootstrap rejects any other owned pending or running job.

CPU closeout instead runs
`closeout/source/run_closeout.py`, using its inherited `run_subset.py` helpers
for admission and retention. Fixtures run before the study. GPU study
contracts reserve the final 600 seconds for retention. Failures remain
evidence; a numerical hypothesis does not become true because a job exits zero.

`retrieve_results.py JOB STUDY NEW_DESTINATION` requires terminal owned
accounting and matching Scratch/WORK custody receipts, verifies every declared
member locally, and retains only the fixed whitelist of shell control files.
Its SFTP deadline scales with the declared member count, from 240 to at most
900 seconds. A timeout retains partial transport logs and files, then stops;
read-only recovery requires a new destination and full byte verification.
Use `recover_transfer.py JOB STUDY ORIGINAL NEW --expected-state STATE
--expected-exit CODE:SIGNAL` for interrupted provenance or official-baseline
collections. It verifies fresh owned terminal accounting against the original
collection, unchanged source/WORK receipts, and every reused or fetched member.
`STUDY` is `ba-closeout`, `ba-setup`, `ba-provenance`, `ba-placement` or
`ba-opguard`. For a failed GPU installation, use the corresponding
`ba-<study>-environment-setup` folder; it has its own compact custody receipt.
An admission failure before the normal custody path requires explicit recovery
of its partial evidence; it is not an empty successful experiment.

The copy40 amendment uses a fresh private node-local environment, keeps the
original locks, and requests 40 minutes with a 2-SU job cap. The failed
same-filesystem hardlink qualification and its unreleased GPU candidates remain
in the history. Its representative cache-link probe returned EPERM; matching
filesystem devices alone did not qualify that installation path. This does
not establish a universal filesystem restriction or a scientific speedup.

The corrected OpGuard fixture entrypoint adds the pinned historical source
directory in its own interpreter. A parent's `sys.path` change is process-local;
it does not reach fresh fixture children. This correction preserves the sandbox
working directory, installed official build, all eight original tests and four
model arms. Its release additionally requires the actual failed fixture review
and fresh terminal/resource reconciliation; it never retries automatically.

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

The new studies use one retained state and a small frozen software-fault grid.
They do not estimate
physical fault frequency, population coverage, learning impact of the tiny
fault, distributed behavior or publication novelty. See the independently
reviewed results and retained failure records before planning expansion.
