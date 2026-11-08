# BB execution infrastructure

These files adapt the reviewed BA transport and private-copy environment code
for BB. They do not define a method, scientific runner, batch, contract,
prerequisite review, or result. Infrastructure precheck and actual allocated
qualification remain separate gates.

`dispatch_once.py` targets branch `experiment/bb-20261108` and the remote control
root `ROOT/slurm/bb-20261108`, where `ROOT` is
`/scratch/09999/aymanmahfuz/activation-checkpoint-integrity`. Its packet and local
record arguments retain the BA schema. An exclusive local intent precedes SSH;
an exclusive remote checkout and intent precede submission. Nonce-framed JSON,
raw transport streams, source pins, ValueGuard account/expiry, quota and owned
queue checks are unchanged. A timeout has an ambiguous outcome and requires
reconciliation. There is no automatic retry.

`retrieve_results.py JOB STUDY DESTINATION` accepts exactly these namespaces:

| Study | Outer shell controls |
| --- | --- |
| `bb-selector` | `bb-selector-{runner.log,installed.txt,modules.txt,exit-code.txt}` |
| `bb-e1` | `bb-e1-{runner.log,installed.txt,modules.txt,exit-code.txt}` |
| `bb-selector-environment-setup` | `bb-selector-{runner.log,modules.txt,exit-code.txt}` |
| `bb-e1-environment-setup` | `bb-e1-{runner.log,modules.txt,exit-code.txt}` |

Collection requires a terminal native accounting record, equal Scratch/WORK
custody receipts, and byte/size/SHA256 verification of every declared member.
Canonical relative ASCII member names reject traversal, controls and SFTP glob
syntax. Archives are made locally from verified transferred files; they are not
remote archive streams sent through a terminal. Raw SSH and SFTP output is
retained even when collection fails.

`recover_transfer.py JOB STUDY ORIGINAL DESTINATION --expected-state STATE
--expected-exit CODE:SIGNAL` uses the same four studies and exact control names.
It requires disjoint original/new folders, identical original/current terminal
accounting and original/Scratch/WORK receipts. It copies only hash-verified
partial members and requests only missing or different members through SFTP.
It preserves the original collection and rechecks every declared member before
acceptance. This is read-only custody recovery, not experiment resubmission.

`copy_runtime.py --study selector|e1` creates a fresh private environment at
`/tmp/aci-bb-copy-JOB/env`. Compact setup evidence is written to
`ROOT/artifacts/JOB/bb-STUDY-environment-setup` and retained through the declared
campaign under WORK. It does not copy the environment or cache into WORK. The
only supported resource combinations are:

| Runtime study | Partition | Wall time | Nodes / tasks / CPUs per task | SU cap |
| --- | --- | --- | --- | --- |
| `selector` | `gpu-a100-dev` | `00:40:00` | `1 / 1 / 8`, exclusive, no requeue | 2 |
| `e1` | `development` | `01:00:00` | `1 / 1 / 8`, exclusive, no requeue | 1 |

Both studies reuse the unchanged 58-package hashlocked
`experiments/ba_20261107/provenance-environment.lock`, Python 3.13.7 and uv 0.12.3.
Installation uses `uv pip sync --require-hashes --link-mode copy` with the same
PyTorch cu126 extra index and `unsafe-best-match` index strategy. It makes no
filesystem, hardlink, cache-sharing, Torch-import or CUDA qualification claim.

The runtime loads `experiments/ba_20261107/allocated_common.py` and
`experiments/ba_20261107/setup_preflight/checks.py` by their exact source paths.
Their legacy `BA_CONTRACT_PATH`, `BA_CONTRACT_SHA256`, `BA_RELEASE_PATH` and
`BA_RELEASE_SHA256` variable names remain the sealed compatibility interface;
they must refer to the new BB contract and release. No global Python search
path is widened. Every BB contract must pin both shared files, the unchanged
58-package lock, this runtime, the actual runner/batch, and all method-specific
source/configuration/input dependencies.

The shared admission gate verifies the live owned allocation, actual batch,
checkout/release/source hashes, registered environment, storage, and remaining
time before installation. The bootstrap additionally requires the owned queue
to contain only itself. Its cutoff leaves 1,260 seconds for a fresh post-install
scientific admission of at least 1,200 seconds, including the unchanged
600-second retention reserve. Setup output has a separate 100 MB bound. Future
scientific wrappers must call the shared admission gate again, then call
`qualify(checkout, out, contract)` before importing Torch or any numerical
dependency. That qualification checks the actual Python/prefix, complete exact
58-version map, bytecode flags and successful setup evidence.

Required prior ACTUAL reviews are declared by the future method contract in
`required_review_verdicts` and `required_review_job_ids`; the release must provide
the exact corresponding `prerequisite_reviews` map. Runtime checks verdict,
decimal job ID and SHA256 syntax plus any pinned job ID. The releasing operator
and independent reviewer must verify the referenced review's actual contents,
scope and custody. Hash syntax alone is not review authentication. The method source precheck must reject contracts that omit their required
ACTUAL review census; this helper does not choose method prerequisites. Serial
dispatch is required because setup rejects other owned jobs.

Only AST, Ruff, command-help and isolated standard-library metadata/transport
fixtures are appropriate locally for this change. Actual environment setup,
Torch/CUDA qualification, method fixtures, numerical comparisons, durable
retention and scientific acceptance require the reviewed allocated DEV job.
