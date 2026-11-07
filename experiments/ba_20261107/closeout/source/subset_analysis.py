"""E3/E4-only record analysis; called after real allocated subset admission."""

import csv
import hashlib
import json
import math
import re
import statistics
import tarfile
import traceback
from pathlib import Path

from analysis_core import POLICIES, analyze_e4, require_allocation
from analysis_original import analyze_e3
from baseline_census import analyze_baselines
from evidence_loading import load_e4
from input_binding import read_json
from e4_provenance import reconstruct_extension_source


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    rows = list(rows)
    keys = sorted({key for row in rows for key in row})
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, allow_nan=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def load_e3(reader, registration):
    """Exact registered-cell loader; no artificial M4 or E5 reader is supplied."""
    require_allocation()
    expected = {
        f"E3-{cell['policy']}-S{cell['seed']}-{cell['onset']}" for cell in registration
    }
    locations = {}
    for name in reader.members:
        if name.startswith("E3/worker-") and name.endswith(
            ("/summary.json", "/attempts.jsonl")
        ):
            match = re.fullmatch(
                r"E3/worker-[012]/([^/]+)/(summary\.json|attempts\.jsonl)", name
            )
            if (
                match is None
                or match[1] not in expected
                or (match[1], match[2]) in locations
            ):
                raise ValueError("Foreign or duplicate E3 cell member")
            locations[(match[1], match[2])] = name
    cells = []
    for identity in sorted(expected):
        summary, journal = (
            locations.get((identity, "summary.json")),
            locations.get((identity, "attempts.jsonl")),
        )
        if (
            summary is None
            or journal is None
            or summary.rsplit("/", 1)[0] != journal.rsplit("/", 1)[0]
        ):
            raise ValueError("Missing or split registered E3 journal/summary")
        parsed = reader.journal(journal)
        cells.append(
            {
                "cell_id": identity,
                "summary": reader.json(summary),
                "attempts": parsed["rows"],
                "journal_evidence": parsed,
            }
        )
    return cells


def fields(descriptor):
    return dict(token.split("=", 1) for token in descriptor.split() if "=" in token)


def e4_extra_validation(raw, readers, original):
    """Check setup, sampled isolation, timing scope and secondary records."""
    require_allocation()
    diagnostics = []
    extension_lineage = reconstruct_extension_source(original)
    extension_pin = extension_lineage["assembled_nvrtc_source_sha256"]
    for number, job in enumerate(("3520049", "3520050", "3520051"), start=1):
        setup = raw["setup"][job]
        if not isinstance(setup, dict):
            raise TypeError("E4 setup record absent")
        descriptor = fields(setup["isolation"]["slurm_job"])
        if not (
            descriptor.get("JobId") == job
            and descriptor.get("UserId", "").startswith("aymanmahfuz(")
            and descriptor.get("Account") == "CCR26063"
            and descriptor.get("Partition") == "gpu-a100"
            and descriptor.get("JobState") == "RUNNING"
            and descriptor.get("NumNodes") == "1"
            and descriptor.get("OverSubscribe") == "NO"
            and descriptor.get("Restarts") == "0"
            and descriptor.get("Requeue") == "0"
        ):
            raise ValueError("E4 recorded exclusive owned allocation differs")
        if (
            setup.get("scope") != "checkpoint_boundaries"
            or setup.get("forensics_in_timers") is not False
            or setup.get("production_batch")
            != {"microbatch": 4, "sequence": 512, "accumulation": 32}
        ):
            raise ValueError("E4 timing or production scope differs")
        if (
            setup.get("extension", {}).get("source_sha256") != extension_pin
            or setup["extension"].get("architecture") != "sm_80"
        ):
            raise ValueError("E4 fingerprint extension lineage differs")
        samples = [setup["isolation"]]
        observed_pids = set()
        for arm in raw["arms"]:
            if arm["block"]["job"] != number:
                continue
            samples.append(arm["isolation"])
            if (
                arm.get("reference_residency")
                != "CPU only, equal GPU residency across paired arms"
            ):
                raise ValueError("E4 reference GPU residency differs")
            for key in (
                "warmup_seconds",
                "inner_update_seconds",
                "experimental_restore_seconds",
            ):
                if (
                    type(arm.get(key)) not in (int, float)
                    or not math.isfinite(arm[key])
                    or arm[key] < 0
                ):
                    raise ValueError("E4 timing component is invalid")
            for key in (
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "warmup_peak_allocated_bytes",
            ):
                if type(arm.get(key)) is not int or arm[key] < 0:
                    raise ValueError("E4 memory record is invalid")
            expected = arm["measured_targets"] / arm["full_loop_seconds"]
            if not math.isclose(
                arm["targets_per_second"], expected, rel_tol=1e-12, abs_tol=0
            ):
                raise ValueError("E4 throughput contradicts timed equal work")
        for sample in samples:
            pids = sample.get("cuda_pids")
            if (
                not isinstance(pids, list)
                or len(pids) > 1
                or any(type(pid) is not int or pid <= 0 for pid in pids)
            ):
                raise ValueError(
                    "E4 sampled CUDA process evidence contradicts isolation"
                )
            observed_pids.update(pids)
        if len(observed_pids) > 1:
            raise ValueError(
                "E4 sampled process identity changes within one worker job"
            )
        diagnostics.append(
            {
                "job_id": job,
                "extension_source_lineage": extension_lineage,
                "samples": len(samples),
                "empty_samples": sum(not sample["cuda_pids"] for sample in samples),
                "observed_pids": sorted(observed_pids),
                "setup_seconds": setup["seconds"],
                "limit": "Recorded source assertions and sampled processes; not continuous universal hardware telemetry",
            }
        )
    return diagnostics


def coverage(e3):
    rows = []
    for cell in e3["all_cells"]:
        raw = [attempt["raw"] for attempt in cell["attempts"]]
        summary = cell["summary"]
        active = sum(row["trigger_active"] for row in raw)
        overlap = sum(row["trigger_active"] and row["selected"] for row in raw)
        rows.append(
            {
                "id": cell["id"],
                **cell["cell"],
                "status": cell["status"],
                "attempts": len(raw),
                "accepted_updates": summary["accepted_updates"],
                "actual_wrong_accepted_updates": summary[
                    "actual_wrong_accepted_updates"
                ],
                "first_alert": summary["first_alert"],
                "delay_attempts": summary["delay_attempts_from_trigger_onset"],
                "trigger_active_attempts": active,
                "selected_attempts": sum(row["selected"] for row in raw),
                "selected_active_opportunities": overlap,
                "checking_coverage_of_active_attempts": overlap / active
                if active
                else None,
                "phase_lock_observed": cell["cell"]["onset"] == "intermittent"
                and cell["cell"]["policy"] in ("period10", "period100")
                and active > 0
                and overlap == 0
                and cell["status"] == "CENSORED_AT300",
                "validation_issues": cell["validation_issues"],
            }
        )
    return rows


def figures(out, e3_rows, e4):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if e4 is not None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for index, policy in enumerate(POLICIES):
            result = e4["primary"][policy]
            blocks = [row for row in e4["all_blocks"] if row["policy"] == policy]
            ax.scatter(
                [row["recomputed_full_loop_ratio"] for row in blocks],
                [index - 0.15 + 0.03 * n for n in range(len(blocks))],
                color="#9a9a9a",
                s=20,
                alpha=0.75,
            )
            for job, log_value in result["job_mean_log_ratios"].items():
                ax.scatter(
                    math.exp(log_value),
                    index,
                    marker="D",
                    s=35,
                    color=("#0072B2", "#009E73", "#D55E00")[int(job) - 1],
                )
            ax.errorbar(
                result["geometric_ratio"],
                index,
                xerr=[
                    [result["geometric_ratio"] - result["lower"]],
                    [result["upper"] - result["geometric_ratio"]],
                ],
                color="black",
                fmt="o",
                capsize=4,
            )
        ax.axvline(1, color="#777777", linestyle="--", linewidth=1)
        ax.set(
            yticks=range(len(POLICIES)),
            yticklabels=POLICIES,
            xlabel="Checked / own matched unchecked elapsed time",
            title="E4: all 50 blocks, three job means, equal-job t95 interval (df=2)",
        )
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"e4-ratios.{suffix}", dpi=180)
        plt.close(fig)
    if e3_rows:
        policies = (
            "disabled",
            "startup",
            "period10",
            "period100",
            "randomized",
            "every",
        )
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
        for ax, onset in zip(axes, ("immediate", "delayed", "intermittent")):
            for index, policy in enumerate(policies):
                cells = [
                    row
                    for row in e3_rows
                    if row["policy"] == policy and row["onset"] == onset
                ]
                for n, row in enumerate(cells):
                    value = row["actual_wrong_accepted_updates"]
                    if value is None:
                        continue
                    offset = 0 if len(cells) == 1 else -0.2 + 0.4 * n / (len(cells) - 1)
                    ax.scatter(
                        value,
                        index + offset,
                        marker="^" if row["status"] == "CENSORED_AT300" else "o",
                        color="#0072B2" if policy != "randomized" else "#CC79A7",
                        s=35,
                    )
            ax.set(title=onset, xlabel="Incorrect accepted optimizer transitions")
            ax.grid(axis="x", alpha=0.2)
        axes[0].set(yticks=range(len(policies)), yticklabels=policies)
        fig.suptitle(
            "E3: all 39 schedule realizations from one model history; triangles censored at 300 attempts"
        )
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"e3-wrong-updates.{suffix}", dpi=180)
        plt.close(fig)


def run(stage, out, readers, early):
    require_allocation()
    stage, out = Path(stage), Path(out)

    def config(name):
        return read_json(stage / "configs" / name)

    results = {
        "status": "ANALYZED_PENDING_INDEPENDENT_ACTUAL_REVIEW",
        "scientific_acceptance": False,
        "scope": "E3_E4_ONLY_RECORD_DERIVED",
        "errors": {},
        "capability_limits": config("Subset-capability-policy.json")[
            "unsupported_and_not_claimed"
        ],
    }
    cells = load_e3(readers["3520048"], config("e3-schedules.json"))
    e3 = analyze_e3(
        config("e3-schedules.json"), cells, early, readers["3520048"].members
    )
    results["e3"] = e3
    if (
        e3["attempt_rows_validated"] != 2973
        or e3["registered_cells"] != 39
        or e3["validation_issues"] != 0
    ):
        results["errors"]["e3"] = (
            "Complete native E3 population or journal validation failed; no schedule promotion"
        )
    rows = coverage(e3)
    results["e3_coverage"] = rows
    # Deduplicate representations while retaining every original journal byte.
    journal_members = {}
    with tarfile.open(out / "e3-all39-verified-journals.tar.gz", "x:gz") as archive:
        for cell in e3["all_cells"]:
            name = cell["journal_evidence"]["member"]
            for member in (name, name.rsplit("/", 1)[0] + "/summary.json"):
                path = readers["3520048"].file(member)
                archive.add(path, arcname=member, recursive=False)
                journal_members[member] = readers["3520048"].members[member]
    verified_members = {}
    with tarfile.open(out / "e3-all39-verified-journals.tar.gz", "r|gz") as archive:
        for member in archive:
            if (
                not member.isfile()
                or member.name not in journal_members
                or member.name in verified_members
            ):
                raise ValueError("Portable E3 journal member census differs")
            h = hashlib.sha256()
            with archive.extractfile(member) as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    h.update(block)
            verified_members[member.name] = {
                "bytes": member.size,
                "sha256": h.hexdigest(),
            }
    if verified_members != journal_members or len(journal_members) != 78:
        raise ValueError("Portable full E3 journal bytes differ")
    results["portable_original_e3_journals"] = {
        "archive": "e3-all39-verified-journals.tar.gz",
        "members": journal_members,
        "all_original_fields_retained": True,
    }
    for cell in e3["all_cells"]:
        evidence = cell["journal_evidence"]
        cell["journal_evidence"] = {
            key: value for key, value in evidence.items() if key != "rows"
        }
        for ordinal, attempt in enumerate(cell["attempts"], start=1):
            raw_row = attempt.pop("raw")
            attempt["native_journal_member"] = evidence["member"]
            attempt["native_journal_ordinal"] = ordinal
            attempt["canonical_native_row_sha256"] = hashlib.sha256(
                json.dumps(
                    raw_row, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
    results["e3_randomized_variation"] = []
    for onset in ("immediate", "delayed", "intermittent"):
        chosen = [
            row
            for row in rows
            if row["policy"] == "randomized" and row["onset"] == onset
        ]
        wrong = [
            row["actual_wrong_accepted_updates"]
            for row in chosen
            if row["actual_wrong_accepted_updates"] is not None
        ]
        delays = [
            row["delay_attempts"] for row in chosen if row["delay_attempts"] is not None
        ]
        results["e3_randomized_variation"].append(
            {
                "onset": onset,
                "fixed_schedule_seeds": len(chosen),
                "confirmed_alerts": sum(
                    row["first_alert"] is not None for row in chosen
                ),
                "wrong_updates_min_median_max": [
                    min(wrong),
                    statistics.median(wrong),
                    max(wrong),
                ]
                if wrong
                else None,
                "uncensored_delay_min_median_max": [
                    min(delays),
                    statistics.median(delays),
                    max(delays),
                ]
                if delays
                else None,
                "scope": "Descriptive schedule variation conditional on one model history; paired reused seeds, no population-history confidence interval",
            }
        )
    write_csv(out / "e3-all39-cells.csv", rows)
    write_csv(
        out / "e3-all2973-attempts.csv",
        (
            {"cell_id": cell["id"], **attempt}
            for cell in e3["all_cells"]
            for attempt in cell["attempts"]
        ),
    )
    try:
        baselines = analyze_baselines(
            readers["3520048"],
            config("e3-abft-products.json"),
            early,
            early["source_sha256"],
        )
        results["baselines"] = baselines
        write_csv(out / "e3-all276-abft-objects.csv", baselines["abft_events"])
        for name, item in baselines["clean_stages"].items():
            write_csv(out / f"e3-{name}-clean-updates.csv", item["updates"])
            write_csv(
                out / f"e3-{name}-clean-test-decisions.csv",
                (
                    {"decision": key, **value}
                    for key, value in item["disjoint_clean_test_decisions"].items()
                ),
            )
    except Exception:  # noqa: BLE001 - preserve section failure and independent section output
        results["errors"]["baselines"] = traceback.format_exc()
    raw = load_e4(readers)
    write_csv(out / "e4-all100-arms.csv", raw["arms"])
    write_csv(
        out / "e4-all4400-measured-updates.csv",
        (
            {"block_id": arm["block"]["id"], "arm": arm["arm"], **update}
            for arm in raw["arms"]
            for update in arm["individual_updates"]
        ),
    )
    e4 = None
    try:
        results["e4_setup_and_isolation"] = e4_extra_validation(
            raw, readers, config("original-contract.json")
        )
        e4 = analyze_e4(raw["pairs"], raw["arms"], config("e4-blocks.json"), early)
        arm_index = {(arm["block"]["id"], arm["arm"]): arm for arm in raw["arms"]}
        for pair in e4["all_blocks"]:
            disabled = arm_index[(pair["id"], "disabled")]
            checked = arm_index[(pair["id"], pair["policy"])]
            if (
                pair["memory_delta_bytes"]
                != checked["peak_allocated_bytes"] - disabled["peak_allocated_bytes"]
            ):
                raise ValueError(
                    "Saved memory difference disagrees with matched arm peaks"
                )
        for result in e4["primary"].values():
            result["throughput_percent"] = (1 / result["geometric_ratio"] - 1) * 100
            result["throughput_percent_lower"] = (1 / result["upper"] - 1) * 100
            result["throughput_percent_upper"] = (1 / result["lower"] - 1) * 100
        for policy, result in e4["primary"].items():
            pairs = [row for row in e4["all_blocks"] if row["policy"] == policy]
            deltas = [row["memory_delta_bytes"] for row in pairs]
            require_memory = all(type(value) is int for value in deltas)
            if not require_memory:
                raise ValueError("E4 memory pair deltas are not integer bytes")
            result["gpu_memory_delta_bytes"] = {
                "minimum": min(deltas),
                "median": statistics.median(deltas),
                "maximum": max(deltas),
                "job_means": {
                    job: statistics.mean(
                        row["memory_delta_bytes"] for row in pairs if row["job"] == job
                    )
                    for job in (1, 2, 3)
                },
                "scope": "Descriptive nested memory records, no independent-update CI",
            }
        results["e4"] = e4
        write_csv(out / "e4-all50-blocks.csv", e4["all_blocks"])
        write_csv(
            out / "e4-registered-policy-estimates.csv",
            ({"policy": policy, **result} for policy, result in e4["primary"].items()),
        )
    except Exception:  # noqa: BLE001 - preserve actual-data failures without promotion
        results["errors"]["e4"] = traceback.format_exc()
    if results["errors"]:
        results["status"] = "ANALYSIS_VALIDATION_FAILURE_RETAINED"
    write_json(out / "results.json", results)
    figures(out, rows, e4)
    lines = [
        "# ValueGuard E3/E4 registered existing-data analysis",
        "",
        "This report validates recorded experiments. Independent actual-result review remains required. M4 repair and E5 training are separate, still pending at preparation; this analysis does not close them.",
        "",
        f"E3: {e3['attempt_rows_validated']} rows across {e3['registered_cells']} schedule cells; validation issues: {e3['validation_issues']}. There is one training history. Eight randomized seeds are reused across onsets, not 24 independent histories. Censored windows and phase-lock misses remain visible in every-row CSVs.",
        "",
        "| Policy | Elapsed ratio | 95% interval, three jobs/df2 | Throughput change |",
        "|---|---:|---:|---:|",
    ]
    if e4:
        for policy, result in e4["primary"].items():
            lines.append(
                f"| {policy} | {result['geometric_ratio']:.6f} | {result['lower']:.6f} to {result['upper']:.6f} | {result['throughput_percent']:+.3f}% |"
            )
    lines += [
        "",
        "E4 averages log ratios within each of three fresh jobs, then weights the three job means equally. Its Student-t interval assumes approximately normal independent job means and has only two degrees of freedom. Startup checks occur in warmup; their post-warmup ratio cannot measure performing startup checks. Sub-1 ratios alone do not demonstrate intrinsic acceleration. All individual timing and GPU-memory records are exported; setup/restore/warmup are separate from the primary loop.",
        "",
        "ABFT has 276 registered objects: 186 journaled clean-test/fault decisions and 90 calibration objects. The stored role thresholds and journal decisions are checked; original native calibration residual maxima were not exported, so this CPU phase cannot claim to reproduce their exact original GPU reductions. Full calibration reconstruction stays explicitly open for the later allocated phase. No universal fault-coverage or competitor-complete-mechanism claim follows.",
        "",
        "Initial checkpoint validation checks fresh whole-file hashes and four-role serialized schema/identity/update/cursor against frozen original native evidence. It does not assert new live model restoration, nonpersistent buffer values or process RNG/GPU placement. Original full-state/RNG comparisons are validated as recorded evidence with pinned source lineage.",
        "",
        "## Validation failures retained",
        "",
        json.dumps(results["errors"], indent=2),
        "",
        "Scientific completion and M6 readiness require independent review of these actual outputs and the outstanding registered M4/E5 evidence.",
    ]
    (out / "Report.md").write_text("\n".join(lines) + "\n")
    return results
