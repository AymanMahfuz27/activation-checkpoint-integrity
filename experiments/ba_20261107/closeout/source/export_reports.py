"""Allocated-only CSV/JSON, static figures and private result reports.

This module consumes preserved reducer outputs. It never trains, evaluates a
model, changes an endpoint or grants scientific acceptance. Rendering is itself
part of the single reserved analysis allocation.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

from analysis_core import (
    ARMS,
    EVALUATIONS,
    HISTORIES,
    POLICIES,
    TARGETS,
    require_allocation,
)


COLORS = {
    "clean-disabled": "#2463A6",
    "buggy-disabled": "#CF9217",
    "buggy-guarded-repaired": "#A64D79",
    "clean-checked": "#587B3C",
}
SHORT = {
    "clean-disabled": "Clean",
    "buggy-disabled": "Buggy",
    "buggy-guarded-repaired": "Guarded + repaired",
    "clean-checked": "Clean + checked",
}
MARKERS = dict(zip(ARMS, ("o", "s", "^", "D")))


def finite(value, *, positive=False):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def number(value, digits=4):
    return f"{value:.{digits}g}" if finite(value) else "N/A"


def write_json(path, value):
    require_allocation()
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows, fields=None):
    """Nested evidence is lossless JSON in a cell; null is an empty CSV cell."""
    require_allocation()
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["status"]
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(
                        value, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def curve_points(outcome, horizontal):
    """Return only observed points, breaking lines at missing/invalid schedules.

    A line connects adjacent scheduled observations only. It is a visual guide,
    never a time-to-target interpolation. Invalid trajectory values remain raw
    descriptive points with an explicit validation label.
    """
    require_allocation()
    if horizontal not in ("branch_update", "elapsed_seconds"):
        raise ValueError(
            "Curve axis must be accepted updates or total measured seconds"
        )
    points, segments, previous = [], [], None
    for row in outcome.get("evaluations", []):
        ce, update, x = (
            row.get("cross_entropy"),
            row.get("branch_update"),
            row.get(horizontal),
        )
        valid = (
            row.get("status") == "PASS"
            and finite(ce)
            and ce < 700
            and update in EVALUATIONS
            and finite(x)
        )
        preservation = row.get("state_preservation", {})
        preservation_valid = all(
            preservation.get(key) is True
            for key in ("valid", "all_finite", "byte_equal", "value_equal")
        )
        if not valid:
            previous = None
            continue
        point = {
            "x": x,
            "ppl": math.exp(ce),
            "branch_update": update,
            "row_state_preservation_valid": preservation_valid,
        }
        points.append(point)
        if (
            previous is not None
            and preservation_valid
            and previous["row_state_preservation_valid"]
        ):
            left = EVALUATIONS.index(previous["branch_update"])
            if (
                left + 1 < len(EVALUATIONS)
                and EVALUATIONS[left + 1] == update
                and x >= previous["x"]
            ):
                segments.append((previous, point))
        previous = point
    return {
        "points": points,
        "segments": segments,
        "source_evaluation_rows": len(outcome.get("evaluations", [])),
        "unplottable_rows_retained_in_raw": len(outcome.get("evaluations", []))
        - len(points),
    }


def terminal_mark(outcome):
    require_allocation()
    if (
        outcome.get("status") == "TARGET_CONFIRMED"
        and outcome.get("validation_status") == "PASS"
        and finite(outcome.get("confirmed_seconds"), positive=True)
    ):
        return {
            "kind": "confirmed",
            "seconds": outcome["confirmed_seconds"],
            "marker": "o",
        }
    bound = outcome.get("observed_bound_seconds")
    if not finite(bound):
        journal = outcome.get("last_journal_elapsed_seconds")
        return {
            "kind": "terminal_missing",
            "seconds": None,
            "last_journal_lower_bound": journal if finite(journal) else None,
            "marker": None,
        }
    if outcome.get("validation_status") == "FAILED_VALIDATION_RETAINED":
        return {"kind": "invalid_evidence", "seconds": bound, "marker": "D"}
    if outcome.get("status") == "FAILED_ARM_RETAINED":
        return {"kind": "failure", "seconds": bound, "marker": "x"}
    if outcome.get("status") in ("CENSORED_UPDATE_LIMIT", "CENSORED_WALL_LIMIT"):
        return {"kind": "censored", "seconds": bound, "marker": ">"}
    return {"kind": "unclassified_terminal", "seconds": bound, "marker": "D"}


def exclusive_cost(outcome):
    require_allocation()
    cost = outcome.get("cost", {})
    keys = (
        "evaluation_seconds",
        "attempt_loop_seconds",
        "unallocated_in_clock_seconds",
    )
    if any(not finite(cost.get(key)) for key in keys) or not finite(
        cost.get("observed_seconds")
    ):
        return None
    if outcome.get(
        "validation_status"
    ) == "FAILED_VALIDATION_RETAINED" or not math.isclose(
        sum(cost[key] for key in keys),
        cost["observed_seconds"],
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        return None
    return {key: cost[key] for key in keys}


def _plotter():
    require_allocation()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#666666",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def _save(plt, figure, folder, name):
    figure.savefig(folder / (name + ".png"), dpi=180, bbox_inches="tight")
    figure.savefig(folder / (name + ".pdf"), bbox_inches="tight")
    plt.close(figure)


def _curves(plt, folder, e5):
    for horizontal, label, filename in (
        (
            "branch_update",
            "Accepted updates after global210",
            "e5-ppl-vs-accepted-updates",
        ),
        (
            "elapsed_seconds",
            "Total measured elapsed seconds (including E0)",
            "e5-ppl-vs-total-seconds",
        ),
    ):
        fig, axes = plt.subplots(
            1, 3, figsize=(16, 5), sharex=True, sharey=True, constrained_layout=True
        )
        for history, ax in zip(HISTORIES, axes):
            plotted = 0
            for arm in ARMS:
                outcome = next(
                    row
                    for row in e5["outcomes"]
                    if (row["history"], row["arm"]) == (history, arm)
                )
                curve = curve_points(outcome, horizontal)
                for a, b in curve["segments"]:
                    ax.plot(
                        [a["x"], b["x"]],
                        [a["ppl"], b["ppl"]],
                        color=COLORS[arm],
                        linewidth=1,
                        alpha=0.7,
                    )
                ax.scatter(
                    [p["x"] for p in curve["points"]],
                    [p["ppl"] for p in curve["points"]],
                    marker=MARKERS[arm],
                    edgecolors=COLORS[arm],
                    facecolors=COLORS[arm]
                    if outcome.get("validation_status") == "PASS"
                    else "none",
                    s=26,
                    label=SHORT[arm],
                )
                plotted += len(curve["points"])
            ax.axhline(
                math.exp(TARGETS[history]),
                color="#444444",
                linestyle="--",
                linewidth=1,
                label="Known validation target",
            )
            ax.set(title=f"History {history}", xlabel=label)
            ax.grid(axis="y")
            if not plotted:
                ax.text(
                    0.5,
                    0.5,
                    "No finite evaluation observed",
                    ha="center",
                    transform=ax.transAxes,
                )
        axes[0].set_ylabel("Perplexity = exp(saved cross entropy)")
        axes[-1].legend(loc="best", fontsize=8)
        fig.suptitle(
            "E5: observed scheduled evaluations; n=3 histories\nOpen points indicate unvalidated/partial trajectories; no inferred target crossing"
        )
        _save(plt, fig, folder, filename)


def _terminal(plt, folder, e5):
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5), sharex=True, sharey=True, constrained_layout=True
    )
    bounds = [terminal_mark(row)["seconds"] for row in e5["outcomes"]]
    maximum = max((value for value in bounds if value is not None), default=1.0)
    for history, ax in zip(HISTORIES, axes):
        for index, arm in enumerate(ARMS):
            row = next(
                row
                for row in e5["outcomes"]
                if (row["history"], row["arm"]) == (history, arm)
            )
            mark = terminal_mark(row)
            if mark["seconds"] is None:
                text = "Terminal time missing"
                if mark["last_journal_lower_bound"] is not None:
                    text += (
                        f"; last journal ≥{number(mark['last_journal_lower_bound'])}s"
                    )
                ax.text(
                    0.02, index, text, transform=ax.get_yaxis_transform(), fontsize=8
                )
            else:
                ax.scatter(
                    mark["seconds"],
                    index,
                    marker=mark["marker"],
                    color=COLORS[arm],
                    s=65,
                )
                ax.annotate(
                    mark["kind"] + ": " + number(mark["seconds"]) + "s",
                    (mark["seconds"], index),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
        ax.set(
            title=f"History {history}",
            xlabel="Measured seconds",
            xlim=(0, maximum * 1.15),
            ylim=(-0.6, 3.7),
        )
        ax.set_yticks(range(4), [SHORT[a] for a in ARMS])
        ax.grid(axis="x")
    fig.suptitle(
        "E5: confirmed target time, observed censor bound or failure\nCircle=confirmed; triangle=censored; X=failure; diamond=invalid/unclassified. Missing is not zero."
    )
    _save(plt, fig, folder, "e5-confirmed-target-or-terminal-status")


def _ratios(plt, folder, e5):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for (name, contrast), ax in zip(e5["contrasts"].items(), axes):
        for index, pair in enumerate(contrast["histories"]):
            if finite(pair["ratio"], positive=True):
                ax.scatter(
                    pair["ratio"],
                    index,
                    marker=("o", "s", "^")[index],
                    color="#2463A6",
                    s=55,
                )
                ax.annotate(
                    number(pair["ratio"]),
                    (pair["ratio"], index),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            else:
                ax.text(
                    0.02,
                    index,
                    "Not estimable: missing required target time",
                    transform=ax.get_yaxis_transform(),
                    fontsize=8,
                )
        interval = contrast["population_interval"]
        if interval is not None:
            ax.plot(
                [interval["lower"], interval["upper"]],
                [3, 3],
                color="#333333",
                linewidth=2,
            )
            ax.scatter(interval["ratio"], 3, marker="D", color="#333333", s=45)
        else:
            ax.text(
                0.02,
                3,
                "No complete-case population CI",
                transform=ax.get_yaxis_transform(),
                fontsize=8,
            )
        ax.set_xscale("log")
        ax.axvline(1, color="#777777", linestyle="--")
        ax.set_yticks(
            range(4), [f"History {h}" for h in HISTORIES] + ["Paired logtime t95, df=2"]
        )
        ax.set(
            title=name.replace("_", " "),
            xlabel=SHORT[contrast["numerator"]]
            + " / "
            + SHORT[contrast["denominator"]],
            ylim=(-0.5, 3.5),
        )
        ax.grid(axis="x")
    fig.suptitle(
        "E5: all three history ratios; <1 means less measured time\nIntervals require every registered pair and strong low-n Student-t assumptions"
    )
    _save(plt, fig, folder, "e5-paired-time-ratios")


def _costs(plt, folder, e5):
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    labels, annotations = [], []
    keys = (
        "evaluation_seconds",
        "attempt_loop_seconds",
        "unallocated_in_clock_seconds",
    )
    palette = ("#2463A6", "#CF9217", "#aaaaaa")
    for index, row in enumerate(e5["outcomes"]):
        labels.append(f"H{row['history']} · {SHORT[row['arm']]}")
        parts = exclusive_cost(row)
        if parts is None:
            ax.text(
                0.02,
                index,
                "No reconciled component breakdown; " + row["status"],
                transform=ax.get_yaxis_transform(),
                fontsize=8,
            )
            continue
        left = 0
        for key, color in zip(keys, palette):
            ax.barh(
                index,
                parts[key],
                left=left,
                color=color,
                height=0.7,
                label=key.replace("_seconds", "").replace("_", " ")
                if index == 0
                else None,
            )
            left += parts[key]
        cost = row["cost"]
        annotations.append(
            (
                left,
                index,
                "Nested in loop: rejected="
                + number(cost.get("rejected_attempt_compute_seconds"))
                + ", rollback="
                + number(cost.get("rollback_seconds"))
                + ", retry="
                + number(cost.get("safe_retry_seconds"))
                + "s",
            )
        )
    for x, y, text in annotations:
        ax.annotate(
            text,
            (x, y),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7,
            va="center",
        )
    ax.set_yticks(range(len(labels)), labels)
    ax.set(
        xlabel="Total measured seconds",
        title="E5: exclusive clock components; all twelve arms retained",
    )
    ax.text(
        0,
        -0.11,
        "Rejected compute, rollback and safe retry are nested annotations, never added again. Residual is unallocated clock time, not attributed overhead.\nExperimental restore/setup/final retention are separate from these measured clocks.",
        transform=ax.transAxes,
        fontsize=8,
    )
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=color, label=key.replace("_seconds", "").replace("_", " "))
            for key, color in zip(keys, palette)
        ],
        loc="best",
        fontsize=8,
    )
    ax.grid(axis="x")
    ax.set_xlim(left=0)
    _save(plt, fig, folder, "e5-total-clock-cost-breakdown")


def _e4(plt, folder, e4):
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = ("#2463A6", "#CF9217", "#A64D79")
    for index, policy in enumerate(POLICIES):
        result = e4["primary"][policy]
        for job in (1, 2, 3):
            blocks = [
                row
                for row in e4["all_blocks"]
                if row["policy"] == policy and row["job"] == job
            ]
            for offset, row in enumerate(blocks):
                x = index + (job - 2) * 0.20 + (offset - (len(blocks) - 1) / 2) * 0.025
                ax.scatter(
                    x,
                    row["recomputed_full_loop_ratio"],
                    s=24,
                    marker=("o", "s", "^")[job - 1],
                    color=colors[job - 1],
                    alpha=0.75,
                )
            mean = math.exp(
                result["job_mean_log_ratios"].get(
                    job, result["job_mean_log_ratios"].get(str(job))
                )
            )
            ax.scatter(
                index + (job - 2) * 0.20,
                mean,
                marker="D",
                color=colors[job - 1],
                s=65,
                edgecolors="#333333",
                linewidth=0.7,
                label=f"Job {job}: blocks / diamond job mean" if index == 0 else None,
            )
        ax.plot(
            [index, index],
            [result["lower"], result["upper"]],
            color="#222222",
            linewidth=2,
        )
        ax.scatter(
            index,
            result["geometric_ratio"],
            marker="_",
            s=120,
            color="#222222",
            zorder=4,
        )
    ax.axhline(1, linestyle="--", color="#555555")
    ax.set_yscale("log")
    ax.set_xticks(range(len(POLICIES)), POLICIES)
    ax.set(
        ylabel="Checked / own-disabled full-loop elapsed ratio",
        title="E4: all50 paired blocks and all3 fresh job means\nBlack estimate/t95 interval gives each job equal weight; df=2",
    )
    ax.text(
        0,
        -0.15,
        "6,400 updates = 4,400 individually journaled measured updates + 2,000 source/counter-supported warmups.\nStartup checks all occur in warmup; its primary ratio does not measure startup-check execution cost.",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y")
    _save(plt, fig, folder, "e4-all-blocks-and-independent-job-estimates")


def _e3(plt, folder, e3):
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 7), sharex=True, constrained_layout=True
    )
    for onset, ax in zip(("immediate", "delayed", "intermittent"), axes):
        cells = [row for row in e3["all_cells"] if row["cell"]["onset"] == onset]
        labels = []
        for index, row in enumerate(cells):
            labels.append(row["cell"]["policy"] + " / S" + str(row["cell"]["seed"]))
            attempts = row.get("attempts", [])
            if not attempts:
                ax.text(
                    0.02,
                    index,
                    "Missing cell",
                    transform=ax.get_yaxis_transform(),
                    fontsize=8,
                )
                continue
            last = attempts[-1]
            valid_reject = (
                last["validated_pre_optimizer_rejection"]
                and not row["validation_issues"]
            )
            censored = (
                row["status"] == "CENSORED_AT300"
                and len(attempts) == 300
                and not row["validation_issues"]
            )
            marker = (
                "o"
                if valid_reject
                else ">"
                if censored
                else "x"
                if last["authoritative_status"] == "ERROR"
                else "D"
            )
            ax.scatter(
                len(attempts),
                index,
                marker=marker,
                color="#2463A6" if valid_reject else "#A64D79",
                s=45,
            )
        ax.set_yticks(range(len(labels)), labels)
        ax.set(title=onset, xlabel="Observed terminal primary attempt", xlim=(0, 310))
        ax.grid(axis="x")
    fig.suptitle(
        "E3: all39 fixed cells, diagnostic8,192 targets/update, one training history\nCircle=validated pre-optimizer rejection; triangle=300-attempt censor; X=ERROR; diamond=other/invalid. Randomized seeds are paired across onsets."
    )
    _save(plt, fig, folder, "e3-all-cells-authoritative-terminal-outcomes")


def _figure_links():
    return [
        "",
        "Figures and raw evidence:",
        "",
        "- [PPL versus accepted updates](figures/e5-ppl-vs-accepted-updates.png) ([PDF](figures/e5-ppl-vs-accepted-updates.pdf))",
        "- [PPL versus total measured seconds](figures/e5-ppl-vs-total-seconds.png) ([PDF](figures/e5-ppl-vs-total-seconds.pdf))",
        "- [Confirmed target times, censoring and failures](figures/e5-confirmed-target-or-terminal-status.png)",
        "- [All paired history time ratios](figures/e5-paired-time-ratios.png)",
        "- [Total clock cost breakdown](figures/e5-total-clock-cost-breakdown.png)",
        "- [All E4 blocks and job estimates](figures/e4-all-blocks-and-independent-job-estimates.png)",
        "- [All E3 terminal cell outcomes](figures/e3-all-cells-authoritative-terminal-outcomes.png)",
        "- [Full results JSON](results.json), [lineage](lineage.json), [controls](controls.json), [all twelve E5 outcomes](e5-all12-outcomes.csv)",
        "",
    ]


def _plain_report(bundle, lineage):
    e1, e3, e4, e5 = (bundle[name] for name in ("e1", "e3", "e4", "e5"))
    text = [
        "# ValueGuard: what the results mean",
        "",
        "The original study is still incomplete. M4 stopped after recording "
        + str(e1["observed_calls"])
        + " of its 638 registered calls, and E2 never ran. E3 and the three timing jobs finished running, but their scientific conclusions still need independent review. M6 is not ready.",
        "",
        "The extra E5 experiment asks a practical question: after paying for checking, rejected work, rollback, repair and evaluation, does protected training reach the same fixed quality target sooner? It compares four approaches in each of three independent histories. The targets were already known and used to stop training; they are not new, untouched tests.",
        "",
        "The table shows every arm. A missing target time means the target was not confirmed, the run failed, or the evidence did not validate. It does not mean zero time or an infinite speedup.",
        "",
        "| History | Approach | Outcome | Updates accepted | Confirmed target time (s) |",
        "|---|---|---|---:|---:|",
    ]
    for row in e5["outcomes"]:
        status = (
            row["status"]
            if row["validation_status"] == "PASS"
            else row["status"] + "; " + row["validation_status"]
        )
        text.append(
            f"| {row['history']} | {SHORT[row['arm']]} | {status} | {number(row.get('accepted_updates'))} | {number(row.get('confirmed_seconds'))} |"
        )
    text += [
        "",
        "A time ratio below 1 means the first approach took less measured time; above 1 means it took more. Every history stays visible. The uncertainty interval is available only when all three required pairs have confirmed target times.",
        "",
        "| Comparison | H1234 / H1 / H2 ratios | 95% uncertainty interval |",
        "|---|---|---|",
    ]
    for name, row in e5["contrasts"].items():
        interval = row["population_interval"]
        bounds = (
            number(interval["lower"]) + " to " + number(interval["upper"])
            if interval
            else "Unavailable: at least one required time is missing"
        )
        text.append(
            "| "
            + SHORT[row["numerator"]]
            + " / "
            + SHORT[row["denominator"]]
            + " | "
            + " / ".join(number(pair["ratio"]) for pair in row["histories"])
            + " | "
            + bounds
            + " |"
        )
        if interval is None:
            meaning = "There is no complete three-history estimate. The missing outcomes remain part of the result."
        elif interval["upper"] < 1:
            meaning = "The estimated interval is below 1, supporting less measured time in this comparison under the three-history statistical assumptions."
        elif interval["lower"] > 1:
            meaning = "The estimated interval is above 1, supporting more measured time in this comparison under the three-history statistical assumptions."
        else:
            meaning = "The interval includes equal time. These three histories do not establish a consistent time advantage or disadvantage."
        text += ["", name.replace("_", " ") + ": " + meaning, ""]
    supported = all(
        row.get("clean_checking_noninterference_supported") is True
        for row in bundle["controls"]
    )
    text += [
        "The clean checked/unchecked control is "
        + (
            "supported across all three histories"
            if supported
            else "not fully established across all three histories"
        )
        + " by the retained first-update and matched final-state evidence. Different endpoints, delayed alerts, missing files and failed validation are kept separate from a passing control. See the technical report and controls.json for the limits on checking-cost claims.",
        "",
        "Using fewer updates is not enough: the protected approach must reach a confirmed target in less total time to show net time savings. Any advantage belongs to the combined detection-and-repair policy, not checking alone or a faster optimizer.",
        "",
        "The original timing experiment uses three fresh jobs as its three independent observations. Its 50 paired blocks show within-job variation. The estimates give each job equal weight, and their intervals rely on strong assumptions with only three jobs. All startup checks occurred during warmup, so its measured ratio cannot tell us the cost of performing startup checks.",
        "",
        "| Checking policy | Checked / disabled time | 95% uncertainty interval |",
        "|---|---:|---:|",
    ]
    for policy in POLICIES:
        row = e4["primary"][policy]
        text.append(
            f"| {policy} | {number(row['geometric_ratio'])} | {number(row['lower'])} to {number(row['upper'])} |"
        )
    text += [
        "",
        "The E3 analysis retains all 39 scheduled cells and "
        + str(e3["attempt_rows_validated"])
        + " primary attempt rows. Errors remain errors; they are not counted as successful prevention. The eight randomized seeds are reused across the three onset conditions, rather than 24 independent training histories.",
        "",
        "The baseline comparison is narrow: checks on one unfused matrix product do not establish coverage of an entire training system. The records include 276 registered products, but only 186 test/fault decisions were journaled. The other 90 are retained calibration objects, not additional recorded detector decisions.",
        "",
        "The full technical report, raw CSV/JSON files and figures preserve failures, missing data, censoring and all history-level ratios. Numerical processing is complete; scientific acceptance still requires independent review. No replacement experiment follows automatically from a negative or inconclusive result.",
        "",
        "Analysis job: " + str(lineage.get("analysis_job_id", "unknown")) + ".",
        "",
    ]
    return "\n".join(text + _figure_links())


def report_text(bundle, lineage, *, plain=False):
    require_allocation()
    if plain:
        return _plain_report(bundle, lineage)
    e1, e2, e3, e4, e5 = (bundle[name] for name in ("e1", "e2", "e3", "e4", "e5"))
    text = [
        "# ValueGuard E1–E5: "
        + ("plain-English results" if plain else "technical results"),
        "",
        "Status: allocated analysis complete, pending independent actual-result review. M4 remains incomplete; full M5 scientific acceptance and M6 readiness are not established.",
        "",
        "M4 retained "
        + str(e1["observed_calls"])
        + " of 638 registered calls (478 main / 160 reference); "
        + str(e1["missing_calls"])
        + " are missing. E2: "
        + e2["status"]
        + " ; all 15 continuations remain visible and no endpoint or population interval is invented. The CPU/CUDA optimizer-step reference-placement confound is source-supported; the exact tensor pair and numerical model corruption are not established.",
        "",
        "E3 validates "
        + str(e3["attempt_rows_validated"])
        + " retained primary rows across 39 cells. Errors and partial optimizer failures remain distinct from validated pre-optimizer rejection. Randomized seeds are paired across onsets; this is one training history. Validation issues: "
        + str(e3["validation_issues"])
        + ".",
        "",
        "E4 uses three fresh jobs as independent units. All 50 pairs and 6,400 updates are retained, including 4,400 journaled measured updates and 2,000 warmups supported by source/counters. Each job contributes equal weight to the paired log-ratio estimate; 95% Student-t intervals have df=2 and strong low-n assumptions.",
        "",
        "| Policy | Checked / disabled time | 95% interval |",
        "|---|---:|---:|",
    ]
    for policy in POLICIES:
        row = e4["primary"][policy]
        text.append(
            f"| {policy} | {number(row['geometric_ratio'])} | {number(row['lower'])} to {number(row['upper'])} |"
        )
    text += [
        "",
        "Startup checks happened during warmup, outside the primary timing window. This ratio cannot measure the cost of executing those checks or cold setup. Exclusivity and sampled isolation do not establish continuous universal hardware telemetry.",
        "",
        "E5 measures detection plus rollback, one safe retry and permanent correction. Targets are frozen known validation/stopping criteria, not untouched tests. The primary time includes E0, checking, rejected work, rollback/retry, evaluation and journaling, and requires two consecutive scheduled postbaseline qualifying evaluations. First crossing is secondary; no interpolation is used.",
        "",
        "| History | Arm | Native status | Validation | Accepted updates | Confirmed seconds | Observed terminal seconds |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in e5["outcomes"]:
        text.append(
            f"| {row['history']} | {row['arm']} | {row['status']} | {row['validation_status']} | {number(row.get('accepted_updates'))} | {number(row.get('confirmed_seconds'))} | {number(row.get('observed_bound_seconds'))} |"
        )
    text += [
        "",
        "Missing times are not zero or infinity. Failed arms are failures, separate from update/wall censoring. The last journal clock alone is not a terminal-duration bound. A missing required target time precludes a complete-case population interval.",
        "",
        "| Contrast | Three paired history ratios | Paired logtime 95% interval |",
        "|---|---|---|",
    ]
    for name, row in e5["contrasts"].items():
        interval = row["population_interval"]
        bounds = (
            number(interval["lower"]) + " to " + number(interval["upper"])
            if interval
            else "Not estimable; required target time missing"
        )
        text.append(
            "| "
            + name
            + " | "
            + ", ".join(
                "H" + str(pair["history"]) + ": " + number(pair["ratio"])
                for pair in row["histories"]
            )
            + " | "
            + bounds
            + " |"
        )
    controls = bundle["controls"]
    supported = all(
        row.get("clean_checking_noninterference_supported") is True for row in controls
    )
    text += [
        "",
        "Actual clean checked/unchecked first-update and matched-endpoint fullstate/buffer/RNG noninterference evidence across all 3 histories: "
        + (
            "supported by retained controls"
            if supported
            else "not established for every history; checking-cost/noninterference claims remain limited"
        )
        + ". Delayed/missing alerts and different endpoints do not receive parity PASS. Inspect controls.json for each comparison.",
        "",
        "Fewer accepted updates alone do not show amortization: total measured seconds must be lower at a confirmed matched quality target. These observations do not establish checking alone or intrinsic optimizer acceleration. Favorable, negative, failed, censored and inconclusive outcomes all remain in the tables and figures.",
        "",
        "The baseline census distinguishes 276 registered scoped ABFT events, 186 journaled test/fault decisions and 90 retained calibration objects. Native calibration residual maxima were not exported; no values or decisions are fabricated. Scoped unfused GEMM and cheap-check measurements do not establish optimized cost or broad model coverage.",
        "",
        "M4: scientifically incomplete. M5: allocated numerical evidence prepared; final frozen-criterion assessment requires independent actual review and all validity gates. M6: not ready. E5 is supplemental and does not change the original M4/M5 criteria.",
        "",
        "Evidence: results.json and raw CSV files preserve every registered outcome, validation issue, attempt and evaluation. Figures are standalone PNG/PDF files. Empty CSV cells mean missing/not applicable; nested evidence is JSON. lineage.json binds raw member/Ranch proofs, source, contract, environment and allocated job. No source evidence is overwritten.",
        "",
        "Analysis job: "
        + str(lineage.get("analysis_job_id", "unknown"))
        + ". Source packet: "
        + str(lineage.get("source_packet_sha256", "unknown"))
        + ".",
        "",
    ]
    return "\n".join(text + _figure_links())


def export_all(output, bundle, lineage):
    require_allocation()
    if (
        bundle.get("verification", {}).get("status")
        != "PASS_RAW_ARCHIVE_AND_PERSISTENT_MEMBER_LINEAGE"
    ):
        raise ValueError("Exports require successful exact all-member evidence gate")
    counts = (
        (len(bundle["e1"]["all_calls"]), 638),
        (len(bundle["e2"]["all_arms"]), 15),
        (len(bundle["e3"]["all_cells"]), 39),
        (len(bundle["e4"]["all_blocks"]), 50),
        (len(bundle["e4_arms"]), 100),
        (len(bundle["e5"]["outcomes"]), 12),
        (len(bundle["controls"]), 3),
    )
    if any(actual != registered for actual, registered in counts) or {
        (row["history"], row["arm"]) for row in bundle["e5"]["outcomes"]
    } != {(history, arm) for history in HISTORIES for arm in ARMS}:
        raise ValueError("Export census differs from registered population")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    figures = output / "figures"
    figures.mkdir()
    write_json(output / "results.json", bundle)
    write_json(output / "lineage.json", lineage)
    write_json(output / "controls.json", bundle["controls"])
    write_csv(output / "e1-all638-calls.csv", bundle["e1"]["all_calls"])
    write_csv(output / "e2-all15-not-reached.csv", bundle["e2"]["all_arms"])
    write_csv(
        output / "e3-all39-cells.csv",
        [
            {key: value for key, value in row.items() if key != "attempts"}
            for row in bundle["e3"]["all_cells"]
        ],
    )
    write_csv(
        output / "e3-all-retained-attempts.csv",
        [
            {"cell_id": row["id"], **attempt}
            for row in bundle["e3"]["all_cells"]
            for attempt in row.get("attempts", [])
        ],
    )
    write_csv(
        output / "e3-baseline-clean-updates.csv",
        [
            {"batch_stage": stage, **row}
            for stage, data in bundle["baselines"]["clean_stages"].items()
            for row in data["updates"]
        ],
    )
    write_csv(
        output / "e3-baseline-cheap-observations.csv",
        [
            {"batch_stage": stage, **row}
            for stage, data in bundle["baselines"]["clean_stages"].items()
            for row in data["cheap_observations"]
        ],
    )
    write_csv(output / "e3-abft-all276-events.csv", bundle["baselines"]["abft_events"])
    write_csv(
        output / "e3-baseline-cheap-decision-validation.csv",
        [
            {"batch_stage": stage, **row}
            for stage, data in bundle["baselines"]["clean_stages"].items()
            for row in data["cheap_decision_validation"]
        ],
    )
    write_csv(output / "e4-all50-blocks.csv", bundle["e4"]["all_blocks"])
    write_csv(output / "e4-all100-arms.csv", bundle["e4_arms"])
    write_csv(
        output / "e4-all4400-measured-updates.csv",
        [
            {"block_id": row["block"]["id"], "arm": row["arm"], **update}
            for row in bundle["e4_arms"]
            for update in row["individual_updates"]
        ],
    )
    write_csv(
        output / "e5-all12-outcomes.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key not in ("evaluations", "attempts")
            }
            for row in bundle["e5"]["outcomes"]
        ],
    )
    for kind in ("evaluations", "attempts"):
        write_csv(
            output / ("e5-all-" + kind + ".csv"),
            [
                {"history": outcome["history"], "arm": outcome["arm"], **row}
                for outcome in bundle["e5"]["outcomes"]
                for row in outcome.get(kind, [])
            ],
        )
    write_csv(
        output / "e5-all-paired-history-ratios.csv",
        [
            {"contrast": name, **pair}
            for name, row in bundle["e5"]["contrasts"].items()
            for pair in row["histories"]
        ],
    )
    write_csv(
        output / "e5-cost-components-and-nested-annotations.csv",
        [
            {
                "history": row["history"],
                "arm": row["arm"],
                "status": row["status"],
                "exclusive_components_valid": exclusive_cost(row) is not None,
                **row.get("cost", {}),
            }
            for row in bundle["e5"]["outcomes"]
        ],
    )
    plt = _plotter()
    _curves(plt, figures, bundle["e5"])
    _terminal(plt, figures, bundle["e5"])
    _ratios(plt, figures, bundle["e5"])
    _costs(plt, figures, bundle["e5"])
    _e4(plt, figures, bundle["e4"])
    _e3(plt, figures, bundle["e3"])
    for name, plain in (
        ("Report-Technical.md", False),
        ("Report-Plain-English.md", True),
    ):
        with (output / name).open("x", encoding="utf-8") as handle:
            handle.write(report_text(bundle, lineage, plain=plain))
    files = {
        str(path.relative_to(output)): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    write_json(
        output / "output-manifest.json",
        {
            "status": "ANALYZED_PENDING_INDEPENDENT_ACTUAL_REVIEW",
            "files": files,
            "scientific_acceptance": False,
            "M4": "INCOMPLETE",
            "M5": "PENDING_INDEPENDENT_ACTUAL_REVIEW",
            "M6": "NOT_READY",
        },
    )
    return {"output": str(output), "files": files, "scientific_acceptance": False}
