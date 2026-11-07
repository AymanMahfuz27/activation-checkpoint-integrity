"""One allocated pass over retained E1-E5 evidence; no training or scoring."""

import hashlib
import json
import math
import traceback
import tarfile
from pathlib import Path

from analysis_core import HISTORIES, analyze_e5, require_allocation
from analysis_e2 import analyze as analyze_e2, evaluation
from analysis_original import analyze_e1
from evidence_loading import load_e5
from state_parity import evaluate_controls
from subset_analysis import run as run_e3e4, write_csv, write_json


def expected_spec(path):
    require_allocation()
    import numpy as np

    tokens = np.load(path, mmap_mode="r", allow_pickle=False)
    if tokens.ndim != 1 or len(tokens) < 1048577:
        raise ValueError("Frozen token source cannot support registered evaluation")
    return {
        "version": "packed-windows-f64-v1",
        "sequence": 512,
        "microbatch": 4,
        "requested_max_tokens": 1048576,
        "scored_targets": 1048576,
        "first_target_index": 1,
        "last_target_index_inclusive": 1048576,
        "context": "Disjoint packed fixed contexts",
        "tail_policy": "Only complete microbatches; omitted targets explicitly counted",
        "token_prefix_sha256": hashlib.sha256(
            np.asarray(tokens[:1048577], dtype="<u4").tobytes()
        ).hexdigest(),
        "log_softmax_dtype": "float32",
        "summation_dtype": "float64",
    }


def thin(value):
    """Compact comparisons; original tensor-level receipts remain in portable tar."""
    if isinstance(value, dict):
        return {
            (
                "tensor_receipt_count"
                if key == "tensors" and isinstance(item, list)
                else key
            ): (
                len(item) if key == "tensors" and isinstance(item, list) else thin(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [thin(item) for item in value]
    return value


def e2_figure(plt, folder, e2):
    require_allocation()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, label in zip(axes, ("largest", "smaller")):
        points = e2["strata"][label]["points"]
        for index, point in enumerate(points):
            for history, value in enumerate(point["history_mean_effects"]):
                if value is not None:
                    ax.scatter(
                        index + (history - 1) * 0.08,
                        value,
                        color=("#0072B2", "#D55E00", "#009E73")[history],
                        s=22,
                    )
            interval = point["recovery_interval"]
            if interval is not None:
                mean = interval["ce_difference"]
                ax.errorbar(
                    index,
                    mean,
                    yerr=[[mean - interval["lower"]], [interval["upper"] - mean]],
                    fmt="ko",
                    capsize=4,
                )
        ax.axhspan(math.log(0.99), math.log(1.01), alpha=0.13, color="green")
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set(
            xticks=range(4),
            xticklabels=[1, 10, 50, 100],
            xlabel="Accepted continuation updates",
            ylabel="Fault minus clean held-out CE (nats/token)",
            title=label,
        )
        ax.set_yscale("symlog", linthresh=0.01)
        ax.grid(alpha=0.15)
    figure.suptitle(
        "E2: three history means; 99.375% recovery intervals; two sites nested per history"
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(Path(folder) / ("e2-effects-recovery." + suffix), dpi=180)
    plt.close(figure)


def run(stage, out, readers, initial, inputs):
    require_allocation()
    from export_reports import _plotter, _curves, _terminal, _ratios, _costs

    def config(name):
        return json.loads((Path(stage) / "configs" / name).read_text())

    result = {
        "status": "ANALYZED_PENDING_INDEPENDENT_ACTUAL_REVIEW",
        "scientific_acceptance": False,
        "errors": {},
        "scope": "E1_E5_EXISTING_RECORDS_ONLY",
        "source_jobs": list(readers),
        "limits": [
            "Three E2/E5 histories with strong small-sample t assumptions",
            "E3 schedule realizations share one model history",
            "ABFT native calibration residual maxima were not exported",
            "CPU saved-state inspection does not assert a live model or process RNG restore",
        ],
    }
    compact = Path(out) / "results"
    compact.mkdir()
    figures = compact / "figures"
    figures.mkdir()
    m4 = readers["3533911"]
    portable = {}
    with tarfile.open(
        compact / "m4-all-original-json-journals.tar.gz", "x:gz"
    ) as archive:
        for name in sorted(m4.members):
            if name.startswith(("E1/", "E2/")) and name.endswith((".json", ".jsonl")):
                archive.add(m4.file(name), arcname=name, recursive=False)
                portable[name] = m4.members[name]
    write_json(compact / "m4-portable-member-map.json", portable)
    try:
        journals = [m4.journal(f"E1/worker-{n}/attempts.jsonl") for n in range(3)]
        if any(j["partial_final_write"] is not None for j in journals):
            raise ValueError("E1 incomplete final journal")
        e1 = analyze_e1(
            config(config("repair-contract.json")["e1"]["attempt_manifest"]),
            [row for j in journals for row in j["rows"]],
        )
        e1["claim_limit"] = (
            "Reference-supported tested fault cases only; byte differences alone do not prove numerical optimizer corruption or general detector completeness"
        )
        refs = {
            (x["registration"]["state"], x["registration"].get("case_id")): x[
                "observation"
            ]
            for x in e1["all_calls"]
            if x["registration"]["mode"] == "plain-reconstruction"
        }
        cases = []
        for item in e1["all_calls"]:
            plan, observation = item["registration"], item["observation"]
            if (
                plan["mode"] not in ("observe-all", "enforce-boundary", "enforce-all")
                or observation is None
            ):
                continue
            ref = refs.get((plan["state"], plan.get("case_id")), {})
            gradient = ref.get("gradient_vs_clean", {})
            finite_change = (
                gradient.get("valid") is True
                and gradient.get("all_finite") is True
                and gradient.get("value_equal") is False
            )
            reference_supported = (
                not item["validation_issues"]
                and observation.get("original_reference_valid") is True
                and observation.get("event_bytes_match") is True
            )
            cases.append(
                {
                    "state": plan["state"],
                    "case_id": plan.get("case_id"),
                    "mode": plan["mode"],
                    "status": item["status"],
                    "reference_supported": reference_supported,
                    "finite_gradient_change_supported": finite_change
                    and reference_supported,
                    "gradient_relative_l2": gradient.get("relative_l2"),
                    "native_pre_optimizer_preservation": item[
                        "native_pre_optimizer_preservation"
                    ],
                    "scope_limit": "Optimizer byte changes are not a numerical optimizer-value proof",
                }
            )
        e1["case_readout"] = cases
        e1["by_mode"] = {
            mode: {
                "calls": sum(x["mode"] == mode for x in cases),
                "validated_rejections": sum(
                    x["mode"] == mode
                    and x["status"] == "REJECTED_PRE_OPTIMIZER"
                    and x["reference_supported"]
                    for x in cases
                ),
                "accepted_finite_gradient_changes": sum(
                    x["mode"] == mode
                    and x["status"] == "ACCEPTED"
                    and x["finite_gradient_change_supported"]
                    for x in cases
                ),
            }
            for mode in ("observe-all", "enforce-boundary", "enforce-all")
        }
        write_csv(compact / "e1-paired-case-readout.csv", cases)
        e1 = thin(e1)
        from e1_disposition import dispositions

        disposition = dispositions(e1)
        write_json(compact / "e1-validation-dispositions.json", disposition)
        write_csv(compact / "e1-validation-dispositions.csv", disposition["rows"])
        result["e1"] = e1
        if (
            not e1["census_complete"]
            or not e1["clean_controls_valid"]
            or any(r["validation_issues"] for r in e1["all_calls"])
        ):
            result["errors"]["e1"] = (
                "Incomplete or invalid registered E1 evidence retained"
            )
        # Full oracle/gradient/state comparisons remain in exact original journals.
        write_json(compact / "e1-results.json", e1)
        write_csv(compact / "e1-all638-calls.csv", e1["all_calls"])
    except Exception:
        result["errors"]["e1"] = traceback.format_exc()
    try:
        holdout = next(
            item["path"]
            for item in inputs["initial_inputs"]
            if item["role"] == "final-holdout"
        )
        e2 = analyze_e2(
            m4, config("repair-contract.json")["e2"], initial, expected_spec(holdout)
        )
        if not e2["complete_registration"]:
            result["errors"]["e2"] = (
                "Registered trajectory incomplete/nonfinite; no population intervals or demonstrated recovery"
            )
        e2_figure(_plotter(), figures, e2)
        e2 = thin(e2)
        result["e2"] = e2
        write_json(compact / "e2-results.json", e2)
        write_csv(compact / "e2-all75-evaluations.csv", e2["endpoint_rows"])
        write_csv(
            compact / "e2-all48-paired-site-effects.csv", e2["individual_effects"]
        )
        write_csv(compact / "e2-all21-controls.csv", e2["controls"])
        write_csv(
            compact / "e2-all15-arms.csv",
            ({"registration": a["registration"], **a["summary"]} for a in e2["arms"]),
        )
    except Exception:
        result["errors"]["e2"] = traceback.format_exc()
    try:
        folder = compact / "e3e4"
        folder.mkdir()
        subset = run_e3e4(stage, folder, readers, initial["early-history1234"])
        result["e3e4"] = subset
        result["errors"].update({"e3e4_" + k: v for k, v in subset["errors"].items()})
    except Exception:
        result["errors"]["e3e4"] = traceback.format_exc()
    try:
        loaded = load_e5(readers)
        spec = expected_spec(
            next(
                x["path"]
                for x in inputs["initial_inputs"]
                if x["role"] == "final-holdout"
            )
        )
        portable_e5 = {}
        with tarfile.open(
            compact / "e5-all-original-json-journals.tar.gz", "x:gz"
        ) as archive:
            for h, metadata in loaded["histories"].items():
                reader = readers[metadata["job"]]
                for name in sorted(reader.members):
                    if name.startswith("E5/") and name.endswith((".json", ".jsonl")):
                        archive.add(
                            reader.file(name),
                            arcname=metadata["job"] + "/" + name,
                            recursive=False,
                        )
                        portable_e5[metadata["job"] + "/" + name] = reader.members[name]
            for record in loaded["records"]:
                reader = readers[loaded["histories"][record["history"]]["job"]]
                for row in record["evaluations"]:
                    exact = reader.json(
                        f"E5/{record['arm']}/evaluation-{row['branch_update']:03}.json"
                    )
                    evaluation(exact, spec, windows=True)
                    if any(row.get(k) != v for k, v in exact.items() if k != "windows"):
                        raise ValueError("E5 exact evaluation JSON and journal differ")
        write_json(compact / "e5-portable-member-map.json", portable_e5)
        counters = {h: initial[f"corrected210-history{h}"] for h in HISTORIES}
        e5 = analyze_e5(loaded["records"], counters)
        controls = []
        for h in HISTORIES:
            controls.append(
                evaluate_controls(
                    h,
                    [x for x in e5["outcomes"] if x["history"] == h],
                    loaded["histories"][h]["parity"],
                    loaded["finals"][h],
                    counters[h],
                )
            )
        e5 = thin(e5)
        result["e5"], result["e5_controls"] = e5, controls
        if loaded["load_issues"] or any(
            x["validation_status"] == "FAILED_VALIDATION_RETAINED"
            for x in e5["outcomes"]
        ):
            result["errors"]["e5"] = {
                "load_issues": loaded["load_issues"],
                "invalid_arms": [
                    [x["history"], x["arm"], x.get("issues")]
                    for x in e5["outcomes"]
                    if x["validation_status"] == "FAILED_VALIDATION_RETAINED"
                ],
            }
        if not all(x["clean_checking_noninterference_supported"] for x in controls):
            result["errors"]["e5_controls"] = (
                "Required clean checking full-state/buffer/RNG parity unsupported; actual comparisons retained"
            )
        write_json(compact / "e5-results.json", e5)
        write_json(compact / "e5-controls.json", controls)
        write_csv(
            compact / "e5-all12-outcomes.csv",
            (
                {k: v for k, v in x.items() if k not in ("attempts", "evaluations")}
                for x in e5["outcomes"]
            ),
        )
        for kind in ("attempts", "evaluations"):
            write_csv(
                compact / f"e5-all-{kind}.csv",
                (
                    {"history": x["history"], "arm": x["arm"], **r}
                    for x in e5["outcomes"]
                    for r in x[kind]
                ),
            )
        write_csv(
            compact / "e5-history-ratios.csv",
            (
                {"contrast": name, **pair}
                for name, x in e5["contrasts"].items()
                for pair in x["histories"]
            ),
        )
        figures = compact / "figures"
        plt = _plotter()
        for plot in (_curves, _terminal, _ratios, _costs):
            plot(plt, figures, e5)
        del loaded
    except Exception:
        result["errors"]["e5"] = traceback.format_exc()
    if result["errors"]:
        result["status"] = "ANALYSIS_VALIDATION_ISSUES_RETAINED"
    write_json(compact / "combined-results.json", result)
    report = [
        "# ValueGuard retained-results analysis",
        "",
        "Status: allocated analysis; independent actual-result review pending.",
        "No training or new model scoring was performed.",
        "",
        "## E2 fixed-horizon effects",
    ]
    for label, data in result.get("e2", {}).get("strata", {}).items():
        report.append(
            f"- {label}: primary100 {json.dumps(data['points'][-1]['primary975'])}; recovery {data['recovery_status']}."
        )
    report.extend(["", "## E5 time to confirmed quality"])
    for row in result.get("e5", {}).get("outcomes", []):
        report.append(
            f"- History{row['history']} {row['arm']}: {row['status']}; {row.get('confirmed_seconds')} seconds; {row['accepted_updates']} accepted updates; validation {row['validation_status']}."
        )
    report.extend(["", "## E5 ratios"])
    for name, row in result.get("e5", {}).get("contrasts", {}).items():
        report.append(f"- {name}: {json.dumps(row)}")
    report.extend(
        [
            "",
            "## Validation issues",
            json.dumps(result["errors"], indent=2),
            "",
            "## Limits",
            *result["limits"],
        ]
    )
    (compact / "analysis-readout.md").write_text("\n".join(report) + "\n")
    return result
