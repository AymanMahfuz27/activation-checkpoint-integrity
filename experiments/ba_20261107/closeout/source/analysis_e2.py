"""Allocated E2 record validation and the original nested-history estimands.

Inputs are verified native journals, frozen protocol, initial role counters and
the held-out spec independently derived from the pinned token file. Outputs
retain every arm/endpoint and the original missingness/multiplicity rules.
"""

import hashlib
import json
import math

from analysis_core import require_allocation
from analysis_original import equal


def need(value, text):
    if not value:
        raise ValueError(text)


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def interval(values, confidence):
    require_allocation()
    if len(values) != 3 or any(
        value is None or not math.isfinite(value) for value in values
    ):
        return None
    from scipy.stats import t

    mean = math.fsum(values) / 3
    variance = math.fsum((value - mean) ** 2 for value in values) / 2
    half = float(t.ppf((1 + confidence) / 2, 2)) * math.sqrt(variance / 3)
    return {
        "n": 3,
        "df": 2,
        "confidence": confidence,
        "history_values": values,
        "ce_difference": mean,
        "lower": mean - half,
        "upper": mean + half,
        "relative_ppl": math.exp(mean) if mean < 709 else None,
        "relative_ppl_lower": math.exp(mean - half) if mean - half < 709 else None,
        "relative_ppl_upper": math.exp(mean + half) if mean + half < 709 else None,
        "assumption": "Three independent approximately normal history means; nested sites are paired, not independent histories",
    }


def comparison(record):
    need(
        isinstance(record, dict)
        and all(
            type(record.get(k)) is bool
            for k in ("valid", "all_finite", "byte_equal", "value_equal")
        )
        and isinstance(record.get("tensors"), list)
        and bool(record["tensors"]),
        "Required E2 diagnostic receipt missing/unsupported",
    )
    need(record["valid"] is True, "E2 comparison metadata invalid")
    return {
        "valid": True,
        "all_finite": record["all_finite"],
        "value_equal": record["value_equal"],
    }


def fault_receipt(row, stratum, seed, enabled, number):
    spec = {
        "case_id": stratum["case_id"],
        "location": stratum["location"],
        "layer": stratum["layer"],
        "kind": stratum["kind"],
        "magnitude": stratum["magnitude"]
        if stratum["magnitude"] is not None
        else 0.001,
        "bit": stratum["bit"] if stratum["bit"] is not None else 31,
        "seed": seed,
        "onset": 1,
        "duration": 1,
        "period": 1,
        "enabled": enabled,
    }
    need(row.get("spec") == spec, "E2 locked fault specification differs")
    injections = row.get("injections")
    need(isinstance(injections, list), "E2 injection receipt absent")
    if not enabled or number != 1:
        need(not injections, "E2 inactive one-shot injection persists")
        return
    need(len(injections) == 1, "E2 one eligible injection required")
    event = injections[0]
    site = next(x["element"] for x in stratum["sites"] if x["seed"] == seed)
    need(
        event.get("case_id") == stratum["case_id"]
        and event.get("spec") == spec
        and event.get("element") == site
        and event.get("tensor") == stratum["identity"]
        and event.get("eligible_elements") == stratum["eligible_elements"]
        and event.get("update") == 1,
        "E2 actual locked injection site/spec differs",
    )


def evaluation(row, expected_spec, *, windows=False):
    require_allocation()
    need(
        row.get("evaluation_spec") == expected_spec
        and row.get("evaluation_spec_sha256") == canonical(expected_spec),
        "E2 held-out window/source/spec differs",
    )
    need(
        row.get("heldout_tokens") == 1048576
        and equal(row.get("state_preservation", {})),
        "E2 evaluator changed recorded full state/RNG or target count",
    )
    ce = row.get("cross_entropy")
    finite = (
        row.get("status") == "PASS" and type(ce) in (float, int) and math.isfinite(ce)
    )
    need(finite or row.get("status") == "NONFINITE", "E2 unknown evaluation outcome")
    if finite:
        if ce < 709:
            need(
                type(row.get("perplexity")) in (int, float)
                and math.isclose(row["perplexity"], math.exp(ce), rel_tol=1e-12),
                "Saved E2 PPL differs from CE",
            )
        else:
            need(row.get("perplexity") is None, "E2 overflow rule differs")
    if windows:
        raw = row.get("windows")
        need(
            isinstance(raw, list)
            and len(raw) == 2048
            and row.get("unscored_available_targets") == 0,
            "E2 packed window census differs",
        )
        total = 0.0
        for number, item in enumerate(raw):
            need(
                item.get("window_id") == number
                and item.get("target_begin") == number * 512 + 1
                and item.get("target_end_exclusive") == (number + 1) * 512 + 1
                and item.get("target_count") == 512,
                "E2 packed window identity/range/count differs",
            )
            nll = item.get("nll_sum")
            if not finite and (nll is None or not math.isfinite(nll)):
                need(
                    item.get("cross_entropy") is None
                    or not math.isfinite(item["cross_entropy"]),
                    "E2 nonfinite window contradiction",
                )
                continue
            need(
                type(nll) in (int, float)
                and math.isfinite(nll)
                and type(item.get("cross_entropy")) in (int, float)
                and math.isclose(
                    item["cross_entropy"], nll / 512, rel_tol=1e-12, abs_tol=1e-12
                ),
                "E2 unsupported or contradictory window NLL",
            )
            total += nll
        if finite:
            need(
                math.isclose(ce, total / 1048576, rel_tol=1e-12, abs_tol=1e-12),
                "E2 window NLL aggregate differs",
            )
    return ce if finite else None


def analyze(reader, protocol, initial_roles, expected_spec):
    require_allocation()
    need(
        protocol["evaluation_updates"] == [0, 1, 10, 50, 100]
        and protocol["individual_ci_confidence"] == 0.975
        and protocol["recovery_individual_ci_confidence"] == 0.99375,
        "Frozen E2 inference differs",
    )
    arms, controls, endpoints, issues = [], [], [], []
    indexes = {}
    for history in protocol["histories"]:
        prefix = f"E2/history-{history}/"
        initial = initial_roles[f"corrected210-history{history}"]
        baseline = reader.json(prefix + "initial-evaluation.json")
        evaluation(baseline, expected_spec, windows=True)
        baseline_short = {
            key: value for key, value in baseline.items() if key != "windows"
        }
        completion = reader.json(prefix + "completion.json")
        planned = [row for row in protocol["arms"] if row["history"] == history]
        need(
            completion["history"] == history
            and completion.get("no_replacements_or_extension") is True
            and len(completion["arms"]) == 5,
            "E2 history completion census differs",
        )
        needed_members = {prefix + p["id"] + "/summary.json" for p in planned}
        actual = {
            name
            for name in reader.members
            if name.startswith(prefix) and name.endswith("/summary.json")
        }
        need(actual == needed_members, "E2 missing or foreign arm")
        control_journal = reader.journal(prefix + "controls.jsonl")
        need(
            control_journal["partial_final_write"] is None
            and len(control_journal["rows"]) == 7
            and reader.json(prefix + "controls.json") == control_journal["rows"],
            "E2 seven complete controls differ",
        )
        kinds = [r["kind"] for r in control_journal["rows"]]
        need(
            kinds.count("clean-observer") == 1
            and kinds.count("trigger-off") == 4
            and kinds.count("guarded-replay") == 2,
            "E2 control roles differ",
        )
        control_ids = []
        for row in control_journal["rows"]:
            kind = row["kind"]
            need(
                row.get("preserve_rng") is True
                and row.get("seed") == 1729
                and row.get("branch_update") == 1
                and row.get("scope") == "checkpoint_boundaries"
                and row.get("policy")
                == ("disabled" if kind == "trigger-off" else "every"),
                "E2 control treatment differs",
            )
            if kind == "clean-observer":
                need(
                    row.get("spec") is None and not row.get("injections"),
                    "E2 clean observer fault active",
                )
                control_ids.append((kind, None, None))
            else:
                spec = row.get("spec", {})
                stratum = next(
                    (
                        x
                        for x in protocol["strata"]
                        if x["case_id"] == spec.get("case_id")
                    ),
                    None,
                )
                need(
                    stratum is not None and spec.get("seed") in protocol["site_seeds"],
                    "E2 foreign control identity",
                )
                fault_receipt(row, stratum, spec["seed"], kind == "guarded-replay", 1)
                control_ids.append((kind, stratum["stratum"], spec["seed"]))
            need(
                row.get("phase")
                == (
                    "BACKWARD_COMPLETE"
                    if row["status"] == "REJECTED_PRE_OPTIMIZER"
                    else "ACCEPTED"
                ),
                "E2 control phase differs",
            )
            if kind in ("clean-observer", "trigger-off"):
                need(
                    row["status"] == "ACCEPTED" and equal(row["state_vs_clean"]),
                    "E2 clean/off full-state parity failed",
                )
                if kind == "clean-observer":
                    need(
                        equal(row["gradient_vs_clean"]),
                        "E2 clean observer gradient parity failed",
                    )
                else:
                    need(
                        not row["injections"] and row["spec"]["enabled"] is False,
                        "E2 trigger-off injection active",
                    )
            elif row["status"] == "REJECTED_PRE_OPTIMIZER":
                need(
                    equal(row["protected_state"])
                    and row["guard_outcome"] == "DETECTED_REJECTED",
                    "E2 rejected control lacks protected state",
                )
            else:
                need(
                    row["status"] == "ACCEPTED"
                    and row["guard_outcome"] == "SUPPORTED_ACCEPTED_MISS",
                    "E2 guarded control error is not a miss",
                )
                comparison(row.get("accepted_transition_vs_clean"))
            controls.append({"history": history, **row})
        expected_controls = (
            {("clean-observer", None, None)}
            | {
                ("trigger-off", x["stratum"], seed)
                for x in protocol["strata"]
                for seed in protocol["site_seeds"]
            }
            | {("guarded-replay", "smaller", seed) for seed in protocol["site_seeds"]}
        )
        need(
            len(control_ids) == len(set(control_ids))
            and set(control_ids) == expected_controls,
            "E2 duplicate or missing control identities",
        )
        for plan in planned:
            arm_prefix = prefix + plan["id"] + "/"
            summary = reader.json(arm_prefix + "summary.json")
            need(
                summary in completion["arms"]
                and summary["arm"] == plan
                and summary["source_input_sha256"] == initial["source_sha256"],
                "E2 identity/input/summary differs",
            )
            need(
                summary["status"] in ("COMPLETE", "INCOMPLETE_RETAINED_NO_REPLACEMENT"),
                "E2 unknown terminal status",
            )
            final_member = arm_prefix + "final-state.pt"
            need(
                reader.members[final_member]["sha256"] == summary["final_state_sha256"],
                "E2 final-state hash differs",
            )
            updates, evals = (
                reader.journal(arm_prefix + "updates.jsonl"),
                reader.journal(arm_prefix + "evaluations.jsonl"),
            )
            need(
                updates["partial_final_write"] is None
                and evals["partial_final_write"] is None,
                "E2 incomplete journal bytes",
            )
            rows = updates["rows"]
            need(len(rows) <= 100, "E2 updates beyond registered horizon")
            accepted = 0
            for number, row in enumerate(rows, 1):
                need(
                    row["arm_id"] == plan["id"]
                    and row["history"] == history
                    and row["branch_update"] == number,
                    "E2 update identity/order differs",
                )
                need(
                    row.get("preserve_rng") is True
                    and row.get("policy") == "disabled"
                    and row.get("scope") == "checkpoint_boundaries"
                    and row.get("seed") == 1729,
                    "E2 production treatment differs",
                )
                if plan["fault"] is not None:
                    stratum = next(
                        x for x in protocol["strata"] if x["stratum"] == plan["stratum"]
                    )
                    fault_receipt(row, stratum, plan["site_seed"], True, number)
                    if row["status"] == "ACCEPTED":
                        need(
                            equal(row["matched_post_rng"]),
                            "E2 matched forward RNG failed",
                        )
                else:
                    need(
                        row.get("spec") is None and not row["injections"],
                        "E2 clean arm has fault",
                    )
                if row["status"] != "ACCEPTED":
                    need(
                        number == len(rows)
                        and row["status"] in ("ERROR", "REJECTED_PRE_OPTIMIZER")
                        and summary["status"] == "INCOMPLETE_RETAINED_NO_REPLACEMENT",
                        "E2 terminal update or trailing journal differs",
                    )
                    break
                if plan["fault"] is not None and number in (1, 10, 50, 100):
                    need(
                        set(row.get("state_vs_clean", {}))
                        == {
                            "model",
                            "adam_exp_avg",
                            "adam_exp_avg_sq",
                            "full_optimizer",
                        },
                        "E2 state/Adam diagnostics missing",
                    )
                    for receipt in list(row["state_vs_clean"].values()) + [
                        row.get("gradient_vs_clean"),
                        row.get("post_clip_gradient_vs_clean"),
                    ]:
                        comparison(receipt)
                accepted += 1
                need(
                    row["phase"] == "ACCEPTED"
                    and row["update"]["update"] == initial["updates"] + accepted
                    and row["update"]["accepted_tokens"]
                    == initial["cursor"] + accepted * 65536
                    and row["update"]["tokens_this_update"] == 65536,
                    "E2 committed counters differ",
                )
            need(
                accepted == summary["accepted_updates"]
                and summary["accepted_targets"] == accepted * 65536
                and (summary["status"] != "COMPLETE" or accepted == 100),
                "E2 equal accepted budget differs",
            )
            saved = reader.tensor(final_member)
            need(
                saved["updates"] == initial["updates"] + accepted
                and saved["cursor"] == initial["cursor"] + accepted * 65536
                and saved["identity"] == initial["identity"],
                "E2 retained final counters/identity differ",
            )
            del saved
            expected_endpoints = [
                n for n in protocol["evaluation_updates"] if n <= accepted
            ]
            need(
                [r["branch_update"] for r in evals["rows"]] == expected_endpoints,
                "E2 endpoint census differs",
            )
            need(
                evals["rows"][0]
                == {
                    "branch_update": 0,
                    "shared_initial": "../initial-evaluation.json",
                    **baseline_short,
                },
                "E2 initial evaluation is not actual shared baseline",
            )
            for row in evals["rows"]:
                ce = evaluation(row, expected_spec)
                if row["branch_update"]:
                    exact = reader.json(
                        arm_prefix + f"evaluation-{row['branch_update']:03}.json"
                    )
                    evaluation(exact, expected_spec, windows=True)
                    need(
                        exact == {k: v for k, v in row.items() if k != "branch_update"}
                        or {k: v for k, v in exact.items() if k != "windows"}
                        == {k: v for k, v in row.items() if k != "branch_update"},
                        "E2 endpoint JSON/journal differs",
                    )
                key = (history, plan["id"], row["branch_update"])
                indexes[key] = ce
                endpoints.append(
                    {
                        "history": history,
                        "arm_id": plan["id"],
                        "stratum": plan.get("stratum", "clean"),
                        "site_seed": plan.get("site_seed"),
                        "branch_update": row["branch_update"],
                        "cross_entropy": ce,
                        "perplexity": row.get("perplexity"),
                        "status": row["status"],
                    }
                )
            arms.append(
                {
                    "registration": plan,
                    "summary": summary,
                    "updates": rows,
                    "evaluations": evals["rows"],
                }
            )
    complete_registration = (
        len(arms) == 15
        and all(
            arm["summary"]["status"] == "COMPLETE" and len(arm["evaluations"]) == 5
            for arm in arms
        )
        and all(value is not None for value in indexes.values())
    )
    effects = []
    strata = {}
    for label in ("largest", "smaller"):
        points = []
        for endpoint in (1, 10, 50, 100):
            histories = []
            for history in protocol["histories"]:
                clean = next(
                    p
                    for p in protocol["arms"]
                    if p["history"] == history and p["fault"] is None
                )
                baseline = indexes.get((history, clean["id"], endpoint))
                sites = []
                for seed in protocol["site_seeds"]:
                    arm = next(
                        p
                        for p in protocol["arms"]
                        if p.get("history") == history
                        and p.get("stratum") == label
                        and p.get("site_seed") == seed
                    )
                    faulty = indexes.get((history, arm["id"], endpoint))
                    difference = (
                        faulty - baseline
                        if faulty is not None and baseline is not None
                        else None
                    )
                    sites.append(difference)
                    effects.append(
                        {
                            "history": history,
                            "stratum": label,
                            "site_seed": seed,
                            "branch_update": endpoint,
                            "fault_minus_clean_ce": difference,
                        }
                    )
                histories.append(
                    math.fsum(sites) / 2
                    if len(sites) == 2 and all(x is not None for x in sites)
                    else None
                )
            recovery = interval(histories, 0.99375) if complete_registration else None
            points.append(
                {
                    "branch_update": endpoint,
                    "history_mean_effects": histories,
                    "exploratory95": interval(histories, 0.95)
                    if complete_registration
                    else None,
                    "recovery_interval": recovery,
                    "primary975": interval(histories, 0.975)
                    if endpoint == 100 and complete_registration
                    else None,
                }
            )
        beyond = False
        recovered = None
        for index, point in enumerate(points):
            current = point["recovery_interval"]
            if current is None:
                continue
            lo, hi = current["relative_ppl_lower"], current["relative_ppl_upper"]
            if lo is None or hi is None:
                continue
            if hi < 0.99 or lo > 1.01:
                beyond = True
            elif beyond and all(
                p["recovery_interval"] is not None
                and p["recovery_interval"]["relative_ppl_lower"] is not None
                and p["recovery_interval"]["relative_ppl_upper"] is not None
                and p["recovery_interval"]["relative_ppl_lower"] >= 0.99
                and p["recovery_interval"]["relative_ppl_upper"] <= 1.01
                for p in points[index:]
            ):
                recovered = point["branch_update"]
                break
        strata[label] = {
            "points": points,
            "recovery_demonstrated_update": recovered,
            "recovery_status": "DEMONSTRATED_AT_SCHEDULED_POINT"
            if recovered is not None
            else "NOT_DEMONSTRATED_BY_100",
        }
    return {
        "registered_arms": 15,
        "complete_registration": complete_registration,
        "arms": arms,
        "controls": controls,
        "endpoint_rows": endpoints,
        "individual_effects": effects,
        "strata": strata,
        "validation_issues": issues,
        "primary_unit": "Three history means of two nested paired sites; shared clean per history",
        "scientific_acceptance": False,
    }
