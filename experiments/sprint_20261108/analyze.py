"""Analyze frozen matched training histories inside an admitted allocation.

The input is the complete training summary and preregistered contrasts. Keep
censored and failed histories visible. A capped-time comparison is available
only when every required history has a valid common-horizon endpoint. Report
all pairs, a paired t interval and an exact two-sided sign test; none makes an
unobserved crossing time identifiable.
"""

import json
import math
from pathlib import Path


def validate_endpoint(row, config):
    endpoint = row.get("endpoint")
    if endpoint is None:
        return False
    if not isinstance(endpoint, dict) or type(endpoint.get("valid")) is not bool:
        raise ValueError("Endpoint validity must be explicit")
    if endpoint["valid"] is False:
        return False
    if row.get("failure") is not None or endpoint.get("failure") is not None:
        raise ValueError("A failed history cannot supply a valid endpoint")
    horizon = endpoint.get("horizon_seconds")
    if (type(horizon) not in (int, float) or not math.isfinite(horizon) or horizon <= 0
            or horizon != config["administrative_wall_horizon_seconds"]
            or endpoint.get("clock") != config["endpoint_clock"]):
        raise ValueError("Endpoint horizon/clock differs from registration")
    event = endpoint.get("event_observed")
    if type(event) is not bool:
        raise ValueError("Endpoint event status must be a boolean")
    stopped = endpoint.get("observed_stop_elapsed_seconds")
    if type(stopped) not in (int, float) or not math.isfinite(stopped) or stopped < 0:
        raise ValueError("Actual endpoint follow-up is not finite")
    restricted = endpoint.get("restricted_seconds")
    if type(restricted) not in (int, float) or not math.isfinite(restricted):
        raise ValueError("Restricted endpoint is not finite")
    if event:
        confirmed = endpoint.get("confirmed_target_elapsed_seconds")
        if (type(confirmed) not in (int, float) or not math.isfinite(confirmed)
                or not 0 <= confirmed <= horizon or stopped < confirmed
                or restricted != confirmed or endpoint.get("stop_reason") != "TARGET_CONFIRMED"
                or endpoint.get("common_horizon_right_censored") is not False):
            raise ValueError("Target event is inconsistent with actual registered follow-up")
    elif (stopped < horizon or endpoint.get("stop_reason") != "COMMON_WALL_HORIZON"
            or endpoint.get("common_horizon_right_censored") is not True
            or endpoint.get("administrative_censor_elapsed_seconds") != horizon
            or endpoint.get("confirmed_target_elapsed_seconds") is not None
            or restricted != horizon):
        raise ValueError("Censored endpoint invents unobserved common-horizon follow-up")
    return True


def analyze(summary, config, output):
    # Numerical dependencies are loaded only by the admitted job entrypoint.
    from scipy import stats

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if summary.get("experiment_id") != config["experiment_id"]:
        raise ValueError("Summary belongs to another registered experiment")
    expected_seeds = config["seeds"]
    if len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("Duplicate history seeds")
    rows = {}
    arm_definitions = {arm["name"]: arm for arm in config["arms"]}
    for row in summary["results"]:
        key = (row["seed"], row["arm"]["name"])
        if row["arm"] != arm_definitions.get(key[1]):
            raise ValueError("Endpoint policy/fault regime differs from registration")
        if key in rows:
            raise ValueError("Duplicate history/arm endpoint")
        rows[key] = row
        validate_endpoint(row, config)
    arms = [arm["name"] for arm in config["arms"]]
    expected = {(seed, arm) for seed in expected_seeds for arm in arms}
    if set(rows) != expected:
        raise ValueError("Incomplete or extraneous history/arm census")
    contrasts = []
    for comparison in config.get("comparisons", []):
        candidate, reference = comparison["candidate"], comparison["reference"]
        pairs = []
        for seed in expected_seeds:
            left, right = rows[(seed, candidate)], rows[(seed, reference)]
            if (left["initial_full_state_sha256"] != right["initial_full_state_sha256"]
                    or left["fault_schedule_sha256"] != right["fault_schedule_sha256"]):
                raise ValueError("Pair lost matched starting state or fault schedule")
            pairs.append({"seed": seed, "candidate": left.get("endpoint"),
                          "reference": right.get("endpoint"),
                          "candidate_failure": left.get("failure"),
                          "reference_failure": right.get("failure")})
        record = {"comparison": comparison, "pairs": pairs,
                  "independent_units": len(pairs), "status": "PENDING_ENDPOINT_VALIDATION"}
        valid = all(pair["candidate"] and pair["reference"]
                    and pair["candidate"].get("valid") is True
                    and pair["reference"].get("valid") is True for pair in pairs)
        if not valid:
            record.update(status="INCOMPLETE_NO_COMPLETE_CASE_INFERENCE", estimate=None)
            contrasts.append(record)
            continue
        horizons = {endpoint["horizon_seconds"] for pair in pairs
                    for endpoint in (pair["candidate"], pair["reference"])}
        clocks = {endpoint["clock"] for pair in pairs
                  for endpoint in (pair["candidate"], pair["reference"])}
        if len(horizons) != 1 or len(clocks) != 1:
            raise ValueError("Common preregistered endpoint horizon/clock differs")
        differences = [pair["candidate"]["restricted_seconds"]
                       - pair["reference"]["restricted_seconds"] for pair in pairs]
        if not all(math.isfinite(value) for value in differences):
            raise ValueError("Nonfinite restricted-time contrast")
        count = len(differences)
        mean = sum(differences) / count
        if count >= 2:
            variance = sum((value - mean) ** 2 for value in differences) / (count - 1)
            standard_error = math.sqrt(variance / count)
            confidence = float(comparison.get("confidence", 0.95))
            critical = float(stats.t.ppf((1 + confidence) / 2, count - 1))
            interval = [mean - critical * standard_error, mean + critical * standard_error]
        else:
            confidence, interval = None, None
        negatives = sum(value < 0 for value in differences)
        positives = sum(value > 0 for value in differences)
        nonzero = negatives + positives
        sign_p = float(stats.binomtest(min(negatives, positives), nonzero, 0.5).pvalue) if nonzero else 1.0
        candidate_hits = sum(pair["candidate"]["event_observed"] for pair in pairs)
        reference_hits = sum(pair["reference"]["event_observed"] for pair in pairs)
        record.update(status="COMPLETE_SCOPED_DESCRIPTIVE_AND_REGISTERED_INFERENCE",
            horizon_seconds=next(iter(horizons)), clock=next(iter(clocks)),
            differences_seconds=differences, mean_candidate_minus_reference_seconds=mean,
            paired_student_t_interval=interval, confidence=confidence,
            exact_two_sided_sign_p=sign_p, nonzero_pairs=nonzero,
            candidate_target_successes=candidate_hits, reference_target_successes=reference_hits,
            assumptions="Fixed task/runtime/recipe; independent training histories. Paired-t interval assumes approximately normal history differences. Sign test tests median/sign probability, not a randomized-treatment mean effect. Zeros excluded only from sign test; all histories retained elsewhere.",
            scope="Capped confirmation time, including administratively censored histories; not an uncensored mean time-to-target or a complete-case speedup.")
        contrasts.append(record)
    result = {"experiment_id": config["experiment_id"], "contrasts": contrasts,
              "scientific_acceptance": False, "no_population_fault_rate_inference": True}
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
