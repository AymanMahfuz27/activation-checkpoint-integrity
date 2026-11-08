"""Replay the frozen E1 clipping cases with a read-only pre-clipping probe.

The GPU job driver has already admitted source, runtime and opaque inputs.
This module leaves the historical engine and clipping implementation intact.
It compares probe-off/probe-on executions and records outcomes, not revised
historical acceptance or a universal norm-overflow explanation.
"""

from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from e1_diagnostics import _allocated_only, _scalar, diagnose_gradients


def _manifest_function():
    """Reuse the qualified canonical typed-leaf manifest, pinned by the driver."""
    path = Path(__file__).resolve().parents[1] / "ba_20261107/opguard_baseline/run_arm.py"
    spec = importlib.util.spec_from_file_location("_bb_e1_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.manifest


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_intervention(row, fault, engine, completion):
    """Check the actual event census and exact registered one-bit mutation.

    Clean arms must have no events. Fault arms must realize the selected
    spec at its declared tensor, update and hash-selected element, with the
    requested bit as the only bit changed in the recorded scalar.
    """
    events = row.get("injections")
    if fault is None:
        if events != []:
            raise RuntimeError("The E1 clean arm contains an intervention")
        return {"expected_events": 0, "actual_events": 0, "valid": True}
    if not isinstance(events, list) or len(events) != 1:
        raise RuntimeError("The E1 fault arm did not realize exactly one event")
    event = events[0]
    base = f"gpt_neox.layers.{fault.layer}.mlp.dense_h_to_4h"
    if fault.location == "gradient":
        identity = base + ".weight.grad"
        eligible = dict(engine.model.named_parameters())[base + ".weight"].numel()
        role, event_key = None, None
    elif fault.location == "recompute":
        identity = base + ".output"
        eligible = (completion.DIAGNOSTIC["microbatch"]
                    * completion.DIAGNOSTIC["sequence"]
                    * engine.model.config.intermediate_size)
        role, event_key = "recompute", [1, "recompute"]
    else:
        raise ValueError("The E1 registry permits only gradient/recompute faults")
    draw = int.from_bytes(hashlib.sha256(
        f"m4m5-site-v1:{fault.seed}:{identity}".encode()
    ).digest()[:8], "big")
    checks = {
        "spec": event.get("spec") == asdict(fault),
        "case_id": event.get("case_id") == fault.case_id,
        "tensor": event.get("tensor") == identity,
        "update": event.get("update") == 1,
        "role": event.get("role") == role,
        "event_key": event.get("event_key") == event_key,
        "eligible_elements": event.get("eligible_elements") == eligible,
        "eligible_bits": event.get("eligible_bits") == 32 * eligible,
        "element": event.get("element") == draw % eligible,
        "exact_bit_xor": (
            type(event.get("old_bits_u32")) is int
            and type(event.get("new_bits_u32")) is int
            and event["old_bits_u32"] ^ event["new_bits_u32"] == 1 << fault.bit
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("E1 intervention realization differs: " + repr(checks))
    return {"expected_events": 1, "actual_events": 1, "valid": True,
            "checks": checks, "identity": identity, "element": draw % eligible}


def run_e1_probes(out, contract):
    """Run exactly twelve arms from two frozen states and compare paired records.

    Per state, execute the clean control and two selected faults, each with
    the probe off and on. The same initial state and historical fault spec
    are restored before each arm. Export typed state/gradient fingerprints,
    actual native transaction phase and the separately recomputed norm data.
    """
    _allocated_only()
    from ac_integrity.research import completion
    from ac_integrity.research.m4m5 import pilot_cases

    completion.source_allocation_guard()
    manifest = _manifest_function()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    selected = contract["method_plan"]["e1_diagnostic"]["selected_cases"]
    if len(selected) != 4 or len({(c["state"], c["case_id"]) for c in selected}) != 4:
        raise ValueError("The frozen four-case E1 census differs")
    roles = {
        "early-warm10": "early-history1234",
        "later-update210": "corrected210-history1234",
    }
    specs = {item.case_id: item for item in pilot_cases(12)}
    inputs = {item["role"]: item for item in contract["inputs"]}
    if not all(state in roles and case["case_id"] in specs
               for case in selected for state in [case["state"]]):
        raise ValueError("E1 case is outside the original state/spec registry")
    stream = completion.stream_from_inputs(contract["inputs"])
    pairs = []
    for state, role in roles.items():
        completion.cutoff()
        engine = completion.load_engine(inputs[role])
        initial = completion.snapshot(engine)
        initial_manifest = manifest(initial, completion)
        protected_initial = manifest(completion.protected(initial), completion)
        case_ids = [None] + [case["case_id"] for case in selected if case["state"] == state]
        if len(case_ids) != 3:
            raise ValueError("Each E1 state requires clean plus two fixed fault contexts")
        for case_id in case_ids:
            records = {}
            for enabled in (False, True):
                completion.cutoff()
                completion.restore(engine, initial)
                before = manifest(completion.snapshot(engine), completion)
                if before != initial_manifest:
                    raise RuntimeError("E1 arm did not restore its exact recorded prestate")
                label = case_id or "clean"
                folder = out / state / label / ("probe-on" if enabled else "probe-off")
                folder.mkdir(parents=True, exist_ok=False)
                observations = []

                def probe(current, mean_loss):
                    if observations:
                        raise RuntimeError("E1 pre-clipping callback ran more than once")
                    observation = diagnose_gradients(
                        (name, parameter.grad)
                        for name, parameter in current.model.named_parameters()
                    )
                    observation["mean_loss"] = _scalar(mean_loss)
                    observation["phase_at_observation"] = "BACKWARD_COMPLETE_BEFORE_NATIVE_CLIP"
                    observation["accepted_updates_before_native_clip"] = current.updates
                    observation["cursor_before_native_clip"] = current.cursor
                    observations.append(observation)

                fault = specs[case_id] if case_id is not None else None
                execution = folder / "execution"
                execution.mkdir()
                row, evidence = completion.execute(
                    engine, stream, execution, spec=fault,
                    policy="disabled", scope="checkpoint_boundaries", enforce=False,
                    preserve_rng=True, gradients=True, branch_update=1,
                    cheap_observer=probe if enabled else None,
                )
                # Persist the returned native row before any validity check can
                # reject missing gradients, callback counts or native outcomes.
                _write(folder / "execution-returned.json", {
                    "row": row, "native_phase": evidence.get("phase"),
                    "evidence_keys": sorted(evidence),
                    "probe_observations": observations,
                    "probe_callback_count": len(observations),
                    "scientific_acceptance": False,
                })
                realization = _validate_intervention(row, fault, engine, completion)
                gradients = evidence.get("gradients")
                if not isinstance(gradients, dict) or not gradients:
                    raise RuntimeError("E1 pre-clipping gradient evidence is absent")
                post = completion.snapshot(engine)
                record = {
                    "state": state,
                    "case_id": case_id,
                    "spec": asdict(fault) if fault is not None else None,
                    "probe_enabled": enabled,
                    "row": row,
                    "native_phase": evidence.get("phase"),
                    "initial_state": before,
                    "post_state": manifest(post, completion),
                    "preclip_gradients": manifest(gradients, completion),
                    "live_post_execution_gradients": manifest({
                        name: parameter.grad
                        for name, parameter in engine.model.named_parameters()
                    }, completion),
                    "intervention_realization": realization,
                    "protected_state_equal_initial": (
                        manifest(completion.protected(post), completion) == protected_initial
                    ),
                    "probe": observations[0] if observations else None,
                    "probe_callback_count": len(observations),
                    "native_clip_outcome": {
                        "status": row["status"],
                        "error": row.get("error"),
                        "phase": evidence.get("phase"),
                        "returned_norm": (
                            row.get("update", {}).get("gradient_norm_before_clip")
                            if row["status"] == "ACCEPTED" else None
                        ),
                    },
                    "historical_validation_changed": False,
                    "scientific_acceptance": False,
                }
                if enabled and len(observations) != 1:
                    raise RuntimeError("E1 probe did not run at its declared transaction phase")
                if case_id is None and row["status"] != "ACCEPTED":
                    raise RuntimeError("The E1 clean native control did not accept")
                if case_id is not None and not (
                    row["status"] == "ERROR"
                    and evidence.get("phase") == "BACKWARD_COMPLETE"
                    and "non-finite" in row.get("error", "").lower()
                ):
                    _write(folder / "arm.json", record)
                    raise RuntimeError("Registered historical clipping outcome did not reproduce")
                _write(folder / "arm.json", record)
                records[enabled] = record
                del gradients, evidence, post
            left, right = records[False], records[True]
            comparisons = {
                "initial_state": left["initial_state"] == right["initial_state"],
                "post_state_including_rng": left["post_state"] == right["post_state"],
                "preclip_gradients": left["preclip_gradients"] == right["preclip_gradients"],
                "live_post_execution_gradients": (
                    left["live_post_execution_gradients"]
                    == right["live_post_execution_gradients"]
                ),
                "events": left["row"]["injections"] == right["row"]["injections"],
                "status": left["row"]["status"] == right["row"]["status"],
                "phase": left["native_phase"] == right["native_phase"],
            }
            pair = {
                "state": state, "case_id": case_id,
                "comparisons": comparisons,
                "noninterference_valid": all(comparisons.values()),
                "norm_diagnostic": right["probe"],
                "native_clip_outcome": right["native_clip_outcome"],
                "protected_state_equal_initial": right["protected_state_equal_initial"],
            }
            pairs.append(pair)
            if not pair["noninterference_valid"]:
                _write(out / "failed-pair.json", pair)
                raise RuntimeError("The E1 diagnostic changed its matched native execution")
        del initial, engine
    result = {
        "experiment_id": "BB-E1-NORM-20261108-v1",
        "execution_complete": True,
        "arms_executed": len(pairs) * 2,
        "paired_contexts": pairs,
        "original_638_call_census_and_26_flags_unchanged": True,
        "native_clipping_implementation_changed": False,
        "scientific_acceptance": False,
        "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW",
    }
    _write(out / "results.json", result)
    return result
