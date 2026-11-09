"""Real optimizer-driven byte LM time-to-target experiment; allocated use only.

Inputs are a frozen JSON config and hash-pinned public corpus. Outputs are
append-only attempt/evaluation journals and an explicit censored or reached
endpoint. Faults are transient software interventions, not hardware-rate
estimates. A target chosen using pilot calibration never inspects final-test
loss. No completion status implies a scientific advantage.
"""

from contextlib import nullcontext
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import struct
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/bb_20261108"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# This existing gate refuses local numerical imports. Root must first perform
# actual physical allocation/source/environment/storage admission.
import selector as core  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes  # noqa: E402
if __package__:
    from .model import LanguageModel  # noqa: E402
else:
    from model import LanguageModel  # noqa: E402


POLICIES = ("disabled", "saved_slot", "stock_metadata", "frontier_consumer", "selected_forward",
            "simple_input", "outputs_only", "full_forward_io", "bb_selected_consumer")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def synchronize():
    torch.cuda.synchronize()


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    return copy.deepcopy(value)


def equal_tree(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
            return False
        a = left.detach().clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8)
        b = right.detach().clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8)
        return bool(torch.equal(a, b))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def snapshot(model, optimizer, accepted_updates):
    return {"model": clone_tree(model.state_dict()), "optimizer": clone_tree(optimizer.state_dict()),
            "cpu_rng": torch.get_rng_state().clone(),
            "cuda_rng": [value.clone() for value in torch.cuda.get_rng_state_all()],
            "python_rng": random.getstate(), "data_cursor": accepted_updates,
            "scheduler": {"kind": "constant", "accepted_updates": accepted_updates}}


def restore(model, optimizer, state):
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(clone_tree(state["optimizer"]))
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    optimizer.zero_grad(set_to_none=True)
    return state["data_cursor"]


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        raw = value.detach().clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8).cpu()
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def state_digest(state):
    """Hash typed complete supplied state; tensor bytes retain signed zeros."""
    digest = hashlib.sha256()

    def visit(value):
        digest.update(type(value).__name__.encode() + b"\0")
        if isinstance(value, torch.Tensor):
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            raw = value.detach().clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8).cpu()
            digest.update(raw.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif type(value) is float:
            digest.update(struct.pack("!d", value))
        else:
            digest.update(repr(value).encode())

    visit(state)
    return digest.hexdigest()


def signature(value):
    with torch.no_grad(), _disable_current_modes():
        digest = core.SignatureStore().capture(value)
        return [int(word) for word in digest.detach().cpu().tolist()]


class FaultController:
    """Place one deterministic, auditable transient fault per selected attempt.

    The replay site runs before GELU's input observer. The native dispatch audit
    separately hashes actual GELU-backward input after saved-tensor unpack and
    before native execution. Consumer-entry faults mutate there instead. Both
    sites use isolated generators, never the training RNG stream.
    """

    def __init__(self, model_layers):
        self.layers = model_layers
        self.active = None
        self.events = []
        self.backward_gelu = 0

    def begin(self, spec):
        self.active = dict(spec) if spec is not None else None
        self.events = []
        self.backward_gelu = 0
        self.audit_hash_seconds = 0.0
        self.intervention_seconds = 0.0
        self.injected_snapshot = None

    def _hash(self, value):
        started = time.perf_counter()
        result = signature(value)
        self.audit_hash_seconds += time.perf_counter() - started
        return result

    def _mutate(self, value, location):
        if self.events:
            raise RuntimeError("Duplicate fault injection")
        spec = self.active
        before = self._hash(value)
        mutation_started = time.perf_counter()
        with torch.no_grad(), _disable_current_modes():
            if spec["kind"] == "sign_flip":
                changed = value.neg()
            elif spec["kind"] == "scale":
                changed = value * spec["severity"]
            elif spec["kind"] == "additive_burst":
                generator = torch.Generator(device=value.device).manual_seed(spec["seed"])
                noise = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
                rms = value.detach().square().mean().sqrt().clamp_min(1e-6)
                changed = value + noise * rms * spec["severity"]
            elif spec["kind"] == "bit_flip":
                changed = value.clone()
                bits = changed.reshape(-1).view(torch.int32)
                index = spec["seed"] % bits.numel()
                bits[index].bitwise_xor_(1 << int(spec["severity"]))
            else:
                raise ValueError("Unsupported declared fault kind")
            # The sham performs the same generator and candidate calculation.
            # Its copy also runs, preserving content while matching mutation work.
            value.copy_(value if spec["sham"] else changed)
        self.intervention_seconds += time.perf_counter() - mutation_started
        after = self._hash(value)
        if spec.get("exact_audit", False):
            with torch.no_grad(), _disable_current_modes():
                self.injected_snapshot = value.detach().clone(memory_format=torch.contiguous_format)
        if (before == after) != spec["sham"]:
            raise RuntimeError("Fault/sham content-change control failed")
        self.events.append({"location": location, "region": str(spec["layer"]),
                            "window": spec["window"], "kind": spec["kind"],
                            "severity": spec["severity"], "seed": spec["seed"],
                            "sham": spec["sham"], "before_signature": before,
                            "after_signature": after, "actual_backward_signature": None,
                            "actual_backward_equals_injected": None,
                            "actual_backward_exact_bytes_equal": None,
                            "equality_evidence": "complete BLAKE3-256 signatures; exact byte comparison additionally enabled by smoke config",
                            "observation_order": "injection; input/consumer observation; native execution"})

    def replay_site(self, value, layer):
        context = core.REGION.get()
        if (self.active is not None and self.active["window"] == "replay_forward"
                and context is not None and context == (str(layer), "recompute")
                and layer == self.active["layer"]):
            self._mutate(value, "after linear producer; before replay GELU input recording")

    def native_entry(self, func, args, kwargs):
        if str(func) != "aten.gelu_backward.default":
            return
        ordinal = self.backward_gelu
        self.backward_gelu += 1
        if self.active is None or ordinal != self.layers - 1 - self.active["layer"]:
            return
        bound = core._bind(func, args, kwargs)
        if self.active["window"] == "consumer_entry":
            self._mutate(bound["self"], "after saved-tensor unpack; before native GELU backward")
        if len(self.events) != 1:
            raise RuntimeError("Native backward reached target without exactly one injection")
        actual = self._hash(bound["self"])
        event = self.events[0]
        event["actual_backward_signature"] = actual
        event["actual_backward_equals_injected"] = actual == event["after_signature"]
        if self.injected_snapshot is not None:
            with torch.no_grad(), _disable_current_modes():
                first = self.injected_snapshot.reshape(-1).view(torch.uint8)
                second = bound["self"].detach().clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8)
                event["actual_backward_exact_bytes_equal"] = bool(torch.equal(first, second))
            if event["actual_backward_exact_bytes_equal"] is not True:
                raise RuntimeError("Native backward exact bytes differ from injected snapshot")
        if not event["actual_backward_equals_injected"]:
            raise RuntimeError("Native backward did not consume the injected activation bytes")

    def finish(self):
        if self.active is not None:
            if len(self.events) != 1 or self.events[0]["actual_backward_equals_injected"] is not True:
                raise RuntimeError("Incomplete fault delivery/consumer evidence")
        elif self.events:
            raise RuntimeError("Clean attempt injected")
        events = copy.deepcopy(self.events)
        self.active = None
        self.injected_snapshot = None
        return events


class FaultAuditMode(TorchDispatchMode):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.controller.native_entry(func, args, kwargs)
        return func(*args, **kwargs)


def observer(policy, plan=None, checker_config=None):
    if policy == "disabled":
        return None
    if policy == "bb_selected_consumer":
        return core.CheckMode(plan, "selected_consumer")
    selection = ROOT / "experiments/sprint_20261108/selection"
    if str(selection) not in sys.path:
        sys.path.insert(0, str(selection))
    if policy in ("saved_slot", "stock_metadata"):
        saved = importlib.import_module("saved_slots")
        checker_config = checker_config or {}
        return saved.SavedSlotCheck(policy="per_unpack" if policy == "saved_slot" else "stock_metadata_deferred",
                                    expected_checkpoint_sha256=checker_config["expected_checkpoint_sha256"])
    frontier = importlib.import_module("frontier")
    mapping = {"frontier_consumer": "consumer", "selected_forward": "selected_forward",
               "simple_input": "simple_input", "outputs_only": "outputs_only",
               "full_forward_io": "full_forward_io"}
    return frontier.DemandFrontierMode(policy=mapping[policy])


def make_engine(config, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    controller = FaultController(config["model"]["layers"])
    model = LanguageModel(config["model"], controller).cuda().float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], foreach=False, fused=False)
    return model, optimizer, controller


def load_data(config):
    data = config["data"]
    path = Path(data["path"])
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != data["sha256"] or data["format"] != "utf8_bytes":
        raise RuntimeError("Corpus byte identity or representation differs")
    fractions = data["split_fractions"]
    if fractions != [0.9, 0.05, 0.05]:
        raise RuntimeError("Unregistered train/calibration/test split")
    train_end = int(len(raw) * fractions[0])
    calibration_end = train_end + int(len(raw) * fractions[1])
    splits = {"train": raw[:train_end], "calibration": raw[train_end:calibration_end],
              "test": raw[calibration_end:]}
    selected = data["evaluation_split"]
    if selected != ("calibration" if config["phase"] == "pilot" else "test"):
        raise RuntimeError("Pilot and confirmatory evaluation splits are not separated")
    tokens = {name: torch.tensor(list(value), dtype=torch.long) for name, value in splits.items()
              if name in ("train", selected)}
    minimum = config["model"]["sequence_length"] + 1
    if any(value.numel() < minimum for value in tokens.values()):
        raise RuntimeError("Corpus split too short")
    return tokens["train"], tokens[selected], {
        "sha256": actual, "bytes": len(raw), "vocabulary": "256 byte values",
        "split_boundaries": [0, train_end, calibration_end, len(raw)],
        "evaluation_split": selected, "test_evaluated": selected == "test"}


def batch_for(tokens, config, update, *, evaluation=False, seed=0):
    batch = config["batch_size"]
    length = config["model"]["sequence_length"]
    available = tokens.numel() - length - 1
    # Training order differs across independent histories but matches all arms
    # of one history. Accepted update alone defines data progress, so retries
    # use exactly the same batch without mutating any iterator state.
    base = 0 if evaluation else (seed * 104729) % max(available, 1)
    starts = [(base + (update * batch + index) * length) % (available + 1)
              for index in range(batch)]
    x = torch.stack([tokens[start:start + length] for start in starts]).cuda()
    y = torch.stack([tokens[start + 1:start + length + 1] for start in starts]).cuda()
    return x, y


def evaluate(model, tokens, config):
    prior_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    per_batch = config["batch_size"] * config["model"]["sequence_length"]
    if config["eval_tokens"] % per_batch:
        raise ValueError("Evaluation token budget must be an exact number of complete batches")
    with torch.inference_mode():
        for index in range(config["eval_tokens"] // per_batch):
            x, y = batch_for(tokens, config, index, evaluation=True)
            logits = model(x)
            # Sum cross entropy over actual targets before division. This is
            # byte perplexity, never a subword/tokenizer-comparable quantity.
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).double(),
                                   y.reshape(-1), reduction="sum")
            total_loss += float(loss)
            total_tokens += y.numel()
    model.train(prior_training)
    ce = total_loss / total_tokens
    finite = math.isfinite(ce)
    return {"cross_entropy_nats_per_byte": ce if finite else None,
            "byte_perplexity": math.exp(ce) if finite and ce < 700 else None,
            "finite": finite, "evaluated_bytes": total_tokens}


def backward_attempt(model, optimizer, controller, x, y, config, policy, fault=None,
                     plan=None, capture_gradients=False):
    optimizer.zero_grad(set_to_none=True)
    controller.begin(fault)
    mode = observer(policy, plan, config.get("checker"))
    audit = FaultAuditMode(controller) if fault is not None else nullcontext()
    with (mode.track_model(model) if mode is not None else nullcontext()), \
            (mode if mode is not None else nullcontext()), audit:
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
    events = controller.finish()
    decision = mode.finish() if mode is not None else {"status": "DISABLED", "failed": False}
    rejected = bool(decision.get("failed", False) or decision.get("status") == "VALUE_MISMATCH")
    gradient_values = ({name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                        if parameter.grad is not None} if capture_gradients else None)
    finite_flags = [torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                    if parameter.grad is not None]
    finite_gradients = bool(torch.stack(finite_flags).all()) if finite_flags else False
    record = {"loss": float(loss) if bool(torch.isfinite(loss)) else None,
              "loss_finite": bool(torch.isfinite(loss)), "finite_gradients": finite_gradients,
              "decision": decision, "fault_events": events, "rejected": rejected,
              "fault_audit_hash_seconds": controller.audit_hash_seconds,
              "fault_intervention_seconds": controller.intervention_seconds,
              "clipping_entered": False, "optimizer_entered": False}
    if not rejected:
        record["clipping_entered"] = True
        # Nonfinite safety is common to every arm and is not ValueGuard credit.
        # A failed common control terminates/censors rather than accepting NaNs.
        if not finite_gradients or not record["loss_finite"]:
            record["common_nonfinite_failure"] = True
            return record, gradient_values
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_norm"], error_if_nonfinite=True)
        record["preclip_norm"] = float(norm)
        record["optimizer_entered"] = True
        optimizer.step()
    return record, gradient_values


def profile_plan(model, optimizer, controller, x, y):
    state = snapshot(model, optimizer, 0)
    mode = core.ProfileMode()
    controller.begin(None)
    with mode.track_model(model), mode:
        logits = model(x)
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).backward()
    plan = mode.finish({"workload": "sprint-byte-transformer", "original_replay": "full"})
    controller.finish()
    restore(model, optimizer, state)
    return plan


def fault_schedule(config, seed):
    rng = random.Random(seed + config["fault"]["seed_offset"])
    return [(rng.random(), rng.randrange(2 ** 31)) for _ in range(config["max_updates"])]


def validate_config(config):
    if config["schema"] != 1 or config["phase"] not in ("pilot", "confirmatory"):
        raise ValueError("Unsupported experiment schema/phase")
    if config["model"]["vocab_size"] != 256 or config["model"]["width"] % config["model"]["heads"]:
        raise ValueError("Invalid declared byte model")
    if config["phase"] == "confirmatory":
        if (config["target_preregistered"] is not True or not config.get("registration_sha256")
                or not isinstance(config["target_perplexity"], (float, int))
                or config["target_perplexity"] <= 1 or 11 in config["seeds"]):
            raise ValueError("Confirmatory target/registration/independent histories not frozen")
    if config["fault"]["window"] not in ("replay_forward", "consumer_entry"):
        raise ValueError("Unknown fault window")
    if not 0 <= config["fault"]["layer"] < config["model"]["layers"]:
        raise ValueError("Fault layer outside model")
    if config["fault"]["kind"] == "bit_flip" and not 0 <= config["fault"]["severity"] <= 30:
        raise ValueError("Bit flip must name an FP32 bit0..30")
    if len(set(arm["name"] for arm in config["arms"])) != len(config["arms"]):
        raise ValueError("Duplicate arm namespace")
    for arm in config["arms"]:
        if arm["policy"] not in POLICIES or not 0 <= arm["fault_probability"] <= 1:
            raise ValueError("Invalid policy/fault probability")


def run_arm(config, arm, seed, train, heldout, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    synchronize()
    started = time.perf_counter()
    model, optimizer, controller = make_engine(config, seed)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial_sha = model_digest(model)
    initial_full_sha = state_digest(snapshot(model, optimizer, 0))
    plan = None
    synchronize()
    common_initialization_seconds = time.perf_counter() - started
    if arm["policy"] == "bb_selected_consumer":
        x, y = batch_for(train, config, 0, seed=seed)
        profile_started = time.perf_counter()
        plan = profile_plan(model, optimizer, controller, x, y)
        synchronize()
        write(directory / "profile.json", {"elapsed_seconds": time.perf_counter() - profile_started,
                                            "plan": plan})
    schedule = fault_schedule(config, seed)
    schedule_sha = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    accepted = attempts = rejections = 0
    reached = None
    consecutive_hits = 0
    failure = None
    evaluations = []
    backend_cold = float(os.environ.get("SPRINT_CHECKER_STARTUP_SECONDS", "0.0"))
    if not math.isfinite(backend_cold) or backend_cold < 0:
        raise ValueError("Invalid measured cold backend startup")
    checker_cold = backend_cold if arm["policy"] != "disabled" else 0.0
    instrumented_cold = backend_cold if arm["policy"] != "disabled" or arm["fault_probability"] > 0 else 0.0

    def eval_point():
        nonlocal reached, consecutive_hits
        metric = evaluate(model, heldout, config)
        synchronize()
        metric.update(accepted_updates=accepted, attempted_updates=attempts, rejected_attempts=rejections,
                      accepted_training_bytes=accepted * config["batch_size"] * config["model"]["sequence_length"],
                      elapsed_seconds=time.perf_counter() - started)
        metric["checker_cold_elapsed_seconds"] = metric["elapsed_seconds"] + checker_cold
        metric["instrumented_cold_elapsed_seconds"] = metric["elapsed_seconds"] + instrumented_cold
        evaluations.append(metric)
        append(directory / "evaluations.jsonl", metric)
        target = config["target_perplexity"]
        if target is not None and metric["byte_perplexity"] is not None and metric["byte_perplexity"] <= target:
            consecutive_hits += 1
        else:
            consecutive_hits = 0
        if reached is None and consecutive_hits >= config["target_confirmation_evaluations"]:
            reached = dict(metric)
        return metric

    try:
        initial_eval = eval_point()
        if (config["target_perplexity"] is not None
                and initial_eval["byte_perplexity"] is not None
                and initial_eval["byte_perplexity"] <= config["target_perplexity"]):
            raise RuntimeError("Target already reached before any training; registration is uninformative")
        if not initial_eval["finite"]:
            raise RuntimeError("Nonfinite initial held-out metric")
        while accepted < config["max_updates"]:
            if time.perf_counter() - started > config["max_seconds_per_arm"]:
                failure = "ARM_TIME_LIMIT"
                break
            x, y = batch_for(train, config, accepted, seed=seed)
            backup = None
            if arm["policy"] != "disabled":
                optimizer.zero_grad(set_to_none=True)
                backup = snapshot(model, optimizer, accepted)
            probability, fault_seed = schedule[accepted]
            spec = None
            if probability < arm["fault_probability"]:
                spec = dict(config["fault"], seed=fault_seed, sham=arm.get("sham", config["fault"]["sham"]))
            attempt_started = time.perf_counter()
            record, _ = backward_attempt(model, optimizer, controller, x, y, config, arm["policy"], spec, plan)
            attempts += 1
            record.update(accepted_update_index=accepted, attempt_index=attempts, retry=False)
            if record["rejected"]:
                rejections += 1
                if backup is None or record["clipping_entered"] or record["optimizer_entered"]:
                    raise RuntimeError("Rejection crossed protected mutation boundary")
                restore_started = time.perf_counter()
                restored_cursor = restore(model, optimizer, backup)
                if restored_cursor != accepted:
                    raise RuntimeError("Data/scheduler cursor restoration differs")
                synchronize()
                record["restore_seconds"] = time.perf_counter() - restore_started
                record["attempt_elapsed_seconds"] = time.perf_counter() - attempt_started
                append(directory / "attempts.jsonl", record)
                retry_started = time.perf_counter()
                retry, _ = backward_attempt(model, optimizer, controller, x, y, config, arm["policy"], None, plan)
                attempts += 1
                retry.update(accepted_update_index=accepted, attempt_index=attempts, retry=True)
                synchronize()
                retry["attempt_elapsed_seconds"] = time.perf_counter() - retry_started
                append(directory / "attempts.jsonl", retry)
                if retry["rejected"] or retry.get("common_nonfinite_failure"):
                    raise RuntimeError("Fault-free same-batch retry failed")
            else:
                synchronize()
                record["attempt_elapsed_seconds"] = time.perf_counter() - attempt_started
                append(directory / "attempts.jsonl", record)
                if record.get("common_nonfinite_failure"):
                    failure = "COMMON_NONFINITE_FAILURE"
                    break
            del backup
            accepted += 1
            if accepted % config["eval_every"] == 0:
                metric = eval_point()
                if not metric["finite"]:
                    failure = "NONFINITE_HELDOUT"
                    break
                # Continue to fixed horizon after crossing: this preserves a
                # complete loss-vs-time/progress comparison and stopping rule.
        if not evaluations or evaluations[-1]["accepted_updates"] != accepted:
            eval_point()
    except Exception as error:
        failure = type(error).__name__ + ": " + str(error)
        write(directory / "failure.json", {"error": failure, "traceback": traceback.format_exc()})
    synchronize()
    elapsed = time.perf_counter() - started
    result = {"seed": seed, "arm": arm, "parameter_count": parameter_count,
              "initial_model_sha256": initial_sha, "final_model_sha256": model_digest(model),
              "initial_full_state_sha256": initial_full_sha,
              "final_full_state_sha256": state_digest(snapshot(model, optimizer, accepted)),
              "fault_schedule_sha256": schedule_sha, "accepted_updates": accepted,
              "attempted_updates": attempts, "rejected_attempts": rejections,
              "elapsed_seconds": elapsed, "target_perplexity": config["target_perplexity"],
              "common_initialization_seconds": common_initialization_seconds,
              "checker_cold_startup_seconds": checker_cold,
              "experimental_audit_cold_startup_seconds": instrumented_cold if arm["policy"] == "disabled" else 0.0,
              "cold_startup_note": "backend measured once before fixtures; checker-cold charges it to checked arms, instrumented-cold also charges fault/sham audit users; both preserve actual warm raw clocks",
              "target_reached": reached is not None, "first_confirmed_crossing": reached,
              "right_censored": reached is None, "censoring_reason": failure or "FIXED_HORIZON",
              "last_evaluation": evaluations[-1] if evaluations else None,
              "failure": failure, "scientific_acceptance": False,
              "timing_boundary": "before per-arm model/optimizer initialization through method profile, all snapshots, checks, failed attempts, restores, retries, journals and heldout evaluation; shared source/runtime/data setup separately reported",
              "time_to_target_is_discrete_evaluation_time": True,
              "fault_schedule_unit": "accepted-update first attempt; rejected transient retries are fault-free",
              "unsupported_is_execution_failure_not_detection": True}
    write(directory / "result.json", result)
    del optimizer, model, controller
    torch.cuda.empty_cache()
    return result


def run(config, output):
    study_started = time.perf_counter()
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA model execution is required for this experiment")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("Training output must be an empty exclusive namespace")
    write(output / "config.json", config)
    train, heldout, data_record = load_data(config)
    runtime = {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
               "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(), "job": os.environ["SLURM_JOB_ID"],
               "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
               "data": data_record, "tf32": False, "deterministic_algorithms": True}
    write(output / "runtime.json", runtime)
    shared_setup_seconds = time.perf_counter() - study_started
    results = []
    for index, seed in enumerate(config["seeds"]):
        arms = config["arms"]
        order = arms[index % len(arms):] + arms[:index % len(arms)]
        if index % 2:
            order = list(reversed(order))
        write(output / ("seed-%d-order.json" % seed), [arm["name"] for arm in order])
        history = []
        for arm in order:
            result = run_arm(config, arm, seed, train, heldout, output / ("seed-%d" % seed) / arm["name"])
            history.append(result)
            results.append(result)
            if result["failure"] and result["failure"] != "ARM_TIME_LIMIT":
                write(output / "incomplete.json", {"reason": "An arm failed; no selective surviving-arm inference", "results": results})
                raise RuntimeError("Training arm failed: " + result["failure"])
        if len({row["initial_model_sha256"] for row in history}) != 1:
            raise RuntimeError("Matched initial state control failed")
        if len({row["initial_full_state_sha256"] for row in history}) != 1:
            raise RuntimeError("Matched full model/optimizer/RNG/data/scheduler initial state failed")
        if len({row["fault_schedule_sha256"] for row in history}) != 1:
            raise RuntimeError("Matched external fault schedule control failed")
    summary = {"experiment_id": config["experiment_id"], "phase": config["phase"],
               "shared_setup_seconds": shared_setup_seconds,
               "total_study_seconds": time.perf_counter() - study_started,
               "results": results, "scientific_acceptance": False,
               "independent_unit": "training seed/history; arm pairs are matched within history",
               "no_learning_claim_from_detection": True}
    write(output / "summary.json", summary)
    return summary


def run_study(config, output):
    """Entry point used after the parent job's admitted fixture suite."""
    return run(config, output)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: runner.py CONFIG_JSON NEW_OUTPUT_DIRECTORY")
    run(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
