"""One fresh-process matched arm using the official configured activation I/O API.

The allocated parent verifies source, inputs, environment, native build and job
resources first. This child never builds, dispatches or substitutes a fallback.
Normal process exit is required to drain the official trace buffer.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import struct
import sys
import traceback


ROLE = ContextVar("opguard_actual_intervention", default=None)
TAG = "ba-layer6-activation"
OFFICIAL_REVISION = "146d6bb523cf8364bbf2788cb319aef1a46f966b"
PRIMARY_EVENT = {
    "element": 6154933,
    "old_bits_u32": 3198802410,
    "new_bits_u32": 3198802411,
    "role": "recompute",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def manifest(value, completion):
    """Hash every typed leaf of full state or pre-clipping gradients, without saves."""
    import numpy as np
    import torch

    result = {}
    for name, leaf in completion.leaves(value):
        require(name not in result, "Duplicate typed state leaf")
        if isinstance(leaf, torch.Tensor):
            require(leaf.layout == torch.strided, "Unsupported nonstrided state")
            raw = (
                leaf.detach()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
            record = {
                "type": "tensor",
                "dtype": str(leaf.dtype),
                "shape": list(leaf.shape),
                "stride": list(leaf.stride()),
                "device": str(leaf.device),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        elif isinstance(leaf, (np.ndarray, np.generic)):
            raw = leaf.tobytes()
            record = {
                "type": type(leaf).__name__,
                "dtype": str(leaf.dtype),
                "shape": list(leaf.shape),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        elif type(leaf) is float:
            record = {"type": "float", "binary64_hex": struct.pack("!d", leaf).hex()}
        elif leaf is None or type(leaf) in (str, bool, int):
            record = {"type": type(leaf).__name__, "value": leaf}
        else:
            raise TypeError(
                "Unsupported state leaf: " + name + " " + type(leaf).__name__
            )
        result[name] = record
    require(result, "Empty state manifest cannot establish parity")
    return result


@contextmanager
def actual_context(completion, contexts):
    """Expose the existing intervention's region/role without a second injector."""
    original = completion.Intervention

    class Tracked(original):
        @contextmanager
        def context(self, region, role):
            with super().context(region, role):
                contexts[(region, role)] += 1
                token = ROLE.set((region, role))
                try:
                    yield
                finally:
                    ROLE.reset(token)

    completion.Intervention = Tracked
    try:
        yield
    finally:
        completion.Intervention = original


def official_runtime(contract, out):
    """Require installed official Python bytes and the actual native XOR backend."""
    require(
        contract["opguard_revision"] == OFFICIAL_REVISION, "Official revision differs"
    )
    require(
        "opguard" not in sys.modules, "OpGuard was imported before arm configuration"
    )
    for name, wanted in {
        "OPGUARD_AUTOSTART": "0",
        "OPGUARD_XOR_REGIONAL": "4",
        "OPGUARD_RUNTIME_CALLSTACK": "1",
    }.items():
        require(os.environ.get(name) == wanted, "Wrong official environment: " + name)
    require(
        "OPGUARD_DETERMINISTIC" not in os.environ,
        "Do not add a determinism intervention",
    )
    require(
        os.environ.get("OPGUARD_LOCAL_DUMP_ENABLED") != "1",
        "Local shadow trace output is unsupported",
    )
    require(
        Path(os.environ["OPGUARD_TRACE_OUTPUT_BASE"]).resolve()
        == (out / "official-trace").resolve(),
        "Official trace output must be this arm's exclusive directory",
    )
    import opguard

    package = Path(opguard.__file__).resolve().parent
    require(
        package.is_relative_to(Path(sys.prefix).resolve()),
        "Official package must be installed in isolated environment",
    )
    required_files = {
        "opguard/wrapped_ops.py",
        "opguard/runtime_tracer.py",
        "opguard/tracer/tensor_dump.py",
        "opguard/tracer/xor_signature.py",
        "opguard/tracer/meta_vars.py",
        "opguard/tracer/trace_buffer.py",
    }
    require(
        required_files <= set(contract["official_runtime_files"]),
        "Official numerical runtime pin census incomplete",
    )
    for name, pin in contract["official_runtime_files"].items():
        relative = Path(name)
        require(
            relative.parts[0] == "opguard" and ".." not in relative.parts,
            "Official runtime pin leaves package",
        )
        require(
            digest(package.parent / relative) == pin,
            "Installed official Python bytes differ: " + name,
        )
    xs = importlib.import_module("opguard.tracer.xor_signature")
    native = xs._xor_reduce_cuda_ext
    require(
        native is not None and hasattr(native, "xor_reduce_regional_cuda"),
        "Official native regional XOR unavailable",
    )
    native_path = Path(native.__file__).resolve()
    require(
        native_path.is_relative_to(Path(sys.prefix).resolve()),
        "Native XOR is outside isolated environment",
    )
    require(xs._fallback_reported is False, "Official XOR fallback already used")
    from opguard.wrapped_ops import make_wrapper
    from opguard.runtime_tracer import flush_trace_buffer
    from opguard.tracer.meta_vars import update_meta

    return {
        "make_wrapper": make_wrapper,
        "flush": flush_trace_buffer,
        "update_meta": update_meta,
        "xor_module": xs,
        "package": str(package),
        "native_path": str(native_path),
        "native_sha256": digest(native_path),
    }


@contextmanager
def activation_observer(engine, runtime):
    """Configure one unique layer activation and count actual native invocations."""
    layers = engine.model.gpt_neox.layers
    activation = layers[6].mlp.act
    require(
        sum(layer.mlp.act is activation for layer in layers) == 1,
        "Activation object is shared across layers; scoped mutation is unsafe",
    )
    original = activation.forward
    had_override = "forward" in activation.__dict__
    old_override = activation.__dict__.get("forward")
    calls, forwards = [], []
    target = "gpt_neox.layers.6"

    def identity():
        context = ROLE.get()
        require(
            context is not None
            and context[0] == target
            and context[1] in ("original", "recompute"),
            "Activation call lacks actual target intervention context",
        )
        return {
            "role": context[1],
            "ordinal": sum(row["role"] == context[1] for row in calls),
            "region": target,
            "layer": 6,
        }

    def completed(module, args, output):
        calls.append(identity())

    hook = activation.register_forward_hook(completed)
    if runtime is not None:
        official_call = runtime["make_wrapper"](
            tag_prefix=TAG, dump_tensors=False, global_sync=False
        )(1)

        def captured(*args, **kwargs):
            label = identity()
            runtime["update_meta"](
                rank=0,
                ba_role=label["role"],
                ba_ordinal=label["ordinal"],
                ba_region=target,
                ba_layer=6,
            )
            executed = []

            def counted_original(*native_args, **native_kwargs):
                require(not executed, "Official wrapper called original forward twice")
                output = original(*native_args, **native_kwargs)
                executed.append(output)
                return output

            output = official_call(counted_original, *args, **kwargs)
            require(
                len(executed) == 1 and output is executed[0],
                "Official wrapper changed native forward result or execution count",
            )
            forwards.append(
                {**label, "original_calls": len(executed), "same_result_object": True}
            )
            return output

        activation.forward = captured
    try:
        yield calls, forwards
    finally:
        hook.remove()
        if runtime is not None:
            if had_override:
                activation.forward = old_override
            else:
                del activation.forward


def run(contract, out, fault, official):
    from ac_integrity.research import completion
    from ac_integrity.research.m4m5 import FaultSpec

    completion.source_allocation_guard()
    require(
        contract["experiment_id"] == "BA-OPGUARD-L6-20261107-v1",
        "Wrong scoped official contract",
    )
    runtime = official_runtime(contract, out) if official else None
    item = next(
        row for row in contract["inputs"] if row["role"] == "corrected210-history1234"
    )
    stream = completion.stream_from_inputs(contract["inputs"])
    engine = completion.load_engine(item)
    initial = manifest(completion.snapshot(engine), completion)
    spec = FaultSpec(
        "recompute-L6-fraction-low-bit0", "recompute", 6, "bit", bit=0, seed=1729
    )
    spec = replace(spec, enabled=fault)
    contexts = Counter()
    execution = out / "execution"
    execution.mkdir()
    with (
        actual_context(completion, contexts),
        activation_observer(engine, runtime) as (calls, forwards),
    ):
        row, evidence = completion.execute(
            engine,
            stream,
            execution,
            spec=spec,
            policy="disabled",
            scope="checkpoint_boundaries",
            enforce=False,
            gradients=True,
            preserve_rng=True,
            branch_update=1,
        )
    require(row["status"] == "ACCEPTED", "Matched arm did not accept")
    require(
        len(calls) == 8
        and all(
            sum(x["role"] == role for x in calls) == 4
            for role in ("original", "recompute")
        ),
        "Missing original/replay activation calls",
    )
    target = "gpt_neox.layers.6"
    require(
        all(contexts[(target, role)] == 4 for role in ("original", "recompute")),
        "Native checkpoint replay count changed",
    )
    if fault:
        require(
            len(row["injections"]) == 1
            and all(
                row["injections"][0].get(key) == value
                for key, value in PRIMARY_EVENT.items()
            ),
            "Historical primary injection bytes did not reproduce",
        )
    else:
        require(row["injections"] == [], "Disabled trigger unexpectedly injected")
    gradients = evidence.get("gradients")
    require(
        isinstance(gradients, dict) and bool(gradients), "Pre-clipping gradients absent"
    )
    result = {
        "experiment_id": contract["experiment_id"],
        "fault": fault,
        "official": official,
        "contract_sha256": os.environ["OPGUARD_BASELINE_CONTRACT_SHA256"],
        "status": "ARM_COMPLETE_AWAITING_NORMAL_PROCESS_EXIT",
        "row": row,
        "calls": calls,
        "official_forward_receipts": forwards,
        "initial_state": initial,
        "post_state": manifest(completion.snapshot(engine), completion),
        "gradients": manifest(gradients, completion),
        "scientific_acceptance": False,
        "contexts": [
            {"region": key[0], "role": key[1], "count": value}
            for key, value in sorted(contexts.items())
        ],
    }
    if runtime is not None:
        runtime["flush"](force_dump=True)
        require(
            runtime["xor_module"]._fallback_reported is False,
            "Native XOR fallback occurred",
        )
        result["native_backend"] = {
            key: runtime[key] for key in ("package", "native_path", "native_sha256")
        }
        result["native_backend"]["fallback_reported"] = False
        require(
            len(forwards) == 8, "Official callable wrapper skipped activation calls"
        )
        result["post_state"] = manifest(completion.snapshot(engine), completion)
    write(out / "arm-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fault", choices=("0", "1"), required=True)
    parser.add_argument("--official", choices=("0", "1"), required=True)
    args = parser.parse_args()
    require(
        sys.platform == "linux" and os.environ.get("SLURM_JOB_ID", "").isdecimal(),
        "Real allocated Linux child required",
    )
    require(
        digest(args.contract) == os.environ["OPGUARD_BASELINE_CONTRACT_SHA256"],
        "Child contract differs from parent seal",
    )
    contract = json.loads(args.contract.read_text())
    historical = Path(contract["source_path"])
    sys.path.insert(0, str(historical / "src"))
    args.out.mkdir(parents=True, exist_ok=False)
    official = args.official == "1"
    if official:
        os.environ["OPGUARD_TRACE_OUTPUT_BASE"] = str(
            (args.out / "official-trace").resolve()
        )
        os.environ["OPGUARD_AUTOSTART"] = "0"
        os.environ["OPGUARD_XOR_REGIONAL"] = "4"
        os.environ["OPGUARD_RUNTIME_CALLSTACK"] = "1"
        os.environ.pop("OPGUARD_EXEC_OUTPUT_DIR", None)
    try:
        run(contract, args.out, args.fault == "1", official)
    except BaseException as error:
        write(
            args.out / "arm-failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "scientific_acceptance": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
