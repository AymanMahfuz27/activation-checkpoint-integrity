"""Stable event identities, module attribution and checkpoint region contexts."""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import threading
import traceback
import weakref
import torch
from torch.utils._pytree import tree_flatten, keystr
from ac_integrity.capture.dispatch import CaptureMode
from ac_integrity.capture.writer import ShardWriter
from ac_integrity.state import backend_state, write_json


class CaptureRuntime:
    def __init__(self, root, config, attempt="0", rank=0):
        self.root = Path(root)
        self.config = config
        self.attempt, self.rank = attempt, rank
        self.step, self.microbatch, self.phase = 0, 0, "setup"
        self.region = ContextVar(f"aci_region_{id(self)}", default=None)
        self.module_stack = ContextVar(f"aci_modules_{id(self)}", default=())
        self.calls = Counter()
        self.ordinals = Counter()
        self.counts = Counter()
        self.bytes = 0
        self.events = 0
        self.unsupported = []
        self.provenance = {}
        self.names = {}
        self.stacks = {}
        self.handles = []
        self.writer = None
        if config.capture.mode == "full":
            c = config.capture
            self.writer = ShardWriter(root, rank, c.queue_size, c.shard_bytes, c.max_bytes,
                                      encoding=c.encoding, deduplicate=c.deduplicate)

    @contextmanager
    def phase_context(self, name):
        previous = self.phase
        self.phase = name
        try:
            yield
        finally:
            self.phase = previous

    def checkpoint_contexts(self, name):
        parent = self.region.get()
        parent_path = parent[0] if parent else "root"
        # Nested checkpoint invocations created again while replaying their
        # parent must use the same call numbers as the original parent pass.
        parent_role = parent[2] if parent else "outside"
        key = (self.step, self.microbatch, parent_path, parent_role, name)
        call = self.calls[key]
        self.calls[key] += 1
        path = f"{parent_path}/{name}:{call}"
        def factory():
            return self._region_context(path, call, "original"), self._region_context(path, call, "recompute")
        return factory

    @contextmanager
    def _region_context(self, path, call, role):
        parent = self.region.get()
        if role == "original" and parent and parent[2].startswith("recompute"):
            role = "recompute_parent"
        token = self.region.set((path, call, role))
        # Place the observer above user dispatch modes in both executions.
        try:
            with CaptureMode(self):
                yield
        finally:
            self.region.reset(token)

    def attach(self, model):
        aliases = {}
        for name, module in model.named_modules(remove_duplicate=False):
            aliases.setdefault(id(module), []).append(name or "model")
        self.names = {id(p): name for name, p in model.named_parameters()}
        self.names.update({id(b): name for name, b in model.named_buffers()})
        for module in model.modules():
            paths = tuple(aliases[id(module)])
            def before(module, args, paths=paths):
                self.module_stack.set(self.module_stack.get() + (paths,))
            def after(module, args, output):
                stack = self.module_stack.get()
                self.module_stack.set(stack[:-1])
            self.handles.append(module.register_forward_pre_hook(before))
            self.handles.append(module.register_forward_hook(after, always_call=True))

    def describe_inputs(self, value):
        tensors = []
        for item in tree_flatten(value)[0]:
            if isinstance(item, torch.Tensor):
                previous = self.provenance.get(id(item))
                event_id = previous[1] if previous and previous[0]() is item else None
                tensors.append({"event_id": event_id, "name": self.names.get(id(item)),
                                "version": item._version,
                                "storage": item.untyped_storage().data_ptr() if item.layout == torch.strided else None})
        return tensors

    def record(self, func, inputs, leaves, schema, args, kwargs):
        region = self.region.get()
        path, call, role = region if region else ("outside", 0, "unpaired")
        phase = role if region else self.phase
        frames = traceback.extract_stack(limit=32)
        # PyTorch's recomputation pack hook inserts detach operations that have
        # no forward counterpart. Capture them, but give framework bookkeeping
        # its own sequence. User detach calls remain eligible for exact pairing.
        import torch.utils.checkpoint as checkpoint_module
        framework = str(func) == "aten.detach.default" and any(
            frame.name == "pack_hook" and frame.filename == checkpoint_module.__file__
            for frame in frames
        )
        if framework:
            phase = "checkpoint_bookkeeping"
        ordinal_key = (self.step, self.microbatch, path, call, phase)
        ordinal = self.ordinals[ordinal_key]
        self.ordinals[ordinal_key] += 1
        stack_text = "".join(traceback.format_list(frames))
        stack_id = hashlib.sha256(stack_text.encode()).hexdigest()
        self.stacks.setdefault(stack_id, stack_text)
        module_parents = [p for names in self.module_stack.get() for p in names]
        def sanitize(value):
            if isinstance(value, torch.Tensor):
                return {"tensor": True}
            if value is None or isinstance(value, (bool, int, str)):
                return value
            if isinstance(value, float):
                return value if __import__("math").isfinite(value) else str(value)
            if isinstance(value, (tuple, list)):
                return [sanitize(v) for v in value]
            if isinstance(value, dict):
                return {str(k): sanitize(v) for k, v in value.items()}
            return str(type(value).__name__)
        for output_path, tensor in leaves:
            if not isinstance(tensor, torch.Tensor):
                continue
            if type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.layout != torch.strided or tensor.is_quantized or tensor.device.type == "meta":
                self.unsupported.append({"operator": str(func), "type": str(type(tensor)), "layout": str(tensor.layout)})
                raise TypeError("Unsupported tensor output; exhaustive capture aborted")
            position = keystr(output_path) or "output"
            pair = f"{self.attempt}/{self.rank}/{self.step}/{self.microbatch}/{path}/{call}/{ordinal}/{position}"
            event_id = f"{self.config.run_id}/{self.attempt}/{self.rank}/{self.step}/{self.microbatch}/{phase}/{path}/{call}/{ordinal}/{position}"
            size = tensor.numel() * tensor.element_size()
            event = {"event_id": event_id, "pair_id": pair if region and not framework else None,
                     "sequence": self.events,
                     "unpaired_reason": "checkpoint_pack_hook_detach" if framework else (None if region else "outside_checkpoint_region"),
                     "operator": str(func), "output_schema": schema, "output_path": position,
                     "step": self.step, "microbatch": self.microbatch, "phase": phase,
                     "checkpoint_role": role, "region": path, "call": call, "op_ordinal": ordinal,
                     "rank": self.rank, "thread": threading.get_ident(),
                     "stream": torch.cuda.current_stream(tensor.device).cuda_stream if tensor.is_cuda else None,
                     "module": module_parents[-1] if module_parents else None,
                     "module_parents": module_parents, "stack_id": stack_id,
                     "shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
                     "layout": str(tensor.layout), "stride": list(tensor.stride()),
                     "storage_offset": tensor.storage_offset(), "storage": tensor.untyped_storage().data_ptr(),
                     "numel": tensor.numel(), "nbytes": size, "requires_grad": tensor.requires_grad,
                     "version": tensor._version, "inputs": inputs,
                     "arguments": sanitize(args), "kwargs": sanitize(kwargs),
                     "grad_enabled": torch.is_grad_enabled(),
                     "autocast_cpu": torch.is_autocast_enabled("cpu"),
                     "autocast_cuda": torch.is_autocast_enabled("cuda"),
                     "backend": backend_state(), "compiling": torch.compiler.is_compiling()}
            self.events += 1
            self.bytes += size
            self.counts[phase] += 1
            identifier = id(tensor)
            def forget(reference, identifier=identifier):
                current = self.provenance.get(identifier)
                if current is not None and current[0] is reference:
                    self.provenance.pop(identifier, None)
            self.provenance[identifier] = (weakref.ref(tensor, forget), event_id)
            if self.writer:
                self.writer.submit(event, tensor)

    def flush(self):
        if self.writer:
            self.writer.flush()

    def summary(self):
        return {"captured_tensors": self.events, "payload_bytes": self.bytes,
                "stored_payload_bytes": self.writer.total_bytes if self.writer else None,
                "reused_payloads": self.writer.reused_payloads if self.writer else 0,
                "by_phase": dict(self.counts), "unsupported": self.unsupported,
                "queue_high_water": self.writer.high_water if self.writer else 0,
                "capture_blocked_seconds": self.writer.blocked_seconds if self.writer else 0.0}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        try:
            if self.writer:
                self.writer.close()
        finally:
            self.provenance.clear()
            write_json(self.root / "stacks.json", self.stacks)
