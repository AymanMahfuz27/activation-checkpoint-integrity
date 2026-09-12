"""Bounded append-only writer. Payload durability precedes committed index rows."""

import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
import zlib
import torch
from ac_integrity.state import fsync_directory


class ShardWriter:
    def __init__(self, root, rank=0, queue_size=8, shard_bytes=268435456,
                 max_bytes=10**10, encoding="raw", deduplicate=False):
        self.root = Path(root)
        self.rank = rank
        self.folder = self.root / "tensors" / f"rank_{rank}"
        self.folder.mkdir(parents=True, exist_ok=False)
        (self.root / "events").mkdir(exist_ok=True)
        self.index = (self.root / "events" / f"rank_{rank}.jsonl").open("x")
        self.queue = queue.Queue(queue_size)
        self.shard_bytes = shard_bytes
        self.max_bytes = max_bytes
        self.total_bytes = 0
        if encoding not in {"raw", "zlib"}:
            raise ValueError("Unsupported lossless encoding")
        self.encoding = encoding
        self.deduplicate = deduplicate
        self.logical_bytes = 0
        self.reused_payloads = 0
        self._locations = {}
        self.high_water = 0
        self.blocked_seconds = 0.0
        self.error = None
        self.closed = False
        self._file = None
        self._number = -1
        self._offset = 0
        self.thread = threading.Thread(target=self._run, name=f"aci-writer-{rank}", daemon=False)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError("Capture writer failed; step must be abandoned") from self.error

    def submit(self, event, tensor):
        """Freeze logical bytes immediately, preserving original metadata separately.

        CUDA copies are ordered on the producing stream. The queued source stays
        alive until its event completes. Queue capacity bounds outstanding copies.
        """
        if self.closed:
            raise RuntimeError("Writer is closed")
        self.check()
        start = time.monotonic()
        # Wait before allocating the next host buffer, bounding memory to queue+2 tensors.
        while self.queue.full():
            self.check()
            time.sleep(0.001)
        source = tensor.detach().resolve_conj().resolve_neg().contiguous()
        event_done = None
        if source.device.type == "cuda":
            host = torch.empty(source.shape, dtype=source.dtype, device="cpu", pin_memory=True)
            host.copy_(source, non_blocking=True)
            event_done = torch.cuda.Event()
            event_done.record(torch.cuda.current_stream(source.device))
        else:
            host = source.cpu().clone()
        item = (event, host, event_done, source)
        while True:
            self.check()
            try:
                self.queue.put(item, timeout=0.05)
                break
            except queue.Full:
                continue
        self.high_water = max(self.high_water, self.queue.qsize())
        self.blocked_seconds += time.monotonic() - start

    def _open_shard(self):
        self._number += 1
        self._offset = 0
        self._file = (self.folder / f"shard_{self._number:06d}.bin.partial").open("xb")

    def _seal_shard(self):
        if self._file is None:
            return
        path = Path(self._file.name)
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        path.rename(path.with_suffix(""))
        fsync_directory(self.folder)
        self._file = None

    def _write(self, item):
        event, host, completion, source = item
        if completion is not None:
            completion.synchronize()
        payload = host.reshape(-1).view(torch.uint8).numpy().tobytes()
        self.logical_bytes += len(payload)
        digest = hashlib.sha256(payload).hexdigest()
        # Hashes index candidates; a byte comparison, not the hash, authorizes
        # reuse. Even a hash collision cannot change the captured tensor.
        for location in self._locations.get(digest, []):
            previous = {"event_id": "dedup-verification", "payload": location}
            if read_payload(self.root, previous, allow_partial=True) == payload:
                event["payload"] = dict(location)
                self.reused_payloads += 1
                self._write_index(event)
                return
        encoded = zlib.compress(payload, level=1) if self.encoding == "zlib" else payload
        if self.total_bytes + len(encoded) > self.max_bytes:
            raise OSError("Capture artifact byte limit exceeded")
        if self._file is None:
            self._open_shard()
        if self._offset and self._offset + len(encoded) > self.shard_bytes:
            self._seal_shard()
            self._open_shard()
        event["payload"] = {"path": f"tensors/rank_{self.rank}/shard_{self._number:06d}.bin",
                            "offset": self._offset, "length": len(encoded),
                            "sha256": digest}
        if self.encoding != "raw":
            event["payload"].update(encoding=self.encoding, raw_length=len(payload))
        self._file.write(encoded)
        self._file.flush()
        os.fsync(self._file.fileno())
        if self.deduplicate:
            self._locations.setdefault(digest, []).append(dict(event["payload"]))
        self._write_index(event)
        self._offset += len(encoded)
        self.total_bytes += len(encoded)

    def _write_index(self, event):
        self.index.write(json.dumps(event, allow_nan=False) + "\n")
        self.index.flush()
        os.fsync(self.index.fileno())

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    self._write(item)
            except BaseException as error:
                self.error = error
            finally:
                self.queue.task_done()

    def flush(self):
        self.queue.join()
        self.check()
        self._seal_shard()

    def close(self):
        if self.closed:
            self.check()
            return
        self.closed = True
        self.queue.put(None)
        self.queue.join()
        self.thread.join()
        try:
            if self.error is None:
                self._seal_shard()
            elif self._file is not None:
                self._file.close()
        finally:
            self.index.close()
        self.check()


def read_payload(root, event, allow_partial=False):
    location = event["payload"]
    root = Path(root).resolve()
    path = (root / location["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Payload path escapes artifact root")
    if allow_partial and not path.exists():
        path = path.with_suffix(path.suffix + ".partial")
    with path.open("rb") as source:
        source.seek(location["offset"])
        payload = source.read(location["length"])
    if len(payload) != location["length"]:
        raise ValueError(f"Payload checksum/length mismatch: {event['event_id']}")
    encoding = location.get("encoding", "raw")
    if encoding == "zlib":
        try:
            decoder = zlib.decompressobj()
            payload = decoder.decompress(payload, location["raw_length"] + 1)
            if (not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
                    or len(payload) != location["raw_length"]):
                raise ValueError("Invalid compressed payload length")
        except zlib.error as error:
            raise ValueError("Compressed payload checksum failure") from error
    elif encoding != "raw":
        raise ValueError("Unsupported payload encoding")
    if hashlib.sha256(payload).hexdigest() != location["sha256"]:
        raise ValueError(f"Payload checksum/length mismatch: {event['event_id']}")
    return payload


def read_tensor(root, event):
    payload = read_payload(root, event)
    dtype = getattr(torch, event["dtype"].removeprefix("torch."))
    if not payload:
        return torch.empty(event["shape"], dtype=dtype)
    return torch.frombuffer(bytearray(payload), dtype=dtype).reshape(event["shape"])
