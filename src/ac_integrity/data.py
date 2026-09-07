"""Immutable token shards with explicit train/validation ranges and batch cursors."""

import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch
from ac_integrity.state import sha256, write_json, fsync_directory


class PackedCorpus:
    def __init__(self, config):
        self.path = Path(config.data.path).resolve()
        manifest_path = self.path / "manifest.json"
        self.digest = sha256(manifest_path)
        if config.data.manifest_sha256 and self.digest != config.data.manifest_sha256:
            raise ValueError("Corpus manifest checksum mismatch")
        self.manifest = json.loads(manifest_path.read_text())
        if sha256(self.path / "records.jsonl") != self.manifest["records_sha256"]:
            raise ValueError("Source record ledger checksum mismatch")
        if self.manifest["fixture"] and config.evidence_class != "smoke":
            raise ValueError("Fixture corpus cannot provide natural/production evidence")
        if self.manifest["vocab_size"] != config.model.vocab_size:
            raise ValueError("Tokenizer vocabulary and model vocabulary differ")
        if not self.manifest["fixture"]:
            for key in ("source", "source_revision", "tokenizer", "tokenizer_revision"):
                if self.manifest[key] != getattr(config.data, key):
                    raise ValueError(f"Corpus provenance differs from configuration: {key}")
        self.shards = []
        self.ends = []
        end = 0
        for shard in self.manifest["shards"]:
            path = self.path / shard["path"]
            if not path.resolve().is_relative_to(self.path):
                raise ValueError("Shard path escapes corpus")
            if sha256(path) != shard["sha256"]:
                raise ValueError(f"Corrupt token shard: {path}")
            array = np.memmap(path, dtype="<u4", mode="r")
            if len(array) != shard["tokens"]:
                raise ValueError("Shard length does not match manifest")
            self.shards.append(array)
            end += len(array)
            self.ends.append(end)
        self.train_end = self.manifest["train_end"]
        self.total = end
        if end != self.manifest["tokens"] or not 0 < self.train_end < end:
            raise ValueError("Invalid train/validation boundary")

    def read(self, start, length):
        if start < 0 or start + length > self.total:
            raise ValueError("Token range outside immutable corpus")
        chunks = []
        while length:
            index = int(np.searchsorted(self.ends, start, side="right"))
            offset = start - (self.ends[index - 1] if index else 0)
            count = min(length, len(self.shards[index]) - offset)
            chunks.append(np.asarray(self.shards[index][offset:offset + count], dtype=np.int64))
            start += count
            length -= count
        return torch.from_numpy(np.concatenate(chunks))

    def batch(self, cursor, batch_size, sequence_length, validation=False):
        begin = self.train_end if validation else 0
        end = self.total if validation else self.train_end
        width = sequence_length + 1
        span = batch_size * width
        if span > end - begin:
            raise ValueError("Corpus split is too small for a batch")
        offset = begin + (cursor % ((end - begin) // span)) * span
        tokens = self.read(offset, span).reshape(batch_size, width)
        return {"input": tokens[:, :-1].contiguous(), "target": tokens[:, 1:].contiguous(),
                "sample_ids": [offset + i * width for i in range(batch_size)]}, cursor + 1


def materialize(config, records, tokenizer, source_metadata):
    """Pack a stable source iterator; record every source row in a durable ledger."""
    root = Path(config.data.path)
    root.mkdir(parents=True, exist_ok=False)
    shard_size = config.data.shard_tokens
    target = config.data.tokens
    if not 0 < config.data.validation_tokens < target:
        raise ValueError("Validation range must be nonempty and smaller than the corpus")
    buffer = []
    total = 0
    shards = []
    def flush():
        nonlocal buffer
        if not buffer:
            return
        name = f"tokens_{len(shards):05d}.bin"
        values = np.asarray(buffer, dtype="<u4")
        with (root / name).open("xb") as output:
            output.write(values.tobytes())
            output.flush()
            import os
            os.fsync(output.fileno())
        shards.append({"path": name, "tokens": len(values), "sha256": sha256(root / name)})
        buffer = []
    with (root / "records.jsonl").open("x") as ledger:
        for source_id, text in records:
            tokens = tokenizer(text)
            tokens = tokens[:target - total]
            if any(t < 0 or t >= config.model.vocab_size for t in tokens):
                raise ValueError("Tokenizer emitted an out-of-vocabulary ID")
            ledger.write(json.dumps({"source_id": source_id, "start": total, "tokens": len(tokens),
                                     "text_sha256": hashlib.sha256(text.encode()).hexdigest()}) + "\n")
            total += len(tokens)
            for token in tokens:
                buffer.append(token)
                if len(buffer) == shard_size:
                    flush()
            if total == target:
                break
        ledger.flush()
        import os
        os.fsync(ledger.fileno())
    if total != target:
        raise ValueError(f"Source exhausted after {total}/{target} tokens; corpus is uncommitted")
    flush()
    manifest = {"schema_version": 1, "fixture": config.data.fixture,
                "preprocessing": "aci-packed-v1-eos-per-record-u32-le",
                "preprocessing_sha256": sha256(__file__), "vocab_size": config.model.vocab_size,
                "tokens": total, "train_end": total - config.data.validation_tokens,
                "validation_end": total, "shards": shards,
                "records_sha256": sha256(root / "records.jsonl"), **source_metadata}
    write_json(root / "manifest.json", manifest)
    fsync_directory(root)
    return {"path": str(root.resolve()), "manifest_sha256": sha256(root / "manifest.json"), "tokens": total}


def prepare(config):
    if config.data.fixture:
        text = "Activation checkpoint replay preserves the training state. "
        def fixture_tokens(value):
            return [b % config.model.vocab_size for b in value.encode()]
        records = ((f"fixture:{i}", text) for i in range(config.data.tokens))
        return materialize(config, records, fixture_tokens, {"source": "local-text-fixture"})
    d = config.data
    for revision in (d.source_revision, d.tokenizer_revision):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Bootstrap must resolve source and tokenizer to immutable 40-character commits")
    if not d.source_files:
        raise ValueError("Specify the exact ordered source_files before downloading")
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        import sentencepiece as spm
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("Install the locked data extra with uv sync --extra data") from error
    tokenizer_path = snapshot_download(d.tokenizer, revision=d.tokenizer_revision,
                                       allow_patterns=["*token*", "*.model", "config.json"])
    tokenizer = spm.SentencePieceProcessor(model_file=str(Path(tokenizer_path) / "tokenizer.model"))
    if tokenizer.vocab_size() != config.model.vocab_size or tokenizer.eos_id() < 0:
        raise ValueError("Expected an exact 32K tokenizer with EOS")
    source_hashes = {}
    def records():
        for filename in d.source_files:
            path = hf_hub_download(d.source, filename, repo_type="dataset", revision=d.source_revision)
            source_hashes[filename] = sha256(path)
            row = 0
            for batch in parquet.ParquetFile(path).iter_batches(batch_size=1024, columns=["text"]):
                for item in batch.column(0).to_pylist():
                    yield f"{filename}:{row}", item
                    row += 1
    tokenizer_files = {p.name: sha256(p) for p in Path(tokenizer_path).iterdir() if p.is_file()}
    return materialize(config, records(),
                       lambda text: tokenizer.encode(text, out_type=int) + [tokenizer.eos_id()],
                       {"source": d.source, "source_revision": d.source_revision,
                        "source_files": source_hashes, "tokenizer": d.tokenizer,
                        "tokenizer_revision": d.tokenizer_revision, "tokenizer_files": tokenizer_files})
