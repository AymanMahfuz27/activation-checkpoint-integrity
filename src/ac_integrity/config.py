"""Validated, versioned experiment configuration. Unknown fields are errors."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import hashlib
import json
import tomllib


@dataclass
class ModelConfig:
    layers: int = 8
    width: int = 512
    heads: int = 8
    kv_heads: int = 4
    intermediate: int = 1408
    vocab_size: int = 32000
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0
    attention: str = "explicit"


@dataclass
class DataConfig:
    path: str = "data/fineweb-edu-100m"
    source: str = "HuggingFaceFW/fineweb-edu"
    source_revision: str = ""
    tokenizer: str = "openlm-research/open_llama_3b"
    tokenizer_revision: str = ""
    manifest_sha256: str = ""
    tokens: int = 100000000
    validation_tokens: int = 1000000
    shard_tokens: int = 1000000
    source_files: list[str] = field(default_factory=list)
    fixture: bool = False


@dataclass
class TrainConfig:
    steps: int = 2000
    sequence_length: int = 256
    microbatch_size: int = 2
    accumulation: int = 4
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.1
    clip_norm: float = 1.0
    warmup_steps: int = 100
    decay_steps: int = 200
    snapshot_steps: list[int] = field(default_factory=lambda: [1, 1000, 2000])
    checkpoint_every: int = 100
    stop_after: int = 0


@dataclass
class CheckpointConfig:
    enabled: bool = True
    blocks: list[int] = field(default_factory=list)
    use_reentrant: bool = False
    preserve_rng_state: bool = True
    early_stop: bool = False
    determinism_check: str = "default"


@dataclass
class CaptureConfig:
    mode: str = "off"
    policy: str = "enforce"
    audit_steps: str | list[int] = "all"
    max_bytes: int = 10000000000
    reserve_bytes: int = 5000000000
    verified_quota_bytes: int = 0
    queue_size: int = 8
    shard_bytes: int = 268435456
    census_path: str = ""


@dataclass
class AdapterConfig:
    name: str = "none"
    trigger: bool = False


@dataclass
class DistributedConfig:
    world_size: int = 1
    pipeline_parallel: int = 1
    tensor_parallel: int = 1
    timeout_seconds: int = 120


@dataclass
class Config:
    schema_version: int = 1
    run_id: str = "research"
    evidence_class: str = "clean"
    seed: int = 17
    device: str = "cpu"
    dtype: str = "float32"
    threads: int = 1
    artifact_root: str = "artifacts/production"
    expected_torch: str = "2.13.0"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    def validate(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported configuration schema")
        if not self.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.run_id):
            raise ValueError("run_id must be a safe single path component")
        m, t = self.model, self.training
        for value in (m.layers, m.width, m.heads, m.kv_heads, m.intermediate, m.vocab_size,
                      t.steps, t.sequence_length, t.microbatch_size, t.accumulation,
                      self.threads, self.capture.queue_size, self.capture.shard_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("Model, batch, step, thread and queue dimensions must be positive integers")
        if m.width % m.heads or m.heads % m.kv_heads or (m.width // m.heads) % 2:
            raise ValueError("RoPE/GQA require even head dimension and divisible head counts")
        if m.attention not in {"explicit", "sdpa"} or not 0 <= m.dropout < 1:
            raise ValueError("Invalid attention or dropout")
        if self.capture.mode not in {"off", "census", "full"}:
            raise ValueError("Unknown capture mode")
        if self.capture.policy not in {"observe", "enforce"}:
            raise ValueError("Unknown mismatch policy")
        if self.capture.audit_steps != "all" and not (
            isinstance(self.capture.audit_steps, list) and
            all(type(s) is int and 1 <= s <= t.steps for s in self.capture.audit_steps)
        ):
            raise ValueError("audit_steps must be all or valid one-based steps")
        if self.checkpoint.use_reentrant or self.checkpoint.early_stop:
            raise ValueError("This audited contract requires non-reentrant, complete recomputation")
        if self.checkpoint.determinism_check not in {"default", "none"}:
            raise ValueError("Unsupported determinism check")
        if any(i < 0 or i >= m.layers for i in self.checkpoint.blocks):
            raise ValueError("Checkpoint block index out of range")
        if self.capture.mode != "off" and m.attention != "explicit":
            raise ValueError("Exhaustive audits require explicit attention")
        if self.dtype != "float32":
            raise ValueError("Research trainer currently validates deterministic FP32 only")
        if self.distributed.world_size != 1 or self.distributed.pipeline_parallel != 1 or self.distributed.tensor_parallel != 1:
            raise ValueError("Research trainer supports one rank; distributed gate is tested separately")
        if self.adapter.name not in {"none", "pytorch_84864"}:
            raise ValueError("Controlled suite is gated on accepted R0/M0.3; unknown or unvalidated adapter")
        if self.data.fixture and self.evidence_class != "smoke":
            raise ValueError("Fixture tokens are only permitted for smoke evidence")
        if min(t.learning_rate, t.epsilon, t.clip_norm) <= 0 or t.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if not 0 <= t.beta1 < 1 or not 0 <= t.beta2 < 1:
            raise ValueError("Invalid Adam betas")
        if min(t.warmup_steps, t.decay_steps, t.checkpoint_every, t.stop_after) < 0:
            raise ValueError("Negative schedule interval")
        if self.capture.max_bytes <= 0 or self.capture.reserve_bytes < 0 or self.capture.verified_quota_bytes < 0:
            raise ValueError("Invalid storage limits")
        return self

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


SECTIONS = {"model": ModelConfig, "data": DataConfig, "training": TrainConfig,
            "checkpoint": CheckpointConfig, "capture": CaptureConfig,
            "adapter": AdapterConfig, "distributed": DistributedConfig}


def from_dict(raw):
    values = dict(raw)
    for name, cls in SECTIONS.items():
        values[name] = cls(**values.get(name, {}))
    return Config(**values).validate()


def load_config(path):
    with Path(path).open("rb") as source:
        return from_dict(tomllib.load(source))


def write_config(config, path):
    """Write the fully resolved flat-section TOML without an optional dependency."""
    raw = asdict(config)
    lines = []
    for key, value in raw.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {json.dumps(value)}")
    for name in SECTIONS:
        lines.append(f"\n[{name}]")
        for key, value in raw[name].items():
            lines.append(f"{key} = {json.dumps(value)}")
    Path(path).write_text("\n".join(lines) + "\n")
