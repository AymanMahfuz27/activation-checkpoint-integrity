"""Choose checked updates before their first forward, without consuming model RNG."""

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class CheckingSchedule:
    name: str = "every"
    startup_updates: int = 5
    period: int = 10
    probability: float = 0.1
    seed: int = 0

    def __post_init__(self):
        if self.name not in {"disabled", "startup", "periodic", "randomized", "every"}:
            raise ValueError("Unknown checking schedule")
        if self.startup_updates < 0 or self.period < 1:
            raise ValueError("Startup must be nonnegative and period positive")
        if not 0 <= self.probability <= 1:
            raise ValueError("Checking probability must be in [0, 1]")

    def selected(self, update):
        if update < 1:
            raise ValueError("Optimizer update IDs are one-based")
        if self.name == "disabled":
            return False
        if self.name == "every":
            return True
        if self.name == "startup":
            return update <= self.startup_updates
        if self.name == "periodic":
            return update % self.period == 0
        # Stateless selection is stable across restart and repeated microbatches.
        # Its private pseudorandom stream never advances Python/NumPy/Torch RNG.
        key = f"aci-checking-v1:{self.seed}:{update}".encode()
        draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / (1 << 64)
        return draw < self.probability

