"""Apply the frozen target rule to prior calibration observations only."""
import json
import math
from pathlib import Path


def derive(path, output):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if [row["accepted_updates"] for row in rows] != list(range(0, 121, 20)):
        raise ValueError("Exact prior clean calibration trajectory required")
    initial = rows[0]["cross_entropy_nats_per_byte"]
    final = rows[-1]["cross_entropy_nats_per_byte"]
    if not all(math.isfinite(x) for x in (initial, final)) or initial - final < 0.3:
        raise ValueError("Pilot clean learning did not qualify")
    target_ce = initial - 0.75 * (initial - final)
    result = {"initial_ce": initial, "final_ce": final,
              "target_ce": target_ce, "target_perplexity": math.ceil(math.exp(target_ce) * 10) / 10,
              "rule": "75% of clean calibration CE improvement, PPL rounded upward to0.1",
              "source_split": "calibration", "scientific_acceptance": False}
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
