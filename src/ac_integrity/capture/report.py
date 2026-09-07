"""Self-contained HTML report with full-payload links and explicit evidence limits."""

import html
import json
from pathlib import Path
from ac_integrity.capture.compare import events
from ac_integrity.capture.writer import read_tensor

LIMITATIONS = [
    "Capture covers eager, dense, ordinary PyTorch tensor outputs visible to operator dispatch.",
    "Fused-kernel temporaries, compiled internals, tensor subclasses and native FP8 are outside this audit.",
    "CPU/MPS copies are synchronous. CUDA uses producing-stream pinned copies with a bounded queue; a reusable preallocated pool is not yet implemented.",
    "A smoke corpus or miniature model does not establish the 40M/125M milestone.",
    "A first divergence alone does not prove its cause; trigger-off replay and gradient/update evidence are required.",
]


def report(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    body = ["<!doctype html><meta charset='utf-8'><title>Activation checkpoint evidence</title>",
            "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#18222d}pre{white-space:pre-wrap;background:#f1f4f7;padding:16px}a{color:#1658a0}</style>",
            "<h1>Activation checkpoint evidence</h1>", "<h2>Run result</h2>",
            "<pre>" + html.escape(json.dumps(summary, indent=2)) + "</pre>",
            "<h2>Evidence files</h2><ul>"]
    for name in ("manifest.json", "environment.json", "data_manifest.json", "config.resolved.toml", "replay.toml", "outcome.pt"):
        if (root / name).exists():
            body.append(f"<li><a href='{name}'>{name}</a></li>")
    body.append("</ul><h2>First divergent tensors</h2>")
    for path in sorted((root / "mismatches").glob("*/first_divergence.json")):
        row = json.loads(path.read_text())
        body.append("<pre>" + html.escape(json.dumps(row, indent=2)) + "</pre>")
        for location in row.get("payloads", []):
            body.append(f"<p><a href='{html.escape(location['path'], quote=True)}'>Complete tensor shard</a>; byte offset {location['offset']}, length {location['length']}.</p>")
    body.append("<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(s)}</li>" for s in LIMITATIONS) + "</ul>")
    (root / "report.html").write_text("\n".join(body))
    return str((root / "report.html").resolve())


def inspect_event(root, event_id):
    for path in (Path(root) / "events").glob("rank_*.jsonl"):
        rank = int(path.stem.split("_")[1])
        for event in events(root, rank):
            if event["event_id"] == event_id:
                tensor = read_tensor(root, event)
                return {"event": event, "first_values": [str(v.item()) for v in tensor.flatten()[:32]]}
    raise ValueError("Unknown event ID")
