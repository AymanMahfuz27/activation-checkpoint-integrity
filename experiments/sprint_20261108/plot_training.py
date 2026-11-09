"""Export observed loss curves without interpolating censored trajectories."""

import json
from pathlib import Path


def plot_curves(training_root, config, output):
    # This entrypoint is called only inside a qualified allocation. Each line
    # is an actual history; no token or evaluation becomes an independent unit.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    training_root, output = Path(training_root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    clock_fields = {"elapsed_seconds": "elapsed_seconds",
                    "checker_cold_elapsed_seconds": "checker_cold_elapsed_seconds",
                    "instrumented_cold_elapsed_seconds": "instrumented_cold_elapsed_seconds"}
    clock = config.get("endpoint_clock", "checker_cold_elapsed_seconds")
    if clock not in clock_fields:
        raise ValueError("Unknown preregistered plot clock")
    field = clock_fields[clock]
    selected = config.get("plot_arms", [arm["name"] for arm in config["arms"]])
    known = {arm["name"] for arm in config["arms"]}
    if not set(selected) <= known:
        raise ValueError("Plot names an unregistered arm")
    figures = []
    for axis_name in (field, "accepted_updates"):
        fig, ax = plt.subplots(figsize=(9.5, 5.8), layout="constrained")
        plotted = []
        for index, name in enumerate(selected):
            for seed_index, seed in enumerate(config["seeds"]):
                source = training_root / ("seed-%d" % seed) / name / "evaluations.jsonl"
                rows = [json.loads(line) for line in source.read_text().splitlines() if line]
                endpoints = source.with_name("endpoint-observations.jsonl")
                if endpoints.exists():
                    observations = [json.loads(line) for line in endpoints.read_text().splitlines() if line]
                    if len(observations) != len(rows):
                        raise ValueError("Metric/endpoint observation count differs")
                    for row, observation in zip(rows, observations):
                        if (row["accepted_updates"] != observation["accepted_updates"]
                                or row["byte_perplexity"] != observation["byte_perplexity"]):
                            raise ValueError("Metric/endpoint observation pairing differs")
                        row.update(observation["clock_values"])
                finite = [row for row in rows if row["byte_perplexity"] is not None]
                if not finite:
                    continue
                ax.plot([row[axis_name] for row in finite],
                        [row["byte_perplexity"] for row in finite],
                        color="C%d" % (index % 10), alpha=0.5,
                        linewidth=1.0, marker=".", markersize=3,
                        label=name if seed_index == 0 else None)
                plotted.append({"seed": seed, "arm": name, "observations": len(finite)})
        target = config.get("target_perplexity")
        if target is not None:
            ax.axhline(target, color="black", linestyle="--", linewidth=1,
                       label="Frozen held-out target")
        ax.set_yscale("log")
        ax.set_ylabel("Held-out byte perplexity (log scale)")
        ax.set_xlabel("Accepted optimizer updates" if axis_name == "accepted_updates"
                      else "Elapsed seconds (%s clock)" % clock.replace("_", " "))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=2)
        ax.set_title(config["experiment_id"] + "; each curve is one training history", fontsize=11)
        stem = "perplexity-vs-" + ("updates" if axis_name == "accepted_updates" else "time")
        for suffix in ("png", "pdf"):
            fig.savefig(output / (stem + "." + suffix), dpi=180)
        plt.close(fig)
        figures.append({"name": stem, "clock": clock, "curves": plotted,
                        "interpolation": "Straight segments are visual guides; no crossing-time interpolation or inference",
                        "clock_boundary": "V2 uses post-journal endpoint observations; V1 uses pre-journal metric timestamps",
                        "independent_unit": "training history"})
    (output / "figure-metadata.json").write_text(json.dumps(figures, indent=2) + "\n")
    return figures
