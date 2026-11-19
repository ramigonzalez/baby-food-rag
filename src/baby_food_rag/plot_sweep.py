"""Draw recall@10 against query latency from a sweep metrics file.

Every point is read from metrics.json. Nothing in the figure is typed in
by hand.
"""

from __future__ import annotations

import json
from pathlib import Path


def plot_recall_latency(metrics_path: Path, png_path: Path) -> None:
    import matplotlib.pyplot as plt

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    configs = report["configs"]
    colors = {
        "bm25": "#1f4e79",
        "dense": "#b85c38",
        "hybrid": "#2d6a4f",
        "rerank": "#6c3483",
    }
    figure, axis = plt.subplots(figsize=(9.5, 6.2))
    for config in configs:
        family = config["family"]
        axis.scatter(
            config["latency_ms"],
            config["recall_at_10"],
            s=46,
            color=colors.get(family, "#333333"),
            zorder=3,
        )
        axis.annotate(
            config["name"],
            (config["latency_ms"], config["recall_at_10"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    axis.set_xlabel("Mean query latency (ms)")
    axis.set_ylabel("Recall@10")
    axis.set_title(report.get("chart_title", "Recall@10 against query latency"))
    axis.grid(True, linewidth=0.4, alpha=0.6)
    figure.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=140)
    plt.close(figure)
