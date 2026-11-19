#!/usr/bin/env python3
"""Redraw the phase 3 chart from metrics.json. No hand-typed scores."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baby_food_rag.plot_sweep import plot_recall_latency  # noqa: E402


def main() -> int:
    default_metrics = "results/phase3-sweep/metrics.json"
    metrics = Path(sys.argv[1] if len(sys.argv) > 1 else default_metrics)
    default_png = metrics.with_name("recall_at_10_vs_latency.png")
    png = Path(sys.argv[2] if len(sys.argv) > 2 else default_png)
    plot_recall_latency(metrics, png)
    print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
