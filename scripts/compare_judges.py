"""Compare judge runs over the same label-sheet items.

    python scripts/compare_judges.py [--out judges.md] LABEL=PATH [LABEL=PATH ...]

Each PATH is a ``metrics.json`` from ``baby-food ask-eval`` or a
``scores.json`` from ``baby-food rejudge``. Only items scored by every run
count. Runs with the same judge model give stability, the others
agreement. No file read here holds answer or guide text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baby_food_rag.rejudge import render_comparison


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="LABEL=PATH")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    runs = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        runs[label] = json.loads(Path(path).read_text(encoding="utf-8"))
    text = render_comparison(runs)
    print(text, end="")
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
