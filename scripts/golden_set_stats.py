"""Print golden set counts and lexical overlap as Markdown tables.

    python scripts/golden_set_stats.py [eval/set.yaml] [results/<run>/per_question.csv]

Overlap is ``golden.quote_overlap``: the share of a question's content
tokens found in its gold quotes. Cohorts split synthetic by batch: ``syn-``
is the phase 2 batch, ``syn2-`` is golden set v2. With a per_question.csv
from ``baby-food eval``, it also prints recall@10 and MRR by cohort and
split at page tolerance 1.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

from baby_food_rag.evaluate import load_questions
from baby_food_rag.golden import quote_overlap

COHORTS = {
    "real": "real",
    "plan": "plan",
    "syn": "synthetic-old",
    "syn2": "synthetic-new",
    "unans2": "synthetic-new",
}


def cohort(question: dict) -> str:
    return COHORTS.get(question["id"].split("-", 1)[0], question["origin"])


def main(path: Path) -> None:
    questions = load_questions(path)
    answerable = [item for item in questions if item["answerable"]]
    print(f"{len(questions)} questions, {len(answerable)} answerable\n")
    _counts("Answerable by cohort and split", answerable, lambda q: (cohort(q), q["split"]))
    _counts("Answerable by first gold source and split", answerable, _source_split)
    _counts("Answerable by status", answerable, lambda q: (q["gold_status"],))
    unanswerable = [item for item in questions if not item["answerable"]]
    _counts("Unanswerable by cohort and split", unanswerable, lambda q: (cohort(q), q["split"]))
    print("| Cohort | n | mean | median | p75 | max |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name in COHORTS.values():
        values = sorted(
            quote_overlap(q["question"], [gold["quote"] for gold in q["gold"]])
            for q in answerable
            if cohort(q) == name
        )
        if not values:
            continue
        p75 = values[int(0.75 * (len(values) - 1))]
        print(
            f"| {name} | {len(values)} | {statistics.mean(values):.3f} | "
            f"{statistics.median(values):.3f} | {p75:.3f} | {values[-1]:.3f} |"
        )


def results_by_cohort(questions: list[dict], path: Path) -> None:
    cohorts = {item["id"]: cohort(item) for item in questions}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["page_tolerance"] == "1"]
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        for split in ("all", row["split"]):
            groups.setdefault((row["retriever"], cohorts[row["id"]], split), []).append(row)
    print("\n| Retriever | Cohort | Split | n | Recall@10 | MRR |")
    print("| --- | --- | --- | ---: | ---: | ---: |")
    for (retriever, name, split), chosen in sorted(groups.items()):
        recall = statistics.mean(float(row["recall_at_10"]) for row in chosen)
        mrr = statistics.mean(float(row["reciprocal_rank"]) for row in chosen)
        print(f"| {retriever} | {name} | {split} | {len(chosen)} | {recall:.4f} | {mrr:.4f} |")


def _source_split(question: dict) -> tuple[str, str]:
    return (question["gold"][0]["source_id"], question["split"])


def _counts(title: str, questions: list[dict], key) -> None:
    print(f"{title}:")
    for group, count in sorted(Counter(key(item) for item in questions).items()):
        print(f"  {' / '.join(str(part) for part in group)}: {count}")
    print()


if __name__ == "__main__":
    set_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/set.yaml")
    main(set_path)
    if len(sys.argv) > 2:
        results_by_cohort(load_questions(set_path), Path(sys.argv[2]))
