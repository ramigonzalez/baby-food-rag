"""Score saved answers again with a judge, without generating them again.

Input is a ``label_sheet.json`` from ``baby-food ask-eval``: each item has
the question, the retrieved chunks, the answer, and its key facts. The
judge sees the same prompts as in the run (``judge.py``). Output is
``scores.json``, with scores, verdict counts, and cost, and ``claims.json``,
with the claim and fact text. Claims restate the answer, so ``claims.json``
is gitignored.

``compare_scores`` measures how far two judge runs over the same items
are apart. It says nothing about which judge is right: that needs the
human labels.
"""

from __future__ import annotations

import json
from pathlib import Path

from baby_food_rag.embed import estimate_tokens
from baby_food_rag.generate import LLM, Answer, Usage
from baby_food_rag.generate_eval import add_reasoning_allowance, score_row
from baby_food_rag.judge import (
    claims_prompt,
    coverage_prompt,
    judge_answer,
    relevance_prompt,
    verify_prompt,
)
from baby_food_rag.types import Chunk, Hit

# A faithfulness change above this counts as a changed item.
CHANGE_THRESHOLD = 0.1


def load_sheet(path: Path) -> dict:
    sheet = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sheet, dict) or not isinstance(sheet.get("items"), list):
        raise ValueError(f"{path} is not a label sheet")
    return sheet


def item_answer(item: dict) -> Answer:
    """The saved answer with its retrieved chunks, as the judge saw them."""
    hits = []
    for number, entry in enumerate(item["retrieved"]):
        chunk = Chunk(
            chunk_id=f"{item['id']}::{number}",
            source_id=entry["source_id"],
            year=0,
            country="",
            page=int(entry["pdf_page"]),
            printed_page=str(entry["page"]),
            text=entry["text"],
            url="",
            title="",
        )
        hits.append(Hit(score=0.0, chunk=chunk))
    return Answer(
        question=item["question"],
        text=item["answer"],
        hits=hits,
        abstained=False,
        model="saved",
        latency_s=0.0,
    )


def item_facts(item: dict) -> list[str]:
    return [entry["fact"] for entry in item.get("key_facts", [])]


def estimate_rejudge(sheet: dict, judge_model: str) -> dict:
    """Tokens and USD before any call.

    The answers are known, so the claim, relevance, and coverage prompts are
    exact. The verify prompt and the outputs use the claims the first judge
    extracted. Tokens are characters / 3, plus the reasoning allowance.
    """
    usage = Usage()
    for item in sheet["items"]:
        answer = item_answer(item)
        claims = [entry["claim"] for entry in item["claims"]]
        facts = item_facts(item)
        verdicts = [{"claim": n, "verdict": "not_found"} for n in range(1, len(claims) + 1)]
        usage.add(_tokens(claims_prompt(answer.question, answer.text)), _json(claims=claims))
        if claims:
            usage.add(_tokens(verify_prompt(claims, answer.hits)), _json(verdicts=verdicts))
        usage.add(_tokens(relevance_prompt(answer.question, answer.text)), _json(relevance=0.5))
        if facts:
            covered = [{"fact": n, "covered": False} for n in range(1, len(facts) + 1)]
            usage.add(
                _tokens(coverage_prompt(answer.question, answer.text, facts)),
                _json(facts=covered),
            )
    add_reasoning_allowance(usage, judge_model)
    return {
        "n_items": len(sheet["items"]),
        "model": judge_model,
        "calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "usd": usage.estimated_usd(judge_model),
    }


def run_rejudge(
    sheet: dict,
    judge: LLM,
    *,
    temperature: float | None = None,
    seed: int | None = None,
    max_usd: float | None = None,
) -> dict:
    """Judge every item. Stops after the item that takes the spend above ``max_usd``."""
    rows = []
    status = "ok"
    for item in sheet["items"]:
        judged = judge_answer(
            judge, item_answer(item), expect_abstention=False, key_facts=item_facts(item)
        )
        rows.append(
            {
                "id": item["id"],
                "split": item.get("split"),
                "origin": item["origin"],
                "faithfulness": judged["faithfulness"],
                "completeness": judged["completeness"],
                "relevance": judged["relevance"],
                "citation_correctness": judged["citation_correctness"],
                "claims": judged["claims"],
                "facts": judged["facts"],
            }
        )
        spent = judge.usage.estimated_usd(judge.model)
        if max_usd is not None and spent is not None and spent > max_usd:
            status = "stopped_at_cap"
            break
    usage = judge.usage
    return {
        "status": status,
        "judge_model": judge.model,
        "temperature": temperature,
        "seed": seed,
        "generator": sheet.get("generator"),
        "n_items": len(rows),
        "usage": {
            "calls": usage.calls,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "usd": usage.estimated_usd(judge.model),
            "reported_usd": usage.reported_usd,
        },
        "means": {
            name: _mean_present(row[name] for row in rows)
            for name in ("faithfulness", "completeness", "relevance", "citation_correctness")
        },
        "rows": rows,
    }


def write_rejudge(report: dict, out_dir: Path) -> None:
    """``scores.json`` has no text. ``claims.json`` keeps claim and fact text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    public = {**report, "rows": [score_row(row) for row in report["rows"]]}
    texts = [
        {"id": row["id"], "claims": row["claims"], "facts": row["facts"]} for row in report["rows"]
    ]
    for name, payload in (("scores.json", public), ("claims.json", texts)):
        (out_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def render_rejudge(report: dict) -> str:
    usage = report["usage"]
    means = report["means"]

    def cell(value: float | None) -> str:
        return "—" if value is None else f"{value:.4f}"

    lines = [
        (
            f"judge {report['judge_model']}  temperature {report['temperature']}  "
            f"seed {report['seed']}  items {report['n_items']}  status {report['status']}"
        ),
        (
            f"faithfulness {cell(means['faithfulness'])}  completeness "
            f"{cell(means['completeness'])}  relevance {cell(means['relevance'])}  "
            f"citation {cell(means['citation_correctness'])}"
        ),
        (
            f"calls {usage['calls']}  input {usage['prompt_tokens']}  output "
            f"{usage['completion_tokens']} (reasoning {usage['reasoning_tokens']})  "
            f"usd {cell(usage['usd'])}  reported {cell(usage['reported_usd'])}"
        ),
    ]
    return "\n".join(lines) + "\n"


def compare_scores(first: list[dict], second: list[dict]) -> dict:
    """Mean absolute difference between two runs, over items scored in both."""
    by_id = {row["id"]: row for row in second}
    pairs = [(row, by_id[row["id"]]) for row in first if row["id"] in by_id]
    result: dict = {"n_items": len(pairs)}
    for name in ("faithfulness", "completeness", "relevance"):
        gaps = [
            abs(a[name] - b[name])
            for a, b in pairs
            if a.get(name) is not None and b.get(name) is not None
        ]
        result[f"n_{name}"] = len(gaps)
        result[f"{name}_mad"] = _mean_present(gaps)
        if name == "faithfulness":
            changed = sum(1 for gap in gaps if gap > CHANGE_THRESHOLD)
            result["faithfulness_changed_share"] = (changed / len(gaps)) if gaps else None
    return result


def render_comparison(runs: dict[str, dict]) -> str:
    """Markdown: per-run means and every pair of runs, on the items all runs scored.

    ``runs`` maps a label to a ``metrics.json`` or ``scores.json`` payload.
    Pairs with the same judge model measure stability; the others agreement.
    """
    common = set.intersection(*({row["id"] for row in run["rows"]} for run in runs.values()))
    rows = {
        label: [row for row in run["rows"] if row["id"] in common] for label, run in runs.items()
    }

    def cell(value: float | None) -> str:
        return "—" if value is None else f"{value:.4f}"

    lines = [
        f"Items scored by every run: {len(common)}.",
        "",
        "| Run | Judge | Temperature | Seed | Faithfulness | Completeness | Relevance |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, run in runs.items():
        means = {
            name: _mean_present(row.get(name) for row in rows[label])
            for name in ("faithfulness", "completeness", "relevance")
        }
        temperature = run.get("temperature")
        lines.append(
            f"| {label} | `{run['judge_model']}` | "
            f"{'not sent' if temperature is None else temperature} | "
            f"{'not sent' if run.get('seed') is None else run['seed']} | "
            f"{cell(means['faithfulness'])} | {cell(means['completeness'])} | "
            f"{cell(means['relevance'])} |"
        )
    lines.extend(
        [
            "",
            "| Pair | Kind | Faithfulness MAD | Faithfulness changed > 0.1 | "
            "Completeness MAD | Relevance MAD |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    labels = list(runs)
    cross = []
    for i, first in enumerate(labels):
        for second in labels[i + 1 :]:
            same = runs[first]["judge_model"] == runs[second]["judge_model"]
            result = compare_scores(rows[first], rows[second])
            if not same:
                cross.append(result)
            share = result["faithfulness_changed_share"]
            lines.append(
                f"| {first} vs {second} | {'same judge' if same else 'cross judge'} | "
                f"{cell(result['faithfulness_mad'])} (n {result['n_faithfulness']}) | "
                f"{'—' if share is None else f'{share:.0%}'} | "
                f"{cell(result['completeness_mad'])} (n {result['n_completeness']}) | "
                f"{cell(result['relevance_mad'])} |"
            )
    if cross:
        faith = _mean_present(item["faithfulness_mad"] for item in cross)
        complete = _mean_present(item["completeness_mad"] for item in cross)
        lines.extend(
            [
                "",
                f"Cross-judge mean over {len(cross)} pairs: faithfulness MAD {cell(faith)}, "
                f"completeness MAD {cell(complete)}.",
            ]
        )
    lines.extend(
        [
            "",
            "MAD is the mean absolute difference per item. No human labels exist yet, so "
            "these numbers show stability and agreement, not which judge is right.",
        ]
    )
    return "\n".join(lines) + "\n"


def _tokens(text: str) -> int:
    return estimate_tokens([text])


def _json(**payload: object) -> int:
    return _tokens(json.dumps(payload, ensure_ascii=False))


def _mean_present(values) -> float | None:
    kept = [float(value) for value in values if value is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)
