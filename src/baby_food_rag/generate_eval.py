"""Run generation and the judge over eval/set.yaml. No invented scores."""

from __future__ import annotations

import json
from pathlib import Path

from baby_food_rag.evaluate import load_questions
from baby_food_rag.generate import DEFAULT_K, LLM, Answer, ask
from baby_food_rag.index import CorpusIndex
from baby_food_rag.judge import abstention_accuracy, judge_answer


def run_generation_eval(
    index: CorpusIndex,
    questions: list[dict],
    llm: LLM,
    *,
    embedder: object | None = None,
    k: int = DEFAULT_K,
) -> dict:
    rows = []
    for question in questions:
        answer = ask(index, question["question"], llm, k=k, embedder=embedder)
        judged = judge_answer(
            llm,
            answer,
            expect_abstention=not question["answerable"],
        )
        rows.append(_row(question, answer, judged))
    return _summarize(rows, model=llm.model, usage=llm.usage)


def write_generation_report(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(render_generation_report(report), encoding="utf-8")


def render_generation_report(report: dict) -> str:
    if report.get("status") == "not_run":
        return (
            f"# Phase 4 generation\n\nNot medical advice.\n\nNot run. {report.get('reason', '')}\n"
        )
    lines = [
        "# Phase 4 generation",
        "",
        "Not medical advice. Numbers below come from this run only.",
        "",
        f"Model: `{report['model']}`. Questions: {report['n_questions']}.",
        f"Mean answer latency: {report['mean_latency_s']:.2f} s.",
        (
            f"Tokens: {report['prompt_tokens']} input, {report['completion_tokens']} output, "
            f"{report['calls']} calls."
        ),
    ]
    cost = report.get("estimated_usd")
    if cost is None:
        lines.append("Cost: not computed (no published rate for this model).")
    else:
        lines.append(f"Estimated cost from measured tokens and the published rate: ${cost:.4f}.")
    lines.extend(["", "## All questions", "", _table(report["overall"]), ""])
    lines.extend(["## Real questions only", "", _table(report["real"]), ""])
    lines.append(
        "Faithfulness is the share of extracted claims the retrieved context supports. "
        "Citation correctness is the share of [source p.N] marks that match a retrieved page. "
        "Abstention accuracy is the share of the unanswerable questions that abstained."
    )
    lines.append("")
    return "\n".join(lines)


def not_run_report(reason: str) -> dict:
    return {"status": "not_run", "reason": reason}


def _row(question: dict, answer: Answer, judged: dict) -> dict:
    return {
        "id": question["id"],
        "origin": question["origin"],
        "answerable": question["answerable"],
        "question": question["question"],
        "answer": answer.body,
        "abstained": answer.abstained,
        "expect_abstention": not question["answerable"],
        "latency_s": answer.latency_s,
        "faithfulness": judged["faithfulness"],
        "citation_correctness": judged["citation_correctness"],
        "relevance": judged["relevance"],
        "abstention_correct": judged["abstention_correct"],
        "n_citations": judged["n_citations"],
        "claims": judged["claims"],
    }


def _summarize(rows: list[dict], *, model: str, usage) -> dict:
    return {
        "status": "ok",
        "model": model,
        "n_questions": len(rows),
        "mean_latency_s": _mean([row["latency_s"] for row in rows]),
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "calls": usage.calls,
        "estimated_usd": usage.estimated_usd(model),
        "overall": _block(rows),
        "real": _block([row for row in rows if row["origin"] == "real"]),
        "rows": rows,
    }


def _block(rows: list[dict]) -> dict:
    unanswerable = [row for row in rows if row["expect_abstention"]]
    answerable = [row for row in rows if not row["expect_abstention"]]
    return {
        "n": len(rows),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "faithfulness": _mean_present(row["faithfulness"] for row in answerable),
        "citation_correctness": _mean_present(row["citation_correctness"] for row in answerable),
        "relevance": _mean_present(row["relevance"] for row in answerable),
        "abstention_accuracy": abstention_accuracy(unanswerable),
        "false_abstentions": sum(1 for row in answerable if row["abstained"]),
    }


def _table(block: dict) -> str:
    def cell(value: float | None) -> str:
        return "—" if value is None else f"{value:.4f}"

    return "\n".join(
        [
            "| | |",
            "| --- | ---: |",
            f"| n | {block['n']} |",
            f"| faithfulness | {cell(block['faithfulness'])} |",
            f"| citation correctness | {cell(block['citation_correctness'])} |",
            f"| relevance | {cell(block['relevance'])} |",
            f"| abstention accuracy | {cell(block['abstention_accuracy'])} |",
            f"| false abstentions | {block['false_abstentions']} |",
        ]
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_present(values) -> float | None:
    kept = [float(value) for value in values if value is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)


def load_eval_questions(path: Path) -> list[dict]:
    return load_questions(path)
