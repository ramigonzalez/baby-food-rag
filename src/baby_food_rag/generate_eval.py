"""Run generation and the judge over eval/set.yaml. No invented scores.

The generator answers and a judge from another model family scores the
answer (see ``judge.py``). Each model keeps its own token count from the
OpenRouter response, and the dollar figure is those counts times
``PRICE_PER_MILLION``.

``estimate_run`` prices a run before any call. It uses the real retrieved
chunks and the exact generation prompt. The judge prompts are built from
the judge templates with a typical answer: ``TYPICAL_ANSWER_CHARS`` of
prose, ``TYPICAL_CITATIONS`` linked citations, and ``TYPICAL_CLAIMS``
claims. Tokens are characters divided by 3, as for the paid embedder.
A model that reasons also gets ``REASONING_TOKENS_PER_CALL`` output tokens
per call. Every answer is assumed to be judged, so a run with abstentions
costs less.

``build_label_sheet`` draws answered items for a person to label, so the
judge can be checked against a human. It carries no judge score.

Answers, claims, and retrieved chunks restate guide text, so they stay
local: ``metrics.json`` keeps scores and page references only, and the
full rows go to ``answers.json``, which is gitignored like the label sheet.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from baby_food_rag.embed import estimate_tokens
from baby_food_rag.evaluate import ORIGINS, load_questions
from baby_food_rag.generate import (
    DEFAULT_K,
    LLM,
    Answer,
    Usage,
    ask,
    cite_label,
    generation_prompt,
    link_citations,
    shown_page,
)
from baby_food_rag.golden import SPLITS
from baby_food_rag.index import CorpusIndex, search
from baby_food_rag.judge import (
    abstention_accuracy,
    claims_prompt,
    coverage_prompt,
    judge_answer,
    relevance_prompt,
    verify_prompt,
)
from baby_food_rag.types import Hit

EVAL_MAX_USD = 1.00
TYPICAL_ANSWER_CHARS = 900
TYPICAL_CITATIONS = 3
TYPICAL_CLAIMS = 6
# Output tokens a model bills per call beyond its visible answer, most likely
# reasoning. Claude Haiku 5.5 billed about 450 per judge call more than its JSON
# needs in the 2-question smoke run of 2026-11-27 (8 calls), rounded up.
REASONING_TOKENS_PER_CALL = {
    "anthropic/claude-haiku-5.5": 500,
    # Billed no reasoning tokens in 98 judge calls on 2026-11-27 (temperature 0, seed 7).
    "google/gemini-2.5-flash": 0,
}
# Row fields that hold generated or guide text. They go to answers.json only.
TEXT_FIELDS = ("question", "answer", "claims", "facts", "key_facts")
LABEL_SHEET_SIZE = 30
LABEL_SHEET_SEED = 20261127


def retrieve_all(
    index: CorpusIndex,
    questions: list[dict],
    *,
    embedder: object | None = None,
    k: int = DEFAULT_K,
) -> list[list[Hit]]:
    """The hybrid top k for each question, the same call ``ask`` makes."""
    return [
        search(index, item["question"], k=k, retriever="hybrid", embedder=embedder)
        for item in questions
    ]


def estimate_run(
    questions: list[dict],
    hits: list[list[Hit]],
    *,
    model: str,
    judge_model: str,
) -> dict:
    """Tokens and USD per model before any call. USD is None without a price."""
    generator = Usage()
    judge = Usage()
    claims = ["x" * (TYPICAL_ANSWER_CHARS // TYPICAL_CLAIMS)] * TYPICAL_CLAIMS
    claims_out = _tokens(json.dumps({"claims": claims}))
    verdicts = [{"claim": n, "verdict": "not_found"} for n in range(1, TYPICAL_CLAIMS + 1)]
    verdicts_out = _tokens(json.dumps({"verdicts": verdicts}))
    relevance_out = _tokens(json.dumps({"relevance": 0.5}))
    for item, found in zip(questions, hits, strict=True):
        question = item["question"]
        written, linked = _typical_answer(found)
        generator.add(_tokens(generation_prompt(question, found)), _tokens(written))
        judge.add(_tokens(claims_prompt(question, linked)), claims_out)
        judge.add(_tokens(verify_prompt(claims, found)), verdicts_out)
        judge.add(_tokens(relevance_prompt(question, linked)), relevance_out)
        facts = item.get("key_facts") or []
        if facts:
            covered = [{"fact": n, "covered": False} for n in range(1, len(facts) + 1)]
            judge.add(
                _tokens(coverage_prompt(question, linked, facts)),
                _tokens(json.dumps({"facts": covered})),
            )
    add_reasoning_allowance(generator, model)
    add_reasoning_allowance(judge, judge_model)
    models = {
        "generator": _usage_block(model, generator),
        "judge": _usage_block(judge_model, judge),
    }
    return {
        "n_questions": len(questions),
        "n_with_key_facts": sum(1 for item in questions if item.get("key_facts")),
        "assumptions": {
            "tokens": "characters / 3",
            "answer_chars": TYPICAL_ANSWER_CHARS,
            "citations": TYPICAL_CITATIONS,
            "claims": TYPICAL_CLAIMS,
            "reasoning_tokens_per_call": {
                name: REASONING_TOKENS_PER_CALL.get(name, 0) for name in (model, judge_model)
            },
            "every_answer_judged": True,
        },
        "models": models,
        "total_usd": _total_usd(models),
    }


def render_estimate(estimate: dict, *, split: str, max_usd: float) -> str:
    lines = [
        f"estimate before any call: {estimate['n_questions']} questions, split {split}, "
        f"{estimate['n_with_key_facts']} with key facts",
        "",
        usage_table(estimate["models"], estimate["total_usd"]),
        "",
        (
            f"Tokens are characters / 3. Each answer is taken as {TYPICAL_ANSWER_CHARS} "
            f"characters with {TYPICAL_CITATIONS} citations and {TYPICAL_CLAIMS} claims, "
            "and every answer is judged."
        ),
    ]
    for name, extra in estimate["assumptions"]["reasoning_tokens_per_call"].items():
        if extra:
            lines.append(f"`{name}` output includes {extra} reasoning tokens per call.")
    total = estimate["total_usd"]
    if total is None:
        lines.append("No price for one of the models, so the cap cannot be checked.")
    elif total > max_usd:
        lines.append(f"Above --max-usd {max_usd:.2f}. Refusing to run.")
    else:
        lines.append(f"Under --max-usd {max_usd:.2f}.")
    return "\n".join(lines) + "\n"


def add_reasoning_allowance(usage: Usage, model: str) -> None:
    extra = usage.calls * REASONING_TOKENS_PER_CALL.get(model, 0)
    usage.completion_tokens += extra
    usage.reasoning_tokens += extra


def within_cap(estimate: dict, max_usd: float) -> bool:
    total = estimate["total_usd"]
    return total is not None and total <= max_usd


def run_generation_eval(
    index: CorpusIndex,
    questions: list[dict],
    llm: LLM,
    *,
    judge: LLM,
    embedder: object | None = None,
    k: int = DEFAULT_K,
    hits: list[list[Hit]] | None = None,
) -> dict:
    if hits is None:
        hits = retrieve_all(index, questions, embedder=embedder, k=k)
    rows = []
    for question, found in zip(questions, hits, strict=True):
        answer = ask(index, question["question"], llm, k=k, embedder=embedder, hits=found)
        judged = judge_answer(
            judge,
            answer,
            expect_abstention=not question["answerable"],
            key_facts=question.get("key_facts") or (),
        )
        rows.append(_row(question, answer, judged))
    return _summarize(rows, generator=llm, judge=judge)


def write_generation_report(report: dict, out_dir: Path) -> None:
    """``metrics.json`` and ``summary.md`` hold no answer or guide text.

    ``answers.json`` keeps the full rows and is gitignored.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = report.get("rows")
    public = dict(report)
    if rows is not None:
        public["rows"] = [score_row(row) for row in rows]
        (out_dir / "answers.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (out_dir / "metrics.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(render_generation_report(report), encoding="utf-8")


def score_row(row: dict) -> dict:
    """A row without text: scores, verdict counts, fact coverage flags, page refs."""
    kept = {key: value for key, value in row.items() if key not in TEXT_FIELDS}
    verdicts = [claim["verdict"] for claim in row.get("claims", [])]
    kept["n_claims"] = len(verdicts)
    for name in ("supported", "contradicted", "not_found"):
        kept[f"n_{name}"] = verdicts.count(name)
    kept["facts_covered"] = [item["covered"] for item in row.get("facts", [])]
    return kept


def write_label_sheet(sheet: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "label_sheet.json"
    path.write_text(json.dumps(sheet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def render_generation_report(report: dict) -> str:
    if report.get("status") == "not_run":
        return (
            "# Phase 1b generation evaluation\n\nNot medical advice.\n\n"
            f"Not run. {report.get('reason', '')}\n"
        )
    lines = [
        "# Phase 1b generation evaluation",
        "",
        "Not medical advice. Numbers below come from this run only.",
        "",
        (
            f"Generator: `{report['model']}`. Judge: `{report['judge_model']}`. "
            f"Split: `{report.get('split', 'all')}`. Questions: {report['n_questions']}."
        ),
        f"Mean answer latency: {report['mean_latency_s']:.2f} s.",
        "",
        "## Tokens and cost",
        "",
        usage_table(report["usage"], report["total_usd"]),
        "",
    ]
    estimate = report.get("estimate")
    if estimate and estimate.get("total_usd") is not None:
        lines.extend([f"Estimate before the run: ${estimate['total_usd']:.4f}.", ""])
    lines.extend(
        [
            "USD is the measured tokens times the OpenRouter rate in `generate.py`.",
            "",
            "## Scores",
            "",
            _score_table(report),
            "",
            (
                "Faithfulness is the share of extracted claims the retrieved context supports, "
                "and hallucination rate is 1 minus faithfulness. "
                "Citation correctness is the share of [source p.N] marks that match a retrieved "
                "page. Abstention accuracy is the share of the unanswerable questions that "
                "abstained. A partial answer is not an abstention: those scores still apply, and "
                "a claim that the guides do not cover a point is supported when the retrieved "
                "text never states that point. Relevance is 0.5 when only part of the question "
                "is covered. Completeness is the share of a question's key facts the answer "
                "states, on the questions that have key facts; an abstention covers none. "
                "Faithfulness, citation correctness, relevance, and completeness are means over "
                "answerable questions."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def not_run_report(reason: str) -> dict:
    return {"status": "not_run", "reason": reason}


def build_label_sheet(
    report: dict,
    index: CorpusIndex,
    *,
    size: int = LABEL_SHEET_SIZE,
    seed: int = LABEL_SHEET_SEED,
) -> dict:
    """Answered items for a human to label, stratified by split and origin.

    The judge's claims and verdicts are shown so agreement can be counted.
    Faithfulness, relevance, completeness, and fact coverage are left out.
    """
    chunks = {chunk.chunk_id: chunk for chunk in index.chunks}
    picked = sample_for_labels(report["rows"], size=size, seed=seed)
    strata: dict[str, int] = {}
    for row in picked:
        key = f"{row.get('split') or '-'}/{row['origin']}"
        strata[key] = strata.get(key, 0) + 1
    items = []
    for row in picked:
        items.append(
            {
                "id": row["id"],
                "split": row.get("split"),
                "origin": row["origin"],
                "answerable": row["answerable"],
                "question": row["question"],
                "retrieved": [
                    {
                        "source_id": ref["source_id"],
                        "page": ref["page"],
                        "pdf_page": ref["pdf_page"],
                        "text": chunks[ref["chunk_id"]].text,
                    }
                    for ref in row["retrieved"]
                ],
                "answer": row["answer"],
                "claims": [
                    {
                        "claim": claim["claim"],
                        "judge_verdict": claim["verdict"],
                        "human_verdict": None,
                    }
                    for claim in row["claims"]
                ],
                "key_facts": [{"fact": fact, "human_covered": None} for fact in row["key_facts"]],
                "human_relevance": None,
                "human_notes": "",
            }
        )
    return {
        "generator": report["model"],
        "judge": report["judge_model"],
        "seed": seed,
        "size": len(items),
        "strata": dict(sorted(strata.items())),
        "instructions": (
            "Use only the retrieved text. For each claim set human_verdict to supported, "
            "contradicted, or not_found. For each key fact set human_covered to true when the "
            "answer states it. Set human_relevance to 0, 0.5, or 1. judge_verdict is the "
            "judge's label, kept to count agreement."
        ),
        "items": items,
    }


def sample_for_labels(rows: list[dict], *, size: int, seed: int) -> list[dict]:
    """Answered rows, ``size`` of them, split across (split, origin) by share.

    Quotas use largest remainders. Inside a stratum, rows are ordered by
    ``sha256("<seed>:<id>")``, so the draw does not depend on row order.
    """
    strata: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["abstained"]:
            continue
        strata.setdefault((row.get("split") or "-", row["origin"]), []).append(row)
    available = sum(len(members) for members in strata.values())
    total = min(size, available)
    if total == 0:
        return []
    quotas = {key: total * len(members) // available for key, members in strata.items()}
    remainders = {key: total * len(members) % available for key, members in strata.items()}
    left = total - sum(quotas.values())
    for key in sorted(strata, key=lambda key: (-remainders[key], key))[:left]:
        quotas[key] += 1
    picked = []
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda row: _rank(seed, row["id"]))
        picked.extend(members[: quotas[key]])
    return picked


def _rank(seed: int, row_id: str) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


def _typical_answer(hits: list[Hit]) -> tuple[str, str]:
    """What the generator writes, and the linked text the judge reads.

    Only the length counts, so the prose is filler.
    """
    marks = "".join(
        " " + cite_label(hit.chunk.source_id, hit.chunk.printed_page, hit.chunk.page)
        for hit in hits[:TYPICAL_CITATIONS]
    )
    written = "x" * TYPICAL_ANSWER_CHARS + marks
    return written, link_citations(written, hits)


def _tokens(text: str) -> int:
    return estimate_tokens([text])


def _usage_block(model: str, usage: Usage) -> dict:
    return {
        "model": model,
        "calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "usd": usage.estimated_usd(model),
        "reported_usd": usage.reported_usd,
    }


def _total_usd(models: dict) -> float | None:
    costs = [block["usd"] for block in models.values()]
    if any(cost is None for cost in costs):
        return None
    return sum(costs)


def usage_table(models: dict, total_usd: float | None) -> str:
    """Output tokens include reasoning tokens; the next column shows that part."""

    def usd(value: float | None) -> str:
        return "—" if value is None else f"${value:.4f}"

    def total(key: str) -> int:
        return sum(block.get(key, 0) for block in models.values())

    lines = [
        "| Role | Model | Calls | Input tokens | Output tokens | of which reasoning | USD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for role, block in models.items():
        lines.append(
            f"| {role} | `{block['model']}` | {block['calls']} | {block['prompt_tokens']} | "
            f"{block['completion_tokens']} | {block.get('reasoning_tokens', 0)} | "
            f"{usd(block['usd'])} |"
        )
    lines.append(
        f"| total | | {total('calls')} | {total('prompt_tokens')} | "
        f"{total('completion_tokens')} | {total('reasoning_tokens')} | {usd(total_usd)} |"
    )
    reported = [block.get("reported_usd") for block in models.values()]
    if all(value is not None for value in reported):
        lines.extend(["", f"OpenRouter reported ${sum(reported):.4f} for these calls."])
    return "\n".join(lines)


def _row(question: dict, answer: Answer, judged: dict) -> dict:
    faithfulness = judged["faithfulness"]
    return {
        "id": question["id"],
        "origin": question["origin"],
        "split": question.get("split"),
        "answerable": question["answerable"],
        "question": question["question"],
        "answer": answer.body,
        "abstained": answer.abstained,
        "expect_abstention": not question["answerable"],
        "latency_s": answer.latency_s,
        "faithfulness": faithfulness,
        "hallucination_rate": None if faithfulness is None else 1 - faithfulness,
        "citation_correctness": judged["citation_correctness"],
        "relevance": judged["relevance"],
        "completeness": judged["completeness"],
        "abstention_correct": judged["abstention_correct"],
        "n_citations": judged["n_citations"],
        "claims": judged["claims"],
        "key_facts": list(question.get("key_facts") or []),
        "key_facts_status": question.get("key_facts_status"),
        "facts": judged["facts"],
        "retrieved": [
            {
                "chunk_id": hit.chunk.chunk_id,
                "source_id": hit.chunk.source_id,
                "page": shown_page(hit.chunk),
                "pdf_page": hit.chunk.page,
            }
            for hit in answer.hits
        ],
    }


def _summarize(rows: list[dict], *, generator: LLM, judge: LLM) -> dict:
    usage = {
        "generator": _usage_block(generator.model, generator.usage),
        "judge": _usage_block(judge.model, judge.usage),
    }
    return {
        "status": "ok",
        "model": generator.model,
        "judge_model": judge.model,
        "n_questions": len(rows),
        "mean_latency_s": _mean([row["latency_s"] for row in rows]),
        "usage": usage,
        "total_usd": _total_usd(usage),
        "overall": _block(rows),
        "by_split": {
            name: _block(group)
            for name in SPLITS
            if (group := [row for row in rows if row.get("split") == name])
        },
        "by_origin": {
            name: _block(group)
            for name in ORIGINS
            if (group := [row for row in rows if row["origin"] == name])
        },
        "rows": rows,
    }


def _block(rows: list[dict]) -> dict:
    unanswerable = [row for row in rows if row["expect_abstention"]]
    answerable = [row for row in rows if not row["expect_abstention"]]
    faithfulness = _mean_present(row["faithfulness"] for row in answerable)
    return {
        "n": len(rows),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "n_with_key_facts": sum(1 for row in answerable if row["completeness"] is not None),
        "faithfulness": faithfulness,
        "hallucination_rate": None if faithfulness is None else 1 - faithfulness,
        "citation_correctness": _mean_present(row["citation_correctness"] for row in answerable),
        "relevance": _mean_present(row["relevance"] for row in answerable),
        "completeness": _mean_present(row["completeness"] for row in answerable),
        "abstention_accuracy": abstention_accuracy(unanswerable),
        "false_abstentions": sum(1 for row in answerable if row["abstained"]),
    }


def _score_table(report: dict) -> str:
    def cell(value: float | None) -> str:
        return "—" if value is None else f"{value:.4f}"

    groups = [("all", report["overall"])]
    groups += [(name, block) for name, block in report["by_split"].items()]
    groups += [(name, block) for name, block in report["by_origin"].items()]
    lines = [
        "| Group | n | Faithfulness | Hallucination | Citation | Relevance | Completeness "
        "| Abstention acc. | False abstentions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in groups:
        lines.append(
            f"| {name} | {block['n']} | {cell(block['faithfulness'])} | "
            f"{cell(block['hallucination_rate'])} | {cell(block['citation_correctness'])} | "
            f"{cell(block['relevance'])} | {cell(block['completeness'])} "
            f"(n {block['n_with_key_facts']}) | {cell(block['abstention_accuracy'])} | "
            f"{block['false_abstentions']} |"
        )
    return "\n".join(lines)


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
