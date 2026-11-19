"""Judge v2 style scores for a generated answer.

Same split as rag-eval-harness judge v2: extract claims that keep the
answer's polarity, then mark each claim supported, contradicted, or
not_found from the retrieved context only. Faithfulness is the share
marked supported. Citation correctness is the share of [SOURCE p.N]
marks that point at a retrieved page. Relevance is a separate 0, 0.5,
or 1 judgment of whether the answer addresses the question.
"""

from __future__ import annotations

import json
import re

from baby_food_rag.generate import DISCLAIMER, LLM, Answer, cite_label
from baby_food_rag.types import Hit

_CITE = re.compile(r"\[([A-Za-z0-9-]+) p\.(\d+)\]")

_CLAIMS = """Task: extract claims.

Split the answer into short, self-contained factual claims. Rules:
- Keep the answer's own position, including negations.
- Do not add claims the answer does not make.
- If the answer is only the abstention sentence, return an empty list.
Return JSON only: {{"claims": ["...", "..."]}}

Question: {question}

Answer: {answer}
"""

_VERIFY = """Task: verify claims.

For each numbered claim, decide whether the context supports it. Use only
the context.
- supported: the context states the claim with the same polarity.
- contradicted: the context states the opposite.
- not_found: the context does not settle it.
Return JSON only, one entry per claim, same order:
{{"verdicts": [{{"claim": 1, "verdict": "supported"}}]}}

Context:
{context}

Claims:
{claims}
"""

_RELEVANCE = """Task: answer relevance.

Does the answer address the question, without judging medical correctness?
Score 1 if it answers the question, 0.5 if it answers only part of it,
0 if it does not address the question. An explicit abstention scores 0
when the question could be answered from ordinary feeding guidance, and
1 when the answer correctly says the guides do not cover it. Here, just
score whether the answer is about the question.
Return JSON only: {{"relevance": 1}}

Question: {question}

Answer: {answer}
"""


def judge_answer(llm: LLM, answer: Answer, *, expect_abstention: bool) -> dict:
    """Score one answer. Abstentions have no claims and skip the claim calls."""
    if answer.abstained:
        return {
            "abstained": True,
            "abstention_correct": expect_abstention,
            "faithfulness": None,
            "citation_correctness": None,
            "relevance": None,
            "claims": [],
            "n_citations": 0,
            "n_citations_correct": 0,
        }
    claims = _claims(llm, answer.question, answer.text)
    verdicts = _verify(llm, claims, answer.hits) if claims else []
    supported = sum(1 for item in verdicts if item["verdict"] == "supported")
    faithfulness = (supported / len(verdicts)) if verdicts else None
    cites = citation_marks(answer.text)
    correct = sum(1 for source_id, page in cites if _page_retrieved(source_id, page, answer.hits))
    relevance = _relevance(llm, answer.question, answer.text)
    return {
        "abstained": False,
        "abstention_correct": not expect_abstention,
        "faithfulness": faithfulness,
        "citation_correctness": (correct / len(cites)) if cites else 0.0,
        "relevance": relevance,
        "claims": verdicts,
        "n_citations": len(cites),
        "n_citations_correct": correct,
    }


def citation_marks(text: str) -> list[tuple[str, str]]:
    return _CITE.findall(text)


def _page_retrieved(source_id: str, page: str, hits: list[Hit]) -> bool:
    for hit in hits:
        shown = hit.chunk.printed_page if hit.chunk.printed_page else str(hit.chunk.page)
        if hit.chunk.source_id == source_id and shown == page:
            return True
    return False


def _claims(llm: LLM, question: str, answer: str) -> list[str]:
    payload = _json(llm.complete(_CLAIMS.format(question=question, answer=_visible(answer))))
    claims = payload.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("claims was not a list")
    return [item.strip() for item in claims if isinstance(item, str) and item.strip()]


def _verify(llm: LLM, claims: list[str], hits: list[Hit]) -> list[dict]:
    numbered = "\n".join(f"{index}. {text}" for index, text in enumerate(claims, start=1))
    blocks = []
    for hit in hits:
        label = cite_label(hit.chunk.source_id, hit.chunk.printed_page, hit.chunk.page)
        blocks.append(f"{label}\n{hit.chunk.text}")
    context = "\n\n".join(blocks)
    payload = _json(llm.complete(_VERIFY.format(context=context or "(none)", claims=numbered)))
    verdicts = payload.get("verdicts", [])
    if not isinstance(verdicts, list) or len(verdicts) != len(claims):
        raise ValueError("verdict count does not match claims")
    parsed = []
    for claim, entry in zip(claims, verdicts, strict=True):
        verdict = str(entry.get("verdict", "")).strip().lower().replace(" ", "_")
        if verdict not in {"supported", "contradicted", "not_found"}:
            raise ValueError(f"bad verdict {verdict!r}")
        parsed.append({"claim": claim, "verdict": verdict})
    return parsed


def _relevance(llm: LLM, question: str, answer: str) -> float:
    payload = _json(llm.complete(_RELEVANCE.format(question=question, answer=_visible(answer))))
    value = payload.get("relevance")
    if value not in {0, 0.5, 1, 0.0, 1.0}:
        raise ValueError(f"bad relevance {value!r}")
    return float(value)


def _visible(answer: str) -> str:
    return answer.replace(DISCLAIMER, "").strip()


def _json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model did not return JSON")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON was not an object")
    return payload


def abstention_accuracy(rows: list[dict]) -> float | None:
    """Share of answerable:false rows whose answer abstained."""
    flagged = [row for row in rows if row.get("expect_abstention")]
    if not flagged:
        return None
    return sum(1 for row in flagged if row.get("abstained")) / len(flagged)
