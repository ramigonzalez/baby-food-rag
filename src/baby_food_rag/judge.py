"""Judge v2 style scores for a generated answer.

Same split as rag-eval-harness judge v2: extract claims that keep the
answer's polarity, then mark each claim supported, contradicted, or
not_found from the retrieved context only. Faithfulness is the share
marked supported. Citation correctness is the share of [SOURCE p.N]
marks that point at a retrieved page. Relevance is a separate 0, 0.5,
or 1 judgment of whether the answer addresses the question.

A partial answer is not an abstention. The metric names stay the same.
Faithfulness, citation correctness, and relevance are computed for it.
Abstention accuracy still uses only the exact abstention sentence on
questions marked answerable: false.

Completeness needs the question's ``key_facts``. The judge reads the
answer, not the context, and marks each fact covered or not. Completeness
is covered facts divided by all facts. An abstention covers none of them,
so it scores 0 without a call. A question with no key facts has no score.

The judge should be a different model family from the generator, because
a model rates its own text higher (Panickssery et al., NeurIPS 2024,
arXiv 2404.13076). The default judge is ``DEFAULT_JUDGE_MODEL``.
"""

from __future__ import annotations

import json

from baby_food_rag.generate import DISCLAIMER, LLM, Answer, citation_marks, cite_label
from baby_food_rag.types import Hit

DEFAULT_JUDGE_MODEL = "anthropic/claude-haiku-5.5"


def same_family(model: str, other: str) -> bool:
    """Same vendor prefix in the OpenRouter id, such as ``openai/``."""
    return model.split("/", 1)[0] == other.split("/", 1)[0]


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
  A claim that the guides do not cover a topic is supported when the
  context does not state that topic, and contradicted when it does.
- contradicted: the context states the opposite.
- not_found: the context does not settle it.
Do not treat a partial answer as an abstention. Score only the claims.
Return JSON only, one entry per claim, same order:
{{"verdicts": [{{"claim": 1, "verdict": "supported"}}]}}

Context:
{context}

Claims:
{claims}
"""

_RELEVANCE = """Task: answer relevance.

Does the answer address the question, without judging medical correctness?
Score 1 if it answers the question, 0.5 if it covers only the part the
guides support and says what they do not cover, 0 if it does not address
the question. An explicit abstention scores 0
when the question could be answered from ordinary feeding guidance, and
1 when the answer correctly says the guides do not cover it. Here, just
score whether the answer is about the question.
Return JSON only: {{"relevance": 1}}

Question: {question}

Answer: {answer}
"""

_COVERAGE = """Task: key fact coverage.

For each numbered key fact, decide whether the answer states it. Use only
the answer. A fact is covered when the answer says the same thing, in any
words, with the same polarity. It is not covered when the answer leaves it
out, says only part of it, or says the opposite.
Return JSON only, one entry per fact, same order:
{{"facts": [{{"fact": 1, "covered": true}}]}}

Question: {question}

Answer: {answer}

Key facts:
{facts}
"""


def judge_answer(
    llm: LLM,
    answer: Answer,
    *,
    expect_abstention: bool,
    key_facts: list[str] | tuple[str, ...] = (),
) -> dict:
    """Score one answer. Abstentions have no claims and skip the claim calls."""
    if answer.abstained:
        return {
            "abstained": True,
            "abstention_correct": expect_abstention,
            "faithfulness": None,
            "citation_correctness": None,
            "relevance": None,
            "completeness": 0.0 if key_facts else None,
            "claims": [],
            "facts": [{"fact": fact, "covered": False} for fact in key_facts],
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
    facts = _coverage(llm, answer.question, answer.text, list(key_facts)) if key_facts else []
    covered = sum(1 for item in facts if item["covered"])
    return {
        "abstained": False,
        "abstention_correct": not expect_abstention,
        "faithfulness": faithfulness,
        "citation_correctness": (correct / len(cites)) if cites else 0.0,
        "relevance": relevance,
        "completeness": (covered / len(facts)) if facts else None,
        "claims": verdicts,
        "facts": facts,
        "n_citations": len(cites),
        "n_citations_correct": correct,
    }


def claims_prompt(question: str, answer: str) -> str:
    return _CLAIMS.format(question=question, answer=_visible(answer))


def verify_prompt(claims: list[str], hits: list[Hit]) -> str:
    numbered = "\n".join(f"{index}. {text}" for index, text in enumerate(claims, start=1))
    blocks = []
    for hit in hits:
        label = cite_label(hit.chunk.source_id, hit.chunk.printed_page, hit.chunk.page)
        blocks.append(f"{label}\n{hit.chunk.text}")
    context = "\n\n".join(blocks)
    return _VERIFY.format(context=context or "(none)", claims=numbered)


def relevance_prompt(question: str, answer: str) -> str:
    return _RELEVANCE.format(question=question, answer=_visible(answer))


def coverage_prompt(question: str, answer: str, facts: list[str]) -> str:
    numbered = "\n".join(f"{index}. {text}" for index, text in enumerate(facts, start=1))
    return _COVERAGE.format(question=question, answer=_visible(answer), facts=numbered)


def _page_retrieved(source_id: str, page: str, hits: list[Hit]) -> bool:
    for hit in hits:
        shown = hit.chunk.printed_page if hit.chunk.printed_page else str(hit.chunk.page)
        if hit.chunk.source_id == source_id and shown == page:
            return True
    return False


def _claims(llm: LLM, question: str, answer: str) -> list[str]:
    payload = _json(llm.complete(claims_prompt(question, answer)))
    claims = payload.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("claims was not a list")
    return [item.strip() for item in claims if isinstance(item, str) and item.strip()]


def _verify(llm: LLM, claims: list[str], hits: list[Hit]) -> list[dict]:
    payload = _json(llm.complete(verify_prompt(claims, hits)))
    verdicts = payload.get("verdicts", [])
    if not isinstance(verdicts, list) or len(verdicts) != len(claims):
        raise ValueError("verdict count does not match claims")
    parsed = []
    for claim, entry in zip(claims, verdicts, strict=True):
        raw = entry.get("verdict", "") if isinstance(entry, dict) else ""
        verdict = str(raw).strip().lower().replace(" ", "_")
        if verdict not in {"supported", "contradicted", "not_found"}:
            raise ValueError(f"bad verdict {verdict!r}")
        parsed.append({"claim": claim, "verdict": verdict})
    return parsed


def _relevance(llm: LLM, question: str, answer: str) -> float:
    payload = _json(llm.complete(relevance_prompt(question, answer)))
    value = payload.get("relevance")
    if value not in {0, 0.5, 1, 0.0, 1.0}:
        raise ValueError(f"bad relevance {value!r}")
    return float(value)


def _coverage(llm: LLM, question: str, answer: str, facts: list[str]) -> list[dict]:
    payload = _json(llm.complete(coverage_prompt(question, answer, facts)))
    entries = payload.get("facts", [])
    if not isinstance(entries, list) or len(entries) != len(facts):
        raise ValueError("coverage count does not match key facts")
    parsed = []
    for fact, entry in zip(facts, entries, strict=True):
        covered = entry.get("covered") if isinstance(entry, dict) else None
        if not isinstance(covered, bool):
            raise ValueError(f"bad coverage {covered!r}")
        parsed.append({"fact": fact, "covered": covered})
    return parsed


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
