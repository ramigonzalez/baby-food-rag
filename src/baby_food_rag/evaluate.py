"""Retrieval baseline over the labeled question set. No generation.

A retrieved page hits a gold page when the source id matches and the PDF
page is within ``page_tolerance`` of the gold page. Tolerance 1, the
default, counts the exact page or one page either side. Chunks collapse
to pages: the rank of a page is the first time that source and PDF page
appear.

Recall@k is the mean, over answerable questions, of how many of that
question's gold pages are hit in the top k pages, divided by how many
gold pages it has. MRR is the mean of 1/rank of the first hit, or 0 when
there is no hit. Unanswerable questions are left out of both.

Hybrid is reciprocal rank fusion (k 60) of the top 50 BM25 chunks and the
top 50 dense chunks, then collapsed to pages. That pool is wider than the
search command's pool of 20 so recall@10 is not cut off by the pool.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from baby_food_rag.embed import l2_normalize
from baby_food_rag.fusion import reciprocal_rank_fusion
from baby_food_rag.index import RETRIEVERS, CorpusIndex, _bm25_tokens
from baby_food_rag.types import Chunk

DEFAULT_TOLERANCE = 1
FUSION_POOL = 50
RRF_K = 60
KS = (5, 10)


def load_questions(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("questions"), list):
        raise ValueError(f"{path} must contain a questions list")
    questions = []
    seen: set[str] = set()
    for item in raw["questions"]:
        questions.append(_question(item))
        if questions[-1]["id"] in seen:
            raise ValueError(f"Duplicate question id {questions[-1]['id']}")
        seen.add(questions[-1]["id"])
    return questions


def run_eval(
    index: CorpusIndex,
    questions: list[dict],
    *,
    embedder: object | None = None,
    tolerance: int = DEFAULT_TOLERANCE,
    retrievers: tuple[str, ...] = RETRIEVERS,
) -> dict:
    if tolerance < 0:
        raise ValueError("page tolerance must be >= 0")
    answerable = [item for item in questions if item["answerable"]]
    rankings = {
        name: _rank_all(index, [item["question"] for item in answerable], name, embedder)
        for name in retrievers
    }
    by_tolerance = {}
    for tol in sorted({0, tolerance}):
        by_tolerance[str(tol)] = _score(answerable, rankings, tol)
    return {
        "page_tolerance": tolerance,
        "hit_rule": (f"same source_id and abs(retrieved pdf_page - gold pdf_page) <= {tolerance}"),
        "fusion_pool": FUSION_POOL,
        "rrf_k": RRF_K,
        "n_questions": len(questions),
        "n_answerable": len(answerable),
        "n_unanswerable": len(questions) - len(answerable),
        "embedder": index.embedder_name,
        "retrievers": by_tolerance[str(tolerance)]["retrievers"],
        "by_tolerance": by_tolerance,
    }


def write_results(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_metrics_csv(report, out_dir / "metrics.csv")
    _write_questions_csv(report, out_dir / "per_question.csv")
    (out_dir / "results.md").write_text(render_results(report), encoding="utf-8")


def render_results(report: dict) -> str:
    tolerance = report["page_tolerance"]
    lines = [
        "# Phase 2 retrieval baseline",
        "",
        "Not medical advice. This file is retrieval only. No answer was generated.",
        "",
        f"Answerable questions: {report['n_answerable']} of {report['n_questions']}.",
        f"Unanswerable questions are excluded from the means: {report['n_unanswerable']}.",
        "",
        "A hit is a retrieved page with the same `source_id` whose PDF page is",
        f"within {tolerance} of a gold PDF page (exact page or ±{tolerance}).",
        "Chunks from the same source and PDF page count once, at the first time",
        "that page appears. Recall@k is the mean, over answerable questions, of",
        "the share of that question's gold pages hit in the top k pages.",
        "MRR is the mean of 1/rank of the first hit, or 0 when there is no hit.",
        "",
        (
            f"Hybrid fuses the top {report['fusion_pool']} BM25 chunks with the top "
            f"{report['fusion_pool']} dense chunks by reciprocal rank fusion "
            f"(k {report['rrf_k']}). Embedder: `{report['embedder']}`."
        ),
        "",
        f"## Tolerance {tolerance}",
        "",
        _table(report["retrievers"]),
        "",
        "## Exact page (tolerance 0)",
        "",
        _table(report["by_tolerance"]["0"]["retrievers"]),
        "",
        "Gold pages were chosen by an agent reading the extracted text, not by a",
        "health professional. `needs_review` questions are included in these means.",
        "",
    ]
    return "\n".join(lines)


def _question(item: object) -> dict:
    if not isinstance(item, dict):
        raise ValueError("A question must be a mapping")
    for key in ("id", "origin", "question", "answerable"):
        if key not in item:
            raise ValueError(f"Question missing {key}")
    origin = item["origin"]
    if origin not in {"real", "plan", "synthetic"}:
        raise ValueError(f"{item['id']} has origin {origin!r}")
    answerable = bool(item["answerable"])
    gold = item.get("gold") or []
    if not isinstance(gold, list):
        raise ValueError(f"{item['id']} gold must be a list")
    if answerable and not gold:
        raise ValueError(f"{item['id']} is answerable and has no gold pages")
    if not answerable and gold:
        raise ValueError(f"{item['id']} is unanswerable and still has gold pages")
    status = item.get("gold_status")
    if answerable and status not in {"verified", "needs_review"}:
        raise ValueError(f"{item['id']} gold_status must be verified or needs_review")
    pages = []
    for entry in gold:
        if not isinstance(entry, dict):
            raise ValueError(f"{item['id']} gold entry is not a mapping")
        for key in ("source_id", "pdf_page", "quote"):
            if key not in entry:
                raise ValueError(f"{item['id']} gold entry missing {key}")
        quote = str(entry["quote"]).strip()
        if not quote:
            raise ValueError(f"{item['id']} has an empty quote")
        pages.append(
            {
                "source_id": str(entry["source_id"]),
                "pdf_page": int(entry["pdf_page"]),
                "printed_page": entry.get("printed_page"),
                "quote": quote,
            }
        )
    return {
        "id": str(item["id"]),
        "origin": origin,
        "question": str(item["question"]).strip(),
        "answerable": answerable,
        "gold_status": status if answerable else None,
        "notes": str(item.get("notes") or "").strip(),
        "gold": pages,
    }


def _rank_all(
    index: CorpusIndex,
    queries: list[str],
    retriever: str,
    embedder: object | None,
) -> list[list[tuple[str, int]]]:
    if retriever == "bm25":
        return [_pages_from_scores(index, index.bm25.score_all(_bm25_tokens(q))) for q in queries]
    if retriever == "dense":
        matrix = _dense_scores(index, queries, embedder)
        return [_pages_from_scores(index, matrix[row]) for row in range(matrix.shape[0])]
    if retriever == "hybrid":
        dense = _dense_scores(index, queries, embedder)
        by_id = {chunk.chunk_id: chunk for chunk in index.chunks}
        ranked = []
        for row, query in enumerate(queries):
            sparse_ids = _top_chunk_ids(index, index.bm25.score_all(_bm25_tokens(query)))
            dense_ids = _top_chunk_ids(index, dense[row])
            fused = reciprocal_rank_fusion(
                [sparse_ids[:FUSION_POOL], dense_ids[:FUSION_POOL]],
                k=RRF_K,
            )
            ordered = [by_id[chunk_id] for chunk_id, _score in fused]
            ranked.append(_collapse(ordered))
        return ranked
    raise ValueError(f"Unknown retriever {retriever!r}")


def _dense_scores(index: CorpusIndex, queries: list[str], embedder: object | None) -> np.ndarray:
    if index.embeddings.ndim != 2 or index.embeddings.shape[0] != len(index.chunks):
        raise RuntimeError("dense eval needs embeddings aligned with the chunks")
    model = embedder
    if model is None:
        from baby_food_rag.embed import load_embedder

        model = load_embedder(index.embedder_name)
    query_matrix = np.asarray(model.embed_queries(queries), dtype=np.float32)
    query_matrix = l2_normalize(query_matrix)
    return query_matrix @ index.embeddings.T


def _pages_from_scores(index: CorpusIndex, scores: np.ndarray) -> list[tuple[str, int]]:
    order = sorted(
        range(len(scores)),
        key=lambda i: (-float(scores[i]), index.chunks[i].chunk_id),
    )
    ordered = [index.chunks[i] for i in order if float(scores[i]) > 0]
    return _collapse(ordered)


def _top_chunk_ids(index: CorpusIndex, scores: np.ndarray) -> list[str]:
    order = sorted(
        range(len(scores)),
        key=lambda i: (-float(scores[i]), index.chunks[i].chunk_id),
    )
    return [index.chunks[i].chunk_id for i in order if float(scores[i]) > 0]


def _collapse(chunks: list[Chunk]) -> list[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    pages: list[tuple[str, int]] = []
    for chunk in chunks:
        key = (chunk.source_id, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        pages.append(key)
    return pages


def _score(
    questions: list[dict],
    rankings: dict[str, list[list[tuple[str, int]]]],
    tolerance: int,
) -> dict:
    retrievers = []
    per_question = []
    for name, lists in rankings.items():
        recalls = {k: [] for k in KS}
        rr_values = []
        for question, pages in zip(questions, lists, strict=True):
            row = _one(question, pages, tolerance)
            per_question.append({"retriever": name, **row})
            for k in KS:
                recalls[k].append(row[f"recall_at_{k}"])
            rr_values.append(row["reciprocal_rank"])
        retrievers.append(
            {
                "retriever": name,
                "n": len(questions),
                "recall_at_5": _mean(recalls[5]),
                "recall_at_10": _mean(recalls[10]),
                "mrr": _mean(rr_values),
            }
        )
    return {"retrievers": retrievers, "per_question": per_question}


def _one(question: dict, pages: list[tuple[str, int]], tolerance: int) -> dict:
    gold = question["gold"]
    first = None
    for rank, page in enumerate(pages, start=1):
        if _hits_any(page, gold, tolerance):
            first = rank
            break
    row = {
        "id": question["id"],
        "origin": question["origin"],
        "reciprocal_rank": (1.0 / first) if first else 0.0,
        "first_rank": first,
    }
    for k in KS:
        row[f"recall_at_{k}"] = _recall(pages[:k], gold, tolerance)
    return row


def _recall(pages: list[tuple[str, int]], gold: list[dict], tolerance: int) -> float:
    if not gold:
        return 0.0
    hit = sum(1 for entry in gold if any(_hits(page, entry, tolerance) for page in pages))
    return hit / len(gold)


def _hits_any(page: tuple[str, int], gold: list[dict], tolerance: int) -> bool:
    return any(_hits(page, entry, tolerance) for entry in gold)


def _hits(page: tuple[str, int], entry: dict, tolerance: int) -> bool:
    return page[0] == entry["source_id"] and abs(page[1] - entry["pdf_page"]) <= tolerance


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _table(rows: list[dict]) -> str:
    lines = [
        "| Retriever | Recall@5 | Recall@10 | MRR | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['retriever']} | {row['recall_at_5']:.4f} | "
            f"{row['recall_at_10']:.4f} | {row['mrr']:.4f} | {row['n']} |"
        )
    return "\n".join(lines)


def _write_metrics_csv(report: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["retriever", "page_tolerance", "recall_at_5", "recall_at_10", "mrr", "n"],
        )
        writer.writeheader()
        for tolerance, block in report["by_tolerance"].items():
            for row in block["retrievers"]:
                writer.writerow(
                    {
                        "retriever": row["retriever"],
                        "page_tolerance": tolerance,
                        "recall_at_5": f"{row['recall_at_5']:.6f}",
                        "recall_at_10": f"{row['recall_at_10']:.6f}",
                        "mrr": f"{row['mrr']:.6f}",
                        "n": row["n"],
                    }
                )


def _write_questions_csv(report: dict, path: Path) -> None:
    fields = [
        "id",
        "origin",
        "retriever",
        "page_tolerance",
        "recall_at_5",
        "recall_at_10",
        "reciprocal_rank",
        "first_rank",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for tolerance, block in report["by_tolerance"].items():
            for row in block["per_question"]:
                writer.writerow(
                    {
                        "id": row["id"],
                        "origin": row["origin"],
                        "retriever": row["retriever"],
                        "page_tolerance": tolerance,
                        "recall_at_5": f"{row['recall_at_5']:.6f}",
                        "recall_at_10": f"{row['recall_at_10']:.6f}",
                        "reciprocal_rank": f"{row['reciprocal_rank']:.6f}",
                        "first_rank": "" if row["first_rank"] is None else row["first_rank"],
                    }
                )
