"""Retrieval baseline over the labeled question set. No generation.

A retrieved page hits a gold page when the source id matches and the PDF
page is within ``page_tolerance`` of the gold page. Tolerance 1, the
default, counts the exact page or one page either side. Each retrieved
page credits at most one gold page, the closest unmatched one, so two
adjacent gold pages are not both counted from a single hit. Chunks
collapse to pages: the rank of a page is the first time that source and
PDF page appear.

Recall@k is the mean, over answerable questions, of how many of that
question's gold pages are hit in the top k pages, divided by how many
gold pages it has. MRR is the mean of 1/rank of the first hit, or 0 when
there is no hit. nDCG@10 uses binary gains: a page that credits a gold
page, under the same one-credit rule, gains 1/log2(rank + 1); the ideal
puts every gold page first. Unanswerable questions are left out of all of
them.

Context-budget recall is the share of gold pages covered by the pages of
the top 8 chunks, which is what ``generate.ask`` sends to the model. It
calls ``index.search`` with ``k=DEFAULT_K``, the same call ``ask`` makes,
so 8 chunks can cover fewer than 8 pages. ``ask`` uses the hybrid
retriever; the dense and BM25 rows show what the same budget would hold.

Hybrid is reciprocal rank fusion (k 60) of the top 50 BM25 chunks and the
top 50 dense chunks, then collapsed to pages. That pool is wider than the
search command's pool of 20 so recall@10 is not cut off by the pool.

Each question may carry ``split: dev`` or ``split: test`` (see
``golden.py``). ``split="all"``, the default, scores every question, so
earlier result folders stay reproducible. Means are also broken down by
split and by origin.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

from baby_food_rag.embed import l2_normalize
from baby_food_rag.fusion import reciprocal_rank_fusion
from baby_food_rag.generate import DEFAULT_K as CONTEXT_BUDGET_K
from baby_food_rag.golden import SPLITS
from baby_food_rag.index import RETRIEVERS, CorpusIndex, _bm25_tokens, query_embedder, search
from baby_food_rag.types import Chunk

DEFAULT_TOLERANCE = 1
FUSION_POOL = 50
RRF_K = 60
ORIGINS = ("real", "plan", "synthetic")
SPLIT_CHOICES = ("all", *SPLITS)


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


def select_split(questions: list[dict], split: str) -> list[dict]:
    """``all`` keeps every question. ``dev`` or ``test`` keeps that split only."""
    if split not in SPLIT_CHOICES:
        raise ValueError(f"split must be one of {', '.join(SPLIT_CHOICES)}")
    if split == "all":
        return list(questions)
    return [item for item in questions if item.get("split") == split]


def run_eval(
    index: CorpusIndex,
    questions: list[dict],
    *,
    embedder: object | None = None,
    tolerance: int = DEFAULT_TOLERANCE,
    retrievers: tuple[str, ...] = RETRIEVERS,
    split: str = "all",
) -> dict:
    if tolerance < 0:
        raise ValueError("page tolerance must be >= 0")
    questions = select_split(questions, split)
    answerable = [item for item in questions if item["answerable"]]
    if embedder is None and any(name != "bm25" for name in retrievers):
        embedder = query_embedder(index)
    queries = [item["question"] for item in answerable]
    rankings = {name: _rank_all(index, queries, name, embedder) for name in retrievers}
    contexts = {name: _context_pages(index, queries, name, embedder) for name in retrievers}
    by_tolerance = {}
    for tol in sorted({0, tolerance}):
        by_tolerance[str(tol)] = _score(answerable, rankings, contexts, tol)
    return {
        "page_tolerance": tolerance,
        "split": split,
        "hit_rule": (f"same source_id and abs(retrieved pdf_page - gold pdf_page) <= {tolerance}"),
        "fusion_pool": FUSION_POOL,
        "rrf_k": RRF_K,
        "context_budget_k": CONTEXT_BUDGET_K,
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
        f"Split: `{report.get('split', 'all')}`.",
        f"Answerable questions: {report['n_answerable']} of {report['n_questions']}.",
        f"Unanswerable questions are excluded from the means: {report['n_unanswerable']}.",
        "",
        "A hit is a retrieved page with the same `source_id` whose PDF page is",
        f"within {tolerance} of a gold PDF page (exact page or ±{tolerance}).",
        "Chunks from the same source and PDF page count once, at the first time",
        "that page appears. One retrieved page credits one gold page, the closest",
        "one still unmatched, so neighboring gold pages are not both scored from",
        "a single hit. Recall@k is the mean, over answerable questions, of",
        "the share of that question's gold pages hit in the top k pages.",
        "MRR is the mean of 1/rank of the first hit, or 0 when there is no hit.",
        "nDCG@10 uses binary gains on pages with the same one-credit rule.",
        (
            f"Ctx@{report.get('context_budget_k', CONTEXT_BUDGET_K)} is the share of gold "
            "pages covered by the pages of the top "
            f"{report.get('context_budget_k', CONTEXT_BUDGET_K)} chunks, the context "
            "`baby-food ask` sends (hybrid)."
        ),
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
        "### By split",
        "",
        _group_table(report["retrievers"], "split"),
        "",
        "### By origin",
        "",
        _group_table(report["retrievers"], "origin"),
        "",
        "### By split and origin",
        "",
        _group_table(report["retrievers"], "both"),
        "",
        "## Exact page (tolerance 0)",
        "",
        _table(report["by_tolerance"]["0"]["retrievers"]),
        "",
        "### By split",
        "",
        _group_table(report["by_tolerance"]["0"]["retrievers"], "split"),
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
    split = item.get("split")
    if split is not None and split not in SPLITS:
        raise ValueError(f"{item['id']} split must be dev or test, not {split!r}")
    human_verified = item.get("human_verified", False)
    if not isinstance(human_verified, bool):
        raise ValueError(f"{item['id']} human_verified must be true or false")
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
        "split": split,
        "human_verified": human_verified,
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
            ranked.append(collapse_pages(ordered))
        return ranked
    raise ValueError(f"Unknown retriever {retriever!r}")


def _context_pages(
    index: CorpusIndex,
    queries: list[str],
    retriever: str,
    embedder: object | None,
) -> list[list[tuple[str, int]]]:
    """Pages of the top chunks from the same ``search`` call ``generate.ask`` makes."""
    return [
        collapse_pages(
            [
                hit.chunk
                for hit in search(
                    index, query, k=CONTEXT_BUDGET_K, retriever=retriever, embedder=embedder
                )
            ]
        )
        for query in queries
    ]


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


def pages_from_scores(chunks: list[Chunk], scores: np.ndarray) -> list[tuple[str, int]]:
    """Positive-score chunks, best first, collapsed to the first sight of each page."""
    order = sorted(
        range(len(scores)),
        key=lambda i: (-float(scores[i]), chunks[i].chunk_id),
    )
    ordered = [chunks[i] for i in order if float(scores[i]) > 0]
    return collapse_pages(ordered)


def collapse_pages(chunks: list[Chunk]) -> list[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    pages: list[tuple[str, int]] = []
    for chunk in chunks:
        key = (chunk.source_id, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        pages.append(key)
    return pages


def score_page_list(pages: list[tuple[str, int]], gold: list[dict], tolerance: int) -> dict:
    first = None
    for rank, page in enumerate(pages, start=1):
        if _hits_any(page, gold, tolerance):
            first = rank
            break
    return {
        "recall_at_5": _recall(pages[:5], gold, tolerance),
        "recall_at_10": _recall(pages[:10], gold, tolerance),
        "reciprocal_rank": (1.0 / first) if first else 0.0,
        "first_rank": first,
        "ndcg_at_10": ndcg(pages, gold, tolerance, k=10),
    }


def ndcg(pages: list[tuple[str, int]], gold: list[dict], tolerance: int, k: int = 10) -> float:
    """Binary gain at each top-k rank that credits a gold page. Ideal: gold pages first."""
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in _credited_ranks(pages[:k], gold, tolerance))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal


def _pages_from_scores(index: CorpusIndex, scores: np.ndarray) -> list[tuple[str, int]]:
    return pages_from_scores(index.chunks, scores)


def _top_chunk_ids(index: CorpusIndex, scores: np.ndarray) -> list[str]:
    order = sorted(
        range(len(scores)),
        key=lambda i: (-float(scores[i]), index.chunks[i].chunk_id),
    )
    return [index.chunks[i].chunk_id for i in order if float(scores[i]) > 0]


def _score(
    questions: list[dict],
    rankings: dict[str, list[list[tuple[str, int]]]],
    contexts: dict[str, list[list[tuple[str, int]]]],
    tolerance: int,
) -> dict:
    retrievers = []
    per_question = []
    for name, lists in rankings.items():
        rows = [
            _one(question, pages, context, tolerance)
            for question, pages, context in zip(questions, lists, contexts[name], strict=True)
        ]
        per_question.extend({"retriever": name, **row} for row in rows)
        retrievers.append(
            {
                "retriever": name,
                **_summary(rows),
                "groups": group_means(rows),
            }
        )
    return {"retrievers": retrievers, "per_question": per_question}


def group_means(rows: list[dict]) -> list[dict]:
    """Means for every split and origin, with ``all`` for either side. Empty groups are skipped."""
    splits = ["all", *[name for name in SPLITS if any(row.get("split") == name for row in rows)]]
    groups = []
    for split in splits:
        for origin in ("all", *ORIGINS):
            chosen = [
                row
                for row in rows
                if split in ("all", row.get("split")) and origin in ("all", row["origin"])
            ]
            if chosen:
                groups.append({"split": split, "origin": origin, **_summary(chosen)})
    return groups


def _summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "recall_at_5": _mean([row["recall_at_5"] for row in rows]),
        "recall_at_10": _mean([row["recall_at_10"] for row in rows]),
        "mrr": _mean([row["reciprocal_rank"] for row in rows]),
        "ndcg_at_10": _mean([row["ndcg_at_10"] for row in rows]),
        "context_budget_recall": _mean([row["context_budget_recall"] for row in rows]),
    }


def _one(
    question: dict,
    pages: list[tuple[str, int]],
    context: list[tuple[str, int]],
    tolerance: int,
) -> dict:
    scored = score_page_list(pages, question["gold"], tolerance)
    return {
        "id": question["id"],
        "origin": question["origin"],
        "split": question.get("split"),
        **scored,
        "context_budget_recall": _recall(context, question["gold"], tolerance),
    }


def _recall(pages: list[tuple[str, int]], gold: list[dict], tolerance: int) -> float:
    if not gold:
        return 0.0
    return _matched_gold_count(pages, gold, tolerance) / len(gold)


def _matched_gold_count(pages: list[tuple[str, int]], gold: list[dict], tolerance: int) -> int:
    return len(_credited_ranks(pages, gold, tolerance))


def _credited_ranks(pages: list[tuple[str, int]], gold: list[dict], tolerance: int) -> list[int]:
    """Ranks (from 1) of the pages that credit a gold page. Each credits at most one."""
    unmatched = set(range(len(gold)))
    ranks: list[int] = []
    for rank, page in enumerate(pages, start=1):
        choice = None
        choice_key = None
        for index in unmatched:
            if not _hits(page, gold[index], tolerance):
                continue
            distance = abs(page[1] - gold[index]["pdf_page"])
            key = (distance, gold[index]["pdf_page"], index)
            if choice_key is None or key < choice_key:
                choice = index
                choice_key = key
        if choice is not None:
            unmatched.remove(choice)
            ranks.append(rank)
    return ranks


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
        "| Retriever | Recall@5 | Recall@10 | MRR | nDCG@10 | Ctx recall | n |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['retriever']} | {row['recall_at_5']:.4f} | "
            f"{row['recall_at_10']:.4f} | {row['mrr']:.4f} | "
            f"{row['ndcg_at_10']:.4f} | {row['context_budget_recall']:.4f} | {row['n']} |"
        )
    return "\n".join(lines)


def _group_table(rows: list[dict], by: str) -> str:
    """``by`` is ``split``, ``origin``, or ``both``. The all/all row is the table above."""
    lines = [
        "| Retriever | Split | Origin | Recall@5 | Recall@10 | MRR | nDCG@10 | Ctx recall | n |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        for group in row.get("groups", []):
            split_all = group["split"] == "all"
            origin_all = group["origin"] == "all"
            wanted = {
                "split": not split_all and origin_all,
                "origin": split_all and not origin_all,
                "both": not split_all and not origin_all,
            }[by]
            if not wanted:
                continue
            lines.append(
                f"| {row['retriever']} | {group['split']} | {group['origin']} | "
                f"{group['recall_at_5']:.4f} | {group['recall_at_10']:.4f} | "
                f"{group['mrr']:.4f} | {group['ndcg_at_10']:.4f} | "
                f"{group['context_budget_recall']:.4f} | {group['n']} |"
            )
    return "\n".join(lines)


def _write_metrics_csv(report: dict, path: Path) -> None:
    fields = [
        "retriever",
        "page_tolerance",
        "split",
        "origin",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "n",
        "ndcg_at_10",
        "context_budget_recall",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for tolerance, block in report["by_tolerance"].items():
            for row in block["retrievers"]:
                for group in row["groups"]:
                    writer.writerow(
                        {
                            "retriever": row["retriever"],
                            "page_tolerance": tolerance,
                            "split": group["split"],
                            "origin": group["origin"],
                            "recall_at_5": f"{group['recall_at_5']:.6f}",
                            "recall_at_10": f"{group['recall_at_10']:.6f}",
                            "mrr": f"{group['mrr']:.6f}",
                            "n": group["n"],
                            "ndcg_at_10": f"{group['ndcg_at_10']:.6f}",
                            "context_budget_recall": f"{group['context_budget_recall']:.6f}",
                        }
                    )


def _write_questions_csv(report: dict, path: Path) -> None:
    fields = [
        "id",
        "origin",
        "split",
        "retriever",
        "page_tolerance",
        "recall_at_5",
        "recall_at_10",
        "reciprocal_rank",
        "first_rank",
        "ndcg_at_10",
        "context_budget_recall",
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
                        "split": row["split"] or "",
                        "retriever": row["retriever"],
                        "page_tolerance": tolerance,
                        "recall_at_5": f"{row['recall_at_5']:.6f}",
                        "recall_at_10": f"{row['recall_at_10']:.6f}",
                        "reciprocal_rank": f"{row['reciprocal_rank']:.6f}",
                        "first_rank": "" if row["first_rank"] is None else row["first_rank"],
                        "ndcg_at_10": f"{row['ndcg_at_10']:.6f}",
                        "context_budget_recall": f"{row['context_budget_recall']:.6f}",
                    }
                )
