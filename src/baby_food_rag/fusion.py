"""Reciprocal rank fusion.

Adapted from rag-eval-harness (MIT):
https://github.com/ramigonzalez/rag-eval-harness/blob/main/src/raglab/retrieval/fusion.py

``score(d) = sum over lists of 1 / (k + rank(d))`` with ranks starting at 1.
Ties break on the chunk id.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_ids: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    if k < 0:
        raise ValueError("rrf k must be >= 0")
    scores: dict[str, float] = {}
    for ids in ranked_ids:
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
