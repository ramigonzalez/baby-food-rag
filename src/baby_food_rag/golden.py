"""Golden set helpers: lexical overlap and the fixed dev/test split.

Overlap is the share of a question's content tokens that also appear in
its gold quotes. Content tokens use the phase 2 BM25 tokenizer (accents
kept, suffix stem), after dropping the closed stopword list in
``analyze.py`` and tokens shorter than 3 characters. ``mel``, ``ovo``,
and ``sal`` stay. A synthetic question written while reading the page
tends to reuse its words, so this number is checked against the real
parent questions.

The split is stratified by ``(answerable, origin, batch, first gold
source)``. The batch is the id prefix (``real``, ``plan``, ``syn``,
``syn2``, ``unans2``), so the first and second synthetic batches are each
spread over both splits. Inside a stratum, questions are ordered by
``sha256("<seed>:<id>")``, so the order does not depend on the file.
Strata are then walked in sorted
order with one running counter per answerable flag, and a question goes
to dev whenever that keeps the dev count at ceil(3/5 of the questions
seen). Every stratum ends up near 60/40 and the total is 60% dev.
"""

from __future__ import annotations

import hashlib

from baby_food_rag.analyze import STOPWORDS, fold_accents
from baby_food_rag.tokenize import stem_token, tokenize

SPLITS = ("dev", "test")
SPLIT_SEED = 20261127
DEV_SHARE = (3, 5)
MIN_TOKEN = 3


def content_tokens(text: str) -> set[str]:
    kept = set()
    for token in tokenize(text):
        if len(token) < MIN_TOKEN or token in STOPWORDS or fold_accents(token) in STOPWORDS:
            continue
        kept.add(stem_token(token))
    return kept


def quote_overlap(question: str, quotes: list[str]) -> float:
    """Share of the question's content tokens found in the quotes. 0 for an empty question."""
    asked = content_tokens(question)
    if not asked:
        return 0.0
    found = {stem_token(token) for quote in quotes for token in tokenize(quote)}
    return len(asked & found) / len(asked)


def split_stratum(question: dict) -> tuple[bool, str, str, str]:
    source = question["gold"][0]["source_id"] if question["gold"] else "-"
    batch = question["id"].split("-", 1)[0]
    return (bool(question["answerable"]), question["origin"], batch, source)


def assign_splits(questions: list[dict], seed: int = SPLIT_SEED) -> dict[str, str]:
    """Question id to ``dev`` or ``test``. Deterministic for a given seed and id set."""
    strata: dict[tuple[bool, str, str, str], list[dict]] = {}
    for question in questions:
        strata.setdefault(split_stratum(question), []).append(question)
    num, den = DEV_SHARE
    seen = {True: 0, False: 0}
    chosen: dict[str, str] = {}
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda item: _rank(seed, item["id"]))
        for question in members:
            index = seen[key[0]]
            before = (index * num + den - 1) // den
            after = ((index + 1) * num + den - 1) // den
            chosen[question["id"]] = "dev" if after > before else "test"
            seen[key[0]] = index + 1
    return chosen


def _rank(seed: int, question_id: str) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()
