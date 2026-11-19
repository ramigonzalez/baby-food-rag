"""Tokenization for BM25.

The harness tokenizer is ``[a-z0-9]+``, which drops Portuguese accents
(``não``, ``açúcar``). This one keeps Unicode letters.
"""

from __future__ import annotations

import re

# Letters (any script) or a digit run. Underscores are not tokens.
_TOKEN = re.compile(r"[0-9]+|[^\W\d_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


def word_count(text: str) -> int:
    return len(text.split())


# Longest suffix first. A stem is kept only when at least 4 characters remain,
# so short words such as "mel", "dar", and "uva" stay whole. This is what makes
# "cortar" meet "cortadas" on the grape page. It is not the harness tokenizer.
_SUFFIXES = (
    "mente",
    "ações",
    "adores",
    "adoras",
    "ando",
    "endo",
    "indo",
    "adas",
    "ados",
    "idas",
    "idos",
    "ções",
    "ada",
    "ado",
    "ida",
    "ido",
    "ares",
    "eres",
    "ires",
    "ar",
    "er",
    "ir",
    "as",
    "os",
    "es",
    "s",
)


def stem_token(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def stem_tokens(tokens: list[str]) -> list[str]:
    return [stem_token(token) for token in tokens]
