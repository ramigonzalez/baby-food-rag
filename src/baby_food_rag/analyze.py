"""Portuguese token options for BM25.

The phase 2 analyzer keeps accents and applies a short suffix list, with
no stopwords. The sweep also tries Snowball Portuguese, accent folding,
and a small closed stopword list. ``pode`` and ``dar`` stay in the index:
in these questions they carry the request, not just grammar.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import snowballstemmer

from baby_food_rag.tokenize import stem_tokens, tokenize

_SNOWBALL = snowballstemmer.stemmer("portuguese")

# Closed list. Not a downloaded corpus.
STOPWORDS = frozenset(
    {
        "a",
        "o",
        "as",
        "os",
        "um",
        "uma",
        "uns",
        "umas",
        "de",
        "da",
        "do",
        "das",
        "dos",
        "e",
        "em",
        "na",
        "no",
        "nas",
        "nos",
        "para",
        "pra",
        "por",
        "com",
        "que",
        "não",
        "nao",
        "se",
        "ao",
        "aos",
        "à",
        "às",
        "ou",
        "como",
        "mais",
        "mas",
        "já",
        "ja",
        "também",
        "tambem",
        "seu",
        "sua",
        "seus",
        "suas",
        "ela",
        "ele",
        "isso",
        "esse",
        "essa",
        "este",
        "esta",
        "isto",
        "pelo",
        "pela",
        "pelos",
        "pelas",
        "sem",
        "sob",
        "sobre",
        "entre",
        "quando",
        "muito",
        "até",
        "ate",
        "depois",
        "antes",
        "lá",
        "la",
        "aqui",
        "mesmo",
        "mesma",
    }
)


@dataclass(frozen=True)
class Analyzer:
    name: str
    stem: str  # "suffix", "snowball", or "none"
    fold: bool
    stop: bool

    def tokens(self, text: str) -> list[str]:
        tokens = tokenize(text)
        if self.fold:
            tokens = [fold_accents(token) for token in tokens]
        # Drop function words before stemming. Snowball maps "comer" to "com",
        # so filtering after the stem would delete the verb.
        if self.stop:
            tokens = [
                token
                for token in tokens
                if token not in STOPWORDS and fold_accents(token) not in STOPWORDS
            ]
        if self.stem == "suffix":
            tokens = stem_tokens(tokens)
        elif self.stem == "snowball":
            tokens = [_SNOWBALL.stemWord(token) for token in tokens]
        elif self.stem != "none":
            raise ValueError(f"Unknown stem {self.stem!r}")
        return tokens


def fold_accents(token: str) -> str:
    decomposed = unicodedata.normalize("NFD", token)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


# Phase 2 BM25: accents kept, suffix stem, no stopwords.
BASELINE_ANALYZER = Analyzer("suffix", stem="suffix", fold=False, stop=False)

BM25_ANALYZERS = (
    BASELINE_ANALYZER,
    Analyzer("plain", stem="none", fold=False, stop=False),
    Analyzer("snowball", stem="snowball", fold=False, stop=False),
    Analyzer("folded", stem="none", fold=True, stop=False),
    Analyzer("snowball-stop", stem="snowball", fold=False, stop=True),
    Analyzer("folded-stop", stem="none", fold=True, stop=True),
)
