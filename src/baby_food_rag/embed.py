"""Dense embedders.

The default model is ``intfloat/multilingual-e5-large`` through fastembed
(ONNX, CPU). fastembed 0.7 lists that model and does not list ``bge-m3``.
E5 needs a prefix: ``passage: `` on chunks and ``query: `` on searches.
The wrapper adds both. fastembed's ``query_embed`` does not.

``HashEmbedder`` is a deterministic bag of tokens for tests and CI. It is
not a semantic model. CI never downloads the e5 weights.

``OpenRouterEmbedder`` calls the OpenRouter embeddings endpoint, so far
only for ``openai/text-embedding-3-large``. It sends the raw text, with no
e5 prefix. Every vector it gets back is stored in a sqlite cache keyed by
sha256 of the model id and the text, so a rerun over unchanged chunks
makes no paid call. Tests pass a fake ``opener`` and never hit the network.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from baby_food_rag.tokenize import tokenize

E5_MODEL = "intfloat/multilingual-e5-large"
E5_PASSAGE_PREFIX = "passage: "
E5_QUERY_PREFIX = "query: "

TE3_LARGE = "openai/text-embedding-3-large"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
# Output size and the OpenRouter price in USD per million input tokens.
OPENROUTER_EMBEDDERS = {
    TE3_LARGE: {"dim": 3072, "usd_per_million": 0.13},
}
EMBEDDER_CHOICES = ("e5", "hash", *OPENROUTER_EMBEDDERS)
DEFAULT_CACHE_PATH = Path("data/cache/embeddings.sqlite")
DEFAULT_MAX_USD = 0.50
EMBED_TIMEOUT_S = 60
EMBED_BATCH_SIZE = 64
# Same retry rule as the chat client in generate.py, plus 529 (provider overloaded).
_RETRY_HTTP = {408, 429, 500, 502, 503, 504, 529}


class MissingAPIKey(RuntimeError):
    pass


class CostCapExceeded(RuntimeError):
    pass


class HashEmbedder:
    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise ValueError("hash embedder dim must be >= 8")
        self.name = "hash"
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, bucket] += sign
        return matrix

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)


class E5Embedder:
    def __init__(self, model_name: str = E5_MODEL, batch_size: int = 16) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)
        self.name = model_name
        self.batch_size = batch_size
        self.dim = int(self.model.embedding_size)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._stack(self.model.embed(prefix_passages(texts), batch_size=self.batch_size))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._stack(self.model.embed(prefix_queries(texts), batch_size=self.batch_size))

    def _stack(self, vectors: object) -> np.ndarray:
        rows = [np.asarray(vector, dtype=np.float32) for vector in vectors]
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(rows)


@dataclass
class EmbeddingUsage:
    """Tokens billed, as the API reports them. Cache hits cost nothing."""

    prompt_tokens: int = 0
    calls: int = 0
    texts_sent: int = 0
    cache_hits: int = 0
    reported_usd: float = 0.0

    def usd(self, usd_per_million: float) -> float:
        return self.prompt_tokens * usd_per_million / 1_000_000

    def as_dict(self, usd_per_million: float) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "calls": self.calls,
            "texts_sent": self.texts_sent,
            "cache_hits": self.cache_hits,
            "usd_per_million": usd_per_million,
            "usd": self.usd(usd_per_million),
            "reported_usd": self.reported_usd,
        }


class EmbeddingCache:
    """Vectors in a sqlite file, keyed by sha256 of the model id and the exact text.

    A connection is opened per call, so a cache shared across Streamlit
    threads is safe. Lookups on a missing file do not create it.
    """

    def __init__(self, path: Path = DEFAULT_CACHE_PATH) -> None:
        self.path = Path(path)

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()

    def get_many(self, model: str, texts: list[str]) -> list[np.ndarray | None]:
        if not self.path.is_file():
            return [None] * len(texts)
        found: list[np.ndarray | None] = []
        with closing(self._connect()) as conn:
            for text in texts:
                row = conn.execute(
                    "SELECT vector FROM embeddings WHERE key = ?", (self.key(model, text),)
                ).fetchone()
                found.append(None if row is None else np.frombuffer(row[0], dtype=np.float32))
        return found

    def put_many(self, model: str, texts: list[str], vectors: list[np.ndarray]) -> None:
        rows = []
        for text, vector in zip(texts, vectors, strict=True):
            array = np.asarray(vector, dtype=np.float32)
            rows.append((self.key(model, text), model, int(array.shape[0]), array.tobytes()))
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, model, dim, vector) VALUES (?, ?, ?, ?)",
                rows,
            )

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, "
            "model TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL)"
        )
        return conn


class OpenRouterEmbedder:
    """Hosted embedder. Same interface as E5Embedder, no prefixes.

    Each batch of texts not in the cache is one POST. A call times out after
    60 seconds and is retried once on a timeout or a 408/429/5xx, like the
    chat client in generate.py. ``usage`` adds up the billed tokens.
    """

    def __init__(
        self,
        model: str = TE3_LARGE,
        api_key: str | None = None,
        *,
        cache: EmbeddingCache | None = None,
        batch_size: int = EMBED_BATCH_SIZE,
        timeout_s: float = EMBED_TIMEOUT_S,
        opener: object | None = None,
    ) -> None:
        spec = OPENROUTER_EMBEDDERS.get(model)
        if spec is None:
            raise ValueError(f"Unknown OpenRouter embedder {model!r}")
        key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise MissingAPIKey("OPENROUTER_API_KEY is not set")
        self.name = model
        self.dim = int(spec["dim"])
        self.usd_per_million = float(spec["usd_per_million"])
        self.batch_size = batch_size
        self.cache = cache
        self.usage = EmbeddingUsage()
        self._key = key
        self._timeout_s = timeout_s
        self._opener = opener or urllib.request.urlopen

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def uncached(self, texts: list[str]) -> list[str]:
        """Texts that would be sent, and billed, by the next embed call."""
        if self.cache is None:
            return list(texts)
        found = self.cache.get_many(self.name, texts)
        return [text for text, vector in zip(texts, found, strict=True) if vector is None]

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = self.cache.get_many(self.name, texts) if self.cache else [None] * len(texts)
        missing = [position for position, row in enumerate(rows) if row is None]
        self.usage.cache_hits += len(texts) - len(missing)
        for start in range(0, len(missing), self.batch_size):
            positions = missing[start : start + self.batch_size]
            batch = [texts[position] for position in positions]
            vectors = self._post(batch)
            for position, vector in zip(positions, vectors, strict=True):
                rows[position] = vector
            if self.cache is not None:
                self.cache.put_many(self.name, batch, vectors)
        return np.vstack(rows).astype(np.float32)

    def _post(self, texts: list[str]) -> list[np.ndarray]:
        last: Exception | None = None
        for attempt in range(2):
            try:
                return self._post_once(texts)
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in _RETRY_HTTP or attempt == 1:
                    raise RuntimeError(f"OpenRouter embeddings HTTP {exc.code}") from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                last = exc
                if attempt == 1:
                    raise RuntimeError("OpenRouter embeddings timed out") from exc
        raise RuntimeError("OpenRouter embeddings timed out") from last

    def _post_once(self, texts: list[str]) -> list[np.ndarray]:
        request = urllib.request.Request(
            OPENROUTER_EMBEDDINGS_URL,
            data=json.dumps({"model": self.name, "input": texts}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with self._opener(request, timeout=self._timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        data = body.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            error = body.get("error") or {}
            message = error.get("message") if isinstance(error, dict) else error
            got = len(data) if isinstance(data, list) else 0
            raise RuntimeError(
                f"OpenRouter embeddings returned {got} vectors for {len(texts)} texts"
                + (f": {message}" if message else "")
            )
        ordered = sorted(data, key=lambda item: int(item["index"]))
        vectors = [np.asarray(item["embedding"], dtype=np.float32) for item in ordered]
        if any(vector.shape != (self.dim,) for vector in vectors):
            raise RuntimeError(f"OpenRouter embeddings are not {self.dim}-dimensional")
        usage = body.get("usage") or {}
        tokens = usage.get("prompt_tokens") or usage.get("total_tokens") or 0
        self.usage.prompt_tokens += int(tokens)
        self.usage.reported_usd += float(usage.get("cost") or 0.0)
        self.usage.calls += 1
        self.usage.texts_sent += len(texts)
        return vectors


def estimate_tokens(texts: list[str]) -> int:
    """Rough count before a paid call: characters divided by 3, rounded up."""
    return math.ceil(sum(len(text) for text in texts) / 3)


def estimate_cost(embedder: object, texts: list[str]) -> dict | None:
    """What embedding ``texts`` would cost, before any call. None for a free embedder.

    Only texts not already in the cache count.
    """
    price = getattr(embedder, "usd_per_million", None)
    if price is None:
        return None
    uncached = getattr(embedder, "uncached", None)
    pending = uncached(texts) if uncached else list(texts)
    tokens = estimate_tokens(pending)
    return {
        "texts": len(texts),
        "uncached": len(pending),
        "estimated_tokens": tokens,
        "usd_per_million": price,
        "estimated_usd": tokens * price / 1_000_000,
    }


def prefix_passages(texts: list[str]) -> list[str]:
    """Text sent to e5 for chunks. The model expects the ``passage: `` prefix."""
    return [E5_PASSAGE_PREFIX + text for text in texts]


def prefix_queries(texts: list[str]) -> list[str]:
    """Text sent to e5 for searches. The model expects the ``query: `` prefix."""
    return [E5_QUERY_PREFIX + text for text in texts]


def embedder_display_name(name: str) -> str:
    """Name to show while embedding. ``e5`` is the full model id."""
    if name in {"e5", E5_MODEL}:
        return E5_MODEL
    return name


def embed_passages(
    embedder: HashEmbedder | E5Embedder | OpenRouterEmbedder,
    texts: list[str],
    *,
    progress: bool = False,
) -> np.ndarray:
    """Embed documents in batches. The bar counts chunks, not batches."""
    batch_size = int(getattr(embedder, "batch_size", 16) or 16)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    dim = int(embedder.dim)
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    rows: list[np.ndarray] = []
    bar = None
    if progress:
        bar = tqdm(total=len(texts), desc="Embedding", unit="chunk", file=sys.stderr, leave=True)
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            rows.append(np.asarray(embedder.embed_documents(batch), dtype=np.float32))
            if bar is not None:
                bar.update(len(batch))
    finally:
        if bar is not None:
            bar.close()
    return np.vstack(rows)


def load_embedder(name: str) -> HashEmbedder | E5Embedder | OpenRouterEmbedder:
    if name == "hash":
        return HashEmbedder()
    if name in {"e5", E5_MODEL}:
        return E5Embedder()
    if name in OPENROUTER_EMBEDDERS:
        return OpenRouterEmbedder(name, cache=EmbeddingCache())
    choices = ", ".join(repr(choice) for choice in EMBEDDER_CHOICES)
    raise ValueError(f"Unknown embedder {name!r}. Use one of {choices}.")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)
