"""Dense embedders.

The default model is ``intfloat/multilingual-e5-large`` through fastembed
(ONNX, CPU). fastembed 0.7 lists that model and does not list ``bge-m3``.
E5 needs a prefix: ``passage: `` on chunks and ``query: `` on searches.
The wrapper adds both. fastembed's ``query_embed`` does not.

``HashEmbedder`` is a deterministic bag of tokens for tests and CI. It is
not a semantic model. CI never downloads the e5 weights.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from baby_food_rag.tokenize import tokenize

E5_MODEL = "intfloat/multilingual-e5-large"
E5_PASSAGE_PREFIX = "passage: "
E5_QUERY_PREFIX = "query: "


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
        prefixed = [E5_PASSAGE_PREFIX + text for text in texts]
        return self._stack(self.model.embed(prefixed, batch_size=self.batch_size))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        prefixed = [E5_QUERY_PREFIX + text for text in texts]
        return self._stack(self.model.embed(prefixed, batch_size=self.batch_size))

    def _stack(self, vectors: object) -> np.ndarray:
        rows = [np.asarray(vector, dtype=np.float32) for vector in vectors]
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(rows)


def load_embedder(name: str) -> HashEmbedder | E5Embedder:
    if name == "hash":
        return HashEmbedder()
    if name in {"e5", E5_MODEL}:
        return E5Embedder()
    raise ValueError(f"Unknown embedder {name!r}. Use 'e5' or 'hash'.")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)
