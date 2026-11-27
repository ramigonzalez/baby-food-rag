"""Build a BM25 + dense index and search it.

Chunks are stored as JSONL under the output directory. Embeddings are a
float32 matrix. BM25 is rebuilt from the chunk text on load, which is
cheap at this corpus size. The index directory is gitignored: it is a
copy of the guides.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from baby_food_rag.bm25 import BM25Index
from baby_food_rag.chunking import CHUNK_OVERLAP, CHUNK_WORDS, chunk_pages
from baby_food_rag.embed import embed_passages, embedder_display_name, l2_normalize, load_embedder
from baby_food_rag.extract import extract_pdf
from baby_food_rag.fusion import reciprocal_rank_fusion
from baby_food_rag.tokenize import stem_tokens, tokenize
from baby_food_rag.types import Chunk, Hit, Page, Source

RETRIEVERS = ("bm25", "dense", "hybrid")
_FUSION_POOL = 50
_RRF_K = 60


@dataclass
class CorpusIndex:
    chunks: list[Chunk]
    embeddings: np.ndarray
    embedder_name: str
    meta: dict

    def __post_init__(self) -> None:
        self._bm25: BM25Index | None = None

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index([_bm25_tokens(chunk.text) for chunk in self.chunks])
        return self._bm25


def build_index(
    sources: list[Source],
    raw_dir: Path,
    out_dir: Path,
    *,
    embedder_name: str = "e5",
) -> CorpusIndex:
    missing = [source.id for source in sources if not (raw_dir / source.filename).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing PDFs for "
            + ", ".join(missing)
            + f". Run the download command first (looked in {raw_dir})."
        )
    _stage("Extracting pages...")
    extracted: list[tuple[Source, list[Page]]] = []
    for source in sources:
        _stage(f"  {source.id}")
        extracted.append((source, extract_pdf(raw_dir / source.filename)))
    _stage("Chunking...")
    chunks: list[Chunk] = []
    stats: list[dict] = []
    for source, pages in extracted:
        page_chunks = chunk_pages(source, pages)
        chunks.extend(page_chunks)
        with_text = sum(1 for page in pages if page.text.strip())
        stats.append(
            {
                "id": source.id,
                "role": source.role,
                "year": source.year,
                "country": source.country,
                "pdf_pages": len(pages),
                "pages_with_text": with_text,
                "chunks": len(page_chunks),
            }
        )
    label = embedder_display_name(embedder_name)
    _stage(f"Embedding {len(chunks)} chunks with {label}...")
    embedder = load_embedder(embedder_name)
    matrix = embed_passages(embedder, [chunk.text for chunk in chunks], progress=True)
    matrix = l2_normalize(np.asarray(matrix, dtype=np.float32))
    meta = {
        "embedder": embedder.name,
        "chunk_words": CHUNK_WORDS,
        "chunk_overlap": CHUNK_OVERLAP,
        "sources": stats,
    }
    _stage("Writing index...")
    save_index(out_dir, chunks, matrix, meta)
    return CorpusIndex(chunks=chunks, embeddings=matrix, embedder_name=embedder.name, meta=meta)


def format_elapsed(seconds: float) -> str:
    """Short duration for the end of ``baby-food index``."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}m {rest:02d}s"


def _stage(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def save_index(out_dir: Path, chunks: list[Chunk], embeddings: np.ndarray, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(_chunk_dict(chunk), ensure_ascii=False) + "\n")
    np.save(out_dir / "embeddings.npy", embeddings)
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_index(out_dir: Path) -> CorpusIndex:
    meta_path = out_dir / "meta.json"
    chunks_path = out_dir / "chunks.jsonl"
    if not meta_path.is_file() or not chunks_path.is_file():
        raise FileNotFoundError(f"No index in {out_dir}. Run the index command first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks = [
        _chunk_from(json.loads(line))
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    embeddings_path = out_dir / "embeddings.npy"
    embeddings = np.load(embeddings_path) if embeddings_path.is_file() else np.zeros((0, 0))
    return CorpusIndex(
        chunks=chunks,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        embedder_name=str(meta.get("embedder", "")),
        meta=meta,
    )


def query_embedder(index: CorpusIndex):
    """Load the query embedder once. Callers reuse it across questions.

    A hash index does not need a shared model. ``search`` builds a cheap
    hash embedder itself when this returns None.
    """
    if index.embedder_name in {"", "hash"}:
        return None
    return load_embedder(index.embedder_name)


def search(
    index: CorpusIndex,
    query: str,
    k: int = 5,
    retriever: str = "bm25",
    embedder: object | None = None,
) -> list[Hit]:
    if k < 1:
        raise ValueError("k must be >= 1")
    if retriever not in RETRIEVERS:
        raise ValueError(f"Unknown retriever {retriever!r}")
    if not index.chunks or not query.strip():
        return []
    if retriever == "bm25":
        return _hits_from_scores(index, index.bm25.score_all(_bm25_tokens(query)), k)
    if retriever == "dense":
        return _dense(index, query, k, embedder)
    return _hybrid(index, query, k, embedder)


def format_hits(query: str, retriever: str, hits: list[Hit]) -> str:
    lines = [f"query: {query}", f"retriever: {retriever}", f"hits: {len(hits)}", ""]
    for rank, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        printed = chunk.printed_page if chunk.printed_page else "-"
        lines.append(
            f"{rank}. score={hit.score:.4f}  {chunk.source_id}  year={chunk.year}  "
            f"country={chunk.country}  printed_page={printed}  pdf_page={chunk.page}"
        )
        lines.append(f"   {chunk.citation_url}")
        lines.append(f"   {snippet(chunk.text, query)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_SNIPPET_SKIP = {
    "como",
    "para",
    "pode",
    "dar",
    "uma",
    "com",
    "que",
    "não",
    "por",
    "dos",
    "das",
}


def snippet(text: str, query: str, width: int = 360) -> str:
    """Window the chunk around a query word so the preview is not only the lead sentence."""
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    words = set(tokenize(flat))
    anchors = [
        token
        for token in tokenize(query)
        if len(token) >= 3 and token not in _SNIPPET_SKIP and token in words
    ]
    if not anchors:
        return flat[: width - 3] + "..."
    anchor = max(anchors, key=len)
    match = re.search(rf"(?<!\w){re.escape(anchor)}(?!\w)", flat, flags=re.IGNORECASE)
    pos = match.start() if match else 0
    start = max(0, pos - width // 3)
    end = min(len(flat), start + width)
    start = max(0, end - width)
    prefix = "..." if start else ""
    suffix = "..." if end < len(flat) else ""
    return prefix + flat[start:end].strip() + suffix


def _bm25_tokens(text: str) -> list[str]:
    return stem_tokens(tokenize(text))


def format_stats(meta: dict) -> str:
    lines = [
        f"embedder: {meta.get('embedder')}",
        f"chunk_words: {meta.get('chunk_words')}  overlap: {meta.get('chunk_overlap')}",
        "",
        f"{'source':<22} {'role':<10} {'country':<8} "
        f"{'pdf_pages':>9} {'with_text':>9} {'chunks':>8}",
    ]
    total_pages = 0
    total_chunks = 0
    for row in meta.get("sources", []):
        lines.append(
            f"{row['id']:<22} {row['role']:<10} {row['country']:<8} "
            f"{row['pdf_pages']:>9} {row['pages_with_text']:>9} {row['chunks']:>8}"
        )
        total_pages += int(row["pdf_pages"])
        total_chunks += int(row["chunks"])
    lines.append(f"{'total':<22} {'':<10} {'':<8} {total_pages:>9} {'':>9} {total_chunks:>8}")
    return "\n".join(lines) + "\n"


def _dense(index: CorpusIndex, query: str, k: int, embedder: object | None) -> list[Hit]:
    if index.embeddings.ndim != 2 or index.embeddings.shape[0] != len(index.chunks):
        raise RuntimeError("dense search needs embeddings.npy aligned with the chunks")
    model = embedder or load_embedder(index.embedder_name)
    query_matrix = np.asarray(model.embed_queries([query]), dtype=np.float32)
    query_matrix = l2_normalize(query_matrix)
    scores = index.embeddings @ query_matrix[0]
    return _hits_from_scores(index, scores, k)


def _hybrid(index: CorpusIndex, query: str, k: int, embedder: object | None) -> list[Hit]:
    pool = max(k, _FUSION_POOL)
    sparse = _hits_from_scores(index, index.bm25.score_all(_bm25_tokens(query)), pool)
    dense = _dense(index, query, pool, embedder)
    fused = reciprocal_rank_fusion(
        [[hit.chunk.chunk_id for hit in sparse], [hit.chunk.chunk_id for hit in dense]],
        k=_RRF_K,
    )
    by_id = {chunk.chunk_id: chunk for chunk in index.chunks}
    hits: list[Hit] = []
    for chunk_id, score in fused[:k]:
        hits.append(Hit(score=score, chunk=by_id[chunk_id]))
    return hits


def _hits_from_scores(index: CorpusIndex, scores: np.ndarray, k: int) -> list[Hit]:
    order = sorted(
        range(len(scores)),
        key=lambda i: (-float(scores[i]), index.chunks[i].chunk_id),
    )
    hits: list[Hit] = []
    for position in order:
        score = float(scores[position])
        if score <= 0:
            break
        hits.append(Hit(score=score, chunk=index.chunks[position]))
        if len(hits) == k:
            break
    return hits


def _chunk_dict(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "year": chunk.year,
        "country": chunk.country,
        "page": chunk.page,
        "printed_page": chunk.printed_page,
        "text": chunk.text,
        "url": chunk.url,
        "title": chunk.title,
    }


def _chunk_from(row: dict) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        source_id=row["source_id"],
        year=int(row["year"]),
        country=row["country"],
        page=int(row["page"]),
        printed_page=row.get("printed_page"),
        text=row["text"],
        url=row["url"],
        title=row.get("title", ""),
    )
