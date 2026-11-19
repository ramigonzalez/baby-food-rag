"""Page-aware recursive chunking.

The splitter follows rag-eval-harness ``RecursiveChunker`` (MIT): try a
coarse separator first, and fall back to a finer one only when a piece is
still over the word budget. Overlap keeps the tail of the previous piece.

Each chunk stays on one PDF page, so the citation page is unambiguous.
The harness chunker concatenates a whole document and cannot do that.
Default budget is 180 words with 40 words of overlap. Guide pages are
longer than the SciFact abstracts that harness used 100/25 on. This is
the phase-1 default, not a measured winner.
"""

from __future__ import annotations

from baby_food_rag.tokenize import word_count
from baby_food_rag.types import Chunk, Page, Source

CHUNK_WORDS = 180
CHUNK_OVERLAP = 40
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def chunk_pages(
    source: Source,
    pages: list[Page],
    *,
    chunk_words: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    if source.year is None:
        raise ValueError(f"{source.id} has no year and cannot be indexed")
    splitter = RecursiveSplitter(chunk_words, overlap)
    chunks: list[Chunk] = []
    for page in pages:
        parts = splitter.split_text(page.text)
        for index, part in enumerate(parts):
            text = part.strip()
            if word_count(text) < 8 and len(text) < 40:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{source.id}::p{page.number}::c{index}",
                    source_id=source.id,
                    year=source.year,
                    country=source.country,
                    page=page.number,
                    printed_page=page.printed_page,
                    text=text,
                    url=source.url,
                    title=source.title,
                )
            )
    return chunks


class RecursiveSplitter:
    def __init__(self, chunk_size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = list(SEPARATORS)

    def split_text(self, text: str) -> list[str]:
        atoms = self._atoms(text.strip(), self.separators)
        if not atoms:
            return []
        chunks: list[str] = []
        current: list[str] = []
        for atom in atoms:
            if not current and word_count(atom) > self.chunk_size:
                chunks.append(atom.strip())
                continue
            if current and word_count(_join(current + [atom])) > self.chunk_size:
                chunks.append(_join(current))
                current = _overlap_tail(current, self.overlap)
                if current and word_count(_join(current + [atom])) > self.chunk_size:
                    current = []
            current.append(atom)
        if current:
            chunks.append(_join(current))
        return [chunk for chunk in chunks if chunk.strip()]

    def _atoms(self, text: str, separators: list[str]) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if word_count(text) <= self.chunk_size:
            return [text]
        separator = next((item for item in separators if item and item in text), None)
        if separator is None:
            return self._hard_words(text)
        rest = separators[separators.index(separator) + 1 :]
        atoms: list[str] = []
        for piece in _split_keep(text, separator):
            if word_count(piece) <= self.chunk_size:
                atoms.append(piece)
            else:
                atoms.extend(self._atoms(piece.strip(), rest))
        return [atom for atom in atoms if atom.strip()]

    def _hard_words(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        size = self.chunk_size
        return [" ".join(words[start : start + size]) for start in range(0, len(words), size)]


def _split_keep(text: str, separator: str) -> list[str]:
    if separator not in text:
        return [text]
    parts = text.split(separator)
    pieces: list[str] = []
    last = len(parts) - 1
    for index, part in enumerate(parts):
        piece = part + separator if index < last else part
        if piece.strip():
            pieces.append(piece)
    return pieces or [text]


def _join(atoms: list[str]) -> str:
    return "".join(atoms).strip()


def _overlap_tail(atoms: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or not atoms:
        return []
    kept: list[str] = []
    total = 0
    for atom in reversed(atoms):
        words = word_count(atom)
        if words == 0:
            continue
        if total + words > overlap:
            break
        kept.append(atom)
        total += words
    kept.reverse()
    return kept
