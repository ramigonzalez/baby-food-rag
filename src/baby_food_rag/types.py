"""Records shared by extraction, chunking, and search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    organ: str
    year: int | None
    country: str
    url: str
    license: str
    role: str
    download: bool
    filename: str
    language: str = ""
    sha256: str | None = None
    pages_expected: int | None = None
    notes: str = ""

    @property
    def in_default_index(self) -> bool:
        return self.role == "core"


@dataclass(frozen=True)
class Page:
    """One PDF page after header, footer, and hyphen cleanup.

    ``number`` is the 1-based page in the file. That is the ``N`` in
    ``url#page=N``. ``printed_page`` is the number drawn on the page when
    we can read one. Cover pages make the two differ.
    """

    number: int
    printed_page: str | None
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    year: int
    country: str
    page: int
    printed_page: str | None
    text: str
    url: str
    title: str

    @property
    def citation_url(self) -> str:
        return f"{self.url}#page={self.page}"


@dataclass(frozen=True)
class Hit:
    score: float
    chunk: Chunk
