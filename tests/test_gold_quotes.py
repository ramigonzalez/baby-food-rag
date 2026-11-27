"""Every verified quote in eval/set.yaml is on its gold PDF page.

CI does not download the guides, so this skips there, like test_guides.py.
Whitespace is normalized on both sides because the extractor joins lines.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pytest

from baby_food_rag.evaluate import load_questions
from baby_food_rag.extract import extract_pdf
from baby_food_rag.manifest import load_manifest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "sources" / "manifest.yaml"
FILES = {source.id: RAW / source.filename for source in load_manifest(MANIFEST)}
QUESTIONS = load_questions(ROOT / "eval" / "set.yaml")
GOLD_SOURCES = {gold["source_id"] for item in QUESTIONS for gold in item["gold"]}
MISSING = sorted(source for source in GOLD_SOURCES if not FILES[source].is_file())

pytestmark = pytest.mark.skipif(bool(MISSING), reason=f"PDFs not downloaded: {MISSING}")


def _squash(text: str) -> str:
    return " ".join(text.split())


@cache
def _pages(source_id: str) -> dict[int, str]:
    return {page.number: _squash(page.text) for page in extract_pdf(FILES[source_id])}


def test_verified_quotes_are_on_the_gold_pdf_page():
    missing = []
    for item in QUESTIONS:
        if item["gold_status"] != "verified":
            continue
        for gold in item["gold"]:
            page = _pages(gold["source_id"]).get(gold["pdf_page"], "")
            if _squash(gold["quote"]) not in page:
                missing.append(f"{item['id']} {gold['source_id']} p{gold['pdf_page']}")
    assert not missing, missing
