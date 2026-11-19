"""Checks against the real PDFs when they are on disk.

CI does not download the guides, so these tests skip there. They use a
few pages called out in the case plan.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from baby_food_rag.extract import extract_pdf

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

MS = RAW / "ms-guia-alimentar-criancas-2019.pdf"
ASBAI = RAW / "asbai-sbp-alergia-alimentar-2025.pdf"
pytestmark = pytest.mark.skipif(not MS.is_file(), reason="MS 2019 PDF is not downloaded")


def test_ms_2019_printed_page_84_is_the_grape_note():
    pages = extract_pdf(MS)
    page = next(item for item in pages if item.printed_page == "84")
    text = page.text.lower()
    assert "uva" in text
    assert "semente" in text
    assert "2 a 4" in text
    assert "guia alimentar para crianças brasileiras menores de 2 anos" not in text
    # The link uses the PDF page, which is not the printed number.
    assert page.number != 84


def test_ms_2019_printed_page_104_is_honey():
    pages = extract_pdf(MS)
    page = next(item for item in pages if item.printed_page == "104")
    text = page.text.lower()
    assert "mel" in text
    assert "2 anos" in text
    assert "botulismo" in text


@pytest.mark.skipif(not ASBAI.is_file(), reason="ASBAI/SBP 2025 PDF is not downloaded")
def test_asbai_order_sentence_is_not_split_across_columns():
    pages = extract_pdf(ASBAI)
    needle = (
        "não há evidências de que a ordem de introdução dos diversos alimentos "
        "sólidos esteja relacionada"
    )
    matches = [page for page in pages if needle in page.text.lower()]
    assert len(matches) == 1
    assert "lactentes" in matches[0].text.lower()
