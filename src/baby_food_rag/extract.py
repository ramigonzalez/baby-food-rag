"""Extract one string per PDF page.

``Page.number`` is the 1-based page in the file, so a citation can be
``url#page=N`` and open that sheet. ``printed_page`` is the page label,
or the number drawn in the margin when the file has no label. In the
2019 MS guide those differ: the grape note is printed page 84 and PDF
page 83.

Running headers and footers are short lines in the top or bottom 12% of
the page whose text, with digits folded, repeats across the file. A line
that is only a page number is dropped too. Words split with a hyphen at
the end of a line are joined when the next line continues in lowercase.
Two-column pages are read down the left column, then the right, so a
sentence is not mixed with the other column. The ASBAI/SBP 2025 article
is that layout.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from baby_food_rag.types import Page

_EDGE = 0.12
_ONLY_NUMBER = re.compile(r"^[0-9]{1,4}$|^[ivxlcdm]{1,8}$", re.IGNORECASE)
_LIGATURES = (
    ("ﬃ", "ffi"),
    ("ﬄ", "ffl"),
    ("ﬁ", "fi"),
    ("ﬂ", "fl"),
    ("ﬀ", "ff"),
)


@dataclass(frozen=True)
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


def extract_pdf(path: Path) -> list[Page]:
    import pymupdf

    document = pymupdf.open(path)
    raw: list[list[Line]] = []
    heights: list[float] = []
    widths: list[float] = []
    labels: list[str] = []
    for page in document:
        raw.append(_page_lines(page))
        heights.append(float(page.rect.height))
        widths.append(float(page.rect.width))
        labels.append((page.get_label() or "").strip())
    repeated = _repeated_edge_templates(raw, heights)
    page_count = len(raw)
    numbers = [
        _edge_numbers(lines, height, page_count) for lines, height in zip(raw, heights, strict=True)
    ]
    printed_pages = choose_printed_pages(labels, numbers, page_count)
    pages: list[Page] = []
    for number, lines in enumerate(raw, start=1):
        height = heights[number - 1]
        kept = [line for line in lines if not _is_running_furniture(line, height, repeated)]
        ordered = order_lines(kept, widths[number - 1], height)
        pages.append(
            Page(number=number, printed_page=printed_pages[number - 1], text=join_lines(ordered))
        )
    return pages


def order_lines(lines: list[Line], width: float, height: float) -> list[Line]:
    """Reading order. Two columns are read left, then right, between full-width lines."""
    if not lines:
        return []
    if not _is_two_column(lines, width):
        return sorted(lines, key=lambda line: (line.y0, line.x0))
    mid = width / 2.0
    full: list[Line] = []
    left: list[Line] = []
    right: list[Line] = []
    for line in lines:
        if _spans_both_columns(line, width):
            full.append(line)
        elif (line.x0 + line.x1) / 2.0 < mid:
            left.append(line)
        else:
            right.append(line)
    full.sort(key=lambda line: line.y0)
    cuts = [0.0] + [(line.y0 + line.y1) / 2.0 for line in full] + [height + 1.0]
    ordered: list[Line] = []
    for index, line in enumerate(full):
        ordered.extend(_band(left, cuts[index], cuts[index + 1]))
        ordered.extend(_band(right, cuts[index], cuts[index + 1]))
        ordered.append(line)
    ordered.extend(_band(left, cuts[-2], cuts[-1]))
    ordered.extend(_band(right, cuts[-2], cuts[-1]))
    return ordered


def join_lines(lines: list[Line]) -> str:
    """Join wrapped lines. A backward jump starts a new paragraph (next column)."""
    if not lines:
        return ""
    paragraphs: list[list[str]] = [[lines[0].text]]
    previous = lines[0]
    for line in lines[1:]:
        gap = line.y0 - previous.y1
        jumped = line.y0 < previous.y0 - 1.0
        line_height = max(previous.y1 - previous.y0, 8.0)
        if jumped or gap > max(6.0, line_height * 0.55):
            paragraphs.append([line.text])
        else:
            paragraphs[-1].append(line.text)
        previous = line
    text = "\n\n".join(part for part in (_glue(parts) for parts in paragraphs) if part)
    return tidy(text)


def tidy(text: str) -> str:
    text = fix_ligatures(text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_ligatures(text: str) -> str:
    """Turn a split ligature such as ``Cientíﬁ co`` back into ``Científico``."""
    for ligature, replacement in _LIGATURES:
        text = re.sub(re.escape(ligature) + r"\s*", replacement, text)
    return text


def _glue(parts: list[str]) -> str:
    merged = parts[0].strip()
    for part in parts[1:]:
        nxt = part.strip()
        if not nxt:
            continue
        if merged.endswith("-") and nxt[0].islower():
            merged = merged[:-1] + nxt
        elif merged.endswith("-") and nxt.startswith("-"):
            merged = merged[:-1] + nxt
        else:
            merged = f"{merged} {nxt}"
    return merged.strip()


def _page_lines(page: object) -> list[Line]:
    data = page.get_text("dict")  # type: ignore[attr-defined]
    lines: list[Line] = []
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        for raw in block["lines"]:
            text = "".join(span.get("text", "") for span in raw["spans"])
            text = _clean_span(text)
            if not text:
                continue
            x0, y0, x1, y1 = raw["bbox"]
            lines.append(Line(float(x0), float(y0), float(x1), float(y1), text))
    return lines


def _clean_span(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "").replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def _is_edge(line: Line, height: float) -> bool:
    if height <= 0:
        return False
    return line.y0 < height * _EDGE or line.y1 > height * (1.0 - _EDGE)


def _template(text: str) -> str:
    folded = text.lower()
    folded = re.sub(r"\d+", "#", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _repeated_edge_templates(pages: list[list[Line]], heights: list[float]) -> set[str]:
    counts: dict[str, int] = {}
    for lines, height in zip(pages, heights, strict=True):
        seen: set[str] = set()
        for line in lines:
            if not _is_edge(line, height):
                continue
            key = _template(line.text)
            if not key or (len(key) < 4 and key != "#"):
                continue
            if key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
    if not pages:
        return set()
    threshold = max(3, math.ceil(0.2 * len(pages)))
    return {key for key, count in counts.items() if count >= threshold}


def _is_running_furniture(line: Line, height: float, repeated: set[str]) -> bool:
    if not _is_edge(line, height):
        return False
    if _ONLY_NUMBER.match(line.text):
        return True
    return _template(line.text) in repeated


def _edge_numbers(lines: list[Line], height: float, page_count: int) -> list[int]:
    numbers: list[int] = []
    for line in lines:
        if not _is_edge(line, height) or not line.text.isdigit():
            continue
        if _acceptable_label(line.text, page_count):
            numbers.append(int(line.text))
    return numbers


def choose_printed_pages(
    labels: list[str],
    numbers: list[list[int]],
    page_count: int,
) -> list[str | None]:
    """Pick the printed page from the running offset, not from a figure digit.

    Most pages of a guide differ from the PDF page by a fixed amount because
    of cover sheets. A figure step such as "1" or "8" also sits in the margin.
    The offset that fits the most pages wins, and a page keeps the candidate
    that matches it. A PDF page label, when the file has one, is used as printed.
    """
    offsets: list[int] = []
    for index, (label, nums) in enumerate(zip(labels, numbers, strict=True), start=1):
        if label.isdigit() and _acceptable_label(label, page_count):
            offsets.append(int(label) - index)
            continue
        close = [num for num in nums if abs(num - index) <= 25]
        if close:
            offsets.append(min(close, key=lambda num: abs(num - index)) - index)
    mode: int | None = None
    if offsets:
        winner, votes = Counter(offsets).most_common(1)[0]
        if votes >= max(3, math.ceil(0.4 * len(offsets))):
            mode = winner
    chosen: list[str | None] = []
    for index, (label, nums) in enumerate(zip(labels, numbers, strict=True), start=1):
        if _acceptable_label(label, page_count):
            chosen.append(label)
            continue
        if mode is not None and (index + mode) in nums:
            chosen.append(str(index + mode))
            continue
        close = [num for num in nums if abs(num - index) <= 25]
        chosen.append(str(close[0]) if len(set(close)) == 1 else None)
    return chosen


def _acceptable_label(label: str, page_count: int) -> bool:
    if not label or not _ONLY_NUMBER.match(label):
        return False
    if label.isdigit() and int(label) > page_count + 30:
        return False
    return True


def _is_two_column(lines: list[Line], width: float) -> bool:
    left = right = 0
    for line in lines:
        if _spans_both_columns(line, width):
            continue
        if (line.x0 + line.x1) / 2.0 < width / 2.0:
            left += 1
        else:
            right += 1
    return left >= 8 and right >= 8


def _spans_both_columns(line: Line, width: float) -> bool:
    mid = width / 2.0
    margin = width * 0.08
    return line.x0 < mid - margin and line.x1 > mid + margin


def _band(lines: list[Line], y_lo: float, y_hi: float) -> list[Line]:
    chosen = [line for line in lines if y_lo <= (line.y0 + line.y1) / 2.0 < y_hi]
    chosen.sort(key=lambda line: (line.y0, line.x0))
    return chosen
