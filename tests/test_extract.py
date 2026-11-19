import pymupdf

from baby_food_rag.extract import (
    Line,
    choose_printed_pages,
    fix_ligatures,
    join_lines,
    order_lines,
    tidy,
)


def test_printed_page_follows_the_running_offset_not_a_figure_digit():
    labels = [""] * 6
    # Printed number is the PDF page plus one. Page 3 also has a figure "1".
    numbers = [[2], [3], [1, 4], [5], [6], [7]]
    chosen = choose_printed_pages(labels, numbers, page_count=6)
    assert chosen == ["2", "3", "4", "5", "6", "7"]


def test_pdf_page_label_is_the_printed_page():
    assert choose_printed_pages(["27"], [[23, 99]], page_count=92) == ["27"]


def test_dehyphenate_and_ligatures():
    lines = [
        Line(40, 80, 200, 92, "A alimentação com-"),
        Line(40, 96, 280, 108, "plementar começa aos seis meses."),
        Line(40, 140, 200, 152, "Cientíﬁ co e reﬂ exo."),
    ]
    text = join_lines(lines)
    assert "complementar começa" in text
    assert "com-" not in text
    assert "Científico" in text
    assert "reflexo" in text
    assert fix_ligatures("diﬀ erent") == "different"


def test_column_order_reads_the_left_column_before_the_right():
    left = [Line(40, 100 + i * 14, 250, 112 + i * 14, f"LEFTALPHA {i}") for i in range(12)]
    left.append(Line(40, 100 + 12 * 14, 250, 112 + 12 * 14, "LEFTEND"))
    right = [Line(320, 100 + i * 14, 540, 112 + i * 14, f"RIGHTBETA {i}") for i in range(12)]
    text = join_lines(order_lines(left + right, 600, 800))
    assert text.index("LEFTEND") < text.index("RIGHTBETA")


def test_synthetic_pdf_strips_header_and_keeps_page(tmp_path):
    path = tmp_path / "fixture.pdf"
    _write_fixture(path)
    from baby_food_rag.extract import extract_pdf

    pages = extract_pdf(path)
    assert len(pages) == 4
    page = pages[1]
    assert page.number == 2
    assert page.printed_page == "2"
    assert "RUNNING HEADER" not in page.text
    assert "complementar" in page.text
    assert page.text.index("LEFTEND") < page.text.index("RIGHTBETA")
    assert tidy(page.text)


def _write_fixture(path):
    doc = pymupdf.open()
    for number in range(1, 5):
        page = doc.new_page(width=600, height=800)
        page.insert_text((40, 28), "RUNNING HEADER")
        page.insert_text((280, 790), str(number))
        if number == 2:
            page.insert_text((40, 90), "A alimentação com-")
            page.insert_text((40, 106), "plementar começa aos seis meses.")
            page.insert_textbox(
                pymupdf.Rect(40, 140, 270, 700),
                " ".join(["LEFTALPHA"] * 80 + ["LEFTEND"]),
            )
            page.insert_textbox(
                pymupdf.Rect(320, 140, 560, 700),
                " ".join(["RIGHTBETA"] * 80),
            )
        else:
            page.insert_text((40, 90), f"Corpo único da página {number} sobre frutas e água.")
    doc.save(path)
    doc.close()
