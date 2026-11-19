from baby_food_rag.chunking import RecursiveSplitter, chunk_pages
from baby_food_rag.types import Page, Source

SOURCE = Source(
    id="MS-2019",
    title="Guia",
    organ="Ministério da Saúde",
    year=2019,
    country="BR",
    url="https://example.test/guia.pdf",
    license="CC BY-NC-SA 4.0",
    role="core",
    download=True,
    filename="guia.pdf",
)


def test_chunks_stay_on_one_page_and_cite_it():
    pages = [
        Page(1, "10", "onlyalpha " * 80),
        Page(2, "11", "onlybeta " * 80),
    ]
    chunks = chunk_pages(SOURCE, pages, chunk_words=40, overlap=10)
    assert chunks
    assert all(chunk.source_id == "MS-2019" for chunk in chunks)
    assert all(chunk.year == 2019 and chunk.country == "BR" for chunk in chunks)
    assert {chunk.page for chunk in chunks} == {1, 2}
    assert all(not ("onlyalpha" in chunk.text and "onlybeta" in chunk.text) for chunk in chunks)
    first = next(chunk for chunk in chunks if chunk.page == 1)
    assert first.printed_page == "10"
    assert first.citation_url == "https://example.test/guia.pdf#page=1"
    assert first.chunk_id.startswith("MS-2019::p1::")


def test_recursive_splitter_respects_the_word_budget():
    sentence = "A criança pode comer fruta madura sem açúcar. "
    text = sentence * 30
    parts = RecursiveSplitter(chunk_size=40, overlap=10).split_text(text)
    assert len(parts) > 1
    assert all(len(part.split()) <= 40 for part in parts)
    # Overlap keeps some words in the next chunk.
    assert parts[0].split()[-1] in parts[1]


def test_short_page_is_one_chunk():
    chunks = chunk_pages(SOURCE, [Page(3, "12", "Mel não é oferecido antes dos dois anos.")])
    assert len(chunks) == 1
    assert chunks[0].page == 3
    assert chunks[0].text.startswith("Mel")
