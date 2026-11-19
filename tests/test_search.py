from baby_food_rag.embed import HashEmbedder, l2_normalize
from baby_food_rag.index import CorpusIndex, format_hits, save_index, search
from baby_food_rag.tokenize import stem_token, tokenize
from baby_food_rag.types import Chunk

CHUNKS = [
    Chunk(
        chunk_id="FIX::p1::c0",
        source_id="FIX",
        year=2019,
        country="BR",
        page=1,
        printed_page="1",
        text="Pode dar mel a uma criança pequena? Não, por causa do botulismo.",
        url="https://example.test/guia.pdf",
        title="Fixture",
    ),
    Chunk(
        chunk_id="FIX::p2::c0",
        source_id="FIX",
        year=2019,
        country="BR",
        page=2,
        printed_page="2",
        text="Pode dar uva se cortar e tirar as sementes.",
        url="https://example.test/guia.pdf",
        title="Fixture",
    ),
    Chunk(
        chunk_id="FIX::p3::c0",
        source_id="UY",
        year=2017,
        country="UY",
        page=3,
        printed_page="3",
        text="Pode dar água nos intervalos das refeições.",
        url="https://example.test/uy.pdf",
        title="Fixture UY",
    ),
]


def _index() -> CorpusIndex:
    embedder = HashEmbedder(dim=256)
    matrix = l2_normalize(embedder.embed_documents([chunk.text for chunk in CHUNKS]))
    return CorpusIndex(chunks=list(CHUNKS), embeddings=matrix, embedder_name="hash", meta={})


def test_tokenize_keeps_portuguese_accents():
    assert tokenize("Açúcar não") == ["açúcar", "não"]


def test_stem_matches_cortar_and_cortadas_and_keeps_mel():
    assert stem_token("cortar") == stem_token("cortadas")
    assert stem_token("mel") == "mel"
    assert stem_token("dar") == "dar"
    assert stem_token("uva") == "uva"


def test_bm25_ranks_honey_first():
    hits = search(_index(), "pode dar mel?", k=3, retriever="bm25")
    assert hits[0].chunk.page == 1
    assert hits[0].chunk.country == "BR"
    assert "mel" in hits[0].chunk.text
    rendered = format_hits("pode dar mel?", "bm25", hits)
    assert "https://example.test/guia.pdf#page=1" in rendered
    assert "printed_page=1" in rendered
    assert "pdf_page=1" in rendered


def test_dense_hash_ranks_grapes_for_uva():
    hits = search(
        _index(), "como cortar uva", k=3, retriever="dense", embedder=HashEmbedder(dim=256)
    )
    assert hits[0].chunk.page == 2
    assert "sementes" in hits[0].chunk.text


def test_hybrid_returns_the_honey_chunk():
    hits = search(_index(), "pode dar mel", k=2, retriever="hybrid", embedder=HashEmbedder(dim=256))
    assert any(hit.chunk.page == 1 for hit in hits)


def test_save_and_search_roundtrip(tmp_path):
    index = _index()
    save_index(tmp_path, index.chunks, index.embeddings, {"embedder": "hash", "sources": []})
    from baby_food_rag.index import load_index

    loaded = load_index(tmp_path)
    hits = search(loaded, "uva sementes", k=1, retriever="bm25")
    assert hits[0].chunk.citation_url.endswith("#page=2")
