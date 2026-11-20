import pymupdf

from baby_food_rag.cli import main
from baby_food_rag.embed import HashEmbedder, embed_passages, embedder_display_name
from baby_food_rag.index import build_index, format_elapsed
from baby_food_rag.types import Source


def test_embedder_display_name_uses_the_e5_model_id():
    assert embedder_display_name("e5") == "intfloat/multilingual-e5-large"
    assert embedder_display_name("hash") == "hash"


def test_elapsed_uses_minutes_after_one_minute():
    assert format_elapsed(12.34) == "12.3 s"
    assert format_elapsed(75) == "1m 15s"


def test_embed_passages_batches_and_draws_a_bar(capsys):
    model = HashEmbedder(dim=32)
    model.batch_size = 2
    seen: list[int] = []
    original = model.embed_documents

    def spy(texts: list[str]):
        seen.append(len(texts))
        return original(texts)

    model.embed_documents = spy
    matrix = embed_passages(model, ["mel na colher de cha"] * 5, progress=True)
    assert matrix.shape == (5, 32)
    assert seen == [2, 2, 1]
    err = capsys.readouterr().err
    assert "Embedding" in err
    assert "chunk" in err


def test_index_prints_stages_and_elapsed(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    pdf = raw / "guia.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), " ".join(["mel"] * 400))
    doc.save(pdf)
    doc.close()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "sources:",
                "  - id: FIX",
                "    title: Fixture",
                "    organ: Test",
                "    year: 2020",
                "    country: BR",
                "    url: https://example.test/guia.pdf",
                "    license: test",
                "    role: core",
                "    download: false",
                "    filename: guia.pdf",
                "",
            ]
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "index",
            "--embedder",
            "hash",
            "--raw",
            str(raw),
            "--out",
            str(tmp_path / "index"),
            "--manifest",
            str(manifest),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    err = captured.err
    assert "Extracting pages..." in err
    assert "FIX" in err
    assert "Chunking..." in err
    assert "Embedding " in err
    assert "hash" in err
    assert "Writing index..." in err
    assert "elapsed " in err
    assert "embedder: hash" in captured.out
    assert (tmp_path / "index" / "chunks.jsonl").is_file()


def test_build_index_message_uses_the_model_id_for_e5(tmp_path, capsys, monkeypatch):
    """The stage line names the model without loading e5 weights."""
    raw = tmp_path / "raw"
    raw.mkdir()
    pdf = raw / "guia.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), " ".join(["uva"] * 40))
    doc.save(pdf)
    doc.close()
    source = Source(
        id="FIX",
        title="Fixture",
        organ="Test",
        year=2020,
        country="BR",
        url="https://example.test/guia.pdf",
        license="test",
        role="core",
        download=False,
        filename="guia.pdf",
    )

    def fake_load(name: str) -> HashEmbedder:
        assert name == "e5"
        return HashEmbedder(dim=32)

    monkeypatch.setattr("baby_food_rag.index.load_embedder", fake_load)
    build_index([source], raw, tmp_path / "out", embedder_name="e5")
    err = capsys.readouterr().err
    assert "Embedding " in err
    assert "intfloat/multilingual-e5-large" in err
