import hashlib
import json
import os
import urllib.error

import numpy as np
import pymupdf
import pytest

from baby_food_rag.cli import main
from baby_food_rag.embed import (
    DEFAULT_CACHE_PATH,
    EMBED_TIMEOUT_S,
    OPENROUTER_EMBEDDINGS_URL,
    TE3_LARGE,
    CostCapExceeded,
    EmbeddingCache,
    MissingAPIKey,
    OpenRouterEmbedder,
    estimate_cost,
    estimate_tokens,
    load_embedder,
)
from baby_food_rag.env import load_env
from baby_food_rag.index import build_index
from baby_food_rag.types import Source

DIM = 3072


def _vector(text: str) -> list[float]:
    vector = [0.0] * DIM
    digest = hashlib.sha256(text.encode()).digest()
    vector[int.from_bytes(digest[:4], "little") % DIM] = 1.0
    vector[int.from_bytes(digest[4:8], "little") % DIM] += 0.5
    return vector


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class FakeAPI:
    """Stands in for urlopen. One vector per input, returned out of order."""

    def __init__(self, failures: list[Exception] | None = None) -> None:
        self.requests: list[dict] = []
        self.failures = list(failures or [])

    def __call__(self, request, timeout: float) -> _Response:
        body = json.loads(request.data)
        self.requests.append(
            {
                "url": request.full_url,
                "auth": request.get_header("Authorization"),
                "timeout": timeout,
                "body": body,
            }
        )
        if self.failures:
            raise self.failures.pop(0)
        data = [
            {"object": "embedding", "index": position, "embedding": _vector(text)}
            for position, text in enumerate(body["input"])
        ]
        tokens = 7 * len(body["input"])
        return _Response(
            {
                "object": "list",
                "data": list(reversed(data)),
                "model": body["model"],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }
        )


def _embedder(api: FakeAPI, cache: EmbeddingCache | None = None, **kwargs) -> OpenRouterEmbedder:
    return OpenRouterEmbedder(api_key="test-key", cache=cache, opener=api, **kwargs)


def test_openrouter_embedder_batches_raw_text_and_records_usage():
    api = FakeAPI()
    embedder = _embedder(api, batch_size=2)
    texts = ["mel antes dos 2 anos", "uva cortada", "ovo cozido", "água no copo", "sal"]
    matrix = embedder.embed_documents(texts)
    assert matrix.shape == (5, DIM)
    assert matrix.dtype == np.float32
    for row, text in zip(matrix, texts, strict=True):
        assert np.allclose(row, _vector(text))
    assert [len(item["body"]["input"]) for item in api.requests] == [2, 2, 1]
    assert all(item["url"] == OPENROUTER_EMBEDDINGS_URL for item in api.requests)
    assert all(item["body"]["model"] == TE3_LARGE for item in api.requests)
    assert all(item["timeout"] == EMBED_TIMEOUT_S for item in api.requests)
    assert all(item["auth"] == "Bearer test-key" for item in api.requests)
    sent = [text for item in api.requests for text in item["body"]["input"]]
    assert sent == texts
    embedder.embed_queries(["pode dar mel?"])
    assert api.requests[-1]["body"]["input"] == ["pode dar mel?"]
    assert embedder.name == TE3_LARGE
    assert embedder.dim == DIM
    assert embedder.usage.prompt_tokens == 42
    assert embedder.usage.calls == 4
    assert embedder.usage.usd(0.13) == pytest.approx(42 * 0.13 / 1_000_000)


def test_cache_makes_a_rerun_with_the_same_texts_free(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache" / "embeddings.sqlite")
    texts = ["mel antes dos 2 anos", "uva cortada", "ovo cozido"]
    first_api = FakeAPI()
    first = _embedder(first_api, cache).embed_documents(texts)
    assert len(first_api.requests) == 1
    rerun_api = FakeAPI()
    rerun = _embedder(rerun_api, cache)
    assert rerun.uncached(texts) == []
    again = rerun.embed_documents(texts)
    assert rerun_api.requests == []
    assert rerun.usage.prompt_tokens == 0
    assert rerun.usage.cache_hits == 3
    assert np.array_equal(first, again)
    rerun.embed_queries(["uva cortada", "pipoca"])
    assert [item["body"]["input"] for item in rerun_api.requests] == [["pipoca"]]


def test_cache_key_is_sha256_of_model_and_text(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite")
    assert cache.key(TE3_LARGE, "mel") == hashlib.sha256(f"{TE3_LARGE}\nmel".encode()).hexdigest()
    assert cache.key(TE3_LARGE, "mel") != cache.key("other/model", "mel")
    assert cache.get_many(TE3_LARGE, ["mel"]) == [None]
    assert not cache.path.exists()
    cache.put_many(TE3_LARGE, ["mel"], [np.ones(4, dtype=np.float32)])
    assert cache.get_many("other/model", ["mel"]) == [None]
    assert np.array_equal(cache.get_many(TE3_LARGE, ["mel"])[0], np.ones(4, dtype=np.float32))


def test_openrouter_embedder_retries_once_after_a_timeout():
    api = FakeAPI(failures=[TimeoutError("hung")])
    embedder = _embedder(api)
    assert embedder.embed_queries(["mel"]).shape == (1, DIM)
    assert len(api.requests) == 2
    assert embedder.usage.calls == 1


def test_openrouter_embedder_gives_up_after_one_retry():
    error = urllib.error.HTTPError(OPENROUTER_EMBEDDINGS_URL, 429, "slow down", None, None)
    api = FakeAPI(failures=[error, error])
    with pytest.raises(RuntimeError, match="429"):
        _embedder(api).embed_queries(["mel"])
    assert len(api.requests) == 2


def test_openrouter_embedder_does_not_retry_a_client_error():
    error = urllib.error.HTTPError(OPENROUTER_EMBEDDINGS_URL, 401, "bad key", None, None)
    api = FakeAPI(failures=[error])
    with pytest.raises(RuntimeError) as caught:
        _embedder(api).embed_queries(["mel"])
    assert "401" in str(caught.value)
    assert "test-key" not in str(caught.value)
    assert len(api.requests) == 1


def test_openrouter_embedder_needs_a_key():
    with pytest.raises(MissingAPIKey):
        OpenRouterEmbedder()


def test_load_embedder_accepts_the_model_id(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    embedder = load_embedder(TE3_LARGE)
    assert isinstance(embedder, OpenRouterEmbedder)
    assert embedder.dim == DIM
    assert embedder.cache.path == DEFAULT_CACHE_PATH
    assert not (tmp_path / DEFAULT_CACHE_PATH).exists()
    with pytest.raises(ValueError, match="Unknown embedder"):
        load_embedder("openai/text-embedding-ada-002")


def test_estimate_counts_only_uncached_texts(tmp_path):
    assert estimate_tokens(["abcdef", "g"]) == 3
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite")
    embedder = _embedder(FakeAPI(), cache)
    texts = ["a" * 300, "b" * 300]
    estimate = estimate_cost(embedder, texts)
    assert estimate["estimated_tokens"] == 200
    assert estimate["estimated_usd"] == pytest.approx(200 * 0.13 / 1_000_000)
    embedder.embed_documents(texts[:1])
    assert estimate_cost(embedder, texts)["uncached"] == 1
    assert estimate_cost(object(), texts) is None


def _fixture_pdf(tmp_path) -> tuple:
    raw = tmp_path / "raw"
    raw.mkdir()
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), " ".join(["mel"] * 400))
    doc.save(raw / "guia.pdf")
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
    return raw, source


def test_build_index_prints_an_estimate_and_refuses_above_the_cap(tmp_path, monkeypatch, capsys):
    raw, source = _fixture_pdf(tmp_path)
    api = FakeAPI()
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    monkeypatch.setattr("baby_food_rag.index.load_embedder", lambda name: _embedder(api, cache))
    with pytest.raises(CostCapExceeded, match="cap"):
        build_index([source], raw, tmp_path / "refused", embedder_name=TE3_LARGE, max_usd=0.0)
    assert api.requests == []
    assert not (tmp_path / "refused").exists()
    assert "Estimate: " in capsys.readouterr().err

    index = build_index([source], raw, tmp_path / "out", embedder_name=TE3_LARGE)
    assert index.embeddings.shape[1] == DIM
    assert index.meta["embedder"] == TE3_LARGE
    usage = index.meta["embedding_usage"]
    assert usage["prompt_tokens"] == 7 * len(index.chunks)
    assert usage["usd_per_million"] == 0.13
    calls = len(api.requests)
    rebuilt = build_index([source], raw, tmp_path / "again", embedder_name=TE3_LARGE)
    assert len(api.requests) == calls
    assert rebuilt.meta["embedding_usage"]["prompt_tokens"] == 0
    assert np.array_equal(rebuilt.embeddings, index.embeddings)
    assert "0 of" in capsys.readouterr().err


def test_cli_index_refuses_a_paid_embedder_above_the_cap(tmp_path, monkeypatch, capsys):
    raw, _source = _fixture_pdf(tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "sources:\n  - id: FIX\n    title: Fixture\n    organ: Test\n    year: 2020\n"
        "    country: BR\n    url: https://example.test/guia.pdf\n    license: test\n"
        "    role: core\n    download: false\n    filename: guia.pdf\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    args = ["index", "--embedder", TE3_LARGE, "--raw", str(raw), "--manifest", str(manifest)]
    code = main([*args, "--out", str(tmp_path / "index"), "--max-usd", "0"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Estimate: " in err
    assert "above the $0.00 cap" in err
    assert not (tmp_path / "index").exists()
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert main([*args, "--out", str(tmp_path / "index")]) == 1
    assert "OPENROUTER_API_KEY is not set" in capsys.readouterr().err


def test_load_env_reads_the_repo_root_file_without_overriding(tmp_path, monkeypatch):
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "manifest.yaml").write_text("sources: []\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "BFR_TEST_FROM_FILE=file\nBFR_TEST_ALREADY_SET=file\n", encoding="utf-8"
    )
    (tmp_path / "app").mkdir()
    monkeypatch.chdir(tmp_path / "app")
    monkeypatch.setenv("BFR_TEST_ALREADY_SET", "shell")
    monkeypatch.setenv("BFR_TEST_FROM_FILE", "placeholder")
    monkeypatch.delenv("BFR_TEST_FROM_FILE")
    assert load_env() is True
    assert os.environ["BFR_TEST_FROM_FILE"] == "file"
    assert os.environ["BFR_TEST_ALREADY_SET"] == "shell"
    assert load_env(tmp_path / "app") is False
