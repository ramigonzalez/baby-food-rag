import csv
import json

import numpy as np
import pytest
import yaml

from baby_food_rag.compare_embedders import (
    DECISION_RULE,
    apply_rule,
    empty_report,
    measure_latency,
    render_comparison,
    run_comparison,
)
from baby_food_rag.embed import HashEmbedder, l2_normalize
from baby_food_rag.index import save_index
from baby_food_rag.types import Chunk


def _dev_block(ci_low: float, ci_high: float, latency_ms: float) -> dict:
    return {
        "comparisons": {"hybrid": {"ci95_low": ci_low, "ci95_high": ci_high}},
        "embedders": {"te3-large": {"latency": {"median_ms": latency_ms}}},
    }


def test_rule_needs_both_the_interval_and_the_latency():
    adopt = apply_rule(_dev_block(0.01, 0.20, 400.0))
    assert adopt["candidate_becomes_default"]
    assert "becomes the default" in adopt["verdict"]
    slow = apply_rule(_dev_block(0.01, 0.20, 1000.0))
    assert not slow["candidate_becomes_default"]
    assert "e5 stays" in slow["verdict"] and "latency" in slow["verdict"]
    tied = apply_rule(_dev_block(-0.05, 0.20, 400.0))
    assert not tied["candidate_becomes_default"]
    assert "does not exclude 0" in tied["verdict"]
    worse = apply_rule(_dev_block(-0.30, -0.01, 400.0))
    assert not worse["candidate_becomes_default"]


def test_preregistered_file_has_the_rule_and_no_numbers():
    text = render_comparison(empty_report())
    assert DECISION_RULE in text
    assert "## Numbers" in text and "Not run yet" in text
    assert "| e5 |" not in text


def test_latency_skips_the_warmup_and_reports_the_median():
    ticks = iter([0.0, 0.1, 1.0, 1.3, 2.0, 2.2])
    calls = []

    class Recorder:
        def embed_queries(self, texts):
            calls.append(texts)

    latency = measure_latency(Recorder(), ["a", "b", "c"], clock=lambda: next(ticks))
    assert calls == [["a"], ["a"], ["b"], ["c"]]
    assert latency["n"] == 3
    assert latency["median_ms"] == pytest.approx(200.0)


def _chunk(page: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"MS-2019::p{page}::c0",
        source_id="MS-2019",
        year=2019,
        country="BR",
        page=page,
        printed_page=str(page + 1),
        text=text,
        url="https://example.test/guia.pdf",
        title="Fixture",
    )


CHUNKS = [
    _chunk(10, "Não ofereça mel antes dos dois anos."),
    _chunk(20, "Uva: retire as sementes e corte em 2 a 4 pedaços."),
    _chunk(30, "Ovo pode entrar a partir dos seis meses."),
    _chunk(40, "Água no copo durante as refeições."),
]


def _hash_index(path, chunks=CHUNKS):
    matrix = l2_normalize(HashEmbedder().embed_documents([chunk.text for chunk in chunks]))
    meta = {"embedder": "hash", "chunk_words": 180, "chunk_overlap": 40, "sources": []}
    save_index(path, chunks, np.asarray(matrix, dtype=np.float32), meta)
    return path


def _questions(path):
    def item(qid, origin, split, text, page):
        return {
            "id": qid,
            "origin": origin,
            "question": text,
            "answerable": True,
            "gold_status": "verified",
            "split": split,
            "gold": [{"source_id": "MS-2019", "pdf_page": page, "quote": "x"}],
        }

    items = [
        item("a", "real", "dev", "pode dar mel", 10),
        item("b", "synthetic", "dev", "como cortar uva", 20),
        item("c", "plan", "dev", "quando entra o ovo", 30),
        item("d", "synthetic", "test", "água no copo", 40),
        {
            "id": "u",
            "origin": "plan",
            "question": "dose de dipirona",
            "answerable": False,
            "split": "dev",
        },
    ]
    path.write_text(yaml.safe_dump({"questions": items}, allow_unicode=True), encoding="utf-8")
    return path


def test_comparison_writes_dev_then_test_once(tmp_path):
    args = {
        "question_path": _questions(tmp_path / "set.yaml"),
        "baseline_index": _hash_index(tmp_path / "base"),
        "candidate_index": _hash_index(tmp_path / "cand"),
        "out_dir": tmp_path / "out",
    }
    report = run_comparison(split="dev", **args)
    dev = report["splits"]["dev"]
    assert dev["n_answerable"] == 3
    assert set(dev["comparisons"]) == {"dense", "hybrid"}
    hybrid = dev["comparisons"]["hybrid"]
    assert hybrid["diff"] == 0.0 and hybrid["n_bootstrap"] == 10_000 and hybrid["seed"] == 0
    assert dev["embedders"]["e5"]["latency"]["n"] == 3
    assert report["decision"]["candidate_becomes_default"] is False
    text = (tmp_path / "out" / "results.md").read_text(encoding="utf-8")
    assert text.index(DECISION_RULE) < text.index("## Dev")
    assert "e5 stays the default embedder" in text
    assert "| e5 | hybrid |" in text and "| te3-large | dense | real |" in text

    run_comparison(split="test", **args)
    with pytest.raises(ValueError, match="runs once"):
        run_comparison(split="test", **args)
    report = run_comparison(split="test", rerun_test=True, **args)
    assert set(report["splits"]) == {"dev", "test"}
    saved = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["decision_rule"] == DECISION_RULE
    text = (tmp_path / "out" / "results.md").read_text(encoding="utf-8")
    assert text.index("## Dev") < text.index("## Test, held-out confirmation (n = 1)")
    with (tmp_path / "out" / "metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["split"] for row in rows} == {"dev", "test"}
    assert {row["embedder"] for row in rows} == {"e5", "te3-large"}
    assert {"ndcg_at_10", "context_budget_recall", "median_query_latency_ms"} <= set(rows[0])
    with (tmp_path / "out" / "per_question.csv").open(encoding="utf-8") as handle:
        questions = list(csv.DictReader(handle))
    # 4 answerable rows (3 dev + 1 test) x 2 embedders x 2 retrievers x 2 tolerances.
    assert len(questions) == 32


def test_comparison_stops_when_the_chunks_differ(tmp_path):
    other = [*CHUNKS[:3], _chunk(40, "Outro texto na mesma página.")]
    with pytest.raises(ValueError, match="same chunks"):
        run_comparison(
            question_path=_questions(tmp_path / "set.yaml"),
            baseline_index=_hash_index(tmp_path / "base"),
            candidate_index=_hash_index(tmp_path / "cand", other),
            out_dir=tmp_path / "out",
            split="dev",
        )
    assert not (tmp_path / "out").exists()
