import json
from pathlib import Path

import pytest
import yaml

from baby_food_rag.embed import HashEmbedder, l2_normalize, prefix_passages, prefix_queries
from baby_food_rag.evaluate import _recall, load_questions, run_eval, write_results
from baby_food_rag.index import CorpusIndex
from baby_food_rag.types import Chunk

ROOT = Path(__file__).resolve().parents[1]


def _chunk(source: str, page: int, text: str, index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{source}::p{page}::c{index}",
        source_id=source,
        year=2019,
        country="BR",
        page=page,
        printed_page=str(page),
        text=text,
        url="https://example.test/guia.pdf",
        title="Fixture",
    )


def _index(chunks: list[Chunk]) -> CorpusIndex:
    embedder = HashEmbedder(dim=64)
    matrix = l2_normalize(embedder.embed_documents([chunk.text for chunk in chunks]))
    return CorpusIndex(
        chunks=chunks,
        embeddings=matrix,
        embedder_name="hash",
        meta={"embedder": "hash"},
    )


def test_one_retrieved_page_credits_only_one_adjacent_gold():
    pages = [("MS-2019", 102)]
    gold = [
        {"source_id": "MS-2019", "pdf_page": 102, "quote": "a"},
        {"source_id": "MS-2019", "pdf_page": 103, "quote": "b"},
    ]
    assert _recall(pages, gold, tolerance=1) == 0.5
    assert _recall([("MS-2019", 102), ("MS-2019", 104)], gold, tolerance=1) == 1.0


def test_e5_prefix_helpers_add_query_and_passage():
    assert prefix_queries(["como cortar uva"]) == ["query: como cortar uva"]
    assert prefix_passages(["uva e acerola"]) == ["passage: uva e acerola"]


def test_recall_counts_neighbor_page_and_skips_unanswerable(tmp_path):
    chunks = [
        _chunk("MS-2019", 10, "Pode dar mel antes dos dois anos por causa do botulismo."),
        _chunk("MS-2019", 11, "Outra página sobre frutas da estação e variação."),
        _chunk("MS-2019", 40, "Pipoca não entra antes dos dois anos."),
    ]
    questions = [
        {
            "id": "hit",
            "origin": "synthetic",
            "question": "pode dar mel",
            "answerable": True,
            "gold_status": "verified",
            "gold": [{"source_id": "MS-2019", "pdf_page": 9, "quote": "mel"}],
        },
        {
            "id": "miss",
            "origin": "synthetic",
            "question": "pipoca no lanche",
            "answerable": True,
            "gold_status": "verified",
            "gold": [{"source_id": "MS-2019", "pdf_page": 40, "quote": "Pipoca"}],
        },
        {
            "id": "out",
            "origin": "plan",
            "question": "qual a dose de dipirona",
            "answerable": False,
            "gold": [],
        },
    ]
    report = run_eval(
        _index(chunks),
        load_questions_from(questions),
        embedder=HashEmbedder(dim=64),
        tolerance=1,
        retrievers=("bm25",),
    )
    assert report["n_answerable"] == 2
    row = report["retrievers"][0]
    # Gold page 9 hits retrieved page 10 when tolerance is 1. Both questions
    # have one gold page, and both are in the top 5, so recall is 1.
    assert row["recall_at_5"] == 1.0
    exact = run_eval(
        _index(chunks),
        load_questions_from(questions),
        embedder=HashEmbedder(dim=64),
        tolerance=0,
        retrievers=("bm25",),
    )
    per = {item["id"]: item for item in exact["by_tolerance"]["0"]["per_question"]}
    assert per["hit"]["recall_at_5"] == 0.0
    assert per["miss"]["recall_at_5"] == 1.0
    assert per["miss"]["reciprocal_rank"] == 1.0 or per["miss"]["first_rank"] == 1
    write_results(report, tmp_path)
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["page_tolerance"] == 1
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "results.md").is_file()


def test_labeled_set_shape():
    root = ROOT
    questions = load_questions(root / "eval" / "set.yaml")
    assert 100 <= len(questions) <= 130
    origins = {}
    for item in questions:
        origins[item["origin"]] = origins.get(item["origin"], 0) + 1
    assert origins["real"] == 10
    assert origins["plan"] == 15
    assert origins["synthetic"] >= 80
    unanswerable = [item for item in questions if not item["answerable"]]
    assert 10 <= len(unanswerable) <= 20
    answerable = [item for item in questions if item["answerable"]]
    assert all(item["split"] in {"dev", "test"} for item in questions)
    assert all(isinstance(item["human_verified"], bool) for item in questions)
    dev_share = sum(item["split"] == "dev" for item in answerable) / len(answerable)
    assert 0.55 <= dev_share <= 0.65
    for origin in ("real", "plan", "synthetic"):
        splits = {item["split"] for item in answerable if item["origin"] == origin}
        assert splits == {"dev", "test"}, origin
    reais = yaml.safe_load((root / "eval" / "perguntas_reais.yaml").read_text(encoding="utf-8"))
    plan = yaml.safe_load((root / "eval" / "questions.yaml").read_text(encoding="utf-8"))
    texts = {item["question"] for item in questions}
    for text in reais["questions"]:
        assert text in texts
    for item in plan["questions"]:
        assert item["question"] in texts
    known = {"MS-2019", "ASBAI-SBP-2025", "SBP-MAN-2024", "SBP-BLW-2017"}
    for item in questions:
        if not item["answerable"]:
            assert item["gold"] == []
            continue
        assert item["gold_status"] in {"verified", "needs_review"}
        for gold in item["gold"]:
            assert gold["source_id"] in known
            assert gold["pdf_page"] >= 1
            assert gold["quote"]


def test_loader_checks_split_and_human_verified():
    base = {
        "id": "x",
        "origin": "synthetic",
        "question": "pode dar mel",
        "answerable": True,
        "gold_status": "verified",
        "gold": [{"source_id": "MS-2019", "pdf_page": 2, "quote": "mel"}],
    }
    loaded = load_questions_from([{**base, "split": "test", "human_verified": True}])
    assert loaded[0]["split"] == "test"
    assert loaded[0]["human_verified"] is True
    assert load_questions_from([base])[0]["split"] is None
    with pytest.raises(ValueError, match="split"):
        load_questions_from([{**base, "split": "train"}])
    with pytest.raises(ValueError, match="human_verified"):
        load_questions_from([{**base, "human_verified": "no"}])


def test_split_filter_and_breakdown(tmp_path):
    chunks = [
        _chunk("MS-2019", 10, "Pode dar mel antes dos dois anos por causa do botulismo."),
        _chunk("MS-2019", 40, "Pipoca não entra antes dos dois anos."),
    ]

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

    questions = load_questions_from(
        [
            item("a", "real", "dev", "pode dar mel", 10),
            item("b", "synthetic", "test", "pipoca no lanche", 40),
            item("c", "synthetic", "dev", "pipoca no lanche", 10),
        ]
    )
    common = {"embedder": HashEmbedder(dim=64), "tolerance": 0, "retrievers": ("bm25",)}
    dev = run_eval(_index(chunks), questions, split="dev", **common)
    assert dev["split"] == "dev"
    assert dev["n_answerable"] == 2
    with pytest.raises(ValueError, match="split"):
        run_eval(_index(chunks), questions, split="train", **common)
    report = run_eval(_index(chunks), questions, **common)
    groups = {(g["split"], g["origin"]): g for g in report["retrievers"][0]["groups"]}
    assert groups[("all", "all")]["n"] == 3
    assert groups[("dev", "all")]["n"] == 2
    assert groups[("test", "synthetic")]["recall_at_5"] == 1.0
    assert groups[("dev", "synthetic")]["recall_at_5"] == 0.0
    assert ("test", "real") not in groups
    write_results(report, tmp_path)
    text = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "### By split" in text
    assert "| bm25 | dev | real |" in text
    header = (tmp_path / "per_question.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("id,origin,split,")


def load_questions_from(items: list[dict]) -> list:
    import tempfile

    path = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump({"questions": items}, path, allow_unicode=True)
    path.close()
    return load_questions(__import__("pathlib").Path(path.name))


def test_first_hit_mrr_uses_page_rank_not_repeated_chunks():
    chunks = [
        _chunk("MS-2019", 1, "pode dar água no copo", 0),
        _chunk("MS-2019", 1, "pode dar água de novo", 1),
        _chunk("MS-2019", 2, "pode dar mel à criança", 0),
    ]
    questions = load_questions_from(
        [
            {
                "id": "mel",
                "origin": "synthetic",
                "question": "pode dar água",
                "answerable": True,
                "gold_status": "verified",
                "gold": [{"source_id": "MS-2019", "pdf_page": 2, "quote": "mel"}],
            }
        ]
    )
    report = run_eval(
        _index(chunks),
        questions,
        embedder=HashEmbedder(dim=64),
        tolerance=0,
        retrievers=("bm25",),
    )
    row = report["by_tolerance"]["0"]["per_question"][0]
    assert row["first_rank"] == 2
    assert row["reciprocal_rank"] == 0.5


@pytest.mark.skipif(
    not (ROOT / "data" / "index" / "chunks.jsonl").is_file(),
    reason="local index is not in CI",
)
def test_gold_quotes_are_on_the_cited_page():
    import json

    pages: dict[tuple[str, int], str] = {}
    for line in (ROOT / "data" / "index" / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        key = (chunk["source_id"], chunk["page"])
        pages[key] = pages.get(key, "") + "\n" + chunk["text"]
    for item in load_questions(ROOT / "eval" / "set.yaml"):
        for gold in item["gold"]:
            text = pages.get((gold["source_id"], gold["pdf_page"]), "")
            assert gold["quote"] in text, f"{item['id']} {gold['source_id']} p{gold['pdf_page']}"
