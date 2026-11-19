from baby_food_rag.embed import HashEmbedder, l2_normalize
from baby_food_rag.generate import (
    ABSTAIN,
    DISCLAIMER,
    Usage,
    ask,
    is_abstention,
    link_citations,
)
from baby_food_rag.generate_eval import (
    not_run_report,
    render_generation_report,
    run_generation_eval,
)
from baby_food_rag.index import CorpusIndex
from baby_food_rag.judge import abstention_accuracy, citation_marks, judge_answer
from baby_food_rag.types import Chunk, Hit


class FakeLLM:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._usage = Usage()
        self.model = "fake"

    @property
    def usage(self) -> Usage:
        return self._usage

    def complete(self, prompt: str) -> str:
        self._usage.add(12, 4)
        return self._replies.pop(0)


def _chunk(page: int, printed: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"MS-2019::p{page}::c0",
        source_id="MS-2019",
        year=2019,
        country="BR",
        page=page,
        printed_page=printed,
        text=text,
        url="https://example.test/guia.pdf",
        title="Guia",
    )


def _index() -> CorpusIndex:
    chunks = [
        _chunk(103, "104", "Não é recomendado oferecer mel à criança menor de 2 anos."),
        _chunk(83, "84", "Uva: retirar as sementes e cortar em 2 a 4 pedaços."),
    ]
    matrix = l2_normalize(HashEmbedder(dim=32).embed_documents([c.text for c in chunks]))
    return CorpusIndex(chunks=chunks, embeddings=matrix, embedder_name="hash", meta={"sources": []})


def test_citation_link_uses_printed_label_and_pdf_page():
    hit = Hit(score=1.0, chunk=_chunk(83, "84", "uva"))
    linked = link_citations("Corte a uva [MS-2019 p.84].", [hit])
    assert "[MS-2019 p.84](https://example.test/guia.pdf#page=83)" in linked


def test_ask_abstains_and_adds_the_disclaimer():
    llm = FakeLLM([ABSTAIN])
    answer = ask(_index(), "qual a dose de dipirona", llm, k=1, embedder=HashEmbedder(dim=32))
    assert answer.abstained
    assert is_abstention(answer.text)
    assert answer.body.endswith(DISCLAIMER)
    assert "pediatra" in DISCLAIMER


def test_ask_links_a_grounded_citation():
    llm = FakeLLM(["Não ofereça mel antes dos 2 anos [MS-2019 p.104]."])
    answer = ask(_index(), "pode dar mel", llm, k=2, embedder=HashEmbedder(dim=32))
    assert not answer.abstained
    assert "https://example.test/guia.pdf#page=103" in answer.text
    assert answer.body.endswith(DISCLAIMER)


def test_judge_scores_a_supported_claim_and_a_matching_page():
    chunk = _chunk(103, "104", "Não é recomendado oferecer mel à criança menor de 2 anos.")
    answer_text = "Não ofereça mel antes dos 2 anos [MS-2019 p.104]."
    from baby_food_rag.generate import Answer

    answer = Answer(
        question="pode dar mel",
        text=answer_text,
        hits=[Hit(score=1.0, chunk=chunk)],
        abstained=False,
        model="fake",
        latency_s=0.1,
    )
    llm = FakeLLM(
        [
            '{"claims": ["Não ofereça mel antes dos 2 anos"]}',
            '{"verdicts": [{"claim": 1, "verdict": "supported"}]}',
            '{"relevance": 1}',
        ]
    )
    judged = judge_answer(llm, answer, expect_abstention=False)
    assert judged["faithfulness"] == 1.0
    assert judged["citation_correctness"] == 1.0
    assert judged["relevance"] == 1.0
    assert citation_marks(answer_text) == [("MS-2019", "104")]


def test_abstention_accuracy_on_unanswerable_rows():
    rows = [
        {"expect_abstention": True, "abstained": True},
        {"expect_abstention": True, "abstained": False},
    ]
    assert abstention_accuracy(rows) == 0.5


def test_generation_eval_with_a_fake_model_writes_no_live_cost():
    questions = [
        {
            "id": "q",
            "origin": "real",
            "question": "pode dar mel",
            "answerable": True,
            "gold_status": "verified",
            "gold": [{"source_id": "MS-2019", "pdf_page": 103, "quote": "mel"}],
            "notes": "",
        }
    ]
    llm = FakeLLM(
        [
            "Não ofereça mel [MS-2019 p.104].",
            '{"claims": ["Não ofereça mel"]}',
            '{"verdicts": [{"claim": 1, "verdict": "supported"}]}',
            '{"relevance": 1}',
        ]
    )
    report = run_generation_eval(_index(), questions, llm, k=1, embedder=HashEmbedder(dim=32))
    assert report["status"] == "ok"
    assert report["overall"]["faithfulness"] == 1.0
    assert report["real"]["n"] == 1
    assert report["estimated_usd"] is None
    missing = render_generation_report(not_run_report("OPENROUTER_API_KEY is not set"))
    assert "Not run" in missing
    assert "0.99" not in missing
