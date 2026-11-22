import json
import urllib.error

from baby_food_rag.embed import HashEmbedder, l2_normalize
from baby_food_rag.generate import (
    ABSTAIN,
    CALL_TIMEOUT_S,
    DISCLAIMER,
    EXAMPLE_QUESTIONS,
    OPENROUTER_URL,
    OpenRouterLLM,
    Usage,
    answer_segments,
    append_successful_turn,
    ask,
    button_width_kwargs,
    drop_abstain_if_cited,
    include_retrieved_passages,
    is_abstention,
    link_citations,
    normalize_citations,
    question_to_send,
    run_question,
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


class _Boom(FakeLLM):
    def complete(self, prompt: str) -> str:
        raise RuntimeError("boom")


def test_failed_ask_does_not_keep_the_question():
    messages: list[dict] = []
    try:
        append_successful_turn(
            messages,
            "pode dar mel",
            _index(),
            _Boom([]),
            k=1,
            embedder=HashEmbedder(dim=32),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("the failing model should raise")
    assert messages == []


def test_successful_turn_appends_question_and_answer_together():
    llm = FakeLLM(["Não ofereça mel [MS-2019 p.104]."])
    messages = append_successful_turn(
        [],
        "pode dar mel",
        _index(),
        llm,
        k=1,
        embedder=HashEmbedder(dim=32),
    )
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "pode dar mel"
    assert "pediatra" in messages[1]["content"]
    assert "p.104" in messages[1]["passages"]


def test_retry_resends_the_failed_question_unchanged():
    failed = "Pode dar mel ao bebê de 8 meses?"
    assert question_to_send(None, failed, retry=True) == failed
    assert question_to_send("  outra pergunta  ", failed, retry=True) == "outra pergunta"
    assert question_to_send(None, failed, retry=False) is None


def test_generation_eval_reuses_the_embedder():
    class Counting(HashEmbedder):
        def __init__(self) -> None:
            super().__init__(dim=32)
            self.calls = 0

        def embed_queries(self, texts: list[str]):
            self.calls += 1
            return super().embed_queries(texts)

    questions = [
        {
            "id": "a",
            "origin": "real",
            "question": "pode dar mel",
            "answerable": False,
            "gold_status": "verified",
            "gold": [],
            "notes": "",
        },
        {
            "id": "b",
            "origin": "plan",
            "question": "uva",
            "answerable": False,
            "gold_status": "verified",
            "gold": [],
            "notes": "",
        },
    ]
    embedder = Counting()
    run_generation_eval(_index(), questions, FakeLLM([ABSTAIN, ABSTAIN]), k=1, embedder=embedder)
    assert embedder.calls == 2


def test_ask_eval_loads_the_embedder_once(monkeypatch, tmp_path):
    from baby_food_rag.cli import main

    index = _index()
    index.embedder_name = "intfloat/multilingual-e5-large"
    sentinel = object()
    monkeypatch.setattr("baby_food_rag.cli.load_index", lambda path: index)
    monkeypatch.setattr("baby_food_rag.cli.load_eval_questions", lambda path: [])
    monkeypatch.setattr("baby_food_rag.cli.OpenRouterLLM", lambda model: FakeLLM([]))
    monkeypatch.setattr("baby_food_rag.index.load_embedder", lambda name: sentinel)
    seen = {}

    def fake_run(index, questions, llm, embedder=None, k=8):
        seen["embedder"] = embedder
        return not_run_report("stopped")

    monkeypatch.setattr("baby_food_rag.cli.run_generation_eval", fake_run)
    code = main(
        ["ask-eval", "--index", str(tmp_path), "--set", str(tmp_path), "--out", str(tmp_path)]
    )
    assert code == 0
    assert seen["embedder"] is sentinel


def test_positional_marks_become_source_pages_and_unknown_marks_drop():
    hits = [
        Hit(score=1.0, chunk=_chunk(103, "104", "Não ofereça mel antes dos 2 anos.")),
        Hit(score=0.5, chunk=_chunk(83, "84", "Uva em pedaços.")),
    ]
    text, dropped = normalize_citations(
        "Mel [1 p.21]. Uva [MS-2019 p.84]. Fora [9 p.1] e [FOO p.3].",
        hits,
    )
    assert text == "Mel [MS-2019 p.104]. Uva [MS-2019 p.84]. Fora e."
    assert dropped == ["[9 p.1]", "[FOO p.3]"]
    linked = link_citations(text, hits)
    assert "[MS-2019 p.104](https://example.test/guia.pdf#page=103)" in linked
    assert "[MS-2019 p.84](https://example.test/guia.pdf#page=83)" in linked
    assert citation_marks(linked) == [("MS-2019", "104"), ("MS-2019", "84")]


def test_printed_page_does_not_bind_to_the_neighbor_pdf_page():
    """Printed 84 is PDF 83. A higher-ranked PDF 84 must not steal the cite."""
    neighbor = Chunk(
        chunk_id="MS-2019::p84::c0",
        source_id="MS-2019",
        year=2019,
        country="BR",
        page=84,
        printed_page="85",
        text="página seguinte",
        url="https://example.test/guia.pdf",
        title="Guia",
    )
    grapes = _chunk(83, "84", "Uva em pedaços.")
    hits = [Hit(score=1.0, chunk=neighbor), Hit(score=0.5, chunk=grapes)]
    text, dropped = normalize_citations("Corte a uva [MS-2019 p.84].", hits)
    assert dropped == []
    assert text == "Corte a uva [MS-2019 p.84]."
    linked = link_citations(text, hits)
    assert linked == "Corte a uva [MS-2019 p.84](https://example.test/guia.pdf#page=83)."
    pdf_only, pdf_dropped = normalize_citations("Corte a uva [MS-2019 p.83].", hits)
    assert pdf_only == "Corte a uva."
    assert pdf_dropped == ["[MS-2019 p.83]"]


def test_embedder_failure_leaves_the_question_unsent():
    messages: list[dict] = []

    def boom(name: str) -> None:
        raise RuntimeError("weights missing")

    try:
        run_question(messages, "pode dar mel", _index(), boom, lambda: FakeLLM([]))
    except RuntimeError as exc:
        assert "weights missing" in str(exc)
    else:
        raise AssertionError("the embedder failure should raise")
    assert messages == []


def test_ask_maps_a_slot_citation_and_keeps_a_partial_answer():
    seen = {}

    class Capture(FakeLLM):
        def complete(self, prompt: str) -> str:
            seen["prompt"] = prompt
            self._usage.add(1, 1)
            return (
                "Não ofereça mel antes dos 2 anos [1 p.9]. Os guias recuperados não dizem a marca."
            )

    answer = ask(_index(), "pode dar mel", Capture([]), k=2, embedder=HashEmbedder(dim=32))
    context = seen["prompt"].split("Trechos:\n", 1)[1]
    assert "Abstenha-se somente quando nenhum trecho" in seen["prompt"]
    assert "só parte" in seen["prompt"]
    assert not context.lstrip().startswith("[1]")
    assert "[MS-2019 p." in context
    assert not answer.abstained
    assert "marca" in answer.text
    first = answer.hits[0].chunk
    label = f"[{first.source_id} p.{first.printed_page}]({first.citation_url})"
    assert label in answer.text
    assert "[1 p." not in answer.text


def test_judge_scores_the_rewritten_citation():
    from baby_food_rag.generate import Answer

    hits = [Hit(score=1.0, chunk=_chunk(83, "84", "Uva: cortar em pedaços."))]
    linked = link_citations("Corte a uva [MS-2019 p.84].", hits)
    answer = Answer(
        question="como servir uva",
        text=linked,
        hits=hits,
        abstained=False,
        model="fake",
        latency_s=0.1,
    )
    llm = FakeLLM(
        [
            '{"claims": ["Corte a uva"]}',
            '{"verdicts": [{"claim": 1, "verdict": "supported"}]}',
            '{"relevance": 1}',
        ]
    )
    judged = judge_answer(llm, answer, expect_abstention=False)
    assert judged["citation_correctness"] == 1.0
    assert judged["n_citations"] == 1
    assert citation_marks(answer.text) == [("MS-2019", "84")]


class _Body:
    def __enter__(self) -> "_Body":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        payload = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }
        return json.dumps(payload).encode()


def test_openrouter_retries_once_after_timeout():
    calls = {"n": 0}

    def opener(request: object, timeout: float) -> _Body:
        calls["n"] += 1
        assert timeout == CALL_TIMEOUT_S
        if calls["n"] == 1:
            raise TimeoutError("hung")
        return _Body()

    llm = OpenRouterLLM(api_key="test-key", opener=opener)
    assert llm.complete("ping") == "ok"
    assert calls["n"] == 2
    assert llm.usage.calls == 1


def test_openrouter_does_not_retry_a_client_error():
    calls = {"n": 0}

    def opener(request: object, timeout: float) -> _Body:
        calls["n"] += 1
        raise urllib.error.HTTPError(OPENROUTER_URL, 400, "bad", None, None)

    llm = OpenRouterLLM(api_key="test-key", opener=opener)
    try:
        llm.complete("ping")
    except RuntimeError as exc:
        assert "400" in str(exc)
        assert "test-key" not in str(exc)
    else:
        raise AssertionError("expected a client error")
    assert calls["n"] == 1


def test_cited_partial_answer_drops_the_abstention_sentence():
    partial = f"Corte a uva [MS-2019 p.84]. {ABSTAIN}"
    kept = drop_abstain_if_cited(partial)
    assert ABSTAIN not in kept
    assert "[MS-2019 p.84]" in kept
    assert not is_abstention(kept)
    assert drop_abstain_if_cited(ABSTAIN) == ABSTAIN
    assert is_abstention(ABSTAIN)


def test_full_abstention_hides_retrieved_passages():
    assert include_retrieved_passages(True) is False
    messages = append_successful_turn(
        [],
        "qual a dose de dipirona",
        _index(),
        FakeLLM([ABSTAIN]),
        k=1,
        embedder=HashEmbedder(dim=32),
    )
    assistant = messages[1]
    assert assistant["abstained"] is True
    assert assistant["passages"] is None
    assert all(part["type"] == "text" for part in assistant["segments"])


def test_citation_segment_opens_the_printed_page_not_the_neighbor():
    neighbor = Chunk(
        chunk_id="MS-2019::p84::c0",
        source_id="MS-2019",
        year=2019,
        country="BR",
        page=84,
        printed_page="85",
        text="página seguinte",
        url="https://example.test/guia.pdf",
        title="Guia seguinte",
    )
    grapes = _chunk(83, "84", "Uva: cortar em pedaços.")
    hits = [Hit(score=1.0, chunk=neighbor), Hit(score=0.5, chunk=grapes)]
    text = (
        "Corte a uva [MS-2019 p.84](https://example.test/guia.pdf#page=83). "
        "Os guias não falam do reflexo de gag."
    )
    segments = answer_segments(text, hits)
    cites = [part for part in segments if part["type"] == "citation"]
    assert len(cites) == 1
    assert cites[0]["page"] == "84"
    assert cites[0]["title"] == "Guia"
    assert cites[0]["url"].endswith("#page=83")
    assert cites[0]["excerpts"] == ["Uva: cortar em pedaços."]
    prose = " ".join(part["text"] for part in segments if part["type"] == "text")
    assert "gag" in prose
    assert "página seguinte" not in prose


def test_example_questions_are_short_and_in_range():
    assert 4 <= len(EXAMPLE_QUESTIONS) <= 6
    assert all(question.strip() for question in EXAMPLE_QUESTIONS)
    assert any("engasgo" in question for question in EXAMPLE_QUESTIONS)


def test_example_buttons_skip_width_on_streamlit_before_1_52():
    old = button_width_kwargs({"label", "key", "use_container_width"})
    assert old == {"use_container_width": True}
    assert "width" not in old
    current = button_width_kwargs({"label", "key", "width", "use_container_width"})
    assert current == {"width": "stretch"}


def test_judge_scores_a_partial_answer_and_names_the_gap_rule():
    from baby_food_rag.judge import _VERIFY

    assert "do not cover" in _VERIFY
    hits = [Hit(score=1.0, chunk=_chunk(83, "84", "Uva: cortar em pedaços."))]
    answer_text = "Corte a uva [MS-2019 p.84]. Os guias não falam do reflexo de gag."
    from baby_food_rag.generate import Answer

    answer = Answer(
        question="engasgo e reflexo de gag",
        text=answer_text,
        hits=hits,
        abstained=False,
        model="fake",
        latency_s=0.1,
    )
    seen = {}

    class Capture(FakeLLM):
        def complete(self, prompt: str) -> str:
            seen.setdefault("prompts", []).append(prompt)
            self._usage.add(1, 1)
            replies = [
                '{"claims": ["Corte a uva", "Os guias não falam do reflexo de gag"]}',
                '{"verdicts": [{"claim": 1, "verdict": "supported"}, '
                '{"claim": 2, "verdict": "supported"}]}',
                '{"relevance": 0.5}',
            ]
            return replies[len(seen["prompts"]) - 1]

    judged = judge_answer(Capture([]), answer, expect_abstention=False)
    assert judged["abstained"] is False
    assert judged["faithfulness"] == 1.0
    assert judged["citation_correctness"] == 1.0
    assert judged["relevance"] == 0.5
    assert "do not cover" in seen["prompts"][1]
