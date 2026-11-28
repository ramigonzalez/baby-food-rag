import json
import urllib.error

import pytest

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
    REASONING_TOKENS_PER_CALL,
    TYPICAL_ANSWER_CHARS,
    build_label_sheet,
    estimate_run,
    not_run_report,
    render_generation_report,
    retrieve_all,
    run_generation_eval,
    sample_for_labels,
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
    llm = FakeLLM(["Não ofereça mel [MS-2019 p.104]."])
    judge = FakeLLM(
        [
            '{"claims": ["Não ofereça mel"]}',
            '{"verdicts": [{"claim": 1, "verdict": "supported"}]}',
            '{"relevance": 1}',
        ]
    )
    report = run_generation_eval(
        _index(), questions, llm, judge=judge, k=1, embedder=HashEmbedder(dim=32)
    )
    assert report["status"] == "ok"
    assert report["overall"]["faithfulness"] == 1.0
    assert report["by_origin"]["real"]["n"] == 1
    assert report["total_usd"] is None
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
    run_generation_eval(
        _index(),
        questions,
        FakeLLM([ABSTAIN, ABSTAIN]),
        judge=FakeLLM([]),
        k=1,
        embedder=embedder,
    )
    assert embedder.calls == 2


def test_ask_eval_loads_the_embedder_once(monkeypatch, tmp_path):
    from baby_food_rag.cli import main

    index = _index()
    index.embedder_name = "intfloat/multilingual-e5-large"
    sentinel = object()
    monkeypatch.setattr("baby_food_rag.cli.load_index", lambda path: index)
    monkeypatch.setattr("baby_food_rag.cli.load_eval_questions", lambda path: [])
    monkeypatch.setattr("baby_food_rag.cli.OpenRouterLLM", lambda model, **kw: FakeLLM([]))
    monkeypatch.setattr("baby_food_rag.index.load_embedder", lambda name: sentinel)
    seen = {}

    def fake_run(index, questions, llm, **kwargs):
        seen["embedder"] = kwargs["embedder"]
        return run_generation_eval(index, questions, llm, **kwargs)

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


def _answer(text: str, *, abstained: bool = False) -> "object":
    from baby_food_rag.generate import Answer

    chunk = _chunk(103, "104", "Não é recomendado oferecer mel à criança menor de 2 anos.")
    return Answer(
        question="pode dar mel",
        text=text,
        hits=[Hit(score=1.0, chunk=chunk)],
        abstained=abstained,
        model="fake",
        latency_s=0.1,
    )


def test_completeness_is_covered_facts_over_all_facts():
    facts = ["Não dar mel antes dos 2 anos.", "O mel pode ter botulismo."]
    seen = []

    class Capture(FakeLLM):
        def complete(self, prompt: str) -> str:
            seen.append(prompt)
            return super().complete(prompt)

    judge = Capture(
        [
            '{"claims": ["Não ofereça mel antes dos 2 anos"]}',
            '{"verdicts": [{"claim": 1, "verdict": "supported"}]}',
            '{"relevance": 1}',
            # Some judges wrap JSON in a code fence.
            '```json\n{"facts": [{"fact": 1, "covered": true}, '
            '{"fact": 2, "covered": false}]}\n```',
        ]
    )
    answer = _answer("Não ofereça mel antes dos 2 anos [MS-2019 p.104].")
    judged = judge_answer(judge, answer, expect_abstention=False, key_facts=facts)
    assert judged["completeness"] == 0.5
    assert judged["facts"] == [
        {"fact": facts[0], "covered": True},
        {"fact": facts[1], "covered": False},
    ]
    assert "1. Não dar mel antes dos 2 anos." in seen[3]
    assert "2. O mel pode ter botulismo." in seen[3]
    assert DISCLAIMER not in seen[3]


def test_completeness_is_zero_for_an_abstention_and_none_without_facts():
    judge = FakeLLM([])
    abstained = judge_answer(
        judge, _answer(ABSTAIN, abstained=True), expect_abstention=False, key_facts=["a", "b"]
    )
    assert abstained["completeness"] == 0.0
    assert judge.usage.calls == 0
    replies = [
        '{"claims": ["Não ofereça mel"]}',
        '{"verdicts": [{"claim": 1, "verdict": "supported"}]}',
        '{"relevance": 1}',
    ]
    judge = FakeLLM(list(replies))
    plain = judge_answer(
        judge, _answer("Não ofereça mel [MS-2019 p.104]."), expect_abstention=False
    )
    assert plain["completeness"] is None
    assert judge.usage.calls == 3


def test_coverage_with_the_wrong_count_raises():
    judge = FakeLLM(
        [
            '{"claims": ["Não ofereça mel"]}',
            '{"verdicts": [{"claim": 1, "verdict": "supported"}]}',
            '{"relevance": 1}',
            '{"facts": [{"fact": 1, "covered": true}]}',
        ]
    )
    try:
        judge_answer(
            judge,
            _answer("Não ofereça mel [MS-2019 p.104]."),
            expect_abstention=False,
            key_facts=["a", "b"],
        )
    except ValueError as exc:
        assert "coverage" in str(exc)
    else:
        raise AssertionError("a short coverage list should raise")


def _eval_questions() -> list[dict]:
    return [
        {
            "id": "q-mel",
            "origin": "real",
            "split": "dev",
            "question": "pode dar mel",
            "answerable": True,
            "gold": [{"source_id": "MS-2019", "pdf_page": 103, "quote": "mel"}],
            "key_facts": ["Não dar mel antes dos 2 anos.", "O mel é açúcar."],
            "key_facts_status": "draft",
        },
        {
            "id": "q-dose",
            "origin": "synthetic",
            "split": "test",
            "question": "qual a dose de dipirona",
            "answerable": False,
            "gold": [],
            "key_facts": [],
            "key_facts_status": None,
        },
    ]


def test_generation_eval_keeps_generator_and_judge_usage_apart_and_groups_rows():
    llm = FakeLLM(["Não ofereça mel antes dos 2 anos [MS-2019 p.104].", ABSTAIN])
    llm.model = "openai/gpt-4o-mini"
    judge = FakeLLM(
        [
            '{"claims": ["Não ofereça mel antes dos 2 anos"]}',
            '{"verdicts": [{"claim": 1, "verdict": "supported"}]}',
            '{"relevance": 1}',
            '{"facts": [{"fact": 1, "covered": true}, {"fact": 2, "covered": false}]}',
        ]
    )
    judge.model = "anthropic/claude-haiku-5.5"
    report = run_generation_eval(
        _index(), _eval_questions(), llm, judge=judge, k=2, embedder=HashEmbedder(dim=32)
    )
    assert report["usage"]["generator"]["calls"] == 2
    assert report["usage"]["judge"]["calls"] == 4
    assert report["usage"]["generator"]["usd"] == (24 * 0.15 + 8 * 0.60) / 1_000_000
    assert report["usage"]["judge"]["usd"] == (48 * 0.10 + 16 * 0.50) / 1_000_000
    assert (
        report["total_usd"]
        == report["usage"]["generator"]["usd"] + (report["usage"]["judge"]["usd"])
    )
    assert report["overall"]["completeness"] == 0.5
    assert report["overall"]["hallucination_rate"] == 0.0
    assert report["by_split"]["dev"]["n"] == 1
    assert report["by_split"]["test"]["abstention_accuracy"] == 1.0
    assert report["by_origin"]["real"]["completeness"] == 0.5
    assert set(report["by_origin"]) == {"real", "synthetic"}
    text = render_generation_report(report)
    assert "| dev | 1 |" in text
    assert "| real | 1 |" in text
    assert "Completeness" in text


def test_label_sheet_shows_claims_and_facts_but_no_judge_score():
    llm = FakeLLM(["Não ofereça mel antes dos 2 anos [MS-2019 p.104].", ABSTAIN])
    judge = FakeLLM(
        [
            '{"claims": ["Não ofereça mel antes dos 2 anos"]}',
            '{"verdicts": [{"claim": 1, "verdict": "supported"}]}',
            '{"relevance": 1}',
            '{"facts": [{"fact": 1, "covered": true}, {"fact": 2, "covered": false}]}',
        ]
    )
    index = _index()
    report = run_generation_eval(
        index, _eval_questions(), llm, judge=judge, k=2, embedder=HashEmbedder(dim=32)
    )
    sheet = build_label_sheet(report, index)
    assert sheet["size"] == 1
    assert sheet["strata"] == {"dev/real": 1}
    item = sheet["items"][0]
    assert set(item) == {
        "id",
        "split",
        "origin",
        "answerable",
        "question",
        "retrieved",
        "answer",
        "claims",
        "key_facts",
        "human_relevance",
        "human_notes",
    }
    assert item["id"] == "q-mel"
    texts = {chunk.text for chunk in index.chunks}
    assert all(entry["text"] in texts for entry in item["retrieved"])
    assert {"source_id", "page", "pdf_page", "text"} == set(item["retrieved"][0])
    assert item["claims"] == [
        {
            "claim": "Não ofereça mel antes dos 2 anos",
            "judge_verdict": "supported",
            "human_verdict": None,
        }
    ]
    assert item["key_facts"] == [
        {"fact": "Não dar mel antes dos 2 anos.", "human_covered": None},
        {"fact": "O mel é açúcar.", "human_covered": None},
    ]
    assert item["human_relevance"] is None


def test_label_sample_is_stratified_seeded_and_skips_abstentions():
    sizes = {
        ("dev", "real"): 4,
        ("test", "real"): 4,
        ("dev", "plan"): 8,
        ("test", "plan"): 4,
        ("dev", "synthetic"): 48,
        ("test", "synthetic"): 32,
    }
    rows = []
    for (split, origin), count in sizes.items():
        for number in range(count):
            rows.append(
                {
                    "id": f"{split}-{origin}-{number}",
                    "split": split,
                    "origin": origin,
                    "abstained": False,
                }
            )
        rows.append(
            {"id": f"{split}-{origin}-x", "split": split, "origin": origin, "abstained": True}
        )
    picked = sample_for_labels(rows, size=30, seed=7)
    assert len(picked) == 30
    assert not any(row["abstained"] for row in picked)
    counts: dict[tuple[str, str], int] = {}
    for row in picked:
        key = (row["split"], row["origin"])
        counts[key] = counts.get(key, 0) + 1
    # 30 of 100 answered: floors 1, 1, 2, 1, 14, 9; the two largest remainders
    # (test/synthetic 60, then dev/plan 40, which wins the tie with dev/synthetic on key order).
    assert counts == {
        ("dev", "real"): 1,
        ("test", "real"): 1,
        ("dev", "plan"): 3,
        ("test", "plan"): 1,
        ("dev", "synthetic"): 14,
        ("test", "synthetic"): 10,
    }
    ids = [row["id"] for row in picked]
    assert ids == [row["id"] for row in sample_for_labels(rows[::-1], size=30, seed=7)]
    assert ids != [row["id"] for row in sample_for_labels(rows, size=30, seed=8)]
    assert len(sample_for_labels(rows[:5], size=30, seed=7)) == 4


def test_estimate_prices_both_models_from_the_prompts():
    index = _index()
    questions = _eval_questions()
    hits = retrieve_all(index, questions, embedder=HashEmbedder(dim=32), k=2)
    estimate = estimate_run(
        questions, hits, model="openai/gpt-4o-mini", judge_model="anthropic/claude-haiku-5.5"
    )
    generator = estimate["models"]["generator"]
    judge = estimate["models"]["judge"]
    assert generator["calls"] == 2
    # Three judge calls per question, plus one for the question with key facts.
    assert judge["calls"] == 7
    # The judge reasons before it answers, and that output is billed.
    assert judge["completion_tokens"] > 7 * REASONING_TOKENS_PER_CALL["anthropic/claude-haiku-5.5"]
    assert generator["completion_tokens"] >= 2 * TYPICAL_ANSWER_CHARS // 3
    assert judge["prompt_tokens"] > generator["prompt_tokens"]
    assert (
        generator["usd"]
        == (generator["prompt_tokens"] * 0.15 + generator["completion_tokens"] * 0.60) / 1_000_000
    )
    assert judge["usd"] == (judge["prompt_tokens"] * 0.10 + judge["completion_tokens"] * 0.50) / (
        1_000_000
    )
    assert estimate["total_usd"] == generator["usd"] + judge["usd"]
    unknown = estimate_run(questions, hits, model="openai/gpt-4o-mini", judge_model="x/unknown")
    assert unknown["models"]["judge"]["usd"] is None
    assert unknown["total_usd"] is None


def _patch_cli(monkeypatch, questions):
    index = _index()
    monkeypatch.setattr("baby_food_rag.cli.load_index", lambda path: index)
    monkeypatch.setattr("baby_food_rag.cli.load_eval_questions", lambda path: questions)
    monkeypatch.setattr("baby_food_rag.cli.query_embedder", lambda index: HashEmbedder(dim=32))

    def no_model(model, **kwargs):
        raise AssertionError("a dry run must not build a model client")

    monkeypatch.setattr("baby_food_rag.cli.OpenRouterLLM", no_model)


def test_ask_eval_dry_run_prints_the_estimate_and_calls_no_model(monkeypatch, tmp_path, capsys):
    from baby_food_rag.cli import main

    _patch_cli(monkeypatch, _eval_questions())
    out = tmp_path / "out"
    args = ["ask-eval", "--dry-run", "--index", str(tmp_path), "--set", str(tmp_path)]
    code = main([*args, "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 0
    assert "2 questions, split all, 1 with key facts" in printed
    assert "`openai/gpt-4o-mini`" in printed
    assert "`anthropic/claude-haiku-5.5`" in printed
    assert "Under --max-usd 1.00" in printed
    assert not out.exists()
    code = main([*args, "--split", "test", "--out", str(out)])
    assert code == 0
    assert "1 questions, split test, 0 with key facts" in capsys.readouterr().out


def test_ask_eval_refuses_above_the_cap_and_a_same_family_judge(monkeypatch, tmp_path, capsys):
    from baby_food_rag.cli import main

    _patch_cli(monkeypatch, _eval_questions())
    base = ["ask-eval", "--index", str(tmp_path), "--set", str(tmp_path)]
    code = main([*base, "--out", str(tmp_path / "out"), "--max-usd", "0.000001"])
    assert code == 1
    assert "Refusing to run" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()
    code = main([*base, "--dry-run", "--judge-model", "openai/gpt-4o"])
    assert code == 2
    assert "same family" in capsys.readouterr().err


def test_metrics_json_keeps_scores_and_answers_json_keeps_the_text(tmp_path):
    from baby_food_rag.generate_eval import write_generation_report

    llm = FakeLLM(["Não ofereça mel antes dos 2 anos [MS-2019 p.104].", ABSTAIN])
    judge = FakeLLM(
        [
            '{"claims": ["Não ofereça mel antes dos 2 anos"]}',
            '{"verdicts": [{"claim": 1, "verdict": "supported"}]}',
            '{"relevance": 1}',
            '{"facts": [{"fact": 1, "covered": true}, {"fact": 2, "covered": false}]}',
        ]
    )
    report = run_generation_eval(
        _index(), _eval_questions(), llm, judge=judge, k=2, embedder=HashEmbedder(dim=32)
    )
    write_generation_report(report, tmp_path)
    metrics = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    for text in ("Não ofereça mel", "Não dar mel", "pode dar mel", "Não é recomendado"):
        assert text not in metrics
    row = json.loads(metrics)["rows"][0]
    assert row["n_claims"] == 1 and row["n_supported"] == 1
    assert row["facts_covered"] == [True, False]
    assert row["retrieved"][0]["source_id"] == "MS-2019"
    answers = json.loads((tmp_path / "answers.json").read_text(encoding="utf-8"))
    assert "Não ofereça mel" in answers[0]["answer"]
    assert report["rows"][0]["answer"] == answers[0]["answer"]


def test_verdicts_that_are_not_objects_are_a_parse_error():
    # A judge may answer ["supported"] instead of [{"verdict": "supported"}]. That must be a
    # ValueError, which rejudge records as the item's error, not an AttributeError.
    judge = FakeLLM(
        [
            '{"claims": ["Não ofereça mel antes dos 2 anos"]}',
            '{"verdicts": ["supported"]}',
        ]
    )
    answer = _answer("Não ofereça mel antes dos 2 anos [MS-2019 p.104].")
    with pytest.raises(ValueError, match="bad verdict"):
        judge_answer(judge, answer, expect_abstention=False)
