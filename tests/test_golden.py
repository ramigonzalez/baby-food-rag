from baby_food_rag.golden import assign_splits, content_tokens, quote_overlap


def test_content_tokens_drop_stopwords_and_short_tokens_but_keep_mel():
    tokens = content_tokens("Pode dar mel pra ela de 7 meses?")
    assert "mel" in tokens
    assert "pra" not in tokens
    assert "de" not in tokens
    assert "7" not in tokens


def test_quote_overlap_is_the_share_of_question_tokens_in_the_quotes():
    # Content tokens: pode, dar, mel, criança. Two of them are in the quote.
    assert quote_overlap("Pode dar mel à criança?", ["Não dar mel."]) == 0.5
    assert quote_overlap("Miojo, pode?", ["macarrão instantâneo"]) == 0.0
    assert quote_overlap("de e o", ["de"]) == 0.0


def _question(qid: str, origin: str, source: str | None) -> dict:
    gold = [{"source_id": source, "pdf_page": 1, "quote": "x"}] if source else []
    return {"id": qid, "origin": origin, "answerable": bool(source), "gold": gold}


def test_assign_splits_is_deterministic_stratified_and_sixty_percent_dev():
    questions = [_question(f"syn2-{i:03d}", "synthetic", "MS-2019") for i in range(20)]
    questions += [_question(f"real-{i:02d}", "real", "SBP-MAN-2024") for i in range(10)]
    questions += [_question(f"unans2-{i:03d}", "synthetic", None) for i in range(5)]
    first = assign_splits(questions)
    assert first == assign_splits(list(reversed(questions)))
    assert first != assign_splits(questions, seed=1)
    answerable = [item["id"] for item in questions if item["answerable"]]
    assert sum(first[qid] == "dev" for qid in answerable) == 18
    for prefix in ("syn2", "real"):
        group = [first[qid] for qid in answerable if qid.startswith(prefix)]
        assert 0.5 <= group.count("dev") / len(group) <= 0.7
    unanswerable = [first[item["id"]] for item in questions if not item["answerable"]]
    assert unanswerable.count("dev") == 3
