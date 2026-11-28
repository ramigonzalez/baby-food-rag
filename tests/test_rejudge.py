"""Judge-only re-scoring over a saved label sheet. HTTP is mocked; no key is used."""

import json

from baby_food_rag.generate import DISCLAIMER, OpenRouterLLM
from baby_food_rag.generate_eval import REASONING_TOKENS_PER_CALL
from baby_food_rag.rejudge import (
    compare_scores,
    estimate_rejudge,
    render_comparison,
    run_rejudge,
    write_rejudge,
)

MEL = "Não é recomendado oferecer mel à criança menor de 2 anos."
URL = "https://example.test/guia.pdf#page=103"


def _sheet() -> dict:
    retrieved = [{"source_id": "MS-2019", "page": "104", "pdf_page": 103, "text": MEL}]
    return {
        "generator": "openai/gpt-4o-mini",
        "judge": "anthropic/claude-haiku-5.5",
        "items": [
            {
                "id": "q-mel",
                "split": "dev",
                "origin": "real",
                "answerable": True,
                "question": "pode dar mel",
                "retrieved": retrieved,
                "answer": f"Não ofereça mel antes dos 2 anos [MS-2019 p.104]({URL}).\n\n"
                f"{DISCLAIMER}",
                "claims": [
                    {
                        "claim": "Não ofereça mel antes dos 2 anos",
                        "judge_verdict": "supported",
                        "human_verdict": None,
                    }
                ],
                "key_facts": [
                    {"fact": "Não dar mel antes dos 2 anos.", "human_covered": None},
                    {"fact": "O mel é açúcar.", "human_covered": None},
                ],
                "human_relevance": None,
                "human_notes": "",
            },
            {
                "id": "q-uva",
                "split": "test",
                "origin": "synthetic",
                "answerable": True,
                "question": "como dar uva",
                "retrieved": retrieved,
                "answer": f"Corte a uva [MS-2019 p.84].\n\n{DISCLAIMER}",
                "claims": [],
                "key_facts": [],
                "human_relevance": None,
                "human_notes": "",
            },
        ],
    }


REPLIES = {
    "Task: extract claims": {
        "pode dar mel": '{"claims": ["Não ofereça mel antes dos 2 anos"]}',
        "como dar uva": '{"claims": ["Corte a uva"]}',
    },
    "Task: verify claims": {
        "mel antes": '{"verdicts": [{"claim": 1, "verdict": "supported"}]}',
        "Corte a uva": '{"verdicts": [{"claim": 1, "verdict": "not_found"}]}',
    },
    "Task: answer relevance": {"": '{"relevance": 1}'},
    "Task: key fact coverage": {
        "": '{"facts": [{"fact": 1, "covered": true}, {"fact": 2, "covered": false}]}'
    },
}


class FakeOpenRouter:
    """Stands in for urlopen: answers by task, records each payload."""

    def __init__(self, reasoning: int = 30, cost: float = 0.0001) -> None:
        self.payloads: list[dict] = []
        self.reasoning = reasoning
        self.cost = cost

    def __call__(self, request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        self.payloads.append(payload)
        assert request.headers["Authorization"] == "Bearer test-key"
        prompt = payload["messages"][0]["content"]
        task = next(name for name in REPLIES if prompt.startswith(name))
        content = next(text for key, text in REPLIES[task].items() if key in prompt)
        body = {
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": self.reasoning},
                "cost": self.cost,
            },
        }
        return _Response(body)


class _Response:
    def __init__(self, body: dict) -> None:
        self._data = json.dumps(body).encode()

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._data


def test_client_sends_temperature_and_seed_only_when_given():
    http = FakeOpenRouter()
    OpenRouterLLM(api_key="test-key", opener=http).complete("Task: answer relevance x")
    judge = OpenRouterLLM(
        "anthropic/claude-haiku-5.5", api_key="test-key", opener=http, temperature=None
    )
    judge.complete("Task: answer relevance x")
    seeded = OpenRouterLLM(
        "google/gemini-2.5-flash", api_key="test-key", opener=http, temperature=0, seed=7
    )
    seeded.complete("Task: answer relevance x")
    generator, haiku, gemini = http.payloads
    assert generator["temperature"] == 0 and "seed" not in generator
    assert "temperature" not in haiku and "seed" not in haiku
    assert gemini["temperature"] == 0 and gemini["seed"] == 7
    assert seeded.usage.reasoning_tokens == 30
    assert seeded.usage.reported_usd == 0.0001


def test_rejudge_scores_saved_answers_without_generating(tmp_path):
    http = FakeOpenRouter()
    judge = OpenRouterLLM(
        "google/gemini-2.5-flash", api_key="test-key", opener=http, temperature=0, seed=7
    )
    report = run_rejudge(_sheet(), judge, temperature=0, seed=7)
    assert report["status"] == "ok"
    first, second = report["rows"]
    assert first["faithfulness"] == 1.0
    assert first["completeness"] == 0.5
    assert first["relevance"] == 1.0
    assert first["citation_correctness"] == 1.0
    assert second["faithfulness"] == 0.0
    assert second["completeness"] is None
    # 4 calls for the item with key facts, 3 for the other; no generation call.
    assert report["usage"]["calls"] == 7
    assert report["usage"]["reasoning_tokens"] == 7 * 30
    assert report["usage"]["usd"] == (700 * 0.30 + 350 * 2.50) / 1_000_000
    assert abs(report["usage"]["reported_usd"] - 0.0007) < 1e-12
    assert all(payload["seed"] == 7 for payload in http.payloads)
    prompts = [payload["messages"][0]["content"] for payload in http.payloads]
    assert not any("Você responde em português" in prompt for prompt in prompts)
    assert not any(DISCLAIMER in prompt for prompt in prompts)
    assert MEL in prompts[1]
    write_rejudge(report, tmp_path)
    scores = (tmp_path / "scores.json").read_text(encoding="utf-8")
    for text in ("Não ofereça mel", "Corte a uva", MEL, "O mel é açúcar"):
        assert text not in scores
    saved = json.loads(scores)
    assert saved["rows"][0]["n_supported"] == 1
    assert saved["rows"][0]["facts_covered"] == [True, False]
    claims = json.loads((tmp_path / "claims.json").read_text(encoding="utf-8"))
    assert claims[0]["claims"][0]["claim"] == "Não ofereça mel antes dos 2 anos"


def test_rejudge_stops_once_the_spend_passes_the_cap():
    judge = OpenRouterLLM(
        "google/gemini-2.5-flash", api_key="test-key", opener=FakeOpenRouter(), temperature=0
    )
    report = run_rejudge(_sheet(), judge, max_usd=0.0001)
    assert report["status"] == "stopped_at_cap"
    assert report["n_items"] == 1


def test_rejudge_estimate_uses_the_saved_claims_and_the_reasoning_allowance():
    estimate = estimate_rejudge(_sheet(), "google/gemini-2.5-flash")
    # Item one: claims, verify, relevance, coverage. Item two has no saved claims.
    assert estimate["calls"] == 6
    allowance = REASONING_TOKENS_PER_CALL["google/gemini-2.5-flash"]
    assert estimate["reasoning_tokens"] == 6 * allowance
    assert (
        estimate["usd"]
        == (estimate["prompt_tokens"] * 0.30 + estimate["completion_tokens"] * 2.50) / 1_000_000
    )


def test_rejudge_cli_dry_run_then_a_mocked_run(monkeypatch, tmp_path, capsys):
    from baby_food_rag.cli import main

    sheet = tmp_path / "label_sheet.json"
    sheet.write_text(json.dumps(_sheet(), ensure_ascii=False), encoding="utf-8")
    http = FakeOpenRouter()
    built = []

    def client(model, **kwargs):
        built.append(kwargs)
        return OpenRouterLLM(model, api_key="test-key", opener=http, **kwargs)

    monkeypatch.setattr("baby_food_rag.cli.OpenRouterLLM", client)
    base = ["rejudge", "--sheet", str(sheet), "--out", str(tmp_path / "out")]
    gemini = ["--judge-model", "google/gemini-2.5-flash", "--temperature", "0", "--seed", "7"]
    assert main([*base, *gemini, "--dry-run"]) == 0
    assert "estimate before any call: 2 items" in capsys.readouterr().out
    assert built == [] and http.payloads == []
    assert main([*base, *gemini, "--max-usd", "0.0000001"]) == 1
    assert "Refusing" in capsys.readouterr().out
    assert main([*base, *gemini]) == 0
    assert built == [{"temperature": 0.0, "seed": 7}]
    assert "faithfulness 0.5000" in capsys.readouterr().out
    assert (tmp_path / "out" / "scores.json").is_file()
    assert main([*base]) == 0
    assert built[-1] == {"temperature": None, "seed": None}
    assert "temperature" not in http.payloads[-1]
    assert main([*base, "--judge-model", "openai/gpt-4o", "--dry-run"]) == 2


def test_compare_scores_is_the_mean_absolute_gap_on_shared_items():
    first = [
        {"id": "a", "faithfulness": 1.0, "completeness": 1.0, "relevance": 1.0},
        {"id": "b", "faithfulness": 0.5, "completeness": None, "relevance": 1.0},
        {"id": "c", "faithfulness": None, "completeness": 0.5, "relevance": 0.5},
        {"id": "only-first", "faithfulness": 0.0, "completeness": 0.0, "relevance": 0.0},
    ]
    second = [
        {"id": "a", "faithfulness": 0.95, "completeness": 0.5, "relevance": 1.0},
        {"id": "b", "faithfulness": 0.25, "completeness": None, "relevance": 0.5},
        {"id": "c", "faithfulness": 0.8, "completeness": 0.5, "relevance": 0.5},
    ]
    result = compare_scores(first, second)
    assert result["n_items"] == 3
    assert result["n_faithfulness"] == 2
    assert abs(result["faithfulness_mad"] - 0.15) < 1e-12
    assert result["faithfulness_changed_share"] == 0.5
    assert result["n_completeness"] == 2
    assert result["completeness_mad"] == 0.25
    assert abs(result["relevance_mad"] - 1 / 6) < 1e-12


def test_render_comparison_labels_same_and_cross_judge_pairs():
    def run(model, faith, temperature=None, seed=None):
        rows = [
            {"id": "a", "faithfulness": faith, "completeness": 1.0, "relevance": 1.0},
            {"id": "b", "faithfulness": 1.0, "completeness": 0.5, "relevance": 1.0},
        ]
        return {"judge_model": model, "temperature": temperature, "seed": seed, "rows": rows}

    text = render_comparison(
        {
            "haiku-1": run("anthropic/claude-haiku-5.5", 1.0),
            "haiku-2": run("anthropic/claude-haiku-5.5", 0.8),
            "gemini-1": run("google/gemini-2.5-flash", 0.5, 0, 7),
            "gemini-2": run("google/gemini-2.5-flash", 0.5, 0, 7),
        }
    )
    assert "Items scored by every run: 2." in text
    assert "| haiku-1 vs haiku-2 | same judge | 0.1000 (n 2) | 50% |" in text
    assert "| gemini-1 vs gemini-2 | same judge | 0.0000 (n 2) | 0% |" in text
    assert "| haiku-1 vs gemini-1 | cross judge | 0.2500 (n 2) | 50% |" in text
    assert "Cross-judge mean over 4 pairs: faithfulness MAD 0.2000" in text
    assert "| haiku-1 | `anthropic/claude-haiku-5.5` | not sent | not sent |" in text
    assert "not which judge is right" in text
