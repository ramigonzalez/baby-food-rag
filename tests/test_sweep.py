import json

from baby_food_rag.analyze import Analyzer, fold_accents
from baby_food_rag.plot_sweep import plot_recall_latency
from baby_food_rag.sweep import bootstrap_difference, render_sweep


def test_snowball_joins_cortar_and_folded_accents_match():
    snowball = Analyzer("snowball", stem="snowball", fold=False, stop=False)
    assert snowball.tokens("cortar") == snowball.tokens("cortadas")
    folded = Analyzer("folded", stem="none", fold=True, stop=False)
    assert folded.tokens("não") == folded.tokens("nao")
    assert fold_accents("açúcar") == "acucar"
    stopped = Analyzer("stop", stem="none", fold=False, stop=True)
    assert "para" not in stopped.tokens("mel para a criança")
    assert "mel" in stopped.tokens("mel para a criança")
    snow_stop = Analyzer("snowball-stop", stem="snowball", fold=False, stop=True)
    assert "com" in snow_stop.tokens("comer fruta")
    assert "par" not in snow_stop.tokens("mel para a criança")


def test_suffix_analyzer_matches_phase2_tokens():
    from baby_food_rag.analyze import BASELINE_ANALYZER
    from baby_food_rag.index import _bm25_tokens

    text = "Pode dar uva cortada sem açúcar."
    assert BASELINE_ANALYZER.tokens(text) == _bm25_tokens(text)


def test_bootstrap_interval_covers_a_real_gap_and_a_tie():
    higher = bootstrap_difference([1.0] * 20, [0.0] * 20, draws=400, seed=1)
    assert higher["diff"] == 1.0
    assert higher["ci95_low"] > 0
    assert higher["significant"]
    tied = bootstrap_difference([0.5, 0.0, 1.0, 0.5], [0.5, 0.0, 1.0, 0.5], draws=400, seed=1)
    assert tied["diff"] == 0.0
    assert tied["ci95_low"] <= 0 <= tied["ci95_high"]
    assert not tied["significant"]


def test_plot_reads_metrics_and_writes_a_png(tmp_path):
    metrics = {
        "chart_title": "Recall@10 against mean query latency",
        "configs": [
            {"name": "bm25-suffix", "family": "bm25", "latency_ms": 2.0, "recall_at_10": 0.3},
            {"name": "dense-e5", "family": "dense", "latency_ms": 40.0, "recall_at_10": 0.5},
        ],
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    png = tmp_path / "recall_at_10_vs_latency.png"
    plot_recall_latency(path, png)
    assert png.read_bytes().startswith(b"\x89PNG")


def test_results_markdown_uses_the_metrics_numbers():
    report = {
        "hardware": {"cpu_model": "test-cpu", "cpus": 4, "device": "cpu"},
        "selection": "Chosen on the same 2 questions.",
        "hit_rule": "within 1",
        "fusion_pool": 50,
        "rrf_k": 60,
        "rerank_pool": 30,
        "not_run": [{"name": "BAAI/bge-m3", "reason": "not shipped"}],
        "comparison": {
            "best": "dense-e5",
            "baseline": "hybrid-baseline",
            "best_recall_at_10": 0.5,
            "baseline_recall_at_10": 0.25,
            "diff_recall_at_10": 0.25,
            "ci95_low": -0.1,
            "ci95_high": 0.4,
            "significant": False,
            "n_bootstrap": 10,
            "seed": 0,
        },
        "configs": [
            {
                "name": "hybrid-baseline",
                "family": "hybrid",
                "recall_at_5": 0.2,
                "recall_at_10": 0.25,
                "mrr": 0.3,
                "latency_ms": 12.0,
                "by_origin": {
                    "real": {"n": 1, "recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0},
                    "plan": {"n": 1, "recall_at_5": 0.4, "recall_at_10": 0.5, "mrr": 0.6},
                    "synthetic": {"n": 0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0},
                },
            }
        ],
    }
    text = render_sweep(report)
    assert "0.2500" in text
    assert "not significant" in text
    assert "BAAI/bge-m3" in text
    assert "test-cpu" in text


def test_results_markdown_shows_split_rows_only_when_questions_have_a_split():
    group = {"n": 2, "recall_at_5": 0.5, "recall_at_10": 0.75, "mrr": 0.5}
    origin = {"n": 0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0}
    config = {
        "name": "hybrid-baseline",
        "family": "hybrid",
        "recall_at_5": 0.5,
        "recall_at_10": 0.75,
        "mrr": 0.5,
        "latency_ms": 1.0,
        "by_origin": {"real": origin, "plan": origin, "synthetic": group},
        "groups": [
            {"split": "all", "origin": "all", **group},
            {"split": "dev", "origin": "all", **group},
        ],
    }
    report = {
        "hardware": {"cpu_model": "test-cpu", "cpus": 4, "device": "cpu"},
        "selection": "Chosen on dev.",
        "hit_rule": "within 1",
        "fusion_pool": 50,
        "rrf_k": 60,
        "rerank_pool": 30,
        "n_answerable": 2,
        "not_run": [],
        "comparison": {
            "best": "hybrid-baseline",
            "baseline": "hybrid-baseline",
            "challenger": "hybrid-baseline",
            "best_recall_at_10": 0.75,
            "baseline_recall_at_10": 0.75,
            "challenger_recall_at_10": 0.75,
            "diff_recall_at_10": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "significant": False,
            "n_bootstrap": 10,
            "seed": 0,
        },
        "configs": [config],
    }
    text = render_sweep(report)
    assert "## By split" in text
    assert "| hybrid-baseline | dev | 2 | 0.5000 | 0.7500 | 0.5000 |" in text
    assert "n is 2." in text
    config["groups"] = config["groups"][:1]
    assert "## By split" not in render_sweep(report)
