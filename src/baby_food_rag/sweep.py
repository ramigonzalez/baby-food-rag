"""Compare retrieval settings on the labeled questions.

The winner is chosen by recall@10 on all answerable questions. There is
no held-out split: with 32 questions a split would be smaller than the
noise. A paired bootstrap says whether that winner beats the phase 2
hybrid baseline by more than chance.

BAAI/bge-m3 dense is not in this comparison. fastembed 0.9.1, the latest
release, does not ship it, and this machine has no GPU and no torch.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from baby_food_rag.analyze import BASELINE_ANALYZER, BM25_ANALYZERS, Analyzer
from baby_food_rag.bm25 import BM25Index
from baby_food_rag.chunking import CHUNK_OVERLAP, CHUNK_WORDS, chunk_pages
from baby_food_rag.embed import l2_normalize
from baby_food_rag.evaluate import (
    FUSION_POOL,
    RRF_K,
    collapse_pages,
    load_questions,
    pages_from_scores,
    score_page_list,
)
from baby_food_rag.extract import extract_pdf
from baby_food_rag.fusion import reciprocal_rank_fusion
from baby_food_rag.index import load_index
from baby_food_rag.manifest import default_index_sources, load_manifest
from baby_food_rag.plot_sweep import plot_recall_latency
from baby_food_rag.types import Chunk

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER = "BAAI/bge-reranker-v2-m3"
RERANK_POOL = 30
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
NOT_RUN = [
    {
        "name": "BAAI/bge-m3",
        "reason": (
            "Dense BGE-M3 is not in fastembed 0.9.1, the latest release on PyPI. "
            "This VM has no GPU and torch is not installed, so the model was not loaded."
        ),
    }
]


def run_sweep(
    *,
    question_path: Path,
    index_dir: Path,
    raw_dir: Path,
    manifest_path: Path,
    out_dir: Path,
    cache_dir: Path,
) -> dict:
    questions = [item for item in load_questions(question_path) if item["answerable"]]
    index = load_index(index_dir)
    current_chunks = index.chunks
    page_chunks = _page_chunks(manifest_path, raw_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("embedding corpora", file=sys.stderr)
    e5_current = _e5_matrix(index, current_chunks, cache_dir)
    e5_model = _load_e5()
    minilm = _PlainEmbedder(MINILM)
    minilm_current = _cached_matrix(minilm, current_chunks, cache_dir / "minilm-180.npy")
    e5_page = _cached_matrix(e5_model, page_chunks, cache_dir / "e5-page.npy")
    minilm_page = _cached_matrix(minilm, page_chunks, cache_dir / "minilm-page.npy")

    print("scoring queries", file=sys.stderr)
    built: list[dict] = []
    bm25_rankers = {}
    for analyzer in BM25_ANALYZERS:
        name = f"bm25-{analyzer.name}"
        ranker = _Bm25Ranker(current_chunks, analyzer)
        bm25_rankers[analyzer.name] = ranker
        built.append(_time_config(name, "bm25", questions, ranker.pages, chunking="180/40"))

    page_bm25 = _Bm25Ranker(page_chunks, BASELINE_ANALYZER)
    built.append(
        _time_config(
            "bm25-suffix-page",
            "bm25",
            questions,
            page_bm25.pages,
            chunking="whole-page",
        )
    )

    dense_rankers = {
        "e5": _DenseRanker(current_chunks, e5_current, e5_model),
        "minilm": _DenseRanker(current_chunks, minilm_current, minilm),
        "e5-page": _DenseRanker(page_chunks, e5_page, e5_model),
        "minilm-page": _DenseRanker(page_chunks, minilm_page, minilm),
    }
    built.append(
        _time_config("dense-e5", "dense", questions, dense_rankers["e5"].pages, chunking="180/40")
    )
    built.append(
        _time_config(
            "dense-minilm",
            "dense",
            questions,
            dense_rankers["minilm"].pages,
            chunking="180/40",
        )
    )
    built.append(
        _time_config(
            "dense-e5-page",
            "dense",
            questions,
            dense_rankers["e5-page"].pages,
            chunking="whole-page",
        )
    )
    built.append(
        _time_config(
            "dense-minilm-page",
            "dense",
            questions,
            dense_rankers["minilm-page"].pages,
            chunking="whole-page",
        )
    )

    hybrid_base = _HybridRanker(current_chunks, bm25_rankers["suffix"], dense_rankers["e5"])
    built.append(
        _time_config("hybrid-baseline", "hybrid", questions, hybrid_base.pages, chunking="180/40")
    )
    best_bm25_name = _best_bm25_analyzer(built)
    if best_bm25_name != BASELINE_ANALYZER.name:
        hybrid_best = _HybridRanker(
            current_chunks, bm25_rankers[best_bm25_name], dense_rankers["e5"]
        )
        built.append(
            _time_config(
                f"hybrid-{best_bm25_name}-e5",
                "hybrid",
                questions,
                hybrid_best.pages,
                chunking="180/40",
                selected_on_same_set=True,
            )
        )

    not_run = list(NOT_RUN)
    try:
        print("loading reranker", file=sys.stderr)
        reranker = _load_reranker()
        rerank = _RerankRanker(hybrid_base, reranker, RERANK_POOL)
        built.append(
            _time_config(
                "rerank-m3-top30",
                "rerank",
                questions,
                rerank.pages,
                chunking="180/40",
            )
        )
    except Exception as exc:
        not_run.append(
            {
                "name": RERANKER,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )

    report = _assemble(built, questions, not_run, best_bm25_name)
    write_sweep(report, out_dir)
    return report


def write_sweep(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    metrics_path.write_text(payload, encoding="utf-8")
    _write_metrics_csv(report, out_dir / "metrics.csv")
    _write_questions_csv(report, out_dir / "per_question.csv")
    (out_dir / "results.md").write_text(render_sweep(report), encoding="utf-8")
    plot_recall_latency(metrics_path, out_dir / "recall_at_10_vs_latency.png")


def render_sweep(report: dict) -> str:
    hardware = report["hardware"]
    comparison = report["comparison"]
    ci_low, ci_high = comparison["ci95_low"], comparison["ci95_high"]
    other = comparison.get("challenger", comparison["best"])
    best_is_baseline = comparison["best"] == comparison["baseline"]
    lower = comparison["significant"] and comparison["diff_recall_at_10"] < 0
    if best_is_baseline and lower:
        verdict = (
            f"{other} is lower than the hybrid baseline and the 95% interval "
            "of that difference stays below 0."
        )
    elif comparison["best"] == comparison["baseline"]:
        verdict = (
            f"No other configuration beat the hybrid baseline. "
            f"The difference for {other} is not significant: the 95% interval includes 0."
        )
    elif comparison["significant"] and comparison["diff_recall_at_10"] > 0:
        verdict = (
            f"{comparison['best']} is higher than the hybrid baseline "
            "and the 95% interval of the difference stays above 0."
        )
    else:
        verdict = (
            f"The difference between {comparison['best']} and the hybrid baseline "
            "is not significant: the 95% interval includes 0."
        )
    lines = [
        "# Phase 3 retrieval sweep",
        "",
        "Not medical advice. Retrieval only. No answer was generated.",
        "",
        report["selection"],
        "",
        (
            f"Hardware: {hardware['cpu_model']}, {hardware['cpus']} CPUs, "
            f"device {hardware['device']}. No GPU."
        ),
        "Latency is the mean online time per answerable query after one warmup query.",
        "It does not include model load or embedding the corpus.",
        "",
        f"Hit rule: {report['hit_rule']}",
        (
            f"Hybrid fuses the top {report['fusion_pool']} chunks from each side "
            f"with reciprocal rank fusion k {report['rrf_k']}."
        ),
        f"The reranker, when it ran, reorders the top {report['rerank_pool']} hybrid chunks.",
        "",
        "## All configurations",
        "",
        _config_table(report["configs"]),
        "",
        "## Answerable real questions only",
        "",
        _origin_table(report["configs"], "real"),
        "",
        "## Best against the hybrid baseline",
        "",
        (
            f"Best by recall@10: `{comparison['best']}` "
            f"({comparison['best_recall_at_10']:.4f}). "
            f"Compared config `{comparison.get('challenger', comparison['best'])}` "
            f"({comparison.get('challenger_recall_at_10', comparison['best_recall_at_10']):.4f}) "
            f"minus baseline `{comparison['baseline']}` "
            f"({comparison['baseline_recall_at_10']:.4f}) "
            f"is {comparison['diff_recall_at_10']:.4f}, "
            f"95% bootstrap interval [{ci_low:.4f}, {ci_high:.4f}] "
            f"({comparison['n_bootstrap']} paired resamples, seed {comparison['seed']}). "
            f"{comparison.get('note', '')}"
        ),
        "",
        verdict,
        "",
    ]
    versus = report.get("hybrid_vs_bm25")
    if versus:
        lines.extend(
            [
                (
                    f"Hybrid baseline minus `bm25-suffix` recall@10 is "
                    f"{versus['diff_recall_at_10']:.4f}, 95% interval "
                    f"[{versus['ci95_low']:.4f}, {versus['ci95_high']:.4f}]."
                ),
                "",
            ]
        )
    lines.append("## Not run")
    lines.append("")
    for item in report["not_run"]:
        lines.append(f"- `{item['name']}`: {item['reason']}")
    lines.extend(
        [
            "",
            "Whole-page dense embeddings are truncated at 512 tokens by the embedding model.",
            "Labels were made by an agent, on this same question set. n is 32.",
            "",
        ]
    )
    return "\n".join(lines)


def bootstrap_difference(
    candidate: list[float],
    baseline: list[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("bootstrap needs two equal non-empty score lists")
    observed = float(np.mean(candidate) - np.mean(baseline))
    rng = np.random.default_rng(seed)
    size = len(candidate)
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    indexes = rng.integers(0, size, size=(draws, size))
    diffs = left[indexes].mean(axis=1) - right[indexes].mean(axis=1)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "significant": bool(low > 0 or high < 0),
        "n_bootstrap": draws,
        "seed": seed,
    }


def _rank_key(item: dict) -> tuple:
    return (item["recall_at_10"], item["mrr"], -item["latency_ms"])


def _assemble(
    built: list[dict], questions: list[dict], not_run: list[dict], best_bm25: str
) -> dict:
    del best_bm25
    baseline = next(item for item in built if item["name"] == "hybrid-baseline")
    best = max(built, key=_rank_key)
    if best["name"] == baseline["name"]:
        challenger = max(
            (item for item in built if item["name"] != baseline["name"]),
            key=_rank_key,
        )
        compared = challenger
        note = "best is the hybrid baseline; the interval is the next config minus that baseline"
    else:
        compared = best
        note = "the interval is the best config minus the hybrid baseline"
    comparison = bootstrap_difference(
        compared["recall_at_10_by_question"],
        baseline["recall_at_10_by_question"],
    )
    comparison["diff_recall_at_10"] = comparison["diff"]
    comparison["best"] = best["name"]
    comparison["challenger"] = compared["name"]
    comparison["baseline"] = baseline["name"]
    comparison["best_recall_at_10"] = best["recall_at_10"]
    comparison["challenger_recall_at_10"] = compared["recall_at_10"]
    comparison["baseline_recall_at_10"] = baseline["recall_at_10"]
    comparison["note"] = note
    bm25 = next(item for item in built if item["name"] == "bm25-suffix")
    versus_bm25 = bootstrap_difference(
        baseline["recall_at_10_by_question"],
        bm25["recall_at_10_by_question"],
    )
    versus_bm25["diff_recall_at_10"] = versus_bm25["diff"]
    versus_bm25["left"] = baseline["name"]
    versus_bm25["right"] = bm25["name"]
    configs = []
    for item in built:
        stored = {
            key: value
            for key, value in item.items()
            if key not in {"rows", "recall_at_10_by_question"}
        }
        configs.append(stored)
    return {
        "chart_title": "Recall@10 against mean query latency",
        "selection": (
            "The best configuration is the highest recall@10 on all "
            f"{len(questions)} answerable questions. There is no held-out split. "
            "If a hybrid uses a BM25 analyzer other than the phase 2 suffix list, "
            "that analyzer was chosen on these same questions."
        ),
        "hardware": _hardware(),
        "page_tolerance": 1,
        "hit_rule": (
            "same source_id and abs(retrieved pdf_page - gold pdf_page) <= 1; "
            "one retrieved page credits one gold page"
        ),
        "fusion_pool": FUSION_POOL,
        "rrf_k": RRF_K,
        "rerank_pool": RERANK_POOL,
        "n_answerable": len(questions),
        "chunk_words": CHUNK_WORDS,
        "chunk_overlap": CHUNK_OVERLAP,
        "not_run": not_run,
        "comparison": comparison,
        "hybrid_vs_bm25": versus_bm25,
        "configs": configs,
        "per_question": [row for item in built for row in item["rows"]],
    }


def _time_config(
    name,
    family,
    questions,
    page_fn,
    *,
    chunking: str,
    selected_on_same_set: bool = False,
) -> dict:
    print(f"timing {name}", file=sys.stderr)
    page_fn(questions[0]["question"])
    latencies = []
    rows = []
    recalls10 = []
    recalls5 = []
    rr_values = []
    for question in questions:
        started = time.perf_counter()
        pages = page_fn(question["question"])
        latencies.append((time.perf_counter() - started) * 1000.0)
        scored = score_page_list(pages, question["gold"], 1)
        recalls10.append(scored["recall_at_10"])
        recalls5.append(scored["recall_at_5"])
        rr_values.append(scored["reciprocal_rank"])
        rows.append(
            {
                "config": name,
                "id": question["id"],
                "origin": question["origin"],
                "recall_at_5": scored["recall_at_5"],
                "recall_at_10": scored["recall_at_10"],
                "reciprocal_rank": scored["reciprocal_rank"],
                "first_rank": scored["first_rank"],
                "latency_ms": latencies[-1],
            }
        )
    by_origin = {}
    for origin in ("real", "plan", "synthetic"):
        group = [row for row in rows if row["origin"] == origin]
        by_origin[origin] = {
            "n": len(group),
            "recall_at_5": _mean([row["recall_at_5"] for row in group]),
            "recall_at_10": _mean([row["recall_at_10"] for row in group]),
            "mrr": _mean([row["reciprocal_rank"] for row in group]),
        }
    return {
        "name": name,
        "family": family,
        "chunking": chunking,
        "selected_on_same_set": selected_on_same_set,
        "n": len(questions),
        "recall_at_5": _mean(recalls5),
        "recall_at_10": _mean(recalls10),
        "mrr": _mean(rr_values),
        "latency_ms": _mean(latencies),
        "by_origin": by_origin,
        "recall_at_10_by_question": recalls10,
        "rows": rows,
    }


def _best_bm25_analyzer(built: list[dict]) -> str:
    candidates = [
        item for item in built if item["family"] == "bm25" and item["chunking"] == "180/40"
    ]
    winner = max(candidates, key=_rank_key)
    # names are bm25-<analyzer name>
    return winner["name"].removeprefix("bm25-")


class _Bm25Ranker:
    def __init__(self, chunks: list[Chunk], analyzer: Analyzer) -> None:
        self.chunks = chunks
        self.analyzer = analyzer
        self.index = BM25Index([analyzer.tokens(chunk.text) for chunk in chunks])

    def pages(self, query: str) -> list[tuple[str, int]]:
        return pages_from_scores(self.chunks, self.index.score_all(self.analyzer.tokens(query)))


class _DenseRanker:
    def __init__(self, chunks: list[Chunk], matrix: np.ndarray, embedder: object) -> None:
        self.chunks = chunks
        self.matrix = matrix
        self.embedder = embedder

    def pages(self, query: str) -> list[tuple[str, int]]:
        vectors = self.embedder.embed_queries([query])
        query_matrix = l2_normalize(np.asarray(vectors, dtype=np.float32))
        scores = self.matrix @ query_matrix[0]
        return pages_from_scores(self.chunks, scores)


class _HybridRanker:
    def __init__(self, chunks: list[Chunk], sparse: _Bm25Ranker, dense: _DenseRanker) -> None:
        self.chunks = chunks
        self.sparse = sparse
        self.dense = dense
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def chunk_order(self, query: str) -> list[Chunk]:
        sparse_scores = self.sparse.index.score_all(self.sparse.analyzer.tokens(query))
        query_matrix = l2_normalize(
            np.asarray(self.dense.embedder.embed_queries([query]), dtype=np.float32)
        )
        dense_scores = self.dense.matrix @ query_matrix[0]
        sparse_ids = _ids_by_score(self.chunks, sparse_scores)[:FUSION_POOL]
        dense_ids = _ids_by_score(self.chunks, dense_scores)[:FUSION_POOL]
        fused = reciprocal_rank_fusion([sparse_ids, dense_ids], k=RRF_K)
        return [self.by_id[chunk_id] for chunk_id, _score in fused]

    def pages(self, query: str) -> list[tuple[str, int]]:
        return collapse_pages(self.chunk_order(query))


class _RerankRanker:
    def __init__(self, hybrid: _HybridRanker, reranker: object, pool: int) -> None:
        self.hybrid = hybrid
        self.reranker = reranker
        self.pool = pool

    def pages(self, query: str) -> list[tuple[str, int]]:
        ordered = self.hybrid.chunk_order(query)
        head = ordered[: self.pool]
        tail = ordered[self.pool :]
        scores = list(self.reranker.rerank(query, [chunk.text for chunk in head]))
        paired = sorted(zip(head, scores, strict=True), key=lambda item: -item[1])
        ranked = [chunk for chunk, _score in paired]
        return collapse_pages(ranked + tail)


class _PlainEmbedder:
    """MiniLM has no query/passage prefix. E5 does, and that stays in E5Embedder."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)
        self.name = model_name

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        rows = [np.asarray(vector, dtype=np.float32) for vector in self.model.embed(texts)]
        return np.vstack(rows) if rows else np.zeros((0, 1), dtype=np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)


def _load_e5():
    from baby_food_rag.embed import E5Embedder

    return E5Embedder()


def _load_reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=RERANKER)


def _e5_matrix(index, chunks: list[Chunk], cache_dir: Path) -> np.ndarray:
    cached = cache_dir / "e5-180.npy"
    if [chunk.text for chunk in index.chunks] == [chunk.text for chunk in chunks]:
        matrix = np.asarray(index.embeddings, dtype=np.float32)
        np.save(cached, matrix)
        return matrix
    return _cached_matrix(_load_e5(), chunks, cached)


def _cached_matrix(embedder: object, chunks: list[Chunk], path: Path) -> np.ndarray:
    if path.is_file():
        matrix = np.load(path)
        if matrix.shape[0] == len(chunks):
            return np.asarray(matrix, dtype=np.float32)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    matrix = l2_normalize(np.asarray(vectors, dtype=np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    return matrix


def _page_chunks(manifest_path: Path, raw_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source in default_index_sources(load_manifest(manifest_path)):
        pages = extract_pdf(raw_dir / source.filename)
        chunks.extend(chunk_pages(source, pages, chunk_words=10_000, overlap=0))
    return chunks


def _ids_by_score(chunks: list[Chunk], scores: np.ndarray) -> list[str]:
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), chunks[i].chunk_id))
    return [chunks[i].chunk_id for i in order if float(scores[i]) > 0]


def _hardware() -> dict:
    model = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu_model": model,
        "cpus": __import__("os").cpu_count(),
        "device": "cpu",
        "gpu": False,
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _config_table(configs: list[dict]) -> str:
    lines = [
        "| Config | Family | Recall@5 | Recall@10 | MRR | Latency ms |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(configs, key=lambda item: (-item["recall_at_10"], item["latency_ms"]))
    for item in ordered:
        lines.append(
            f"| {item['name']} | {item['family']} | {item['recall_at_5']:.4f} | "
            f"{item['recall_at_10']:.4f} | {item['mrr']:.4f} | {item['latency_ms']:.1f} |"
        )
    return "\n".join(lines)


def _origin_table(configs: list[dict], origin: str) -> str:
    lines = [
        "| Config | n | Recall@5 | Recall@10 | MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(configs, key=lambda item: -item["by_origin"][origin]["recall_at_10"])
    for item in ordered:
        block = item["by_origin"][origin]
        lines.append(
            f"| {item['name']} | {block['n']} | {block['recall_at_5']:.4f} | "
            f"{block['recall_at_10']:.4f} | {block['mrr']:.4f} |"
        )
    return "\n".join(lines)


def _write_metrics_csv(report: dict, path: Path) -> None:
    fields = [
        "name",
        "family",
        "chunking",
        "n",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "latency_ms",
        "real_n",
        "real_recall_at_10",
        "real_mrr",
        "plan_recall_at_10",
        "synthetic_recall_at_10",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in report["configs"]:
            writer.writerow(
                {
                    "name": item["name"],
                    "family": item["family"],
                    "chunking": item["chunking"],
                    "n": item["n"],
                    "recall_at_5": f"{item['recall_at_5']:.6f}",
                    "recall_at_10": f"{item['recall_at_10']:.6f}",
                    "mrr": f"{item['mrr']:.6f}",
                    "latency_ms": f"{item['latency_ms']:.3f}",
                    "real_n": item["by_origin"]["real"]["n"],
                    "real_recall_at_10": f"{item['by_origin']['real']['recall_at_10']:.6f}",
                    "real_mrr": f"{item['by_origin']['real']['mrr']:.6f}",
                    "plan_recall_at_10": f"{item['by_origin']['plan']['recall_at_10']:.6f}",
                    "synthetic_recall_at_10": (
                        f"{item['by_origin']['synthetic']['recall_at_10']:.6f}"
                    ),
                }
            )


def _write_questions_csv(report: dict, path: Path) -> None:
    fields = [
        "config",
        "id",
        "origin",
        "recall_at_5",
        "recall_at_10",
        "reciprocal_rank",
        "first_rank",
        "latency_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["per_question"]:
            writer.writerow(
                {
                    "config": row["config"],
                    "id": row["id"],
                    "origin": row["origin"],
                    "recall_at_5": f"{row['recall_at_5']:.6f}",
                    "recall_at_10": f"{row['recall_at_10']:.6f}",
                    "reciprocal_rank": f"{row['reciprocal_rank']:.6f}",
                    "first_rank": "" if row["first_rank"] is None else row["first_rank"],
                    "latency_ms": f"{row['latency_ms']:.3f}",
                }
            )
