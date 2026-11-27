"""Phase 1a: does a hosted embedder beat e5 on these Portuguese guides?

The variable is the embedding model: the local ``intfloat/multilingual-e5-large``
(the baseline, ``data/index``) against ``openai/text-embedding-3-large``
through OpenRouter (the candidate, ``data/index-te3-large``). Fixed: golden
set v2, 180/40 chunks, the suffix BM25, and the fusion. Both indexes must
hold the same chunks, or the run stops.

Dense and hybrid are both scored, since they were tied on the v2 test
split. The paired bootstrap is ``sweep.bootstrap_difference`` (10,000
resamples, seed 0) on per-question recall@10, candidate minus baseline.

Query latency is one uncached ``embed_queries`` call per answerable
question, after one warmup call. For the hosted model the cache is
bypassed, so each timed call is a real, billed request.

The decision rule is fixed in ``DECISION_RULE`` and applied to dev only.
The test split is meant to be run once, as the held-out confirmation; a
second test run needs ``rerun_test=True``. Results for both splits share
one folder: each run adds its split to ``metrics.json`` and rewrites the
other files from it.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from baby_food_rag.embed import OPENROUTER_EMBEDDERS, OpenRouterEmbedder, load_embedder
from baby_food_rag.evaluate import (
    CONTEXT_BUDGET_K,
    FUSION_POOL,
    ORIGINS,
    RRF_K,
    load_questions,
    run_eval,
    select_split,
)
from baby_food_rag.index import CorpusIndex, load_index, query_embedder
from baby_food_rag.sweep import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, bootstrap_difference

BASELINE = "e5"
CANDIDATE = "te3-large"
RETRIEVERS = ("dense", "hybrid")
SPLITS = ("dev", "test")
TOLERANCE = 1
LATENCY_LIMIT_MS = 1000.0
DECISION_RULE = (
    "te3-large becomes the default embedder only if, on dev, the 95% paired CI of the "
    "hybrid recall@10 difference excludes 0 in its favour, and its median query latency "
    "is under 1 second. Otherwise e5 stays."
)


def run_comparison(
    *,
    question_path: Path,
    baseline_index: Path,
    candidate_index: Path,
    out_dir: Path,
    split: str,
    rerun_test: bool = False,
) -> dict:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    report = load_report(out_dir)
    if split == "test" and "test" in report["splits"] and not rerun_test:
        raise ValueError(
            f"The test split is already in {out_dir / 'metrics.json'}. "
            "It is the held-out confirmation and runs once. Pass --rerun-test to replace it."
        )
    questions = load_questions(question_path)
    baseline = load_index(baseline_index)
    candidate = load_index(candidate_index)
    check_same_chunks(baseline, candidate)
    texts = [item["question"] for item in select_split(questions, split) if item["answerable"]]
    embedders = {}
    for label, index, path in (
        (BASELINE, baseline, baseline_index),
        (CANDIDATE, candidate, candidate_index),
    ):
        print(f"{label}: scoring {split} with {index.embedder_name}", file=sys.stderr)
        embedder = query_embedder(index) or load_embedder(index.embedder_name)
        evaluation = run_eval(
            index,
            questions,
            embedder=embedder,
            tolerance=TOLERANCE,
            retrievers=RETRIEVERS,
            split=split,
        )
        print(f"{label}: timing {len(texts)} query embeddings", file=sys.stderr)
        timer = _latency_embedder(index, embedder)
        latency = measure_latency(timer, texts)
        embedders[label] = {
            "embedder": index.embedder_name,
            "index": str(path),
            "latency": latency,
            "query_usage": _usage([embedder, timer]),
            "eval": evaluation,
        }
    report["splits"][split] = {
        "split": split,
        "n_answerable": len(texts),
        "embedders": embedders,
        "comparisons": paired_comparisons(embedders),
    }
    report["hardware"] = _hardware()
    report["index_usage"] = candidate.meta.get("embedding_usage")
    if "dev" in report["splits"]:
        report["decision"] = apply_rule(report["splits"]["dev"])
    write_comparison(report, out_dir)
    return report


def empty_report() -> dict:
    return {
        "title": "Phase 1a embedder comparison",
        "decision_rule": DECISION_RULE,
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "retrievers": list(RETRIEVERS),
        "page_tolerance": TOLERANCE,
        "fusion_pool": FUSION_POOL,
        "rrf_k": RRF_K,
        "context_budget_k": CONTEXT_BUDGET_K,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
        "latency_limit_ms": LATENCY_LIMIT_MS,
        "splits": {},
    }


def load_report(out_dir: Path) -> dict:
    path = out_dir / "metrics.json"
    if not path.is_file():
        return empty_report()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("decision_rule") != DECISION_RULE:
        raise ValueError(f"{path} was written under a different decision rule")
    return report


def check_same_chunks(baseline: CorpusIndex, candidate: CorpusIndex) -> None:
    left = [(chunk.chunk_id, chunk.text) for chunk in baseline.chunks]
    right = [(chunk.chunk_id, chunk.text) for chunk in candidate.chunks]
    if left != right:
        raise ValueError("The two indexes do not hold the same chunks. Rebuild one of them.")
    for key in ("chunk_words", "chunk_overlap"):
        if baseline.meta.get(key) != candidate.meta.get(key):
            raise ValueError(f"The two indexes differ on {key}")


def measure_latency(embedder: object, texts: list[str], *, clock=time.perf_counter) -> dict:
    """Milliseconds per single-query ``embed_queries`` call, after one untimed warmup."""
    if not texts:
        return {"n": 0, "median_ms": 0.0, "mean_ms": 0.0, "p90_ms": 0.0, "max_ms": 0.0}
    embedder.embed_queries([texts[0]])
    times = []
    for text in texts:
        started = clock()
        embedder.embed_queries([text])
        times.append((clock() - started) * 1000.0)
    ordered = sorted(times)
    return {
        "n": len(times),
        "median_ms": float(statistics.median(times)),
        "mean_ms": float(statistics.fmean(times)),
        "p90_ms": float(ordered[int(0.9 * (len(ordered) - 1))]),
        "max_ms": float(ordered[-1]),
    }


def paired_comparisons(embedders: dict) -> dict:
    """Candidate minus baseline on per-question recall@10, for each retriever."""
    result = {}
    for retriever in RETRIEVERS:
        left = _per_question(embedders[CANDIDATE]["eval"], retriever)
        right = _per_question(embedders[BASELINE]["eval"], retriever)
        if [row["id"] for row in left] != [row["id"] for row in right]:
            raise ValueError("per-question rows are not in the same order")
        boot = bootstrap_difference(
            [row["recall_at_10"] for row in left],
            [row["recall_at_10"] for row in right],
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
        )
        result[retriever] = {
            "metric": "recall_at_10",
            "candidate": CANDIDATE,
            "baseline": BASELINE,
            "candidate_mean": _mean([row["recall_at_10"] for row in left]),
            "baseline_mean": _mean([row["recall_at_10"] for row in right]),
            **boot,
        }
    return result


def apply_rule(dev: dict) -> dict:
    """The pre-registered rule, on the dev block only."""
    hybrid = dev["comparisons"]["hybrid"]
    latency = dev["embedders"][CANDIDATE]["latency"]["median_ms"]
    ci_in_favour = hybrid["ci95_low"] > 0
    fast_enough = latency < LATENCY_LIMIT_MS
    adopt = ci_in_favour and fast_enough
    reasons = []
    if not ci_in_favour:
        reasons.append(
            f"the dev hybrid recall@10 interval [{hybrid['ci95_low']:.4f}, "
            f"{hybrid['ci95_high']:.4f}] does not exclude 0 in te3-large's favour"
        )
    if not fast_enough:
        reasons.append(f"the median query latency is {latency:.0f} ms, not under 1000 ms")
    if adopt:
        verdict = (
            "te3-large meets the rule: the dev hybrid recall@10 interval "
            f"[{hybrid['ci95_low']:.4f}, {hybrid['ci95_high']:.4f}] is above 0 and the median "
            f"query latency is {latency:.0f} ms. By the rule it becomes the default embedder. "
            "The code default was not changed in this phase."
        )
    else:
        verdict = "e5 stays the default embedder: " + "; and ".join(reasons) + "."
    return {
        "rule": DECISION_RULE,
        "split": "dev",
        "hybrid_ci95_low": hybrid["ci95_low"],
        "hybrid_ci95_high": hybrid["ci95_high"],
        "ci_excludes_zero_in_favour": ci_in_favour,
        "candidate_median_latency_ms": latency,
        "latency_under_limit": fast_enough,
        "candidate_becomes_default": adopt,
        "verdict": verdict,
    }


def write_comparison(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_metrics_csv(report, out_dir / "metrics.csv")
    _write_questions_csv(report, out_dir / "per_question.csv")
    (out_dir / "results.md").write_text(render_comparison(report), encoding="utf-8")


def render_comparison(report: dict) -> str:
    lines = [
        "# Phase 1a: embedder comparison",
        "",
        "Not medical advice. Retrieval only. No answer was generated and no chat model",
        "was called.",
        "",
        "## Pre-registered decision rule",
        "",
        "Written and committed before any comparison number was computed:",
        "",
        f"> {report['decision_rule']}",
        "",
        "The rule is applied to the dev split only. The test split is run once afterwards",
        "for both embedders, as the held-out confirmation, and does not change the decision.",
        "",
        "## Setup",
        "",
        (
            "Variable: the embedding model. `e5` is `intfloat/multilingual-e5-large` "
            "(local, ONNX through fastembed, with `query: ` and `passage: ` prefixes), index "
            "`data/index`. `te3-large` is `openai/text-embedding-3-large` through the "
            "OpenRouter embeddings endpoint (3072 dimensions, no prefixes), index "
            "`data/index-te3-large`."
        ),
        "",
        (
            "Fixed: golden set v2 (`eval/set.yaml`), 180/40 word chunks (the run stops unless "
            "both indexes hold the same chunks), the suffix BM25 (k1 1.5, b 0.75), and hybrid "
            f"reciprocal rank fusion (k {report['rrf_k']}) of the top {report['fusion_pool']} "
            "BM25 and dense chunks."
        ),
        "",
        (
            f"A hit is the same `source_id` and a PDF page within {report['page_tolerance']} "
            "of a gold page; one retrieved page credits one gold page. Recall@k and MRR are "
            "as in phase 2. nDCG@10 uses binary gains on pages under the same one-credit rule. "
            f"Ctx recall is the share of gold pages covered by the pages of the top "
            f"{report['context_budget_k']} chunks from `index.search`, the call "
            "`baby-food ask` makes; only the hybrid row is what the chat sends today."
        ),
        "",
        (
            "Paired bootstrap: te3-large minus e5 on per-question recall@10, "
            f"{report['bootstrap']['draws']:,} resamples of the questions, seed "
            f"{report['bootstrap']['seed']}, 2.5 and 97.5 percentiles."
        ),
        "",
        (
            "Latency: one `embed_queries` call per answerable question, after one untimed "
            "warmup. e5 runs on this CPU with the model already loaded. te3-large bypasses the "
            "cache, so each call is a real HTTPS request to OpenRouter from this machine."
        ),
    ]
    hardware = report.get("hardware")
    if hardware:
        lines.extend(
            [
                "",
                f"Hardware: {hardware['cpu_model']}, {hardware['cpus']} CPUs, "
                f"{hardware['system']}. e5 runs on the CPU (ONNX); no GPU is used.",
            ]
        )
    lines.append("")
    splits = report.get("splits", {})
    if not splits:
        lines.extend(
            [
                "## Numbers",
                "",
                "Not run yet. This file holds the rule above, committed before the comparison.",
                "",
            ]
        )
        return "\n".join(lines)
    if "dev" in splits:
        lines.extend(_split_section(splits["dev"], "Dev"))
        decision = report.get("decision")
        if decision:
            lines.extend(["## Decision (rule applied to dev)", "", decision["verdict"], ""])
    if "test" in splits:
        lines.extend(_split_section(splits["test"], "Test, held-out confirmation"))
        lines.extend(
            [
                "The test split was run once, after the dev decision. It is reported as a",
                "check and does not change the decision.",
                "",
            ]
        )
        if "dev" in splits:
            lines.extend([*_agreement(splits["dev"], splits["test"]), ""])
    lines.extend(_cost_section(report))
    lines.extend(
        [
            "## Limitations",
            "",
            "- Gold pages and the synthetic questions were written by an agent; the owner",
            "  reviewed a sample of 20 and no clinician has checked them.",
            "- Real parent questions are 4 per split, too few to read on their own.",
            "- te3-large latency is network time from this machine to OpenRouter and the",
            "  provider behind it. It varies with the connection and the hour.",
            "- Cost uses the API's token counts times the listed price, not an invoice.",
            "",
        ]
    )
    return "\n".join(lines)


def _split_section(block: dict, title: str) -> list[str]:
    lines = [
        f"## {title} (n = {block['n_answerable']})",
        "",
        "| Embedder | Retriever | Recall@5 | Recall@10 | MRR | nDCG@10 | Ctx recall | n |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for retriever in RETRIEVERS:
        for label in (BASELINE, CANDIDATE):
            row = _summary(block["embedders"][label]["eval"], retriever)
            lines.append(
                f"| {label} | {retriever} | {row['recall_at_5']:.4f} | "
                f"{row['recall_at_10']:.4f} | {row['mrr']:.4f} | {row['ndcg_at_10']:.4f} | "
                f"{row['context_budget_recall']:.4f} | {row['n']} |"
            )
    lines.extend(
        [
            "",
            "### By origin",
            "",
            "| Embedder | Retriever | Origin | Recall@5 | Recall@10 | MRR | nDCG@10 | "
            "Ctx recall | n |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for origin in ORIGINS:
        for retriever in RETRIEVERS:
            for label in (BASELINE, CANDIDATE):
                group = _group(block["embedders"][label]["eval"], retriever, origin)
                if group is None:
                    continue
                lines.append(
                    f"| {label} | {retriever} | {origin} | {group['recall_at_5']:.4f} | "
                    f"{group['recall_at_10']:.4f} | {group['mrr']:.4f} | "
                    f"{group['ndcg_at_10']:.4f} | {group['context_budget_recall']:.4f} | "
                    f"{group['n']} |"
                )
    lines.extend(
        [
            "",
            "### te3-large minus e5, recall@10 (paired bootstrap)",
            "",
            "| Retriever | e5 | te3-large | Difference | 95% CI | Excludes 0 |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for retriever in RETRIEVERS:
        item = block["comparisons"][retriever]
        lines.append(
            f"| {retriever} | {item['baseline_mean']:.4f} | {item['candidate_mean']:.4f} | "
            f"{item['diff']:+.4f} | [{item['ci95_low']:+.4f}, {item['ci95_high']:+.4f}] | "
            f"{'yes' if item['significant'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "### Query-embedding latency",
            "",
            "| Embedder | Median ms | Mean ms | p90 ms | Max ms | n |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in (BASELINE, CANDIDATE):
        latency = block["embedders"][label]["latency"]
        lines.append(
            f"| {label} | {latency['median_ms']:.1f} | {latency['mean_ms']:.1f} | "
            f"{latency['p90_ms']:.1f} | {latency['max_ms']:.1f} | {latency['n']} |"
        )
    lines.append("")
    return lines


def _agreement(dev: dict, test: dict) -> list[str]:
    """Does the test interval point the same way as dev, retriever by retriever?"""
    lines = []
    for retriever in RETRIEVERS:
        left, right = dev["comparisons"][retriever], test["comparisons"][retriever]
        verdicts = [_direction(left), _direction(right)]
        same = "agree" if verdicts[0] == verdicts[1] else "do not agree"
        lines.append(
            f"- {retriever}: dev [{left['ci95_low']:+.4f}, {left['ci95_high']:+.4f}] "
            f"({verdicts[0]}), test [{right['ci95_low']:+.4f}, {right['ci95_high']:+.4f}] "
            f"({verdicts[1]}). The two splits {same}."
        )
    return lines


def _direction(item: dict) -> str:
    if item["ci95_low"] > 0:
        return "te3-large higher"
    if item["ci95_high"] < 0:
        return "te3-large lower"
    return "includes 0"


def _cost_section(report: dict) -> list[str]:
    rows = []
    if report.get("index_usage"):
        rows.append(("Index build (from the index meta.json)", report["index_usage"]))
    for split in SPLITS:
        block = report["splits"].get(split)
        usage = block["embedders"][CANDIDATE]["query_usage"] if block else None
        if usage:
            rows.append((f"{split} run: queries and timing calls", usage))
    if not rows:
        return []
    lines = [
        "## Cost (te3-large; e5 is local)",
        "",
        "| Item | Calls | Tokens | USD at the listed price | USD reported by OpenRouter |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    total_tokens = 0
    total_usd = 0.0
    total_reported = 0.0
    for name, usage in rows:
        total_tokens += usage["prompt_tokens"]
        total_usd += usage["usd"]
        total_reported += usage.get("reported_usd", 0.0)
        lines.append(
            f"| {name} | {usage['calls']} | {usage['prompt_tokens']:,} | "
            f"${usage['usd']:.4f} | {_reported(usage.get('reported_usd', 0.0))} |"
        )
    price = rows[0][1]["usd_per_million"]
    lines.extend(
        [
            f"| Total | | {total_tokens:,} | ${total_usd:.4f} | {_reported(total_reported)} |",
            "",
            f"Listed price: ${price:.2f} per million input tokens. A rerun with the same chunks",
            "and questions reads every vector from `data/cache/embeddings.sqlite`; only the",
            "uncached timing calls are billed again.",
            "",
        ]
    )
    return lines


def _reported(usd: float) -> str:
    return f"${usd:.4f}" if usd else "not sent"


def _latency_embedder(index: CorpusIndex, embedder: object) -> object:
    """The hosted model is timed without its cache, so every call is a real request."""
    if index.embedder_name in OPENROUTER_EMBEDDERS:
        return OpenRouterEmbedder(index.embedder_name, cache=None)
    return embedder


def _usage(embedders: list[object]) -> dict | None:
    priced = [item for item in embedders if getattr(item, "usd_per_million", None) is not None]
    if not priced:
        return None
    price = priced[0].usd_per_million
    totals = {
        "prompt_tokens": 0,
        "calls": 0,
        "texts_sent": 0,
        "cache_hits": 0,
        "reported_usd": 0.0,
    }
    seen: set[int] = set()
    for item in priced:
        if id(item) in seen:
            continue
        seen.add(id(item))
        usage = item.usage.as_dict(price)
        for key in totals:
            totals[key] += usage[key]
    totals["usd_per_million"] = price
    totals["usd"] = totals["prompt_tokens"] * price / 1_000_000
    return totals


def _per_question(evaluation: dict, retriever: str) -> list[dict]:
    rows = evaluation["by_tolerance"][str(TOLERANCE)]["per_question"]
    return [row for row in rows if row["retriever"] == retriever]


def _summary(evaluation: dict, retriever: str) -> dict:
    return next(row for row in evaluation["retrievers"] if row["retriever"] == retriever)


def _group(evaluation: dict, retriever: str, origin: str) -> dict | None:
    for group in _summary(evaluation, retriever)["groups"]:
        if group["split"] == "all" and group["origin"] == origin:
            return group
    return None


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _hardware() -> dict:
    model = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            found = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            model = found or model
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "cpu_model": model,
        "cpus": os.cpu_count(),
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


def _scored_blocks(report: dict):
    """(split, embedder label, embedder block, tolerance, scored) for every stored eval."""
    for split, block in report["splits"].items():
        for label, item in block["embedders"].items():
            for tolerance, scored in item["eval"]["by_tolerance"].items():
                yield split, label, item, tolerance, scored


def _write_metrics_csv(report: dict, path: Path) -> None:
    fields = [
        "split",
        "embedder",
        "model",
        "retriever",
        "page_tolerance",
        "origin",
        "n",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "context_budget_recall",
        "median_query_latency_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, label, item, tolerance, scored in _scored_blocks(report):
            for row in scored["retrievers"]:
                for group in row["groups"]:
                    if group["split"] != "all":
                        continue
                    writer.writerow(
                        {
                            "split": split,
                            "embedder": label,
                            "model": item["embedder"],
                            "retriever": row["retriever"],
                            "page_tolerance": tolerance,
                            "origin": group["origin"],
                            "n": group["n"],
                            **_formatted(group, _METRICS),
                            "median_query_latency_ms": f"{item['latency']['median_ms']:.3f}",
                        }
                    )


def _write_questions_csv(report: dict, path: Path) -> None:
    fields = [
        "split",
        "embedder",
        "retriever",
        "page_tolerance",
        "id",
        "origin",
        "recall_at_5",
        "recall_at_10",
        "reciprocal_rank",
        "first_rank",
        "ndcg_at_10",
        "context_budget_recall",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, label, _item, tolerance, scored in _scored_blocks(report):
            for row in scored["per_question"]:
                writer.writerow(
                    {
                        "split": split,
                        "embedder": label,
                        "retriever": row["retriever"],
                        "page_tolerance": tolerance,
                        "id": row["id"],
                        "origin": row["origin"],
                        **_formatted(row, _QUESTION_METRICS),
                        "first_rank": "" if row["first_rank"] is None else row["first_rank"],
                    }
                )


_METRICS = ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10", "context_budget_recall")
_QUESTION_METRICS = (
    "recall_at_5",
    "recall_at_10",
    "reciprocal_rank",
    "ndcg_at_10",
    "context_budget_recall",
)


def _formatted(row: dict, keys: tuple[str, ...]) -> dict:
    return {key: f"{row[key]:.6f}" for key in keys}
