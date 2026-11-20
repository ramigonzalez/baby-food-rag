"""Command line: download, index, search, and Portuguese answers.

Answers cite the retrieved page. They are not medical advice.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from baby_food_rag.download import download_sources, select_for_download
from baby_food_rag.evaluate import load_questions, render_results, run_eval, write_results
from baby_food_rag.generate import (
    DEFAULT_K,
    DEFAULT_MODEL,
    MissingAPIKey,
    OpenRouterLLM,
    ask,
    format_answer_markdown,
)
from baby_food_rag.generate_eval import (
    load_eval_questions,
    not_run_report,
    render_generation_report,
    run_generation_eval,
    write_generation_report,
)
from baby_food_rag.index import (
    build_index,
    format_elapsed,
    format_hits,
    format_stats,
    load_index,
    query_embedder,
    search,
)
from baby_food_rag.manifest import find_manifest, load_manifest, sources_for_index
from baby_food_rag.sweep import run_sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="baby-food",
        description="Search official infant-feeding guides. Not medical advice.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="Download PDFs listed as download: true")
    download.add_argument("--dest", type=Path, default=Path("data/raw"))
    download.add_argument("--id", action="append", default=None, help="Download one source id")
    download.add_argument(
        "--force", action="store_true", help="Fetch again even if the file exists"
    )
    download.add_argument("--manifest", type=Path, default=None)

    index = sub.add_parser("index", help="Extract, chunk, and embed the default core")
    index.add_argument(
        "--include",
        action="append",
        default=[],
        choices=["optional", "contrast", "reference"],
        help="Also index this group. Default is the Brazilian core only.",
    )
    index.add_argument("--embedder", default="e5", choices=["e5", "hash"])
    index.add_argument("--raw", type=Path, default=Path("data/raw"))
    index.add_argument("--out", type=Path, default=Path("data/index"))
    index.add_argument("--manifest", type=Path, default=None)

    search_cmd = sub.add_parser("search", help="Return the top chunks for a query")
    search_cmd.add_argument("query")
    search_cmd.add_argument("-k", type=int, default=5)
    search_cmd.add_argument("--retriever", default="hybrid", choices=["bm25", "dense", "hybrid"])
    search_cmd.add_argument("--index", type=Path, default=Path("data/index"))

    stats = sub.add_parser("stats", help="Pages and chunks in an index")
    stats.add_argument("--index", type=Path, default=Path("data/index"))

    evaluate = sub.add_parser("eval", help="Score BM25, dense, and hybrid on the labeled set")
    evaluate.add_argument("--set", type=Path, default=Path("eval/set.yaml"), dest="question_set")
    evaluate.add_argument("--index", type=Path, default=Path("data/index"))
    evaluate.add_argument("--out", type=Path, default=Path("results/phase2-baseline"))
    evaluate.add_argument(
        "--page-tolerance",
        type=int,
        default=1,
        help="A gold page hits when the PDF page is within this distance. Default 1.",
    )

    sweep = sub.add_parser("sweep", help="Compare BM25, dense, hybrid, and rerank settings")
    sweep.add_argument("--set", type=Path, default=Path("eval/set.yaml"), dest="question_set")
    sweep.add_argument("--index", type=Path, default=Path("data/index"))
    sweep.add_argument("--raw", type=Path, default=Path("data/raw"))
    sweep.add_argument("--manifest", type=Path, default=None)
    sweep.add_argument("--out", type=Path, default=Path("results/phase3-sweep"))
    sweep.add_argument("--cache", type=Path, default=Path("data/index/sweep-cache"))

    ask_cmd = sub.add_parser("ask", help="Answer in Portuguese from the hybrid index")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("-k", type=int, default=DEFAULT_K)
    ask_cmd.add_argument("--model", default=DEFAULT_MODEL)
    ask_cmd.add_argument("--index", type=Path, default=Path("data/index"))

    ask_eval = sub.add_parser("ask-eval", help="Generate and judge the labeled set")
    ask_eval.add_argument("--set", type=Path, default=Path("eval/set.yaml"), dest="question_set")
    ask_eval.add_argument("--index", type=Path, default=Path("data/index"))
    ask_eval.add_argument("--out", type=Path, default=Path("results/phase4-generation"))
    ask_eval.add_argument("--model", default=DEFAULT_MODEL)
    ask_eval.add_argument("-k", type=int, default=DEFAULT_K)

    args = parser.parse_args(argv)
    if args.command == "download":
        return _download(args)
    if args.command == "index":
        return _index(args)
    if args.command == "search":
        return _search(args)
    if args.command == "stats":
        return _stats(args)
    if args.command == "eval":
        return _eval(args)
    if args.command == "sweep":
        return _sweep(args)
    if args.command == "ask":
        return _ask(args)
    if args.command == "ask-eval":
        return _ask_eval(args)
    parser.error(f"unknown command {args.command}")
    return 2


def _download(args: argparse.Namespace) -> int:
    manifest = args.manifest or find_manifest()
    try:
        chosen = select_for_download(load_manifest(manifest), args.id)
    except KeyError as exc:
        print(f"Unknown source id {exc.args[0]}", file=sys.stderr)
        return 2
    results = download_sources(chosen, args.dest, force=args.force)
    failed = 0
    for result in results:
        line = f"{result.status.upper():7} {result.source_id}  {result.message}"
        if result.sha256:
            line += f"  sha256={result.sha256}"
        print(line)
        if result.status == "failed":
            failed += 1
    if failed:
        print(f"{failed} download(s) failed. The others were kept.", file=sys.stderr)
        return 1
    return 0


def _index(args: argparse.Namespace) -> int:
    manifest = args.manifest or find_manifest()
    sources = sources_for_index(load_manifest(manifest), set(args.include))
    if not sources:
        print("No sources selected.", file=sys.stderr)
        return 2
    started = time.perf_counter()
    try:
        index = build_index(sources, args.raw, args.out, embedder_name=args.embedder)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(format_stats(index.meta), end="")
    print(f"wrote {args.out}")
    print(f"elapsed {format_elapsed(time.perf_counter() - started)}", file=sys.stderr)
    return 0


def _search(args: argparse.Namespace) -> int:
    try:
        index = load_index(args.index)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    hits = search(index, args.query, k=args.k, retriever=args.retriever)
    print(format_hits(args.query, args.retriever, hits), end="")
    return 0


def _eval(args: argparse.Namespace) -> int:
    try:
        index = load_index(args.index)
        questions = load_questions(args.question_set)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    embedder = query_embedder(index)
    report = run_eval(
        index,
        questions,
        embedder=embedder,
        tolerance=args.page_tolerance,
    )
    write_results(report, args.out)
    print(render_results(report), end="")
    print(f"wrote {args.out}")
    return 0


def _sweep(args: argparse.Namespace) -> int:
    try:
        report = run_sweep(
            question_path=args.question_set,
            index_dir=args.index,
            raw_dir=args.raw,
            manifest_path=args.manifest or find_manifest(),
            out_dir=args.out,
            cache_dir=args.cache,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    comparison = report["comparison"]
    compared = comparison.get("challenger", comparison["best"])
    compared_score = comparison.get("challenger_recall_at_10", comparison["best_recall_at_10"])
    print(
        f"{compared} minus {comparison['baseline']} "
        f"recall@10 {comparison['diff_recall_at_10']:.4f} "
        f"({compared} {compared_score:.4f}, "
        f"{comparison['baseline']} {comparison['baseline_recall_at_10']:.4f}) "
        f"ci95 [{comparison['ci95_low']:.4f}, {comparison['ci95_high']:.4f}]"
    )
    print(f"wrote {args.out}")
    return 0


def _ask(args: argparse.Namespace) -> int:
    try:
        index = load_index(args.index)
        llm = OpenRouterLLM(model=args.model)
    except MissingAPIKey as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    answer = ask(index, args.question, llm, k=args.k, embedder=query_embedder(index))
    print(format_answer_markdown(answer))
    print(f"\nlatency_s={answer.latency_s:.2f} model={answer.model}")
    return 0


def _ask_eval(args: argparse.Namespace) -> int:
    out_dir = args.out
    try:
        llm = OpenRouterLLM(model=args.model)
    except MissingAPIKey as exc:
        report = not_run_report(str(exc))
        write_generation_report(report, out_dir)
        print(render_generation_report(report), end="")
        print(f"wrote {out_dir}", file=sys.stderr)
        return 1
    try:
        index = load_index(args.index)
        questions = load_eval_questions(args.question_set)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    report = run_generation_eval(
        index,
        questions,
        llm,
        k=args.k,
        embedder=query_embedder(index),
    )
    write_generation_report(report, out_dir)
    print(render_generation_report(report), end="")
    print(f"wrote {out_dir}")
    return 0


def _stats(args: argparse.Namespace) -> int:
    try:
        index = load_index(args.index)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(format_stats(index.meta), end="")
    return 0
