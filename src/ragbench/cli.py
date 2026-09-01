"""Command line interface and terminal reporting for RAGBench."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
from ragbench import __version__
from ragbench.benchmark import BenchmarkRunner
from ragbench.dataset import load_dataset_from_json
from ragbench.models import BenchmarkConfig, BenchmarkReport


def format_ascii_table(report: BenchmarkReport) -> str:
    """Format benchmark report into a clean, human-readable ASCII table."""
    lines: List[str] = []
    lines.append("=" * 88)
    lines.append(f" RAGBench Evaluation Report: {report.dataset_name}")
    lines.append(f" Documents: {report.num_documents} | Queries: {report.num_queries}")
    lines.append("=" * 88)

    if not report.results:
        lines.append(" No benchmark results available.")
        lines.append("=" * 88)
        return "\n".join(lines)

    first_res = report.results[0]
    k_keys = sorted(first_res.metrics.recall_at_k.keys())
    k_headers = " | ".join([f"R@{k:<2}" for k in k_keys])

    header = f"{'Chunking':<12} | {'Retrieval':<12} | {'Chunks':<6} | {k_headers} | {'MRR':<6} | {'Lat(ms)':<7}"
    lines.append(header)
    lines.append("-" * len(header))

    for res in report.results:
        chunk_name = res.chunking_strategy[:12]
        ret_name = res.retrieval_strategy[:12]
        n_chunks = str(res.num_chunks)
        recalls_str = " | ".join([f"{res.metrics.recall_at_k.get(k, 0.0):.3f}" for k in k_keys])
        mrr_str = f"{res.metrics.mrr:.3f}"
        lat_str = f"{res.metrics.avg_latency_ms:.2f}"

        row = f"{chunk_name:<12} | {ret_name:<12} | {n_chunks:<6} | {recalls_str} | {mrr_str:<6} | {lat_str:<7}"
        lines.append(row)

    lines.append("=" * 88)
    return "\n".join(lines)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ragbench",
        description="RAGBench: Benchmark chunking and retrieval strategies on RAG datasets."
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"ragbench {__version__}"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Path to JSON dataset file"
    )
    parser.add_argument(
        "--chunkers", "-c",
        nargs="+",
        default=["fixed", "recursive"],
        help="Chunking strategies to evaluate (e.g. fixed recursive)"
    )
    parser.add_argument(
        "--retrievers", "-r",
        nargs="+",
        default=["bm25", "vector"],
        help="Retrieval strategies to evaluate (e.g. bm25 vector tfidf)"
    )
    parser.add_argument(
        "--k-values", "-k",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="K thresholds for Recall@K and Precision@K"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in characters (default: 500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in characters (default: 50)"
    )
    parser.add_argument(
        "--vector-model",
        type=str,
        default=None,
        help="sentence-transformers model name for dense vector search"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save report (JSON format)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON to stdout instead of ASCII table"
    )

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)

    dataset_path = Path(parsed.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at '{parsed.dataset}'", file=sys.stderr)
        return 1

    try:
        dataset = load_dataset_from_json(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return 1

    config = BenchmarkConfig(
        chunking_strategies=parsed.chunkers,
        retrieval_strategies=parsed.retrievers,
        k_values=parsed.k_values,
        chunk_size=parsed.chunk_size,
        chunk_overlap=parsed.chunk_overlap,
        vector_model_name=parsed.vector_model,
    )

    runner = BenchmarkRunner(config=config)
    report = runner.run(dataset=dataset)

    if parsed.json:
        print(report.model_dump_json(indent=2))
    else:
        print(format_ascii_table(report))

    if parsed.output:
        out_path = Path(parsed.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"Report saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
