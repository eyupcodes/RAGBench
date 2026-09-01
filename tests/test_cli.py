"""Tests for command-line interface and table reporting."""

import json
import pytest
from ragbench.cli import format_ascii_table, main, parse_args
from ragbench.models import (
    BenchmarkReport,
    EvaluationMetrics,
    StrategyResult,
)


@pytest.fixture
def sample_report():
    metrics = EvaluationMetrics(
        recall_at_k={1: 0.5, 3: 1.0, 5: 1.0},
        precision_at_k={1: 0.5, 3: 0.33, 5: 0.2},
        mrr=0.75,
        avg_latency_ms=15.42,
    )
    res1 = StrategyResult(
        chunking_strategy="fixed",
        retrieval_strategy="bm25",
        num_chunks=12,
        metrics=metrics,
    )
    res2 = StrategyResult(
        chunking_strategy="recursive",
        retrieval_strategy="vector",
        num_chunks=10,
        metrics=metrics,
    )
    return BenchmarkReport(
        dataset_name="cli_demo",
        num_documents=4,
        num_queries=2,
        results=[res1, res2],
    )


def test_format_ascii_table(sample_report):
    table_str = format_ascii_table(sample_report)
    assert "RAGBench Evaluation Report: cli_demo" in table_str
    assert "fixed" in table_str
    assert "recursive" in table_str
    assert "bm25" in table_str
    assert "0.750" in table_str


def test_format_ascii_table_empty():
    empty_report = BenchmarkReport(dataset_name="empty", num_documents=0, num_queries=0, results=[])
    table_str = format_ascii_table(empty_report)
    assert "No benchmark results available." in table_str


def test_parse_args_defaults():
    args = parse_args(["--dataset", "data.json"])
    assert args.dataset == "data.json"
    assert args.chunkers == ["fixed", "recursive"]
    assert args.retrievers == ["bm25", "vector"]
    assert args.chunk_size == 500
    assert args.chunk_overlap == 50


def test_cli_main_success(tmp_path, capsys):
    dataset_content = {
        "name": "cli_test_ds",
        "documents": [
            {"id": "doc1", "text": "Python is a dynamic programming language."},
            {"id": "doc2", "text": "PostgreSQL is a relational database."},
        ],
        "queries": [
            {"id": "q1", "query": "Python language", "relevant_doc_ids": ["doc1"]},
        ]
    }
    ds_file = tmp_path / "ds.json"
    ds_file.write_text(json.dumps(dataset_content), encoding="utf-8")
    out_file = tmp_path / "report.json"

    exit_code = main([
        "--dataset", str(ds_file),
        "--chunkers", "fixed",
        "--retrievers", "tfidf",
        "--k-values", "1", "3",
        "--output", str(out_file),
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "RAGBench Evaluation Report: cli_test_ds" in captured.out
    assert out_file.exists()


def test_cli_main_json_output(tmp_path, capsys):
    dataset_content = {
        "name": "json_test_ds",
        "documents": [{"id": "d1", "text": "Sample text"}],
        "queries": [{"id": "q1", "query": "Sample", "relevant_doc_ids": ["d1"]}]
    }
    ds_file = tmp_path / "ds_json.json"
    ds_file.write_text(json.dumps(dataset_content), encoding="utf-8")

    exit_code = main([
        "--dataset", str(ds_file),
        "--chunkers", "fixed",
        "--retrievers", "tfidf",
        "--json"
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    parsed_json = json.loads(captured.out)
    assert parsed_json["dataset_name"] == "json_test_ds"
    assert len(parsed_json["results"]) == 1


def test_cli_main_missing_dataset_file(capsys):
    exit_code = main(["--dataset", "non_existent_dataset.json"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: Dataset file not found" in captured.err
