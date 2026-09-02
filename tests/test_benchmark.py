"""Tests for benchmark execution engine."""

import pytest
from ragbench.benchmark import BenchmarkRunner
from ragbench.metrics import ExactMatchAnswerEvaluator
from ragbench.models import (
    BenchmarkConfig,
    BenchmarkDataset,
    Document,
    Query,
    RetrievalResult,
)


@pytest.fixture
def sample_dataset():
    docs = [
        Document(
            id="doc_lang_1",
            text="Python is a multi-paradigm, dynamically typed programming language supporting functional and OOP.",
        ),
        Document(
            id="doc_lang_2",
            text="Rust is a systems programming language with compile-time memory safety without a garbage collector.",
        ),
        Document(
            id="doc_lang_3",
            text="PostgreSQL is an advanced open-source object-relational database with strong ACID transactions.",
        ),
    ]
    queries = [
        Query(
            id="q_1",
            query="Which language offers compile-time memory safety?",
            relevant_doc_ids=["doc_lang_2"],
            ground_truth_answer="Rust",
        ),
        Query(
            id="q_2",
            query="Tell me about relational database ACID transactions",
            relevant_doc_ids=["doc_lang_3"],
            ground_truth_answer="PostgreSQL",
        ),
    ]
    return BenchmarkDataset(
        name="programming_benchmark",
        description="Dataset for testing language and database retrieval",
        documents=docs,
        queries=queries,
    )


def test_benchmark_runner_execution(sample_dataset):
    config = BenchmarkConfig(
        chunking_strategies=["fixed", "recursive"],
        retrieval_strategies=["bm25", "tfidf"],
        k_values=[1, 3],
        chunk_size=100,
        chunk_overlap=20,
    )
    runner = BenchmarkRunner(config=config)
    report = runner.run(dataset=sample_dataset)

    assert report.dataset_name == "programming_benchmark"
    assert report.num_documents == 3
    assert report.num_queries == 2
    assert len(report.results) == 4

    for res in report.results:
        assert res.chunking_strategy in ["fixed", "recursive"]
        assert res.retrieval_strategy in ["bm25", "tfidf"]
        assert res.num_chunks > 0
        assert res.metrics.mrr >= 0.0
        assert 1 in res.metrics.recall_at_k
        assert 3 in res.metrics.recall_at_k


def test_benchmark_runner_with_answer_hook(sample_dataset):
    config = BenchmarkConfig(
        chunking_strategies=["fixed"],
        retrieval_strategies=["bm25"],
        k_values=[1],
    )
    runner = BenchmarkRunner(config=config)

    def dummy_generator(query: str, results: list[RetrievalResult]) -> str:
        if results and "rust" in results[0].chunk.text.lower():
            return "Rust"
        return "Unknown"

    report = runner.run(
        dataset=sample_dataset,
        answer_evaluator=ExactMatchAnswerEvaluator(),
        answer_generator=dummy_generator,
    )

    assert len(report.results) == 1
    assert report.results[0].metrics.answer_score is not None


def test_benchmark_runner_hybrid(sample_dataset):
    config = BenchmarkConfig(
        chunking_strategies=["fixed"],
        retrieval_strategies=["hybrid"],
        k_values=[1, 3],
        chunk_size=100,
        chunk_overlap=20,
    )
    runner = BenchmarkRunner(config=config)
    report = runner.run(dataset=sample_dataset)
    assert len(report.results) == 1
    assert report.results[0].retrieval_strategy == "hybrid"
    assert report.results[0].metrics.mrr >= 0.0


def test_benchmark_runner_empty_dataset():
    empty_ds = BenchmarkDataset(name="empty", documents=[], queries=[])
    runner = BenchmarkRunner()
    report = runner.run(dataset=empty_ds)

    assert report.num_documents == 0
    assert report.results == []
