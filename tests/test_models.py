"""Tests for RAGBench data models and schemas."""

import pytest
from pydantic import ValidationError
from ragbench.models import (
    BenchmarkConfig,
    BenchmarkDataset,
    BenchmarkReport,
    Chunk,
    Document,
    EvaluationMetrics,
    Query,
    RetrievalResult,
    StrategyResult,
)


def test_document_creation_valid():
    doc = Document(id="doc_1", text="Sample document text", metadata={"source": "test"})
    assert doc.id == "doc_1"
    assert doc.text == "Sample document text"
    assert doc.metadata["source"] == "test"


def test_document_missing_id():
    with pytest.raises(ValidationError):
        Document(text="No ID provided")


def test_chunk_creation_valid():
    chunk = Chunk(
        id="doc_1_chunk_0",
        doc_id="doc_1",
        text="Sample chunk",
        chunk_index=0,
        metadata={"start": 0}
    )
    assert chunk.id == "doc_1_chunk_0"
    assert chunk.doc_id == "doc_1"
    assert chunk.chunk_index == 0


def test_query_defaults():
    query = Query(id="q_1", query="Where is Paris?")
    assert query.id == "q_1"
    assert query.relevant_doc_ids == []
    assert query.ground_truth_answer is None


def test_benchmark_dataset():
    dataset = BenchmarkDataset(
        name="test_ds",
        documents=[Document(id="d1", text="text 1")],
        queries=[Query(id="q1", query="query 1", relevant_doc_ids=["d1"])]
    )
    assert dataset.name == "test_ds"
    assert len(dataset.documents) == 1
    assert len(dataset.queries) == 1


def test_benchmark_config_defaults():
    config = BenchmarkConfig()
    assert "fixed" in config.chunking_strategies
    assert "recursive" in config.chunking_strategies
    assert "bm25" in config.retrieval_strategies
    assert "vector" in config.retrieval_strategies
    assert config.chunk_size == 500
    assert config.chunk_overlap == 50


def test_benchmark_report_serialization():
    metrics = EvaluationMetrics(
        recall_at_k={1: 0.5, 3: 1.0},
        precision_at_k={1: 0.5, 3: 0.33},
        mrr=0.75,
        avg_latency_ms=12.5,
    )
    strat_res = StrategyResult(
        chunking_strategy="fixed",
        retrieval_strategy="bm25",
        num_chunks=10,
        metrics=metrics,
    )
    report = BenchmarkReport(
        dataset_name="sample",
        num_documents=5,
        num_queries=2,
        results=[strat_res],
    )
    json_str = report.model_dump_json()
    assert "sample" in json_str
    assert "fixed" in json_str
    assert "0.75" in json_str
