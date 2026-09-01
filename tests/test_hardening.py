"""Edge case and hardening tests for RAGBench."""

import pytest
from ragbench.benchmark import BenchmarkRunner
from ragbench.chunkers import FixedSizeChunker, RecursiveCharacterChunker
from ragbench.metrics import (
    ExactMatchAnswerEvaluator,
    TokenOverlapAnswerEvaluator,
    calculate_precision_at_k,
    calculate_recall_at_k,
    evaluate_retrieval_batch,
)
from ragbench.models import (
    BenchmarkConfig,
    BenchmarkDataset,
    Chunk,
    Document,
    Query,
    RetrievalResult,
)
from ragbench.retrievers import BM25Retriever, TFIDFVectorRetriever, VectorRetriever


def test_chunking_with_unicode_and_special_characters():
    text = "🚀 Artificial Intelligence is transforming technology.\n\nÇalışma prensipleri ve RAG mimarisi oldukça önemlidir!"
    doc = Document(id="doc_unicode", text=text)
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    assert "🚀" in chunks[0].text
    all_chunks_text = " ".join([c.text for c in chunks])
    assert "Çalışma" in all_chunks_text
    assert "önemlidir" in all_chunks_text


def test_retrieval_with_zero_matches():
    chunks = [
        Chunk(id="c1", doc_id="d1", text="Cats and dogs are common domestic animals.", chunk_index=0),
    ]
    bm25 = BM25Retriever(chunks=chunks)
    results = bm25.retrieve(query="quantum mechanics astrophysics", top_k=5)
    assert len(results) == 0

    tfidf = TFIDFVectorRetriever(chunks=chunks)
    results_tfidf = tfidf.retrieve(query="quantum mechanics astrophysics", top_k=5)
    assert len(results_tfidf) == 0


def test_metrics_zero_relevant_documents():
    query = Query(id="q_empty_rel", query="Something", relevant_doc_ids=[])
    metrics = evaluate_retrieval_batch(
        query_results=[[]],
        queries=[query],
        k_values=[1, 3, 5],
    )
    assert metrics.recall_at_k[1] == 0.0
    assert metrics.mrr == 0.0


def test_precision_recall_negative_or_zero_k():
    res = [RetrievalResult(chunk=Chunk(id="c1", doc_id="d1", text="a", chunk_index=0), score=1.0, rank=1)]
    assert calculate_precision_at_k(res, {"d1"}, k=0) == 0.0
    assert calculate_precision_at_k(res, {"d1"}, k=-1) == 0.0


def test_token_overlap_evaluator_empty_inputs():
    evaluator = TokenOverlapAnswerEvaluator()
    assert evaluator.evaluate("", "") == 1.0
    assert evaluator.evaluate("something", "") == 0.0
    assert evaluator.evaluate("", "something") == 0.0


def test_benchmark_runner_single_doc_many_queries():
    doc = Document(id="d1", text="Machine learning optimizes models using data.")
    queries = [
        Query(id=f"q{i}", query=f"query {i} learning", relevant_doc_ids=["d1"])
        for i in range(10)
    ]
    dataset = BenchmarkDataset(name="stress_q", documents=[doc], queries=queries)
    runner = BenchmarkRunner(config=BenchmarkConfig(chunking_strategies=["fixed"], retrieval_strategies=["tfidf"]))
    report = runner.run(dataset)

    assert report.num_queries == 10
    assert len(report.results) == 1
    assert report.results[0].metrics.mrr > 0.0
