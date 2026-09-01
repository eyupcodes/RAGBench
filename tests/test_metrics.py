"""Tests for evaluation metrics and answer evaluation hooks."""

import pytest
from ragbench.metrics import (
    ExactMatchAnswerEvaluator,
    TokenOverlapAnswerEvaluator,
    calculate_mrr,
    calculate_precision_at_k,
    calculate_recall_at_k,
    evaluate_retrieval_batch,
)
from ragbench.models import Chunk, Query, RetrievalResult


@pytest.fixture
def sample_retrieval_results():
    c1 = Chunk(id="d1_c0", doc_id="doc_1", text="chunk 1", chunk_index=0)
    c2 = Chunk(id="d2_c0", doc_id="doc_2", text="chunk 2", chunk_index=0)
    c3 = Chunk(id="d3_c0", doc_id="doc_3", text="chunk 3", chunk_index=0)
    return [
        RetrievalResult(chunk=c1, score=0.9, rank=1),
        RetrievalResult(chunk=c2, score=0.7, rank=2),
        RetrievalResult(chunk=c3, score=0.5, rank=3),
    ]


def test_calculate_recall_at_k(sample_retrieval_results):
    relevant = {"doc_1", "doc_3"}

    assert calculate_recall_at_k(sample_retrieval_results, relevant, k=1) == 0.5
    assert calculate_recall_at_k(sample_retrieval_results, relevant, k=2) == 0.5
    assert calculate_recall_at_k(sample_retrieval_results, relevant, k=3) == 1.0


def test_calculate_precision_at_k(sample_retrieval_results):
    relevant = {"doc_1", "doc_3"}

    assert calculate_precision_at_k(sample_retrieval_results, relevant, k=1) == 1.0
    assert calculate_precision_at_k(sample_retrieval_results, relevant, k=2) == 0.5
    assert pytest.approx(calculate_precision_at_k(sample_retrieval_results, relevant, k=3), 0.01) == 0.666


def test_calculate_mrr(sample_retrieval_results):
    assert calculate_mrr(sample_retrieval_results, {"doc_1"}) == 1.0
    assert calculate_mrr(sample_retrieval_results, {"doc_2"}) == 0.5
    assert pytest.approx(calculate_mrr(sample_retrieval_results, {"doc_3"}), 0.01) == 0.333
    assert calculate_mrr(sample_retrieval_results, {"doc_999"}) == 0.0


def test_exact_match_evaluator():
    evaluator = ExactMatchAnswerEvaluator()
    assert evaluator.evaluate("Paris", "paris") == 1.0
    assert evaluator.evaluate("London", "Paris") == 0.0


def test_token_overlap_evaluator():
    evaluator = TokenOverlapAnswerEvaluator()
    assert evaluator.evaluate("The capital is Paris", "Paris is the capital") == 1.0
    assert evaluator.evaluate("Completely unrelated text", "Paris France") == 0.0
    score = evaluator.evaluate("Paris France Europe", "Paris France")
    assert 0.5 < score < 1.0


def test_evaluate_retrieval_batch(sample_retrieval_results):
    queries = [
        Query(id="q1", query="Query 1", relevant_doc_ids=["doc_1"], ground_truth_answer="Ans 1"),
        Query(id="q2", query="Query 2", relevant_doc_ids=["doc_2"], ground_truth_answer="Ans 2"),
    ]
    query_results = [sample_retrieval_results, sample_retrieval_results]
    latencies = [10.0, 20.0]

    evaluator = ExactMatchAnswerEvaluator()
    metrics = evaluate_retrieval_batch(
        query_results=query_results,
        queries=queries,
        k_values=[1, 3],
        latencies_ms=latencies,
        answer_evaluator=evaluator,
        generated_answers=["Ans 1", "Wrong Ans"],
    )

    assert metrics.recall_at_k[1] == 0.5
    assert metrics.mrr == 0.75
    assert metrics.avg_latency_ms == 15.0
    assert metrics.answer_score == 0.5
