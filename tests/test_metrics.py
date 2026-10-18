"""Tests for evaluation metrics and answer evaluation hooks."""

import math

import pytest

from ragbench.metrics import (
    ExactMatchAnswerEvaluator,
    TokenOverlapAnswerEvaluator,
    calculate_hit_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
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
    assert (
        pytest.approx(
            calculate_precision_at_k(sample_retrieval_results, relevant, k=3), 0.01
        )
        == 0.666
    )


def test_calculate_hit_at_k(sample_retrieval_results):
    assert calculate_hit_at_k(sample_retrieval_results, {"doc_3"}, k=2) == 0.0
    assert calculate_hit_at_k(sample_retrieval_results, {"doc_3"}, k=3) == 1.0
    assert calculate_hit_at_k(sample_retrieval_results, set(), k=3) == 0.0
    assert calculate_hit_at_k(sample_retrieval_results, {"doc_1"}, k=0) == 0.0


def test_calculate_ndcg_at_k_ideal_ranking(sample_retrieval_results):
    assert calculate_ndcg_at_k(sample_retrieval_results, {"doc_1", "doc_2"}, 3) == 1.0
    assert calculate_ndcg_at_k(sample_retrieval_results, {"doc_1"}, 1) == 1.0


def test_calculate_ndcg_at_k_penalizes_late_hits(sample_retrieval_results):
    # one relevant doc at rank 3: DCG = 1/log2(4) = 0.5, ideal DCG = 1
    assert calculate_ndcg_at_k(sample_retrieval_results, {"doc_3"}, 3) == pytest.approx(
        0.5
    )
    # relevant docs at ranks 1 and 3 out of 2: (1 + 0.5) / (1 + 1/log2(3))
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert calculate_ndcg_at_k(
        sample_retrieval_results, {"doc_1", "doc_3"}, 3
    ) == pytest.approx(expected)


def test_calculate_ndcg_at_k_cuts_off_at_k(sample_retrieval_results):
    assert calculate_ndcg_at_k(sample_retrieval_results, {"doc_3"}, 2) == 0.0


def test_calculate_ndcg_at_k_counts_a_document_once():
    chunks = [
        RetrievalResult(
            chunk=Chunk(id=f"a_c{i}", doc_id="a", text="t", chunk_index=i),
            score=1.0 - i / 10,
            rank=i + 1,
        )
        for i in range(3)
    ]
    assert calculate_ndcg_at_k(chunks, {"a", "b"}, 3) == pytest.approx(
        1.0 / (1 + 1 / math.log2(3))
    )


def test_calculate_ndcg_at_k_degenerate_inputs(sample_retrieval_results):
    assert calculate_ndcg_at_k(sample_retrieval_results, set(), 3) == 0.0
    assert calculate_ndcg_at_k(sample_retrieval_results, {"doc_1"}, 0) == 0.0
    assert calculate_ndcg_at_k([], {"doc_1"}, 3) == 0.0


def test_calculate_mrr(sample_retrieval_results):
    assert calculate_mrr(sample_retrieval_results, {"doc_1"}) == 1.0
    assert calculate_mrr(sample_retrieval_results, {"doc_2"}) == 0.5
    assert (
        pytest.approx(calculate_mrr(sample_retrieval_results, {"doc_3"}), 0.01) == 0.333
    )
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
        Query(
            id="q1",
            query="Query 1",
            relevant_doc_ids=["doc_1"],
            ground_truth_answer="Ans 1",
        ),
        Query(
            id="q2",
            query="Query 2",
            relevant_doc_ids=["doc_2"],
            ground_truth_answer="Ans 2",
        ),
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
    assert metrics.hit_at_k == {1: 0.5, 3: 1.0}
    assert metrics.ndcg_at_k[1] == 0.5
    assert metrics.ndcg_at_k[3] == pytest.approx(0.8155, abs=1e-4)
    assert metrics.mrr == 0.75
    assert metrics.avg_latency_ms == 15.0
    assert metrics.answer_score == 0.5


def test_evaluate_retrieval_batch_rejects_mismatched_result_count(
    sample_retrieval_results,
):
    queries = [
        Query(id="q1", query="Query 1", relevant_doc_ids=["doc_1"]),
        Query(id="q2", query="Query 2", relevant_doc_ids=["doc_2"]),
    ]
    with pytest.raises(ValueError, match="1 entries but there are 2 queries"):
        evaluate_retrieval_batch(
            query_results=[sample_retrieval_results],
            queries=queries,
        )
