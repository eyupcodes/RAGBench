"""Evaluation metrics and answer evaluation hooks for RAGBench."""

import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Callable, Dict, List, Optional, Set
from ragbench.models import EvaluationMetrics, Query, RetrievalResult


def _get_relevant_doc_ids(query: Query) -> Set[str]:
    """Extract ground truth document IDs for a query."""
    return set(query.relevant_doc_ids)


def calculate_recall_at_k(results: List[RetrievalResult], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Recall@K: proportion of relevant documents retrieved in top-k results.
    """
    if not relevant_ids:
        return 0.0

    top_k_results = results[:k]
    retrieved_doc_ids = {r.chunk.doc_id for r in top_k_results}
    hits = len(retrieved_doc_ids.intersection(relevant_ids))
    return hits / len(relevant_ids)


def calculate_precision_at_k(results: List[RetrievalResult], relevant_ids: Set[str], k: int) -> float:
    """Calculate Precision@K: proportion of top-k retrieved documents that are relevant."""
    if k <= 0:
        return 0.0

    top_k_results = results[:k]
    if not top_k_results:
        return 0.0

    retrieved_doc_ids = {r.chunk.doc_id for r in top_k_results}
    hits = len(retrieved_doc_ids.intersection(relevant_ids))
    return hits / len(top_k_results)


def calculate_mrr(results: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) for a single query."""
    if not relevant_ids or not results:
        return 0.0

    for rank, res in enumerate(results, start=1):
        if res.chunk.doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


class BaseAnswerEvaluator(ABC):
    """Abstract base class for answer evaluation hooks."""

    @abstractmethod
    def evaluate(self, generated_answer: str, ground_truth: str) -> float:
        """Evaluate generated answer against ground truth. Returns score between 0.0 and 1.0."""
        pass


class ExactMatchAnswerEvaluator(BaseAnswerEvaluator):
    """Binary exact match answer evaluator (case-insensitive, trimmed)."""

    def evaluate(self, generated_answer: str, ground_truth: str) -> float:
        return 1.0 if generated_answer.strip().lower() == ground_truth.strip().lower() else 0.0


class TokenOverlapAnswerEvaluator(BaseAnswerEvaluator):
    """Token-level F1 overlap evaluator (deterministic, no external LLM required)."""

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def evaluate(self, generated_answer: str, ground_truth: str) -> float:
        gen_tokens = self._tokenize(generated_answer)
        gt_tokens = self._tokenize(ground_truth)

        if not gen_tokens or not gt_tokens:
            return 1.0 if gen_tokens == gt_tokens else 0.0

        gen_counter = Counter(gen_tokens)
        gt_counter = Counter(gt_tokens)

        overlap = sum((gen_counter & gt_counter).values())
        if overlap == 0:
            return 0.0

        precision = overlap / len(gen_tokens)
        recall = overlap / len(gt_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1


def evaluate_retrieval_batch(
    query_results: List[List[RetrievalResult]],
    queries: List[Query],
    k_values: List[int] = [1, 3, 5, 10],
    latencies_ms: Optional[List[float]] = None,
    answer_evaluator: Optional[BaseAnswerEvaluator] = None,
    generated_answers: Optional[List[str]] = None,
) -> EvaluationMetrics:
    """Calculate aggregate benchmark metrics across all queries."""
    if not queries:
        return EvaluationMetrics()

    n_queries = len(queries)
    recall_sums: Dict[int, float] = {k: 0.0 for k in k_values}
    precision_sums: Dict[int, float] = {k: 0.0 for k in k_values}
    mrr_sum = 0.0

    for query, results in zip(queries, query_results):
        rel_ids = _get_relevant_doc_ids(query)
        for k in k_values:
            recall_sums[k] += calculate_recall_at_k(results, rel_ids, k)
            precision_sums[k] += calculate_precision_at_k(results, rel_ids, k)
        mrr_sum += calculate_mrr(results, rel_ids)

    mean_recall = {k: round(val / n_queries, 4) for k, val in recall_sums.items()}
    mean_precision = {k: round(val / n_queries, 4) for k, val in precision_sums.items()}
    mean_mrr = round(mrr_sum / n_queries, 4)
    avg_latency = round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0.0

    answer_score = None
    if answer_evaluator and generated_answers and len(generated_answers) == n_queries:
        scores = []
        for q, gen_ans in zip(queries, generated_answers):
            if q.ground_truth_answer is not None:
                score = answer_evaluator.evaluate(gen_ans, q.ground_truth_answer)
                scores.append(score)
        if scores:
            answer_score = round(sum(scores) / len(scores), 4)

    return EvaluationMetrics(
        recall_at_k=mean_recall,
        precision_at_k=mean_precision,
        mrr=mean_mrr,
        avg_latency_ms=avg_latency,
        answer_score=answer_score,
    )
