"""Benchmark execution engine orchestrating chunking, indexing, retrieval, and evaluation."""

import time
from typing import Callable, List, Optional
from ragbench.chunkers import get_chunker
from ragbench.metrics import BaseAnswerEvaluator, evaluate_retrieval_batch
from ragbench.models import (
    BenchmarkConfig,
    BenchmarkDataset,
    BenchmarkReport,
    RetrievalResult,
    StrategyResult,
)
from ragbench.retrievers import get_retriever


class BenchmarkRunner:
    """Orchestrates RAG benchmark comparisons across chunking and retrieval strategies."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()

    def run(
        self,
        dataset: BenchmarkDataset,
        config: Optional[BenchmarkConfig] = None,
        answer_evaluator: Optional[BaseAnswerEvaluator] = None,
        answer_generator: Optional[Callable[[str, List[RetrievalResult]], str]] = None,
    ) -> BenchmarkReport:
        """
        Execute benchmark evaluation across all configured strategy combinations.
        """
        active_config = config or self.config
        results: List[StrategyResult] = []

        if not dataset.documents:
            return BenchmarkReport(
                dataset_name=dataset.name,
                num_documents=0,
                num_queries=len(dataset.queries),
                results=[],
            )

        for chunk_strat in active_config.chunking_strategies:
            # 1. Chunk documents
            chunker = get_chunker(
                name=chunk_strat,
                chunk_size=active_config.chunk_size,
                chunk_overlap=active_config.chunk_overlap,
            )
            chunks = chunker.chunk_documents(dataset.documents)

            for ret_strat in active_config.retrieval_strategies:
                # 2. Build retrieval index
                retriever = get_retriever(
                    name=ret_strat,
                    chunks=chunks,
                    vector_model_name=active_config.vector_model_name,
                    rrf_k=active_config.rrf_k,
                )

                # 3. Execute retrieval for all queries
                max_k = max(active_config.k_values) if active_config.k_values else 10
                query_results: List[List[RetrievalResult]] = []
                latencies: List[float] = []
                generated_answers: List[str] = []

                for query in dataset.queries:
                    t0 = time.perf_counter()
                    ret_res = retriever.retrieve(query=query.query, top_k=max_k)
                    t1 = time.perf_counter()

                    latencies.append((t1 - t0) * 1000.0)
                    query_results.append(ret_res)

                    if answer_generator:
                        gen_ans = answer_generator(query.query, ret_res)
                        generated_answers.append(gen_ans)

                # 4. Evaluate metrics
                metrics = evaluate_retrieval_batch(
                    query_results=query_results,
                    queries=dataset.queries,
                    k_values=active_config.k_values,
                    latencies_ms=latencies,
                    answer_evaluator=answer_evaluator,
                    generated_answers=generated_answers if generated_answers else None,
                )

                results.append(
                    StrategyResult(
                        chunking_strategy=chunk_strat,
                        retrieval_strategy=ret_strat,
                        num_chunks=len(chunks),
                        metrics=metrics,
                    )
                )

        return BenchmarkReport(
            dataset_name=dataset.name,
            num_documents=len(dataset.documents),
            num_queries=len(dataset.queries),
            results=results,
        )
