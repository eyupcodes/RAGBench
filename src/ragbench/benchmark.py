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
from ragbench.rerankers import get_reranker


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

            expanded_retrieval_strategies: List[str] = []
            for ret_s in active_config.retrieval_strategies:
                if (
                    ret_s.lower().strip() in ["vector", "dense", "vector_search"]
                    and active_config.vector_model_names
                ):
                    for m in active_config.vector_model_names:
                        expanded_retrieval_strategies.append(f"{ret_s}:{m}")
                else:
                    expanded_retrieval_strategies.append(ret_s)

            for ret_strat in expanded_retrieval_strategies:
                model_for_retriever = active_config.vector_model_name
                embedding_model: Optional[str] = None
                clean_s = ret_strat.lower().strip()
                if clean_s.startswith("vector:") or clean_s.startswith("dense:"):
                    _, parsed_model = ret_strat.split(":", 1)
                    model_for_retriever = parsed_model.strip()
                    embedding_model = model_for_retriever
                elif clean_s in ["vector", "dense", "vector_search"]:
                    embedding_model = active_config.vector_model_name

                # 2. Build retrieval index
                retriever = get_retriever(
                    name=ret_strat,
                    chunks=chunks,
                    vector_model_name=model_for_retriever,
                    rrf_k=active_config.rrf_k,
                )
                reranker = get_reranker(
                    name=active_config.reranking_strategy,
                    model_name=active_config.reranker_model_name,
                )

                # 3. Execute retrieval for all queries
                max_k = max(active_config.k_values) if active_config.k_values else 10
                use_reranker = bool(
                    active_config.reranking_strategy
                    and active_config.reranking_strategy.lower() not in ["none", "noop"]
                )
                candidate_k = max(max_k * 3, 20) if use_reranker else max_k
                query_results: List[List[RetrievalResult]] = []
                latencies: List[float] = []
                generated_answers: List[str] = []

                for query in dataset.queries:
                    t0 = time.perf_counter()
                    ret_res = retriever.retrieve(query=query.query, top_k=candidate_k)
                    if use_reranker:
                        ret_res = reranker.rerank(query=query.query, results=ret_res, top_k=max_k)
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
                        reranking_strategy=active_config.reranking_strategy,
                        embedding_model=embedding_model,
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
