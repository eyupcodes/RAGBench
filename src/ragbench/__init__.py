"""RAGBench: A modular benchmark toolkit for RAG retrieval and chunking strategies."""

__version__ = "0.1.0"

from ragbench.models import (
    Document,
    Chunk,
    Query,
    BenchmarkDataset,
    RetrievalResult,
    BenchmarkConfig,
    BenchmarkReport,
    EvaluationMetrics,
)
from ragbench.benchmark import BenchmarkRunner

__all__ = [
    "Document",
    "Chunk",
    "Query",
    "BenchmarkDataset",
    "RetrievalResult",
    "BenchmarkConfig",
    "BenchmarkReport",
    "EvaluationMetrics",
    "BenchmarkRunner",
]
