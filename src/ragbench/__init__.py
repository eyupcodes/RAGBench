"""RAGBench: A modular benchmark toolkit for RAG retrieval and chunking strategies."""

__version__ = "0.4.0"

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
from ragbench.rerankers import (
    BaseReranker,
    NoOpReranker,
    LexicalReranker,
    CrossEncoderReranker,
    get_reranker,
)

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
    "BaseReranker",
    "NoOpReranker",
    "LexicalReranker",
    "CrossEncoderReranker",
    "get_reranker",
]
