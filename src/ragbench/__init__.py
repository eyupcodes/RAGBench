"""RAGBench: A modular benchmark toolkit for RAG retrieval and chunking strategies."""

__version__ = "0.5.0"

from ragbench.benchmark import BenchmarkRunner
from ragbench.models import (
    BenchmarkConfig,
    BenchmarkDataset,
    BenchmarkReport,
    Chunk,
    Document,
    EvaluationMetrics,
    Query,
    RetrievalResult,
)
from ragbench.rerankers import (
    BaseReranker,
    CrossEncoderReranker,
    LexicalReranker,
    NoOpReranker,
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
