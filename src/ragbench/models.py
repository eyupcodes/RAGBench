"""Data models and configuration schemas for RAGBench."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """A raw document in the benchmark dataset."""
    id: str = Field(..., description="Unique identifier for the document")
    text: str = Field(..., description="Full text content of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class Chunk(BaseModel):
    """A chunked segment of a document."""
    id: str = Field(..., description="Unique chunk ID (e.g. doc_1_chunk_0)")
    doc_id: str = Field(..., description="Parent document ID")
    text: str = Field(..., description="Text content of this chunk")
    chunk_index: int = Field(..., description="0-indexed position in parent document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata preserved from document")


class Query(BaseModel):
    """A test query with ground truth references for retrieval evaluation."""
    id: str = Field(..., description="Unique query ID")
    query: str = Field(..., description="The search query text")
    relevant_doc_ids: List[str] = Field(
        default_factory=list,
        description="IDs of relevant documents for this query"
    )
    relevant_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Optional ground truth chunk IDs if chunk-level relevance is known"
    )
    ground_truth_answer: Optional[str] = Field(
        None,
        description="Optional expected answer text for answer evaluation hooks"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkDataset(BaseModel):
    """A benchmark dataset containing documents and test queries."""
    name: str = Field("default", description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    documents: List[Document] = Field(default_factory=list)
    queries: List[Query] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """A retrieved chunk and its relevance score."""
    chunk: Chunk
    score: float
    rank: int


class BenchmarkConfig(BaseModel):
    """Configuration for running a benchmark evaluation."""
    chunking_strategies: List[str] = Field(
        default_factory=lambda: ["fixed", "recursive"],
        description="List of chunking strategy names to benchmark"
    )
    retrieval_strategies: List[str] = Field(
        default_factory=lambda: ["bm25", "vector"],
        description="List of retrieval strategy names to benchmark"
    )
    k_values: List[int] = Field(
        default_factory=lambda: [1, 3, 5, 10],
        description="K thresholds for Recall@K and Precision@K"
    )
    chunk_size: int = Field(500, description="Target chunk size in characters/tokens")
    chunk_overlap: int = Field(50, description="Chunk overlap in characters/tokens")
    vector_model_name: Optional[str] = Field(
        None,
        description="sentence-transformers model name or None for deterministic fallback"
    )


class EvaluationMetrics(BaseModel):
    """Calculated metrics for a single strategy run."""
    recall_at_k: Dict[int, float] = Field(default_factory=dict)
    precision_at_k: Dict[int, float] = Field(default_factory=dict)
    mrr: float = Field(0.0, description="Mean Reciprocal Rank")
    avg_latency_ms: float = Field(0.0, description="Average query retrieval latency in ms")
    answer_score: Optional[float] = Field(
        None,
        description="Score from answer evaluation hook if provided"
    )


class StrategyResult(BaseModel):
    """Results for one combination of (chunking_strategy, retrieval_strategy)."""
    chunking_strategy: str
    retrieval_strategy: str
    num_chunks: int
    metrics: EvaluationMetrics


class BenchmarkReport(BaseModel):
    """Complete benchmark execution report."""
    dataset_name: str
    num_documents: int
    num_queries: int
    results: List[StrategyResult] = Field(default_factory=list)
