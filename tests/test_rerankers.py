"""Tests for reranker strategies and factory."""

from unittest.mock import MagicMock

import pytest

from ragbench.models import Chunk, RetrievalResult
from ragbench.rerankers import (
    CrossEncoderReranker,
    LexicalReranker,
    NoOpReranker,
    get_reranker,
)


@pytest.fixture
def sample_candidates():
    c1 = Chunk(
        id="c1", doc_id="d1", text="Python is dynamic and versatile.", chunk_index=0
    )
    c2 = Chunk(
        id="c2",
        doc_id="d2",
        text="Rust provides compile-time memory safety.",
        chunk_index=0,
    )
    c3 = Chunk(
        id="c3",
        doc_id="d3",
        text="PostgreSQL handles relational ACID data.",
        chunk_index=0,
    )
    return [
        RetrievalResult(chunk=c1, score=0.5, rank=1),
        RetrievalResult(chunk=c2, score=0.4, rank=2),
        RetrievalResult(chunk=c3, score=0.3, rank=3),
    ]


def test_noop_reranker(sample_candidates):
    reranker = NoOpReranker()
    out = reranker.rerank(query="memory safety", results=sample_candidates)
    assert len(out) == 3
    assert out[0].chunk.id == "c1"
    assert out[1].chunk.id == "c2"
    assert out[0].rank == 1
    assert out[1].rank == 2


def test_noop_reranker_top_k(sample_candidates):
    reranker = NoOpReranker()
    out = reranker.rerank(query="anything", results=sample_candidates, top_k=2)
    assert len(out) == 2
    assert out[0].chunk.id == "c1"
    assert out[1].chunk.id == "c2"


def test_noop_reranker_empty():
    reranker = NoOpReranker()
    out = reranker.rerank(query="anything", results=[])
    assert out == []


def test_lexical_reranker(sample_candidates):
    reranker = LexicalReranker()
    # c2 mentions "memory" and "safety"
    out = reranker.rerank(query="memory safety", results=sample_candidates)
    assert len(out) == 3
    assert out[0].chunk.id == "c2"
    assert out[0].rank == 1
    assert out[0].score > out[1].score


def test_lexical_reranker_empty():
    reranker = LexicalReranker()
    assert reranker.rerank(query="query", results=[]) == []


def test_cross_encoder_reranker_with_mock_model(sample_candidates):
    mock_model = MagicMock()
    # Return scores favoring c2 then c3 then c1
    mock_model.predict.return_value = [0.1, 0.95, 0.4]

    reranker = CrossEncoderReranker(model=mock_model)
    out = reranker.rerank(
        query="Rust memory safety", results=sample_candidates, top_k=2
    )

    mock_model.predict.assert_called_once()
    assert len(out) == 2
    assert out[0].chunk.id == "c2"
    assert out[0].rank == 1
    assert out[0].score == 0.95
    assert out[1].chunk.id == "c3"
    assert out[1].rank == 2
    assert out[1].score == 0.4


def test_cross_encoder_reranker_fallback_no_model(sample_candidates):
    # When model is None and sentence-transformers is not instantiated, fallback gracefully
    reranker = CrossEncoderReranker(model_name=None, model=None)
    out = reranker.rerank(query="memory safety", results=sample_candidates)
    assert len(out) == 3
    # Lexical fallback should prioritize c2
    assert out[0].chunk.id == "c2"
    assert out[0].rank == 1


def test_cross_encoder_reranker_empty():
    mock_model = MagicMock()
    reranker = CrossEncoderReranker(model=mock_model)
    assert reranker.rerank(query="q", results=[]) == []
    mock_model.predict.assert_not_called()


def test_get_reranker_factory():
    assert isinstance(get_reranker(None), NoOpReranker)
    assert isinstance(get_reranker("none"), NoOpReranker)
    assert isinstance(get_reranker("noop"), NoOpReranker)
    assert isinstance(get_reranker("lexical"), LexicalReranker)
    assert isinstance(get_reranker("cross-encoder"), CrossEncoderReranker)
    assert isinstance(get_reranker("ce"), CrossEncoderReranker)
    assert isinstance(get_reranker("cross_encoder"), CrossEncoderReranker)

    with pytest.raises(ValueError, match="Unknown reranking strategy"):
        get_reranker("invalid_reranker_name")
