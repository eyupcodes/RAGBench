"""Tests for retrieval strategies."""

import pytest
from ragbench.models import Chunk
from ragbench.retrievers import (
    BM25Retriever,
    HybridRetriever,
    PurePythonBM25,
    TFIDFVectorRetriever,
    VectorRetriever,
    get_retriever,
)


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            id="doc_py_0",
            doc_id="doc_py",
            text="Python is a widely used high-level programming language known for readability.",
            chunk_index=0
        ),
        Chunk(
            id="doc_rust_0",
            doc_id="doc_rust",
            text="Rust is a systems language emphasizing memory safety and thread concurrency without garbage collection.",
            chunk_index=0
        ),
        Chunk(
            id="doc_db_0",
            doc_id="doc_db",
            text="PostgreSQL is an open-source relational database management system emphasizing extensibility and SQL compliance.",
            chunk_index=0
        ),
    ]


def test_pure_python_bm25_scoring():
    corpus = [
        ["python", "programming", "language"],
        ["rust", "memory", "safety"],
        ["postgresql", "relational", "database"]
    ]
    bm25 = PurePythonBM25(corpus)
    scores = bm25.get_scores(["memory", "safety"])
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_bm25_retriever(sample_chunks):
    retriever = BM25Retriever(chunks=sample_chunks)
    results = retriever.retrieve("memory safety in systems programming", top_k=2)

    assert len(results) >= 1
    assert results[0].chunk.doc_id == "doc_rust"
    assert results[0].rank == 1
    assert results[0].score > 0.0


def test_tfidf_vector_retriever(sample_chunks):
    retriever = TFIDFVectorRetriever(chunks=sample_chunks)
    results = retriever.retrieve("relational database SQL", top_k=2)

    assert len(results) >= 1
    assert results[0].chunk.doc_id == "doc_db"
    assert results[0].score > 0.0


def test_vector_retriever_fallback(sample_chunks):
    retriever = VectorRetriever(chunks=sample_chunks)
    results = retriever.retrieve("Python readability", top_k=2)

    assert len(results) >= 1
    assert results[0].chunk.doc_id == "doc_py"


def test_retriever_empty_query(sample_chunks):
    retriever = BM25Retriever(chunks=sample_chunks)
    results = retriever.retrieve("   ", top_k=5)
    assert results == []


def test_retriever_empty_chunks():
    retriever = BM25Retriever(chunks=[])
    results = retriever.retrieve("query", top_k=5)
    assert results == []


def test_get_retriever_factory(sample_chunks):
    r_bm25 = get_retriever("bm25", chunks=sample_chunks)
    assert isinstance(r_bm25, BM25Retriever)

    r_vec = get_retriever("vector", chunks=sample_chunks)
    assert isinstance(r_vec, VectorRetriever)

    r_tfidf = get_retriever("tfidf", chunks=sample_chunks)
    assert isinstance(r_tfidf, TFIDFVectorRetriever)

    with pytest.raises(ValueError, match="Unknown retrieval strategy"):
        get_retriever("non_existent_retriever", chunks=sample_chunks)


def test_hybrid_retriever_basic(sample_chunks):
    retriever = HybridRetriever(chunks=sample_chunks)
    results = retriever.retrieve("memory safety", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk.doc_id == "doc_rust"
    assert results[0].score > 0.0


def test_hybrid_retriever_empty_chunks():
    retriever = HybridRetriever(chunks=[])
    assert retriever.retrieve("query", top_k=5) == []


def test_hybrid_retriever_empty_query(sample_chunks):
    retriever = HybridRetriever(chunks=sample_chunks)
    assert retriever.retrieve("   ", top_k=5) == []


def test_hybrid_retriever_custom_rrf_k(sample_chunks):
    retriever = HybridRetriever(chunks=sample_chunks, rrf_k=30)
    assert retriever.rrf_k == 30
    results = retriever.retrieve("Python programming", top_k=1)
    assert len(results) == 1


def test_get_retriever_hybrid_factory(sample_chunks):
    r_hybrid = get_retriever("hybrid", chunks=sample_chunks)
    assert isinstance(r_hybrid, HybridRetriever)
    r_rrf = get_retriever("rrf", chunks=sample_chunks, rrf_k=30)
    assert isinstance(r_rrf, HybridRetriever)
    assert r_rrf.rrf_k == 30
    r_alias = get_retriever("hybrid_rrf", chunks=sample_chunks)
    assert isinstance(r_alias, HybridRetriever)
