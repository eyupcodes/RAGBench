"""Tests for chunking strategies."""

import pytest
from ragbench.chunkers import (
    FixedSizeChunker,
    RecursiveCharacterChunker,
    get_chunker,
)
from ragbench.models import Document


def test_fixed_size_chunker_basic():
    doc = Document(id="doc_1", text="abcdefghijklmnopqrstuvwxyz")
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.chunk_document(doc)

    # 26 chars with chunk_size=10, overlap=2 (step=8):
    # chunk 0: 0-10 ("abcdefghij")
    # chunk 1: 8-18 ("ijklmnopqr")
    # chunk 2: 16-26 ("qrstuvwxyz")
    assert len(chunks) == 3
    assert chunks[0].text == "abcdefghij"
    assert chunks[0].chunk_index == 0
    assert chunks[0].doc_id == "doc_1"
    assert chunks[1].text == "ijklmnopqr"
    assert chunks[2].text == "qrstuvwxyz"


def test_fixed_size_chunker_empty():
    doc = Document(id="doc_empty", text="   ")
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.chunk_document(doc)
    assert chunks == []


def test_chunker_invalid_parameters():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        FixedSizeChunker(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        FixedSizeChunker(chunk_size=10, chunk_overlap=-1)

    with pytest.raises(ValueError, match="must be strictly less"):
        FixedSizeChunker(chunk_size=10, chunk_overlap=10)


def test_recursive_chunker_paragraphs():
    text = "Paragraph one content here.\n\nParagraph two content here.\n\nParagraph three content here."
    doc = Document(id="doc_p", text=text)
    chunker = RecursiveCharacterChunker(chunk_size=40, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3
    assert all(c.doc_id == "doc_p" for c in chunks)
    assert "Paragraph one" in chunks[0].text


def test_recursive_chunker_fallback_to_sentence():
    text = "Sentence one. Sentence two is longer. Sentence three is here."
    doc = Document(id="doc_s", text=text)
    chunker = RecursiveCharacterChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2


def test_chunk_documents_batch():
    docs = [
        Document(id="d1", text="Short text 1"),
        Document(id="d2", text="Short text 2"),
    ]
    chunker = get_chunker("fixed", chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_documents(docs)

    assert len(chunks) == 2
    assert chunks[0].doc_id == "d1"
    assert chunks[1].doc_id == "d2"


def test_get_chunker_factory():
    c_fixed = get_chunker("fixed")
    assert isinstance(c_fixed, FixedSizeChunker)

    c_rec = get_chunker("recursive")
    assert isinstance(c_rec, RecursiveCharacterChunker)

    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunker("unknown_strategy")
