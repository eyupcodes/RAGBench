"""Chunking strategies for RAGBench."""

import re
from abc import ABC, abstractmethod
from typing import List, Optional
from ragbench.models import Chunk, Document


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a document into chunks."""
        pass

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split multiple documents into chunks."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


class FixedSizeChunker(BaseChunker):
    """Fixed-size character chunking with configurable overlap."""

    def chunk_document(self, document: Document) -> List[Chunk]:
        text = document.text
        if not text.strip():
            return []

        chunks: List[Chunk] = []
        start = 0
        text_len = len(text)
        step = self.chunk_size - self.chunk_overlap
        chunk_idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = f"{document.id}_chunk_{chunk_idx}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        doc_id=document.id,
                        text=chunk_text,
                        chunk_index=chunk_idx,
                        metadata={
                            **document.metadata,
                            "start_char": start,
                            "end_char": end,
                            "chunker": "fixed",
                        }
                    )
                )
                chunk_idx += 1

            start += step
            if end >= text_len:
                break

        return chunks


class RecursiveCharacterChunker(BaseChunker):
    """
    Hierarchical recursive character chunking splitting on structural boundaries
    (paragraphs -> sentences -> words -> characters).
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the first matching separator."""
        final_chunks: List[str] = []
        if not text:
            return final_chunks

        separator = ""
        new_separators: List[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                new_separators = []
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_text(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    for j in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        good_splits.append(s[j:j + self.chunk_size])

        current_chunk: List[str] = []
        current_length = 0

        for piece in good_splits:
            piece_len = len(piece)
            sep_len = len(separator) if current_chunk else 0

            if current_length + piece_len + sep_len <= self.chunk_size:
                current_chunk.append(piece)
                current_length += piece_len + sep_len
            else:
                if current_chunk:
                    joined = separator.join(current_chunk).strip()
                    if joined:
                        final_chunks.append(joined)

                    overlap_chunk: List[str] = []
                    overlap_len = 0
                    for rev_p in reversed(current_chunk):
                        if overlap_len + len(rev_p) <= self.chunk_overlap:
                            overlap_chunk.insert(0, rev_p)
                            overlap_len += len(rev_p)
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_length = sum(len(p) for p in current_chunk) + (len(separator) * (len(current_chunk) - 1) if current_chunk else 0)

                current_chunk.append(piece)
                current_length += len(piece) + (len(separator) if len(current_chunk) > 1 else 0)

        if current_chunk:
            joined = separator.join(current_chunk).strip()
            if joined:
                final_chunks.append(joined)

        return final_chunks

    def chunk_document(self, document: Document) -> List[Chunk]:
        text = document.text
        if not text.strip():
            return []

        raw_chunks = self._split_text(text, self.separators)
        chunks: List[Chunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        id=f"{document.id}_chunk_{idx}",
                        doc_id=document.id,
                        text=chunk_text.strip(),
                        chunk_index=idx,
                        metadata={
                            **document.metadata,
                            "chunker": "recursive",
                        }
                    )
                )

        return chunks


def get_chunker(
    name: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> BaseChunker:
    """Factory function for retrieving a chunker instance by name."""
    clean_name = name.lower().strip()
    if clean_name == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif clean_name in ["recursive", "recursive_character"]:
        return RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(
            f"Unknown chunking strategy '{name}'. Supported strategies: 'fixed', 'recursive'"
        )
