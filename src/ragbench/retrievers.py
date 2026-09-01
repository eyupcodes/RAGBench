"""Retrieval strategies for RAGBench (BM25, TF-IDF Vector fallback, and Embedding Vector)."""

import math
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple
from ragbench.models import Chunk, RetrievalResult


class BaseRetriever(ABC):
    """Abstract base class for all retrieval strategies."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._index()

    @abstractmethod
    def _index(self) -> None:
        """Build the retrieval index from the provided chunks."""
        pass

    @abstractmethod
    def _search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        """Search the index and return (Chunk, score) tuples."""
        pass

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve top_k most relevant chunks for a given query."""
        if not self.chunks or top_k <= 0 or not query.strip():
            return []

        raw_results = self._search(query=query, top_k=top_k)
        results: List[RetrievalResult] = []
        for rank, (chunk, score) in enumerate(raw_results, start=1):
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    rank=rank
                )
            )
        return results


def _tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer in lowercase."""
    return re.findall(r"\b\w+\b", text.lower())


class PurePythonBM25:
    """Zero-dependency BM25 implementation (Okapi BM25) for deterministic fallback."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 1.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self._calc_idf(corpus)

    def _calc_idf(self, corpus: List[List[str]]) -> None:
        df: Dict[str, int] = Counter()
        for doc in corpus:
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for word in frequencies.keys():
                df[word] += 1

        for word, freq in df.items():
            # Standard Lucene/Okapi smoothed IDF
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for i, doc_freq in enumerate(self.doc_freqs):
                if token in doc_freq:
                    tf = doc_freq[token]
                    doc_len = self.doc_lens[i]
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    scores[i] += idf_val * (numerator / denominator)
        return scores


class BM25Retriever(BaseRetriever):
    """BM25 keyword search retriever."""

    def _index(self) -> None:
        self.tokenized_corpus = [_tokenize(chunk.text) for chunk in self.chunks]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self.tokenized_corpus)
            self._is_rank_bm25 = True
        except ImportError:
            self._bm25 = PurePythonBM25(self.tokenized_corpus)
            self._is_rank_bm25 = False

    def _search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [(self.chunks[i], scores[i]) for i in ranked_indices if scores[i] > 0.0]


class TFIDFVectorRetriever(BaseRetriever):
    """Deterministic TF-IDF cosine-similarity vector retriever (zero external dependency)."""

    def _index(self) -> None:
        self.doc_tokens = [_tokenize(c.text) for c in self.chunks]
        self.n_docs = len(self.chunks)
        df: Dict[str, int] = Counter()
        for doc in self.doc_tokens:
            unique_words = set(doc)
            for w in unique_words:
                df[w] += 1

        self.idf: Dict[str, float] = {
            w: math.log((1.0 + self.n_docs) / (1.0 + freq)) + 1.0
            for w, freq in df.items()
        }

        self.doc_vectors: List[Dict[str, float]] = []
        self.doc_norms: List[float] = []

        for doc in self.doc_tokens:
            tf = Counter(doc)
            vec: Dict[str, float] = {}
            for w, count in tf.items():
                if w in self.idf:
                    vec[w] = (count / len(doc)) * self.idf[w] if doc else 0.0
            norm = math.sqrt(sum(v * v for v in vec.values()))
            self.doc_vectors.append(vec)
            self.doc_norms.append(norm if norm > 0.0 else 1.0)

    def _search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        q_tf = Counter(q_tokens)
        q_vec: Dict[str, float] = {}
        for w, count in q_tf.items():
            if w in self.idf:
                q_vec[w] = (count / len(q_tokens)) * self.idf[w]
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0.0:
            return []

        scores: List[float] = []
        for i, d_vec in enumerate(self.doc_vectors):
            dot_product = sum(q_val * d_vec.get(w, 0.0) for w, q_val in q_vec.items())
            sim = dot_product / (q_norm * self.doc_norms[i])
            scores.append(sim)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], scores[i]) for i in ranked_indices if scores[i] > 0.0]


class VectorRetriever(BaseRetriever):
    """
    Dense vector retriever using sentence-transformers when available,
    otherwise falling back automatically to TFIDFVectorRetriever.
    """

    def __init__(self, chunks: List[Chunk], model_name: Optional[str] = None):
        self.model_name = model_name
        self._dense_model = None
        self._dense_embeddings = None
        self._fallback_retriever: Optional[TFIDFVectorRetriever] = None
        super().__init__(chunks)

    def _index(self) -> None:
        if not self.chunks:
            return

        if self.model_name is not None:
            try:
                from sentence_transformers import SentenceTransformer
                self._dense_model = SentenceTransformer(self.model_name)
                texts = [c.text for c in self.chunks]
                self._dense_embeddings = self._dense_model.encode(texts, normalize_embeddings=True)
                return
            except Exception:
                pass

        self._fallback_retriever = TFIDFVectorRetriever(self.chunks)

    def _search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        if self._dense_model is not None and self._dense_embeddings is not None:
            import numpy as np
            q_emb = self._dense_model.encode([query], normalize_embeddings=True)[0]
            scores = np.dot(self._dense_embeddings, q_emb)
            ranked = np.argsort(-scores)[:top_k]
            return [(self.chunks[i], float(scores[i])) for i in ranked]

        if self._fallback_retriever is not None:
            return self._fallback_retriever._search(query, top_k)

        return []


def get_retriever(
    name: str,
    chunks: List[Chunk],
    vector_model_name: Optional[str] = None
) -> BaseRetriever:
    """Factory function for instantiating retrievers."""
    clean_name = name.lower().strip()
    if clean_name == "bm25":
        return BM25Retriever(chunks=chunks)
    elif clean_name in ["vector", "dense", "vector_search"]:
        return VectorRetriever(chunks=chunks, model_name=vector_model_name)
    elif clean_name in ["tfidf", "sparse_vector"]:
        return TFIDFVectorRetriever(chunks=chunks)
    else:
        raise ValueError(
            f"Unknown retrieval strategy '{name}'. Supported: 'bm25', 'vector', 'tfidf'"
        )
