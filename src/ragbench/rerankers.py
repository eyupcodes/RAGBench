"""Reranking strategies for refining retrieved candidates in RAGBench."""

import re
from abc import ABC, abstractmethod
from typing import Any

from ragbench.models import RetrievalResult


class BaseReranker(ABC):
    """Abstract base class for candidate reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results for a given query."""
        pass


class NoOpReranker(BaseReranker):
    """Pass-through reranker preserving original order and scores."""

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not results:
            return []
        selected = results[:top_k] if top_k is not None else list(results)
        return [
            RetrievalResult(chunk=r.chunk, score=r.score, rank=idx)
            for idx, r in enumerate(selected, start=1)
        ]


class LexicalReranker(BaseReranker):
    """Zero-dependency lexical reranker scoring query term frequency in candidate chunks."""

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        q_tokens = re.findall(r"\w+", query.lower())
        if not q_tokens:
            return NoOpReranker().rerank(query, results, top_k)

        scored: list[RetrievalResult] = []
        for r in results:
            text_tokens = re.findall(r"\w+", r.chunk.text.lower())
            overlap_score = sum(text_tokens.count(tok) for tok in q_tokens)
            norm_score = overlap_score / max(len(text_tokens), 1)
            scored.append(RetrievalResult(chunk=r.chunk, score=norm_score, rank=0))

        scored.sort(key=lambda x: x.score, reverse=True)
        selected = scored[:top_k] if top_k is not None else scored
        for idx, item in enumerate(selected, start=1):
            item.rank = idx
        return selected


class CrossEncoderReranker(BaseReranker):
    """Reranker using sentence-transformers CrossEncoder with automatic lexical fallback."""

    def __init__(
        self,
        model_name: str | None = None,
        model: Any | None = None,
    ):
        self.model_name = model_name
        self._model = model
        self._fallback = LexicalReranker()

        if self._model is None and self.model_name is not None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = None

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        if self._model is not None:
            pairs = [[query, r.chunk.text] for r in results]
            raw_scores = self._model.predict(pairs)
            scored: list[RetrievalResult] = []
            for r, s in zip(results, raw_scores, strict=True):
                scored.append(RetrievalResult(chunk=r.chunk, score=float(s), rank=0))
            scored.sort(key=lambda x: x.score, reverse=True)
            selected = scored[:top_k] if top_k is not None else scored
            for idx, item in enumerate(selected, start=1):
                item.rank = idx
            return selected

        return self._fallback.rerank(query=query, results=results, top_k=top_k)


def get_reranker(
    name: str | None = None,
    model_name: str | None = None,
) -> BaseReranker:
    """Factory function for retrieving reranker instances."""
    if not name or name.strip().lower() in ["none", "noop"]:
        return NoOpReranker()

    clean_name = name.strip().lower()
    if clean_name in ["lexical", "overlap"]:
        return LexicalReranker()
    elif clean_name in ["cross-encoder", "cross_encoder", "ce"]:
        return CrossEncoderReranker(model_name=model_name)
    else:
        raise ValueError(
            f"Unknown reranking strategy '{name}'. Supported: 'none', 'lexical', 'cross-encoder'"
        )
