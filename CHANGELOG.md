# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Dataset loaders now raise `DatasetLoadError` on duplicate document or query IDs instead of silently accepting them; the directory loader reports colliding file names such as `a/notes.md` and `b/notes.md` (#3).

## [0.1.0] - 2026-09-01

### Added
- **Dataset Loader**: Support for loading documents and test queries from structured JSON files and directories.
- **Chunking Strategies**:
  - `FixedSizeChunker`: Character-level chunking with configurable overlap.
  - `RecursiveCharacterChunker`: Hierarchical splitting on natural structural boundaries (paragraphs, sentences, words).
- **Retrieval Strategies**:
  - `BM25Retriever`: Okapi BM25 implementation with pure Python fallback.
  - `TFIDFVectorRetriever`: Deterministic cosine-similarity vector retrieval with zero external dependencies.
  - `VectorRetriever`: Dense embedding retrieval using sentence-transformers with automatic fallback.
- **Evaluation Metrics**:
  - Recall@K and Precision@K for arbitrary K thresholds.
  - Mean Reciprocal Rank (MRR).
  - Retrieval latency tracking in milliseconds.
  - Answer evaluation hooks (`ExactMatchAnswerEvaluator`, `TokenOverlapAnswerEvaluator`).
- **CLI & Reporting**:
  - Command-line runner `ragbench` with configurable arguments.
  - Formatted ASCII comparison table and JSON export options.
  - Comprehensive unit and integration test suite with 90% test coverage.

## [0.4.0] - 2026-09-02

### Added
- **Multi-Embedding Comparison**: Compare multiple dense embedding models across vector retrieval simultaneously.
- **Retriever Syntax**: Support for `vector:<model_name>` and `dense:<model_name>` notation in `get_retriever`.
- **BenchmarkConfig & Models**: Added `vector_model_names` to `BenchmarkConfig` and `embedding_model` to `StrategyResult`.
- **BenchmarkRunner**: Automatic expansion of vector retrieval strategies across configured embedding models.
- **CLI**: `--vector-models` flag supporting multiple model names with dynamic column widths in ASCII table.

## [0.3.0] - 2026-09-02

### Added
- **Reranking Engine**: `BaseReranker`, `NoOpReranker`, `LexicalReranker`, and `CrossEncoderReranker` in `src/ragbench/rerankers.py`.
- **BenchmarkConfig**: `reranking_strategy` and `reranker_model_name` fields.
- **BenchmarkRunner**: Automatic candidate pool retrieval and reranking pipeline.
- **CLI**: `--reranker` and `--reranker-model` flags with dynamic Reranker column in ASCII report.

## [0.2.0] - 2026-09-02

### Added
- **Hybrid Retrieval (RRF)**: `HybridRetriever` combining BM25 and TF-IDF via Reciprocal Rank Fusion (`rrf_k` default 60). Accessible as `hybrid`, `rrf`, `hybrid_rrf`.
- **BenchmarkConfig**: New `rrf_k` field for tuning RRF constant.
- **CLI**: `--rrf-k` flag and `hybrid` retriever option.
