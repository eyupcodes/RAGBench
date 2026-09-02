# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [0.2.0] - 2026-09-02

### Added
- **Hybrid Retrieval (RRF)**: `HybridRetriever` combining BM25 and TF-IDF via Reciprocal Rank Fusion (`rrf_k` default 60). Accessible as `hybrid`, `rrf`, `hybrid_rrf`.
- **BenchmarkConfig**: New `rrf_k` field for tuning RRF constant.
- **CLI**: `--rrf-k` flag and `hybrid` retriever option.
