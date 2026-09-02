# RAGBench ⚡

A modular, lightweight benchmark toolkit for systematically comparing **chunking**, **embedding**, and **retrieval** strategies on Retrieval-Augmented Generation (RAG) datasets.

---

## 1. Bu Proje Ne Yapıyor? (What It Does)

**RAGBench**, RAG (Retrieval-Augmented Generation) sistemlerinde kullanılan farklı metin bölme (chunking) ve arama/erişim (retrieval) tekniklerini aynı veri seti üzerinde deterministik olarak karşılaştırır.

- **Chunking Stratejileri**: Fixed-size windowing vs. Recursive structural boundaries.
- **Retrieval Stratejileri**: Okapi BM25 keyword matching vs. Sparse TF-IDF cosine vs. Dense sentence-transformer embeddings vs. Hybrid RRF fusion.
- **Doğruluk ve Hız Metrikleri**: Recall@K, Precision@K, MRR (Mean Reciprocal Rank), query latency (ms) ve isteğe bağlı answer generation evaluation hooks.

## 2. Neden Kullanılır? (Why Use It)

RAG performansı ve model halüsinasyonu yalnızca kullanılan LLM modeline bağlı değildir; **retrieval katmanındaki mühendislik kararları kritik fark yaratır**.

RAGBench sayesinde:
- Chunk boyutunun ve örtüşme oranının (overlap) retrieval doğruluğuna etkisini ölçebilirsiniz.
- BM25 anahtar kelime araması ile semantik vektör aramasının güçlü/zayıf kaldığı sorguları tespit edebilirsiniz.
- Harici bir SaaS bağımlılığı olmadan, yerel makinede veya CI pipeline'larında deterministik testler çalıştırabilirsiniz.

## 3. Nasıl Kurulur? (Installation)

### Gereksinimler
- Python 3.10+

### Kurulum

```bash
# Depoyu klonlayın
git clone https://github.com/your-username/ragbench.git
cd ragbench

# Temel kurulum (sıfır harici C bağımlılığı)
pip install -e .

# Test ve ek bağımlılıklarla birlikte kurulum
pip install -e ".[all]"
```

## 4. Nasıl Çalıştırılır? (Quick Start & Usage)

### CLI Üzerinden Hızlı Test

Depoda hazır bulunan örnek teknik dokümantasyon veri seti ile hemen çalıştırın:

```bash
ragbench --dataset examples/data/sample_tech_docs.json --chunkers fixed recursive --retrievers bm25 tfidf --k-values 1 3 5
```

#### Örnek Terminal Çıktısı (ASCII Table)

```text
========================================================================================
 RAGBench Evaluation Report: tech_knowledge_base
 Documents: 5 | Queries: 5
========================================================================================
Chunking     | Retrieval    | Chunks | R@1   | R@3   | R@5   | MRR    | Lat(ms)
----------------------------------------------------------------------------
fixed        | bm25         | 5      | 1.000 | 1.000 | 1.000 | 1.000  | 0.11   
fixed        | tfidf        | 5      | 1.000 | 1.000 | 1.000 | 1.000  | 0.02   
recursive    | bm25         | 5      | 1.000 | 1.000 | 1.000 | 1.000  | 0.06   
recursive    | tfidf        | 5      | 1.000 | 1.000 | 1.000 | 1.000  | 0.01   
========================================================================================
```

JSON formatında çıktı almak veya dosyaya kaydetmek için:

```bash
ragbench --dataset examples/data/sample_tech_docs.json --json --output report.json
```

---

### Python API Olarak Kullanım

```python
from ragbench import BenchmarkRunner, BenchmarkConfig
from ragbench.dataset import load_dataset_from_json
from ragbench.cli import format_ascii_table

# 1. Veri setini yükle
dataset = load_dataset_from_json("examples/data/sample_tech_docs.json")

# 2. Konfigürasyon tanımla
config = BenchmarkConfig(
    chunking_strategies=["fixed", "recursive"],
    retrieval_strategies=["bm25", "tfidf"],
    k_values=[1, 3, 5],
    chunk_size=300,
    chunk_overlap=40
)

# 3. Benchmark'ı çalıştır
runner = BenchmarkRunner(config=config)
report = runner.run(dataset=dataset)

# 4. Sonuçları yazdır
print(format_ascii_table(report))
```

---

## Veri Seti Formatı (Dataset Schema)

Veri setinizi basit bir JSON dosyası olarak hazırlayabilirsiniz:

```json
{
  "name": "my_benchmark",
  "description": "My domain-specific RAG benchmark dataset",
  "documents": [
    {
      "id": "doc_1",
      "text": "Full text of document 1...",
      "metadata": {"category": "tech"}
    }
  ],
  "queries": [
    {
      "id": "q_1",
      "query": "What is the primary topic of document 1?",
      "relevant_doc_ids": ["doc_1"],
      "ground_truth_answer": "Expected answer for LLM evaluation hook"
    }
  ]
}
```

---

## Mimari Özeti (Architecture)

```text
┌─────────────────────────────────────────────────────────────┐
│                       BenchmarkRunner                       │
├──────────────────────────────┬──────────────────────────────┤
│      Chunking Engine         │      Retrieval Engine        │
│  - FixedSizeChunker          │  - BM25Retriever             │
│  - RecursiveCharacterChunker │  - TFIDFVectorRetriever      │
│                              │  - HybridRetriever (RRF)       │
│                              │  - VectorRetriever (dense)   │
├──────────────────────────────┴──────────────────────────────┤
│                     Evaluation Engine                       │
│  - Recall@K, Precision@K, MRR, Query Latency (ms)           │
│  - Answer Hooks (ExactMatch, TokenOverlap F1)               │
└─────────────────────────────────────────────────────────────┘
```

---

## Testleri Çalıştırma

```bash
pytest --cov=ragbench --cov-report=term-missing
```

---

## Yol Haritası (Roadmap)

- [x] V1 — Dataset loader (JSON & Directory)
- [x] V1 — 2 Chunking strategy (`fixed`, `recursive`)
- [x] V1 — BM25 (Okapi & pure-Python fallback)
- [x] V1 — Vector retrieval (TF-IDF cosine & SentenceTransformers)
- [x] V1 — Evaluation metrics (Recall@K, Precision@K, MRR, Latency)
- [x] V1 — Answer evaluation hook (Token overlap / Exact match)
- [x] V1 — CLI & ASCII report generator
- [x] V2 — Hybrid retrieval (RRF / Reciprocal Rank Fusion)
- [ ] V2 — Cross-encoder reranking integration (FlashRank / Cohere / BGE)
- [ ] V2 — Multi-embedding model comparisons

---

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında dağıtılmaktadır.
