# RAGBench

## Konumlandırma

- **Kategori:** AI / Retrieval
- **Zorluk:** Orta
- **Portföy amacı:** RAG'in yalnız API çağrısı değil retrieval engineering tarafını da bildiğini gösterir.

## Amaç

Chunking, embedding, retrieval ve reranking seçeneklerini aynı dataset üzerinde karşılaştırmak.

## Çözdüğü Problem

RAG performansı yalnız kullanılan LLM'e bağlı değildir; retrieval katmanındaki seçimler kritik fark yaratır.

## MVP Kapsamı

- Dataset loader
- 2 chunking strategy
- BM25
- Vector retrieval
- Recall@K/MRR
- Answer evaluation hook
- CLI report

## Önerilen Teknik Yığın

- Python 3.12+
- sentence-transformers
- rank-bm25
- FAISS veya Chroma
- Pydantic
- pytest

## Ölçülecek Metrikler

- Recall@K
- Precision@K
- MRR
- latency
- answer score

## V1 Kabul Kriterleri

- Temiz bir makinede README adımlarıyla kurulabilmeli.
- Ana kullanım senaryosu tek komut veya kısa bir akışla çalışmalı.
- Kritik çekirdek davranışlar testlerle doğrulanmalı.
- Hatalı input kontrollü biçimde ele alınmalı.
- Örnek input ve örnek output repoda bulunmalı.
- GitHub Actions üzerinde temel test/lint akışı çalışmalı.
- Secret veya kişisel veri repository'ye girmemeli.

## Repo Yapısı

```text
ragbench/
├── src/
├── tests/
├── examples/
├── docs/
├── .github/
│   └── workflows/
├── README.md
├── LICENSE
├── CHANGELOG.md
└── .gitignore
```

Gereksiz klasörler sırf şablon için açılmamalı.

## İlk Milestone'lar

### M1 — Foundation
- Proje skeleton
- CLI/app entrypoint
- Config modeli
- Test altyapısı
- CI

### M2 — Core
- MVP'nin ana fonksiyonları
- Temel hata yönetimi
- Örnek veri

### M3 — Quality
- Edge-case testleri
- README
- Example output
- Release hazırlığı

### M4 — Release
- `v0.1.0`
- GitHub topics
- Açıklama ve ekran görüntüsü
- Issues için başlangıç etiketleri

## Sonraki Geliştirmeler

- hybrid retrieval
- reranker
- multiple embeddings
- HTML comparison
- dataset generators

## Bilinçli Olarak Yapılmayacaklar

- İlk sürümde gereksiz SaaS/account sistemi
- Sırf "AI project" görünmesi için zorunlu LLM entegrasyonu
- Kullanılmayan mikroservisler
- Erken optimizasyon
- Gizli/local sistemlerden proprietary mantık kopyalama

## GitHub Sunumu

README ilk ekranı şu dört şeyi hızlı göstermeli:

1. Bu araç ne yapıyor?
2. Neden var?
3. 30 saniyede nasıl denenir?
4. Örnek çıktı nasıl görünüyor?

Önerilen repository topics:

`ai-retrieval`, `developer-tools`, `portfolio`, `open-source`

## Commit Standardı

Örnek:

```text
feat: add core runner
test: cover invalid configuration
fix: handle timeout correctly
docs: add quick-start example
chore: configure CI
```

## Tamamlanmış Sayılma Şartı

Proje, yalnız kod yazıldığı için tamamlanmış sayılmaz. Aşağıdakiler olmadan `Done` değildir:

- Çalışan MVP
- Test
- README
- Example
- CI
- Release tag
