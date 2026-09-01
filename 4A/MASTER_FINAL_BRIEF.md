# RAGBench — Master Final Brief

## 1. Ürün Tanımı

Chunking, embedding, retrieval ve reranking seçeneklerini aynı dataset üzerinde karşılaştırmak.

## 2. Problem

RAG performansı yalnız kullanılan LLM'e bağlı değildir; retrieval katmanındaki seçimler kritik fark yaratır.

## 3. Hedef Kullanıcı

- Developer
- AI/ML engineer veya ilgili teknik kullanıcı
- Kendi projelerinde küçük ve bağımsız araç isteyen kullanıcı

## 4. Ana Değer Önerisi

RAG'in yalnız API çağrısı değil retrieval engineering tarafını da bildiğini gösterir.

## 5. V1 Kapsamı

- Dataset loader
- 2 chunking strategy
- BM25
- Vector retrieval
- Recall@K/MRR
- Answer evaluation hook
- CLI report

## 6. V1 Dışı

- hybrid retrieval
- reranker
- multiple embeddings
- HTML comparison
- dataset generators

Bu maddeler V1 zorunluluğu değildir; yalnız sonraki sürüm adaylarıdır.

## 7. Teknik İlkeler

- Local-first veya CLI-first yaklaşım tercih edilir.
- Core logic UI'dan ayrılmalıdır.
- Provider/adapter kullanılan yerlerde interface sınırı korunmalıdır.
- Deterministic davranış mümkün olduğunda AI değerlendirmesinden önce gelmelidir.
- Test edilebilirlik mimari karar olarak ele alınmalıdır.
- Secret'lar yalnız environment üzerinden alınmalıdır.
- Kullanıcı verisi varsayılan olarak dış servise gönderilmemelidir.

## 8. Kalite Kriterleri

- README quick start çalışır.
- CI yeşil.
- Kritik core path testli.
- Hatalar kullanıcıya anlaşılır mesaj verir.
- En az bir gerçek örnek senaryo repoda bulunur.
- `v0.1.0` release üretilebilir.

## 9. Riskler

- Scope creep
- Çok erken dashboard/UI ekleme
- Provider API değişiklikleri
- Benchmark/evaluation projelerinde subjektif score'a aşırı güven
- Test verisinin yetersiz olması

## 10. Başarı Tanımı

V1, bir yabancının repository'yi README üzerinden kurup ana senaryoyu çalıştırabilmesi ve aracın neyi ölçtüğünü/çözdüğünü anlayabilmesiyle başarılıdır.
