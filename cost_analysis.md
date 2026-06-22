# Cost Analysis: ChromaDB vs Managed Vector Databases

## Assumptions
- Embedding dimensions: 384 (all-MiniLM-L6-v2)
- Average vector size: ~1.5 KB (384 float32 dimensions + metadata overhead)
- Managed DB comparison: Pinecone (starter/serverless tier pricing as of 2024)
- Query volume: 1,000 queries/day (~30,000/month)
- Self-hosted: running on a $10/month VPS or local machine

---

## Storage Cost Comparison

| Scale         | ChromaDB (self-hosted) | Pinecone Serverless   | Weaviate Cloud        |
|---------------|------------------------|----------------------|-----------------------|
| 100K vectors  | ~$0/month (local)      | ~$8/month            | ~$25/month            |
| 1M vectors    | ~$10/month (VPS disk)  | ~$80/month           | ~$120/month           |
| 10M vectors   | ~$40/month (VPS disk)  | ~$800/month          | ~$900/month           |

### How these numbers were calculated
- ChromaDB: cost = VPS storage cost only (~$0.02/GB). 
  - 100K vectors × 1.5KB = ~150MB → negligible
  - 1M vectors × 1.5KB = ~1.5GB → ~$0.03 storage, rounded to VPS base cost
  - 10M vectors × 1.5KB = ~15GB → ~$0.30 storage, absorbed in VPS cost
- Pinecone: ~$0.08 per 1M vector-months (serverless write) + read costs
- Weaviate Cloud: based on publicly listed sandbox/starter tier pricing

---

## Query Latency Comparison

| Store          | p50 Latency | p95 Latency | Notes                        |
|----------------|-------------|-------------|------------------------------|
| ChromaDB       | ~45ms       | ~120ms      | Local, no network overhead   |
| FAISS          | ~10ms       | ~30ms       | In-memory, fastest retrieval |
| Pinecone       | ~80ms       | ~200ms      | Network round trip included  |
| Weaviate Cloud | ~100ms      | ~250ms      | Network round trip included  |

*Actual measured latencies from this project are in results/eval_results.json*

---

## Embedding Cost (one-time ingestion)

| Model                        | Cost per 1M tokens | Our corpus (~50K tokens) |
|------------------------------|--------------------|--------------------------|
| all-MiniLM-L6-v2 (local)     | $0.00              | $0.00                    |
| OpenAI text-embedding-3-small | $0.02              | ~$0.001                  |
| OpenAI text-embedding-3-large | $0.13              | ~$0.0065                 |

**Decision: local embedding eliminates recurring ingestion cost entirely.**

---

## LLM Generation Cost (Groq / LLaMA3-8b)

| Volume           | Groq (free tier) | OpenAI GPT-4o       | Anthropic Claude 3 Haiku |
|------------------|------------------|---------------------|--------------------------|
| 30K queries/month | $0.00            | ~$90/month          | ~$45/month               |

*Assumes ~500 prompt tokens + ~200 completion tokens per query*

---

## Total Monthly Cost Estimate

| Scale        | This Stack (ChromaDB + Groq + MiniLM) | Pinecone + OpenAI |
|--------------|---------------------------------------|-------------------|
| 100K vectors | ~$10/month (VPS only)                 | ~$98/month        |
| 1M vectors   | ~$10/month (VPS only)                 | ~$170/month       |
| 10M vectors  | ~$40/month (upgraded VPS)             | ~$890/month       |

---

## When Would You Switch Back to a Managed DB?

1. **Team size scales** — managed DBs handle auth, backups, monitoring out of the box. Self-hosting those adds engineering overhead that costs more than the DB subscription at scale.
2. **Multi-region queries** — Pinecone/Weaviate have global edge nodes. ChromaDB on a single VPS has one latency profile regardless of where the user is.
3. **Vector count exceeds ~50M** — HNSW index in ChromaDB starts consuming significant RAM. Managed DBs handle this transparently.
4. **SLA requirements** — a production B2B product needs 99.9% uptime guarantees. Self-hosted ChromaDB puts that burden on you.

## Was Retrieval or Generation the Weak Link?

Based on evaluation results:
- Retrieval tends to be the weak link for **specific factual questions** where the answer lives in one specific chunk and chunking boundaries split the context incorrectly.
- Generation tends to fail when **retrieved chunks are marginally relevant** — the model either refuses to answer or stitches together a plausible-sounding but unfaithful response.
- The hybrid approach (ChromaDB cosine similarity + metadata filtering) improves precision but does not solve chunking boundary problems — a sliding window or sentence-aware chunker would be the next improvement.