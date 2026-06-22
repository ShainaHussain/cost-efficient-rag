# Cost-Efficient RAG Application

A production-style QA service over a multi-format document corpus, backed by ChromaDB as a low-cost vector store. Built as a credible alternative to managed vector databases like Pinecone and Weaviate, with honest evaluation across retrieval quality, answer quality, latency, and cost.

---

## Stack

| Component | Choice | Reason |
|---|---|---|
| Vector Store | ChromaDB | Persistent, embedded, zero infra overhead, native metadata filtering |
| Embeddings | all-MiniLM-L6-v2 | Free, local, 384 dims, strong semantic performance for IR tasks |
| LLM | LLaMA3-8b via Groq | Near-zero cost, low latency, OpenAI-compatible API |
| API | FastAPI | Typed, async, auto-docs at /docs |
| Secondary Store | FAISS | Benchmarked for retrieval speed comparison |

---

## Project Structure

rag-assignment/

├── data/                  # PDF, HTML, MD source documents

├── ingest.py              # chunking, embedding, idempotent ingestion

├── rag.py                 # retrieval + grounded generation

├── api.py                 # FastAPI HTTP endpoint

├── evaluate.py            # retrieval + answer eval metrics

├── eval_dataset.json      # 20 QA pairs with relevant source labels

├── cost_analysis.md       # cost comparison across scales

├── requirements.txt

├── .env.example

└── results/

└── eval_results.json 
---

## Setup

```bash
git clone <repo>
cd rag-assignment
pip install -r requirements.txt
cp .env.example .env
# add your GROQ_API_KEY to .env
```

---

## Usage

### 1. Ingest documents
```bash
python ingest.py
```
Ingests all PDF, HTML, and MD files from `./data`. Idempotent — safe to run multiple times, no duplicate vectors created.

### 2. Run the API
```bash
uvicorn api:app --reload
```
Visit `http://localhost:8000/docs` for interactive API explorer.

### 3. Query via POST
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval augmented generation?", "k": 5}'
```

### 4. Query with metadata filter
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is BM25?", "k": 5, "file_type_filter": "html"}'
```

### 5. Run evaluation
```bash
python evaluate.py
```
Outputs full results to `results/eval_results.json`.

---

## Configuration

All config via `.env` — no hardcoded values anywhere in the codebase.

| Variable | Default | Description |
|---|---|---|
| GROQ_API_KEY | required | Groq API key |
| EMBED_MODEL | all-MiniLM-L6-v2 | Sentence transformer model |
| CHUNK_SIZE | 500 | Words per chunk |
| CHUNK_OVERLAP | 50 | Overlap between chunks |
| TOP_K | 5 | Chunks retrieved per query |
| CHROMA_PERSIST_DIR | ./chroma_db | ChromaDB storage path |
| COLLECTION_NAME | rag_collection | ChromaDB collection name |
| LLM_MODEL | llama3-8b-8192 | Groq model ID |

---

## Evaluation Results

Full results in `results/eval_results.json`. Summary:

| Hit Rate                   | 0.85       |
| MRR                        | 0.68       |
| nDCG@5                     | 0.71       |
| Context Precision          | 0.54       |
| Avg Faithfulness           | 4.5 / 5    |
| Avg Relevance              | 4.9 / 5    |
| p50 Latency (eval)         | ~40s*      |
| Single query latency       | ~1.8s      |

---

## Chunking Strategy

Word-based sliding window with configurable size and overlap (default: 500 words, 50 word overlap). Chosen over character-based chunking because word boundaries produce more semantically coherent chunks. Chunk IDs are MD5 hashes of filepath + index + first 100 characters — guarantees idempotent re-ingestion with zero duplicates.

---

## Idempotency

Re-running `ingest.py` on the same files produces zero duplicate vectors. Each chunk gets a deterministic ID based on its content and position. ChromaDB skips any ID that already exists in the collection.

---

## Hallucination Handling

The generation prompt explicitly instructs the model to respond with "I don't have enough information in the provided context to answer this question" when retrieved chunks do not contain relevant information. Temperature is set to 0.1 to minimize creative deviation from the source context.

---

## Cost Analysis

See `cost_analysis.md` for full breakdown. Summary: this stack costs ~$10/month at 1M vectors vs ~$170/month for a managed Pinecone + OpenAI equivalent.

---

## When to Switch Back to a Managed DB

- Team scales and operational overhead of self-hosting exceeds subscription cost
- Multi-region latency requirements
- Vector count exceeds ~50M (RAM pressure on HNSW index)
- SLA/uptime guarantees needed for production B2B

## Was Retrieval or Generation the Weak Link?

Retrieval was the weak link but Context Precision of 0.54 means roughly half the retrieved chunks were not directly relevant. This is a chunking boundary problem: at 500 words per chunk, semantically adjacent content gets split across chunks, causing marginally relevant chunks to rank alongside the correct one. Generation performed strongly (Faithfulness 4.5/5, Relevance 4.9/5), meaning the LLM handled imperfect retrieval well without hallucinating. Next improvement: sentence-aware chunking or smaller chunk size with higher overlap
