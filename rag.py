import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from groq import Groq
import time

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOP_K = int(os.getenv("TOP_K", 5))
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")

client = Groq(api_key=GROQ_API_KEY)

def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

def retrieve(query: str, k: int = TOP_K, file_type_filter: str = None):
    collection = get_collection()
    where = {"file_type": file_type_filter} if file_type_filter else None
    start = time.time()
    results = collection.query(
        query_texts=[query],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    latency = round((time.time() - start) * 1000, 2)
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "file_type": results["metadatas"][0][i]["file_type"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": round(results["distances"][0][i], 4)
        })
    return chunks, latency

def generate(query: str, k: int = TOP_K, file_type_filter: str = None):
    chunks, retrieval_latency = retrieve(query, k, file_type_filter)

    if not chunks:
        return {
            "answer": "I could not find any relevant context to answer this question.",
            "sources": [],
            "retrieval_latency_ms": retrieval_latency,
            "total_latency_ms": retrieval_latency,
            "token_usage": {},
            "chunk_count": 0
        }

    # build context with source labels
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(f"[Source {i+1}: {chunk['source']}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are a precise question answering assistant.
Answer the question using ONLY the context provided below.
Cite which source(s) you used by referencing [Source N] inline.
If the context does not contain enough information to answer, say exactly:
"I don't have enough information in the provided context to answer this question."
Do not make up any information.

Context:
{context}

Question: {query}

Answer:"""

    start = time.time()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512
    )
    generation_latency = round((time.time() - start) * 1000, 2)
    total_latency = round(retrieval_latency + generation_latency, 2)

    answer = response.choices[0].message.content.strip()
    token_usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_latency_ms": retrieval_latency,
        "generation_latency_ms": generation_latency,
        "total_latency_ms": total_latency,
        "token_usage": token_usage,
        "chunk_count": len(chunks)
    }

if __name__ == "__main__":
    query = input("Enter your question: ")
    result = generate(query)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources used:")
    for s in result["sources"]:
        print(f"  - {s['source']} (chunk {s['chunk_index']}, distance {s['distance']})")
    print(f"\nLatency: {result['total_latency_ms']}ms | Tokens: {result['token_usage']}")