import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from rag import generate, retrieve

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cost-Efficient RAG API",
    description="QA service over a document corpus backed by ChromaDB",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str
    k: int = int(os.getenv("TOP_K", 5))
    file_type_filter: str = None

class QueryResponse(BaseModel):
    answer: str
    chunk_count: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    token_usage: dict
    sources: list

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    logger.info(f"Query received: '{request.question}' | k={request.k} | filter={request.file_type_filter}")
    
    result = generate(
        query=request.question,
        k=request.k,
        file_type_filter=request.file_type_filter
    )
    
    logger.info(
        f"Query completed | latency={result['total_latency_ms']}ms | "
        f"chunks={result['chunk_count']} | tokens={result['token_usage'].get('total_tokens', 0)}"
    )
    
    return QueryResponse(**result)

@app.get("/query")
def query_get(
    question: str = Query(..., description="Your question"),
    k: int = Query(int(os.getenv("TOP_K", 5)), description="Number of chunks to retrieve"),
    file_type_filter: str = Query(None, description="Filter by file type: pdf, html, md")
):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    logger.info(f"Query received: '{question}' | k={k} | filter={file_type_filter}")
    
    result = generate(query=question, k=k, file_type_filter=file_type_filter)
    
    logger.info(
        f"Query completed | latency={result['total_latency_ms']}ms | "
        f"chunks={result['chunk_count']} | tokens={result['token_usage'].get('total_tokens', 0)}"
    )
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)