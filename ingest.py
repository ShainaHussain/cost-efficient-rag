import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader
from bs4 import BeautifulSoup

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def extract_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_html(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")

def extract_md(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(filepath)
    elif ext in [".html", ".htm"]:
        return extract_html(filepath)
    elif ext == ".md":
        return extract_md(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def make_chunk_id(filepath: str, chunk_index: int, chunk_text: str) -> str:
    content = f"{filepath}::{chunk_index}::{chunk_text[:100]}"
    return hashlib.md5(content.encode()).hexdigest()

def ingest_file(filepath: str, collection):
    filename = Path(filepath).name
    file_type = Path(filepath).suffix.lower().strip(".")
    print(f"Ingesting: {filename}")
    text = extract_text(filepath)
    if not text.strip():
        print(f"  Warning: no text extracted from {filename}")
        return 0
    chunks = chunk_text(text)
    ids, documents, metadatas = [], [], []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        chunk_id = make_chunk_id(filepath, i, chunk)
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({
            "source": filename,
            "file_type": file_type,
            "chunk_index": i,
            "total_chunks": len(chunks)
        })
    existing = collection.get(ids=ids)["ids"]
    existing_set = set(existing)
    new_ids = [id for id in ids if id not in existing_set]
    new_docs = [documents[i] for i, id in enumerate(ids) if id not in existing_set]
    new_meta = [metadatas[i] for i, id in enumerate(ids) if id not in existing_set]
    if new_ids:
        collection.add(ids=new_ids, documents=new_docs, metadatas=new_meta)
        print(f"  Added {len(new_ids)} new chunks (skipped {len(existing_set)} duplicates)")
    else:
        print(f"  All {len(ids)} chunks already exist — skipping")
    return len(new_ids)

def ingest_directory(data_dir: str = "./data"):
    collection = get_collection()
    supported = {".pdf", ".html", ".htm", ".md"}
    files = [f for f in Path(data_dir).iterdir() if f.suffix.lower() in supported]
    if not files:
        print(f"No supported files found in {data_dir}")
        return
    total = 0
    for f in files:
        total += ingest_file(str(f), collection)
    print(f"\nDone. Total new chunks added: {total}")
    print(f"Collection size: {collection.count()} chunks")

if __name__ == "__main__":
    ingest_directory()