# backend/api/chain.py
#
# RAG chain with hybrid retrieval: vector similarity + BM25 keyword search.
#
# Uses the raw chromadb client directly for both vector search and BM25 doc loading
# — avoids the langchain-chroma package which caps at chromadb<0.6.0.

import os
from typing import List, Tuple

import chromadb
import httpx
from chromadb.config import Settings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_HOST       = os.getenv("CHROMA_HOST",       "chromadb")
CHROMA_PORT       = int(os.getenv("CHROMA_PORT",   "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ai_professor")
OLLAMA_HOST       = os.getenv("OLLAMA_HOST",       "ollama")
OLLAMA_PORT       = int(os.getenv("OLLAMA_PORT",   "11434"))
EMBED_MODEL       = os.getenv("EMBED_MODEL",       "nomic-embed-text")

VECTOR_K = int(os.getenv("VECTOR_K", "5"))
BM25_K   = int(os.getenv("BM25_K",   "5"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_collection():
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _embed(text: str) -> List[float]:
    """Get embedding vector from Ollama for a query string."""
    resp = httpx.post(
        f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _load_all_docs() -> List[Document]:
    """Load all documents from ChromaDB for BM25 index."""
    collection = _get_collection()
    result = collection.get(include=["documents", "metadatas"])
    docs = []
    for text, meta in zip(result["documents"], result["metadatas"]):
        if text:
            docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve_context(query: str) -> Tuple[str, List[dict]]:
    """
    Hybrid retrieval: vector similarity + BM25 keyword search.
    Returns (context_text, sources).
    """
    collection = _get_collection()

    # ── Vector search ─────────────────────────────────────────────────────────
    try:
        query_embedding = _embed(query)
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=VECTOR_K,
            include=["documents", "metadatas"],
        )
        vector_docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(
                vector_results["documents"][0],
                vector_results["metadatas"][0],
            )
        ]
    except Exception as e:
        print(f"[retrieval] vector search error: {e}")
        vector_docs = []

    # ── BM25 keyword search ───────────────────────────────────────────────────
    try:
        all_docs = _load_all_docs()
        if all_docs:
            bm25 = BM25Retriever.from_documents(all_docs)
            bm25.k = BM25_K
            bm25_docs = bm25.invoke(query)
        else:
            bm25_docs = []
    except Exception as e:
        print(f"[retrieval] BM25 error: {e}")
        bm25_docs = []

    # ── Merge and deduplicate ─────────────────────────────────────────────────
    seen = set()
    merged = []
    for doc in vector_docs + bm25_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    docs = merged[:VECTOR_K]  # cap at VECTOR_K total

    context_text = "\n\n---\n\n".join(doc.page_content for doc in docs)
    sources = [doc.metadata for doc in docs]
    return context_text, sources