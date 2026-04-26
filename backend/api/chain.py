# backend/api/chain.py
#
# RAG chain with hybrid retrieval: vector similarity + BM25 keyword search.
#
# Uses the raw chromadb client directly — avoids langchain-chroma version conflicts.
# BM25 index is cached in memory and only rebuilt when new docs are added.

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

# ── BM25 cache ────────────────────────────────────────────────────────────────
# Built once, reused for every request. Only rebuilds when collection size
# changes (i.e. after new documents are indexed into ChromaDB).
_bm25_cache = None
_bm25_cache_size = 0


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
    resp = httpx.post(
        f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _load_all_docs() -> List[Document]:
    collection = _get_collection()
    result = collection.get(include=["documents", "metadatas"])
    docs = []
    for text, meta in zip(result["documents"], result["metadatas"]):
        if text:
            docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


def _get_bm25():
    """
    Return a cached BM25Retriever. Rebuilds only when the collection
    size changes. First call takes ~2s; subsequent calls are instant.
    """
    global _bm25_cache, _bm25_cache_size
    try:
        collection = _get_collection()
        current_size = collection.count()

        if _bm25_cache is not None and current_size == _bm25_cache_size:
            return _bm25_cache  # cache hit

        if current_size == 0:
            return None

        print(f"[retrieval] building BM25 index over {current_size} chunks...", flush=True)
        docs = _load_all_docs()
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = BM25_K
        _bm25_cache = retriever
        _bm25_cache_size = current_size
        print("[retrieval] BM25 index ready.", flush=True)
        return _bm25_cache

    except Exception as e:
        print(f"[retrieval] BM25 cache error: {e}")
        return None


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
        bm25 = _get_bm25()
        bm25_docs = bm25.invoke(query) if bm25 else []
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

    docs = merged[:VECTOR_K]

    context_text = "\n\n---\n\n".join(doc.page_content for doc in docs)
    sources = [doc.metadata for doc in docs]
    return context_text, sources