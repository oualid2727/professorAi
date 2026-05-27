# backend/api/chain.py
#
# Multi-stage RAG with hybrid retrieval, reranking, compression, and CRAG.
#
# Pipeline:
#   1. Query rewriting (multi-query + HyDE)              → query_rewriter.py
#   2. Hybrid retrieval per rewrite (vector + BM25)      → this file
#   3. Cross-encoder reranking                           → reranker.py
#   4. CRAG retrieval evaluator                          → crag.py
#   5. Contextual compression                            → compressor.py
#
# Toggle MULTISTAGE_ENABLED to fall back to the original simple pipeline
# (useful for demos / latency-sensitive environments without GPU).

import os
from typing import List, Tuple

import chromadb
import httpx
from chromadb.config import Settings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from api.query_rewriter import generate_rewrites
from api.reranker import rerank
from api.compressor import compress
from api.crag import evaluate_retrieval, get_prompt_suffix


# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_HOST       = os.getenv("CHROMA_HOST",       "chromadb")
CHROMA_PORT       = int(os.getenv("CHROMA_PORT",   "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ai_professor")
OLLAMA_HOST       = os.getenv("OLLAMA_HOST",       "ollama")
OLLAMA_PORT       = int(os.getenv("OLLAMA_PORT",   "11434"))
EMBED_MODEL       = os.getenv("EMBED_MODEL",       "nomic-embed-text")

VECTOR_K = int(os.getenv("VECTOR_K", "5"))
BM25_K   = int(os.getenv("BM25_K",   "5"))

# Multi-stage settings
MULTISTAGE_ENABLED   = os.getenv("MULTISTAGE_ENABLED", "true").lower() == "true"
CANDIDATES_PER_QUERY = int(os.getenv("CANDIDATES_PER_QUERY", "10"))


# ── BM25 cache ────────────────────────────────────────────────────────────────

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
    """Return a cached BM25Retriever, rebuilt only when collection size changes."""
    global _bm25_cache, _bm25_cache_size
    try:
        collection = _get_collection()
        current_size = collection.count()

        if _bm25_cache is not None and current_size == _bm25_cache_size:
            return _bm25_cache

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


# ── Single-query retrieval (used by both pipelines) ──────────────────────────

def _retrieve_candidates_single(query: str, k: int) -> List[Document]:
    """Hybrid retrieval for one query — returns up to 2*k deduped candidates."""
    collection = _get_collection()

    # Vector
    try:
        qvec = _embed(query)
        vres = collection.query(
            query_embeddings=[qvec],
            n_results=k,
            include=["documents", "metadatas"],
        )
        vec_docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(vres["documents"][0], vres["metadatas"][0])
        ]
    except Exception as e:
        print(f"[retrieval] vector error: {e}")
        vec_docs = []

    # BM25
    try:
        bm25 = _get_bm25()
        if bm25:
            bm25.k = k
            bm25_docs = bm25.invoke(query)
        else:
            bm25_docs = []
    except Exception as e:
        print(f"[retrieval] bm25 error: {e}")
        bm25_docs = []

    seen, merged = set(), []
    for doc in vec_docs + bm25_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged


# ── Simple pipeline (original behavior — fallback) ───────────────────────────

def retrieve_context_simple(query: str) -> Tuple[str, List[dict], str]:
    """Original hybrid retrieval. Returns (context, sources, grade)."""
    docs = _retrieve_candidates_single(query, VECTOR_K)
    docs = docs[:VECTOR_K]
    context_text = "\n\n---\n\n".join(doc.page_content for doc in docs)
    sources = [doc.metadata for doc in docs]
    return context_text, sources, "correct"


# ── Multi-stage pipeline with CRAG ───────────────────────────────────────────

def retrieve_context_multistage(query: str) -> Tuple[str, List[dict], str]:
    """
    Full multi-stage pipeline:
      query rewriting → hybrid retrieval → rerank → CRAG → compression
    Returns (context, sources, grade) where grade is one of:
      'correct'   — proceed normally
      'ambiguous' — generate but the LLM should hedge
      'incorrect' — refuse to answer from course material
    """
    # ── Stage 1: query rewriting ─────────────────────────────────────────────
    queries = generate_rewrites(query)

    # ── Stage 2: hybrid retrieval per query, merged ──────────────────────────
    all_candidates: List[Document] = []
    seen = set()
    for q in queries:
        for doc in _retrieve_candidates_single(q, CANDIDATES_PER_QUERY):
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                all_candidates.append(doc)
    print(f"[multistage] {len(all_candidates)} unique candidates after retrieval")

    if not all_candidates:
        return "", [], "incorrect"

    # ── Stage 3: cross-encoder reranking ─────────────────────────────────────
    reranked = rerank(query, all_candidates)
    print(f"[multistage] top reranker scores: "
          f"{[round(s, 3) for _, s in reranked[:3]]}")

    # ── Stage 4: CRAG evaluator ──────────────────────────────────────────────
    grade = evaluate_retrieval(query, reranked)

    # If the evaluator says nothing is relevant, return empty context.
    # The WebSocket handler will append the refusal instruction to the prompt.
    if grade == "incorrect":
        return "", [], grade

    # ── Stage 5: contextual compression ──────────────────────────────────────
    top_docs = [doc for doc, _ in reranked]
    compressed = compress(query, top_docs)

    context_text = "\n\n---\n\n".join(d.page_content for d in compressed)
    sources = []
    for (doc, score), comp in zip(reranked, compressed):
        meta = dict(doc.metadata)
        meta["score"] = round(score, 3)
        meta["chunk_text"] = comp.page_content
        sources.append(meta)

    return context_text, sources, grade


# ── Public entry point ────────────────────────────────────────────────────────

def retrieve_context(query: str) -> Tuple[str, List[dict], str]:
    """
    Returns (context, sources, grade).
    The grade is used by main.py to decide which prompt suffix to append.
    """
    if MULTISTAGE_ENABLED:
        return retrieve_context_multistage(query)
    return retrieve_context_simple(query)