# backend/api/chain.py
#
# RAG chain with hybrid retrieval: vector similarity + BM25 keyword search.
#
# Why hybrid?
#   - Vector search is great for semantic questions ("explain gradient descent")
#     but misses exact terms ("what is the definition of eigenvalue").
#   - BM25 is great for exact keyword matches but has no semantic understanding.
#   - Combining both with equal weight gives the best of both worlds.
#
# Architecture:
#   ChromaRetriever  (vector, k=5) ─┐
#                                    ├─ EnsembleRetriever → top-5 merged docs → LLM
#   BM25Retriever    (keyword, k=5) ─┘

import os
from typing import List

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document


# ── Config (read from environment, with sensible defaults) ────────────────────

CHROMA_HOST       = os.getenv("CHROMA_HOST",       "chromadb")
CHROMA_PORT       = int(os.getenv("CHROMA_PORT",   "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ai_professor")
OLLAMA_HOST       = os.getenv("OLLAMA_HOST",       "ollama")
OLLAMA_PORT       = int(os.getenv("OLLAMA_PORT",   "11434"))
EMBED_MODEL       = os.getenv("EMBED_MODEL",       "nomic-embed-text")

# How many documents each retriever fetches before merging
VECTOR_K = int(os.getenv("VECTOR_K", "5"))
BM25_K   = int(os.getenv("BM25_K",   "5"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_chroma_client():
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )


def _load_all_docs_from_chroma() -> List[Document]:
    """
    Fetch every document stored in the ChromaDB collection so BM25Retriever
    can build its keyword index over the full corpus.
    BM25 is an in-memory index — it needs all docs up front.
    """
    client = _get_chroma_client()
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION)

    result = collection.get(include=["documents", "metadatas"])

    docs = []
    for text, meta in zip(result["documents"], result["metadatas"]):
        if text:
            docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


# ── Public API ────────────────────────────────────────────────────────────────

def build_retriever() -> EnsembleRetriever:
    """
    Build and return a hybrid retriever that combines:
      - ChromaDB vector similarity search  (weight 0.5)
      - BM25 keyword search                (weight 0.5)

    Both retrievers fetch k=5 documents independently.
    EnsembleRetriever merges and deduplicates the results using
    Reciprocal Rank Fusion before returning the final list.
    """
    # ── Vector retriever ──────────────────────────────────────────────────────
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}",
    )
    vectorstore = Chroma(
        client=_get_chroma_client(),
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
    )
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": VECTOR_K}
    )

    # ── BM25 retriever ────────────────────────────────────────────────────────
    # Load all docs from Chroma to build the BM25 index.
    # In production with millions of docs you'd cache this; for a local
    # professor project the full corpus fits comfortably in memory.
    all_docs = _load_all_docs_from_chroma()

    if not all_docs:
        # Collection is empty — return just the vector retriever to avoid crash
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = BM25_K

    # ── Ensemble ──────────────────────────────────────────────────────────────
    # weights=[0.5, 0.5] means both retrievers contribute equally.
    # Tune toward vector (e.g. [0.3, 0.7]) if your questions are more
    # semantic, or toward BM25 (e.g. [0.7, 0.3]) if they use exact terms.
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )


def retrieve_context(query: str) -> tuple[str, List[dict]]:
    """
    Run hybrid retrieval for a query.
    Returns:
        context_text  — all retrieved chunks joined into one string for the prompt
        sources       — list of metadata dicts (filename, subject, chapter…)
    """
    retriever = build_retriever()
    docs = retriever.invoke(query)

    context_text = "\n\n---\n\n".join(doc.page_content for doc in docs)
    sources = [doc.metadata for doc in docs]
    return context_text, sources