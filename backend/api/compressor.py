# backend/api/compressor.py
#
# Stage 4 of multi-stage RAG: drop sentences from each chunk that aren't
# relevant to the query. Keeps the LLM context focused and short.
#
# Uses Ollama's nomic-embed-text (already running) — no new model load.

import os
import re
from typing import List

import httpx
import numpy as np
from langchain_core.documents import Document

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Keep sentences with cosine similarity above this threshold
SIMILARITY_THRESHOLD = float(os.getenv("COMPRESS_THRESHOLD", "0.55"))
# Always keep at least this many top-scoring sentences per chunk
MIN_SENTENCES_PER_CHUNK = int(os.getenv("COMPRESS_MIN_SENTENCES", "2"))


def _split_sentences(text: str) -> List[str]:
    # Simple multilingual sentence splitter — good enough for FR/EN/AR
    # Splits on .!?؟ followed by whitespace or end-of-string
    parts = re.split(r"(?<=[.!?؟])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 15]


def _embed(text: str) -> np.ndarray:
    try:
        resp = httpx.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return np.array(resp.json()["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"[compressor] embed error: {e}")
        return np.zeros(768, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compress(query: str, documents: List[Document]) -> List[Document]:
    """
    For each document, keep only the sentences relevant to the query.
    If compression strips everything, keep the original chunk.
    """
    if not documents:
        return []

    query_vec = _embed(query)
    if not np.any(query_vec):
        return documents  # embedding failed, return as-is

    compressed: List[Document] = []
    for doc in documents:
        sentences = _split_sentences(doc.page_content)
        if len(sentences) <= 1:
            compressed.append(doc)
            continue

        scored = []
        for sent in sentences:
            sent_vec = _embed(sent)
            score = _cosine(query_vec, sent_vec)
            scored.append((sent, score))

        # Keep sentences above threshold OR top-N if too few pass
        kept = [s for s, sc in scored if sc >= SIMILARITY_THRESHOLD]
        if len(kept) < MIN_SENTENCES_PER_CHUNK:
            scored.sort(key=lambda x: x[1], reverse=True)
            kept = [s for s, _ in scored[:MIN_SENTENCES_PER_CHUNK]]

        new_text = " ".join(kept)
        compressed.append(Document(page_content=new_text, metadata=doc.metadata))

    return compressed