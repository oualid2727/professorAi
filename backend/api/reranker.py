# backend/api/reranker.py
#
# Stage 3 of multi-stage RAG: rerank candidates using a cross-encoder.
#
# Cross-encoders read (query, passage) pairs jointly — much more accurate than
# bi-encoder vector similarity, but slow. We use it on ~30 candidates from
# hybrid retrieval and keep the top 5.
#
# Model: BAAI/bge-reranker-v2-m3
#   - 568M params
#   - multilingual (100+ languages including Arabic, French, English)
#   - downloads ~1.1 GB on first use, cached afterwards

import os
from typing import List, Tuple

from langchain_core.documents import Document

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_K   = int(os.getenv("RERANK_TOP_K", "5"))

_reranker = None


def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker if _reranker is not False else None

    try:
        from sentence_transformers import CrossEncoder
        device = _get_device()
        print(f"[reranker] loading {RERANKER_MODEL} on {device}...")
        _reranker = CrossEncoder(RERANKER_MODEL, device=device, max_length=512)
        print(f"[reranker] ready on {device}.")
    except Exception as e:
        print(f"[reranker] load failed: {e}")
        _reranker = False
        return None

    return _reranker


def rerank(
    query: str,
    documents: List[Document],
    top_k: int = RERANK_TOP_K,
) -> List[Tuple[Document, float]]:
    """
    Score each (query, doc) pair with the cross-encoder and return the top_k
    documents with their scores, sorted descending.
    Falls back to original order if the reranker can't load.
    """
    if not documents:
        return []

    model = _get_reranker()
    if model is None:
        # Graceful fallback — preserve order, score 0
        return [(doc, 0.0) for doc in documents[:top_k]]

    pairs = [(query, doc.page_content) for doc in documents]
    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        print(f"[reranker] predict error: {e}")
        return [(doc, 0.0) for doc in documents[:top_k]]

    scored = list(zip(documents, [float(s) for s in scores]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]