# backend/api/trace.py
#
# Pipeline tracer: instruments retrieve_context_multistage to record what
# happened at each stage. Used by the diagnostic UI to visualize the pipeline.
#
# Usage:
#   from api.trace import trace_query
#   trace = trace_query("Explique le théorème central limite")
#   # trace is a dict with: rewrites, candidates, reranked, grade, compressed
#
# This is a parallel pipeline to the production one — it duplicates the
# retrieval logic so it can record intermediate state without modifying
# the hot path. Worth the duplication for jury demo purposes.

import os
import time
from typing import Dict, Any, List

from langchain_core.documents import Document

from api.query_rewriter import generate_rewrites
from api.reranker import rerank
from api.compressor import compress
from api.crag import evaluate_retrieval, CRAG_MODE
from api.chain import _retrieve_candidates_single, CANDIDATES_PER_QUERY


def _doc_summary(doc: Document, max_len: int = 200) -> Dict[str, Any]:
    """Truncate doc content for the trace payload."""
    text = doc.page_content
    return {
        "preview":  text[:max_len] + ("…" if len(text) > max_len else ""),
        "full":     text,
        "length":   len(text),
        "metadata": {
            "filename":  doc.metadata.get("filename",  ""),
            "subject":   doc.metadata.get("subject",   ""),
            "chapter":   doc.metadata.get("chapter",   ""),
            "professor": doc.metadata.get("professor", ""),
            "language":  doc.metadata.get("language",  ""),
        },
    }


def trace_query(query: str) -> Dict[str, Any]:
    """
    Run the multi-stage pipeline with full instrumentation.
    Returns a dict suitable for the diagnostic UI.
    """
    trace: Dict[str, Any] = {
        "query":     query,
        "timings":   {},
        "config": {
            "candidates_per_query": CANDIDATES_PER_QUERY,
            "crag_mode":            CRAG_MODE,
            "reranker":             os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            "embed_model":          os.getenv("EMBED_MODEL",    "nomic-embed-text"),
        },
    }

    # ── Stage 1: query rewriting ─────────────────────────────────────────────
    t0 = time.time()
    rewrites = generate_rewrites(query)
    trace["timings"]["rewrite_ms"] = int((time.time() - t0) * 1000)
    trace["rewrites"] = [
        {
            "text":      r,
            "is_hyde":   i == len(rewrites) - 1 and len(rewrites) > 1,
            "is_origin": i == 0,
        }
        for i, r in enumerate(rewrites)
    ]

    # ── Stage 2: retrieval per rewrite ───────────────────────────────────────
    t0 = time.time()
    all_candidates: List[Document] = []
    seen = set()
    per_query_counts: List[Dict[str, Any]] = []

    for q in rewrites:
        local_docs = _retrieve_candidates_single(q, CANDIDATES_PER_QUERY)
        new_count = 0
        for doc in local_docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                all_candidates.append(doc)
                new_count += 1
        per_query_counts.append({
            "query":     q[:120] + ("…" if len(q) > 120 else ""),
            "retrieved": len(local_docs),
            "new":       new_count,
        })

    trace["timings"]["retrieval_ms"] = int((time.time() - t0) * 1000)
    trace["retrieval"] = {
        "per_query":           per_query_counts,
        "unique_candidates":   len(all_candidates),
        "total_with_dupes":    sum(p["retrieved"] for p in per_query_counts),
    }

    if not all_candidates:
        trace["candidates"] = []
        trace["reranked"]   = []
        trace["grade"]      = "incorrect"
        trace["compressed"] = []
        return trace

    trace["candidates"] = [
        {**_doc_summary(doc), "index": i}
        for i, doc in enumerate(all_candidates)
    ]

    # ── Stage 3: reranking ───────────────────────────────────────────────────
    t0 = time.time()
    reranked = rerank(query, all_candidates)
    trace["timings"]["rerank_ms"] = int((time.time() - t0) * 1000)
    trace["reranked"] = [
        {
            **_doc_summary(doc),
            "score":           round(score, 4),
            "rank":            i + 1,
            "original_index":  all_candidates.index(doc),
        }
        for i, (doc, score) in enumerate(reranked)
    ]

    # ── Stage 4: CRAG ────────────────────────────────────────────────────────
    t0 = time.time()
    grade = evaluate_retrieval(query, reranked)
    trace["timings"]["crag_ms"] = int((time.time() - t0) * 1000)
    trace["grade"] = grade
    trace["grade_thresholds"] = {
        "correct":   float(os.getenv("CRAG_CORRECT_THRESHOLD",   "2.0")),
        "incorrect": float(os.getenv("CRAG_INCORRECT_THRESHOLD", "-2.0")),
    }
    trace["max_score"] = round(max((s for _, s in reranked), default=0.0), 4)

    # ── Stage 5: compression ─────────────────────────────────────────────────
    if grade == "incorrect":
        trace["compressed"] = []
    else:
        t0 = time.time()
        top_docs = [doc for doc, _ in reranked]
        compressed = compress(query, top_docs)
        trace["timings"]["compress_ms"] = int((time.time() - t0) * 1000)
        trace["compressed"] = [
            {
                "before":          original.page_content,
                "after":           comp.page_content,
                "before_length":   len(original.page_content),
                "after_length":    len(comp.page_content),
                "reduction_pct":   round(
                    100 * (1 - len(comp.page_content) / max(len(original.page_content), 1)),
                    1,
                ),
                "metadata":        _doc_summary(original)["metadata"],
            }
            for original, comp in zip(top_docs, compressed)
        ]

    trace["timings"]["total_ms"] = sum(trace["timings"].values())
    return trace