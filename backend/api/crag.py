# backend/api/crag.py
#
# CRAG (Corrective RAG) — Yan et al., 2024.
#
# Adds a retrieval evaluator on top of the multi-stage RAG pipeline.
# Grades the retrieved chunks against the query and returns one of:
#   'correct'    — confident the answer is in the chunks, proceed normally
#   'ambiguous'  — partial match, generate but the LLM should hedge
#   'incorrect'  — nothing relevant; refuse to answer from course material
#
# Classic CRAG falls back to web search on 'incorrect'. We don't — this is a
# closed-domain academic professor, so the corrective action is an honest
# refusal. This is documented in the README as a deliberate domain adaptation.
#
# Two evaluation modes:
#   (a) score-based  — uses the cross-encoder scores from the reranker (fast, free)
#   (b) llm-based    — asks Llama 3 to grade each chunk (slower, more nuanced)
# Default is score-based because the reranker scores are already produced.

import os
from typing import List, Tuple, Literal

import httpx
from langchain_core.documents import Document

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "ollama")
OLLAMA_PORT  = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Mode: 'score' (default) or 'llm'
CRAG_MODE = os.getenv("CRAG_MODE", "score").lower()

# Thresholds for the score-based evaluator.
# These are calibrated for BAAI/bge-reranker-v2-m3 which outputs logits
# in roughly [-10, +10]. After sigmoid they sit in [0, 1].
# Raw logits are easier to threshold: > 0 means relevant, > 5 means very relevant.
CRAG_CORRECT_THRESHOLD   = float(os.getenv("CRAG_CORRECT_THRESHOLD",   "2.0"))
CRAG_INCORRECT_THRESHOLD = float(os.getenv("CRAG_INCORRECT_THRESHOLD", "-2.0"))

Grade = Literal["correct", "ambiguous", "incorrect"]


# ── Mode (a): score-based evaluator ──────────────────────────────────────────

def _evaluate_by_score(scored_docs: List[Tuple[Document, float]]) -> Grade:
    """
    Use the cross-encoder scores from the reranker stage to grade retrieval.

    Decision logic (based on the highest-scoring chunk):
      max_score >  CRAG_CORRECT_THRESHOLD     → 'correct'
      max_score <  CRAG_INCORRECT_THRESHOLD   → 'incorrect'
      otherwise                               → 'ambiguous'
    """
    if not scored_docs:
        return "incorrect"

    max_score = max(score for _, score in scored_docs)

    if max_score > CRAG_CORRECT_THRESHOLD:
        return "correct"
    if max_score < CRAG_INCORRECT_THRESHOLD:
        return "incorrect"
    return "ambiguous"


# ── Mode (b): LLM-based evaluator ────────────────────────────────────────────

_GRADER_PROMPT = """You are a strict retrieval evaluator. Decide whether the
following passages contain enough information to answer the student's question.

Reply with ONLY ONE WORD:
- "yes"     if the passages clearly contain the answer
- "partial" if they mention the topic but don't fully answer
- "no"      if they are off-topic or unrelated

Question: {question}

Passages:
{passages}

Answer:"""


def _evaluate_by_llm(query: str, docs: List[Document]) -> Grade:
    """Ask Llama 3 to grade the retrieval. Slower but more nuanced."""
    if not docs:
        return "incorrect"

    # Build a compact passage block — first 400 chars per doc
    passages = "\n\n---\n\n".join(
        f"[{i + 1}] {doc.page_content[:400]}"
        for i, doc in enumerate(docs[:5])
    )

    try:
        resp = httpx.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": _GRADER_PROMPT.format(question=query, passages=passages),
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=20,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip().lower()

        # Be tolerant of phrasing variations
        if answer.startswith("yes"):
            return "correct"
        if answer.startswith("no"):
            return "incorrect"
        if answer.startswith("partial"):
            return "ambiguous"

        # Couldn't parse — be conservative
        print(f"[crag] unparseable grader response: {answer!r}")
        return "ambiguous"

    except Exception as e:
        print(f"[crag] llm evaluator error: {e}")
        return "ambiguous"


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_retrieval(
    query: str,
    scored_docs: List[Tuple[Document, float]],
) -> Grade:
    """
    Grade the reranked candidates. Routes to score-based or LLM-based mode
    depending on CRAG_MODE.

    Parameters
    ----------
    query       : the student's original question
    scored_docs : output of reranker.rerank() — list of (Document, score)
    """
    if CRAG_MODE == "llm":
        docs = [doc for doc, _ in scored_docs]
        grade = _evaluate_by_llm(query, docs)
    else:
        grade = _evaluate_by_score(scored_docs)

    print(f"[crag] retrieval grade: {grade} (mode={CRAG_MODE})")
    return grade


# ── Corrective actions ────────────────────────────────────────────────────────

REFUSAL_PROMPT_SUFFIX = (
    "\n\nIMPORTANT: The course material does not contain information to answer "
    "this question. Politely inform the student that this topic has not been "
    "covered in the course, and suggest they ask their instructor or consult "
    "additional resources. Do NOT invent an answer."
)

HEDGE_PROMPT_SUFFIX = (
    "\n\nIMPORTANT: The course material only partially covers this question. "
    "Answer based on what is available, but explicitly tell the student which "
    "parts of their question are NOT covered by the course material."
)


def get_prompt_suffix(grade: Grade) -> str:
    """Return the instruction to append to the LLM prompt based on the grade."""
    if grade == "incorrect":
        return REFUSAL_PROMPT_SUFFIX
    if grade == "ambiguous":
        return HEDGE_PROMPT_SUFFIX
    return ""