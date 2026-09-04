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
from api.llm_client import generate as llm_generate

import httpx
from langchain_core.documents import Document



# Mode: 'score' (default) or 'llm'
CRAG_MODE = os.getenv("CRAG_MODE", "score").lower()

# Thresholds for the score-based evaluator.
#
# BAAI/bge-reranker-v2-m3 outputs SIGMOID-NORMALIZED scores in [0, 1], not
# raw unbounded logits — this was confirmed empirically early in the project
# (see report, Chapitre 3, calibration du module CRAG) and again through a
# 24-question real-corpus evaluation (report, Chapitre 5) that produced the
# exact thresholds below.
#
# IMPORTANT: earlier defaults here were "2.0"/"-2.0" (logit-scale values,
# unreachable by a [0,1]-bounded score) — every single retrieval fell through
# to "ambiguous" regardless of actual relevance. That bug shipped as the
# *default* even though the correct 0.7/0.3 values were separately set via
# environment variables in the GPU-specific docker-compose files. Since those
# env vars aren't set on every machine this runs on, the defaults below are
# now the real, correct values directly — no environment override required
# for the system to behave correctly out of the box.
#
# Calibration methodology (24-question real evaluation on the full course
# corpus, see run_full_evaluation.py):
#   - 8 out-of-scope questions scored 0.000-0.002 (tight, clean cluster)
#   - 8 in-scope questions scored mostly 0.6-0.998 (one outlier at 0.070)
#   - 8 designed-ambiguous questions scored mostly 0.012-0.233 (one outlier
#     at 0.602)
#   - CRAG_INCORRECT_THRESHOLD = 0.01 sits in the clean gap between the
#     out-of-scope cluster (≤0.002) and everything else (≥0.012).
#   - CRAG_CORRECT_THRESHOLD = 0.25 sits in the gap between the
#     ambiguous-question cluster (≤0.233) and the in-scope cluster (≥0.295).
#   - This raised concordance between expected and obtained verdicts from
#     33.3% (8/24, under the old buggy defaults) to 91.7% (22/24).
CRAG_CORRECT_THRESHOLD   = float(os.getenv("CRAG_CORRECT_THRESHOLD",   "0.25"))
CRAG_INCORRECT_THRESHOLD = float(os.getenv("CRAG_INCORRECT_THRESHOLD", "0.01"))

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
    """Ask the LLM to grade the retrieval. Slower but more nuanced."""
    if not docs:
        return "incorrect"

    # Build a compact passage block — first 400 chars per doc
    passages = "\n\n---\n\n".join(
        f"[{i + 1}] {doc.page_content[:400]}"
        for i, doc in enumerate(docs[:5])
    )

    try:
        answer = llm_generate(
            _GRADER_PROMPT.format(question=query, passages=passages),
            temperature=0.0,
            timeout=20,
        ).lower()

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