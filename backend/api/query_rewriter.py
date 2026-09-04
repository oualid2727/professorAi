# backend/api/query_rewriter.py
#
# Stage 1 of multi-stage RAG: rewrite the user's question into multiple forms
# to increase recall. Combines:
#   - Multi-query: 3 paraphrased versions of the question
#   - HyDE: a pseudo-answer the LLM thinks would answer the question
#
# All rewrites are then passed to the retriever; results are merged.
# LLM calls route through llm_client, respecting LLM_BACKEND (ollama|groq).

import os
import json
from typing import List

from api.llm_client import generate as llm_generate

NUM_REWRITES = int(os.getenv("NUM_REWRITES", "3"))


_REWRITE_PROMPT = """You are helping retrieve relevant passages from a course.
Generate {n} different reformulations of the student's question. Each should:
- use different vocabulary / synonyms
- include technical terms a textbook would use
- be a complete question

Return ONLY a JSON array of strings, no preamble.

Student question: {question}

JSON array:"""


_HYDE_PROMPT = """You are a professor. Write a short (3-4 sentences) textbook-style
answer to the following question. The answer doesn't need to be perfectly correct;
its purpose is to match the vocabulary of real course material.

Question: {question}

Answer:"""


def _llm_generate(prompt: str, temperature: float = 0.3) -> str:
    try:
        return llm_generate(prompt, temperature=temperature, timeout=30)
    except Exception as e:
        print(f"[rewriter] llm error: {e}")
        return ""


def generate_rewrites(question: str) -> List[str]:
    """Return [original, rewrite_1, rewrite_2, ..., hyde_doc]."""
    queries = [question]

    # ── Multi-query ──────────────────────────────────────────────────────────
    raw = _llm_generate(_REWRITE_PROMPT.format(n=NUM_REWRITES, question=question))
    try:
        # Strip markdown fences if the model added any
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        # Find the JSON array
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            rewrites = json.loads(cleaned[start:end + 1])
            if isinstance(rewrites, list):
                queries.extend([str(r) for r in rewrites if r])
    except Exception as e:
        print(f"[rewriter] failed to parse rewrites: {e}")

    # ── HyDE pseudo-document ─────────────────────────────────────────────────
    hyde = _llm_generate(_HYDE_PROMPT.format(question=question), temperature=0.5)
    if hyde:
        queries.append(hyde)

    print(f"[rewriter] {len(queries)} queries generated")
    return queries