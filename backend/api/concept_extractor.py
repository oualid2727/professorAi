# backend/api/concept_extractor.py
#
# Pedagogical RAG — Phase 1: LLM-based concept extraction.
#
# Reads a piece of text (a PDF chunk or a TP template) and returns:
#   {
#     "prerequisites": ["variables", "function definition", ...],
#     "introduces":    ["recursion", "base case", ...]
#   }
#
# Two modes:
#   - Free-form (vocabulary=None): the LLM picks any English noun phrases.
#     Used in Phase 2 to build the canonical vocabulary by clustering.
#   - Constrained (vocabulary=[...]): the LLM is told to ONLY use concepts
#     from the provided list. Used in Phase 3 once the vocabulary exists.
#
# Why English: per our design, canonical concept IDs are English regardless
# of the source language (French PDFs, Arabic questions). The LLM is told
# to translate non-English terminology into standard English noun phrases.
#
# Reliability: uses Ollama's JSON output mode (format='json') so the LLM
# is constrained to produce valid JSON. We also have robust fallback
# parsing in case of malformed output.

import json
import os
import re
from typing import Dict, List, Optional, Sequence

import httpx


# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "ollama")
OLLAMA_PORT  = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Concept phrase constraints — keep them short and meaningful.
MIN_PHRASE_LEN = 2
MAX_PHRASE_LEN = 50
# Max number of concepts the LLM should return per role per chunk.
MAX_CONCEPTS_PER_ROLE = 8

# How long to wait for Ollama to respond. Concept extraction is one of many
# LLM calls during ingestion, so we want to fail fast on bad chunks rather
# than block the whole batch.
EXTRACTION_TIMEOUT = float(os.getenv("EXTRACTION_TIMEOUT", "45.0"))


# ── Prompt construction ───────────────────────────────────────────────────────

_PROMPT_FREE = """You are a curriculum analyst. Read the passage below and identify:
  1. PREREQUISITES: concepts the reader must already know to understand this passage
  2. INTRODUCES:    concepts the passage explains, defines, or teaches

Rules:
- Use SHORT English noun phrases (1-4 words). Examples: "recursion", "for loop", "matrix multiplication", "base case".
- If the passage is in French or another language, translate the concept names to standard English terminology.
- Return at most {max_n} concepts per category. Pick the most central ones.
- If the passage doesn't clearly assume prior knowledge, return an empty prerequisites array.
- If the passage doesn't teach anything specific (e.g., it's just a code example or boilerplate), return an empty introduces array.
- Do NOT return generic concepts like "programming" or "math" — be specific.
- Do NOT include the passage's domain (e.g., "Python") as a concept.

Return JSON in this exact format:
{{
  "prerequisites": ["...", "..."],
  "introduces":    ["...", "..."]
}}

Passage:
\"\"\"
{text}
\"\"\"

JSON:"""


_PROMPT_CONSTRAINED = """You are a curriculum analyst. Read the passage below and identify which concepts from the provided vocabulary apply.

VOCABULARY (use these concept names EXACTLY as written; do not invent new ones):
{vocabulary_list}

For the passage below, return:
  1. PREREQUISITES: which vocabulary concepts the reader must already know
  2. INTRODUCES:    which vocabulary concepts the passage explains or teaches

Rules:
- ONLY use concepts from the vocabulary above. Do not invent new concept names.
- If no vocabulary concept clearly applies, return empty arrays.
- Return at most {max_n} concepts per category.

Return JSON in this exact format:
{{
  "prerequisites": ["...", "..."],
  "introduces":    ["...", "..."]
}}

Passage:
\"\"\"
{text}
\"\"\"

JSON:"""


def _build_prompt(text: str, vocabulary: Optional[Sequence[str]] = None) -> str:
    if vocabulary:
        vocab_str = "\n".join(f"- {v}" for v in vocabulary)
        return _PROMPT_CONSTRAINED.format(
            vocabulary_list=vocab_str,
            text=text.strip(),
            max_n=MAX_CONCEPTS_PER_ROLE,
        )
    return _PROMPT_FREE.format(
        text=text.strip(),
        max_n=MAX_CONCEPTS_PER_ROLE,
    )


# ── Ollama call ───────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> str:
    """
    Call Ollama with JSON output mode. Returns the raw response string.
    Returns empty string on failure (caller handles).
    """
    try:
        resp = httpx.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",      # ← key reliability win — Ollama enforces JSON
                "options": {
                    "temperature": 0.1,    # near-deterministic; we want consistency
                    "num_predict": 400,    # plenty for a JSON object with ~8 concepts/role
                },
            },
            timeout=EXTRACTION_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        print(f"[extractor] ollama error: {e}")
        return ""


# ── Parsing ───────────────────────────────────────────────────────────────────

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _parse_concepts(raw: str) -> Dict[str, List[str]]:
    """
    Robust JSON parsing. Even with format='json', LLMs sometimes wrap output
    in markdown fences or add explanation. We strip and try multiple paths.
    Returns the standard shape always: {prerequisites: [], introduces: []}.
    """
    if not raw:
        return {"prerequisites": [], "introduces": []}

    # Strip markdown fences if present
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    # Try direct parse first
    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: find the first {...} block in the string
        m = _JSON_OBJECT_RE.search(cleaned)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    if not isinstance(parsed, dict):
        print(f"[extractor] could not parse JSON: {raw[:120]!r}")
        return {"prerequisites": [], "introduces": []}

    # Coerce null or wrong-type values to empty list before cleaning
    prereqs = parsed.get("prerequisites", [])
    introduces = parsed.get("introduces", [])
    if not isinstance(prereqs, (list, tuple, str)):
        prereqs = []
    if not isinstance(introduces, (list, tuple, str)):
        introduces = []

    return {
        "prerequisites": _clean_concept_list(prereqs),
        "introduces":    _clean_concept_list(introduces),
    }


# ── Phrase cleaning ───────────────────────────────────────────────────────────

# Strip excess whitespace, dedupe, enforce length limits.
_WS_RE = re.compile(r"\s+")


def _clean_phrase(phrase: str) -> Optional[str]:
    """
    Normalize one concept phrase. Returns None if it should be dropped.
    Keeps the human-readable form (e.g. 'for loop'), not the slug.
    """
    if not isinstance(phrase, str):
        return None
    p = _WS_RE.sub(" ", phrase.strip().lower())
    if len(p) < MIN_PHRASE_LEN or len(p) > MAX_PHRASE_LEN:
        return None
    # Drop pure punctuation, single chars, etc.
    if not re.search(r"[a-z]", p):
        return None
    return p


def _clean_concept_list(items: Sequence) -> List[str]:
    """Clean, dedupe (preserving order), and truncate to MAX_CONCEPTS_PER_ROLE."""
    seen, out = set(), []
    for item in items:
        cleaned = _clean_phrase(item)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
            if len(out) >= MAX_CONCEPTS_PER_ROLE:
                break
    return out


# ── Slug helper (used by Phase 2 normalizer) ──────────────────────────────────

_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify_concept(phrase: str) -> str:
    """
    Convert a human-readable concept phrase to a stable snake_case ID.
    Used by Phase 2 when assigning canonical concept_ids.
    Examples:
      "For Loop"          → "for_loop"
      "Matrix Operations" → "matrix_operations"
      "C-style strings"   → "c_style_strings"
    """
    s = phrase.lower().strip()
    s = _SLUG_NON_ALNUM.sub("_", s)
    s = s.strip("_")
    s = re.sub(r"_+", "_", s)
    return s or "unknown_concept"


# ── Public API ────────────────────────────────────────────────────────────────

def extract_concepts(
    text: str,
    vocabulary: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    """
    Extract prerequisites and introduces concepts from a chunk of text.

    Args:
        text:        the passage to analyze (one chunk of course material).
        vocabulary:  optional list of canonical concept names. If provided,
                     the LLM is constrained to use only these. If None,
                     free-form extraction (Phase 2 vocabulary-building mode).

    Returns:
        Always a dict with two keys: 'prerequisites' and 'introduces'.
        Lists may be empty. Never raises.
    """
    if not text or not text.strip():
        return {"prerequisites": [], "introduces": []}

    # Truncate very long chunks to keep prompts under context limits.
    # Llama 3 8B has 8K context, our chunks are normally ~2000 chars,
    # but defensive truncation at 4000 chars keeps prompt+vocab+response safe.
    if len(text) > 4000:
        text = text[:4000]

    prompt = _build_prompt(text, vocabulary)
    raw = _call_ollama(prompt)
    return _parse_concepts(raw)