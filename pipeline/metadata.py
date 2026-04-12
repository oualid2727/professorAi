# pipeline/metadata.py
#
# Metadata enrichment for the silver layer.
# Handles three concerns independently so each can be tested or swapped:
#   1. Filename parsing  →  professor / subject / chapter
#   2. Ollama fallback   →  subject (when filename gives nothing)
#   3. Language detect   →  language code  (e.g. "en", "fr", "ar")

from __future__ import annotations

import re
from pathlib import Path


# ── 1. Filename parsing ───────────────────────────────────────────────────────
#
# Convention:  Professor_Subject_Chapter.pdf
# Examples:
#   Smith_LinearAlgebra_Ch3.pdf   → ("Smith", "LinearAlgebra", "Ch3")
#   Intro_To_Physics.pdf          → ("Intro", "To", "Physics")   (best-effort)
#   lecture_notes.pdf             → ("Unknown", "Unknown", "Unknown")  (only 2 parts)
#
# Rules:
#   - Strip the extension first.
#   - Split on underscores.
#   - Part 0 → professor,  part 1 → subject,  part 2 → chapter.
#   - If a part is missing, return "Unknown" for that field.
#   - CamelCase parts are left as-is (LinearAlgebra stays LinearAlgebra).

def parse_filename(filename: str) -> dict[str, str]:
    stem = Path(filename).stem          # e.g. "Smith_LinearAlgebra_Ch3"
    parts = stem.split("_")

    def get(index: int) -> str:
        val = parts[index].strip() if index < len(parts) else ""
        return val if val else "Unknown"

    return {
        "professor": get(0),
        "subject":   get(1),
        "chapter":   get(2),
    }


# ── 2. Ollama subject fallback ────────────────────────────────────────────────
#
# Called only when filename parsing returns "Unknown" for subject.
# Sends the first ~800 chars of extracted text to a local Ollama model and
# asks for a one-line subject tag.
#
# Returns a short string like "Linear Algebra" or "Unknown" on failure.

def detect_subject_ollama(
    raw_text: str,
    ollama_host: str = "ollama",
    ollama_port: int = 11434,
    model: str = "llama3",
) -> str:
    try:
        import httpx  # already in backend requirements; install in spark container too

        snippet = raw_text[:800].strip()
        if not snippet:
            return "Unknown"

        prompt = (
            "You are a document classifier. "
            "Read the following excerpt from a course document and reply with "
            "ONLY a short subject label (2–4 words, no punctuation). "
            "Examples: Linear Algebra, Introduction to Python, Organic Chemistry.\n\n"
            f"Excerpt:\n{snippet}\n\nSubject:"
        )

        resp = httpx.post(
            f"http://{ollama_host}:{ollama_port}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        subject = resp.json().get("response", "").strip()

        # Sanitise: keep only the first line, strip quotes/punctuation
        subject = subject.splitlines()[0]
        subject = re.sub(r'["""\'.,;:!?]', "", subject).strip()
        return subject if subject else "Unknown"

    except Exception:
        return "Unknown"


# ── 3. Language detection ─────────────────────────────────────────────────────
#
# Uses langdetect on the first 500 chars of extracted text.
# Returns an ISO-639-1 code ("en", "fr", "ar", …) or "unknown" on failure.
# langdetect is non-deterministic by default; we seed it for reproducibility.

def detect_language(raw_text: str) -> str:
    try:
        from langdetect import detect, DetectorFactory  # noqa: PLC0415
        DetectorFactory.seed = 42                        # make results reproducible
        snippet = raw_text[:500].strip()
        if not snippet:
            return "unknown"
        return detect(snippet)
    except Exception:
        return "unknown"


# ── Public entry point used by the Spark UDF ─────────────────────────────────

def enrich_metadata(
    filename: str,
    raw_text: str,
    ollama_host: str = "ollama",
    ollama_port: int = 11434,
    model: str = "llama3",
) -> dict[str, str]:
    """
    Returns a dict with keys: professor, subject, chapter, language.
    Called once per document row inside a Spark UDF.
    """
    parsed = parse_filename(filename)

    # Only hit Ollama when the filename gave us nothing for subject
    if parsed["subject"] == "Unknown":
        parsed["subject"] = detect_subject_ollama(
            raw_text, ollama_host, ollama_port, model
        )

    parsed["language"] = detect_language(raw_text)
    return parsed