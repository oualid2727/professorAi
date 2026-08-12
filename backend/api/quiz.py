# backend/api/quiz.py
#
# Auto quiz generator. Given a topic, retrieves the relevant course material
# (reusing the existing multi-stage RAG retrieval) and asks the LLM to write
# multiple-choice questions GROUNDED in that material — not from its general
# knowledge. This is the key point: the quiz tests the student's actual course,
# not generic facts.
#
# Exposed via POST /quiz/generate in main.py.
#
# Honest behavior: if the topic isn't covered by the course material (CRAG
# grade == "incorrect"), we refuse rather than invent a quiz about something
# the course never taught.

import json
import os
import re
from typing import Dict, List, Optional

import httpx

try:
    from api.chain import retrieve_context
except ImportError:  # running as a plain script / tests
    from chain import retrieve_context


# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "ollama")
OLLAMA_PORT  = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

DEFAULT_NUM_QUESTIONS = 5
MIN_NUM_QUESTIONS = 1
MAX_NUM_QUESTIONS = 10
OPTIONS_PER_QUESTION = 4

QUIZ_TIMEOUT = float(os.getenv("QUIZ_TIMEOUT", "120.0"))


# ── Prompt ────────────────────────────────────────────────────────────────────

_QUIZ_PROMPT = """You are a teacher creating a multiple-choice quiz for your students.

Using ONLY the course material below, write {n} multiple-choice questions that test understanding of the topic: "{topic}".

Rules:
- Base every question strictly on the course material provided. Do NOT use outside knowledge.
- Each question must have exactly {opts} answer options.
- Exactly ONE option is correct.
- Make the wrong options plausible but clearly incorrect according to the material.
- Write everything in {language}.
- Add a short explanation (one sentence) for why the correct answer is right.
- Vary difficulty: some recall, some understanding.

Return JSON in EXACTLY this format, nothing else:
{{
  "questions": [
    {{
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_index": 0,
      "explanation": "..."
    }}
  ]
}}

--- COURSE MATERIAL ---
{context}
--- END COURSE MATERIAL ---

JSON:"""


def _build_prompt(topic: str, context: str, n: int, language_name: str) -> str:
    return _QUIZ_PROMPT.format(
        topic=topic.strip(),
        context=context.strip(),
        n=n,
        opts=OPTIONS_PER_QUESTION,
        language=language_name,
    )


# ── Ollama ────────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> str:
    try:
        resp = httpx.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.4,   # a little variety in questions
                    "num_predict": 1500,  # room for several questions
                },
            },
            timeout=QUIZ_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        print(f"[quiz] ollama error: {e}")
        return ""


# ── Parsing & validation ──────────────────────────────────────────────────────

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _parse_quiz(raw: str) -> List[Dict]:
    """
    Parse and VALIDATE the quiz JSON. Returns a list of clean question dicts.
    Malformed questions are dropped, not fixed. Never raises.
    """
    if not raw:
        return []

    cleaned = raw.replace("```json", "").replace("```", "").strip()
    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(cleaned)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    if not isinstance(parsed, dict):
        print(f"[quiz] could not parse JSON: {raw[:120]!r}")
        return []

    raw_questions = parsed.get("questions", [])
    if not isinstance(raw_questions, list):
        return []

    clean: List[Dict] = []
    for q in raw_questions:
        validated = _validate_question(q)
        if validated:
            clean.append(validated)
    return clean


def _validate_question(q) -> Optional[Dict]:
    """Return a clean question dict, or None if it's malformed."""
    if not isinstance(q, dict):
        return None

    question = q.get("question")
    options  = q.get("options")
    correct  = q.get("correct_index")
    explanation = q.get("explanation", "")

    # question must be a non-empty string
    if not isinstance(question, str) or not question.strip():
        return None

    # options must be a list of >= 2 non-empty strings
    if not isinstance(options, list) or len(options) < 2:
        return None
    clean_options = [str(o).strip() for o in options if str(o).strip()]
    if len(clean_options) < 2:
        return None

    # correct_index must be an int pointing at a real option
    try:
        correct = int(correct)
    except (TypeError, ValueError):
        return None
    if correct < 0 or correct >= len(clean_options):
        return None

    if not isinstance(explanation, str):
        explanation = ""

    return {
        "question": question.strip(),
        "options": clean_options,
        "correct_index": correct,
        "explanation": explanation.strip(),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def generate_quiz(
    topic: str,
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    language_name: str = "French",
) -> Dict:
    """
    Generate a course-grounded multiple-choice quiz on `topic`.

    Returns a dict:
      {
        "topic": str,
        "language": str,
        "grounded": bool,           # True if built from course material
        "questions": [ {...}, ... ],
        "sources": [ {...}, ... ],  # the chunks used, for transparency
        "error": str | None,
      }
    Never raises.
    """
    topic = (topic or "").strip()
    if not topic:
        return _error("No topic provided.", topic, language_name)

    # Clamp question count
    try:
        n = int(num_questions)
    except (TypeError, ValueError):
        n = DEFAULT_NUM_QUESTIONS
    n = max(MIN_NUM_QUESTIONS, min(MAX_NUM_QUESTIONS, n))

    # 1. Retrieve course material for the topic (reuse multi-stage RAG)
    try:
        context, sources, grade = retrieve_context(topic)
    except Exception as e:
        print(f"[quiz] retrieval error: {e}")
        return _error("Could not retrieve course material.", topic, language_name)

    # 2. Honest refusal: topic not in the course
    if grade == "incorrect" or not context.strip():
        return {
            "topic": topic,
            "language": language_name,
            "grounded": False,
            "questions": [],
            "sources": sources or [],
            "error": "This topic does not appear to be covered by the course material, so I can't make a quiz on it.",
        }

    # 3. Generate
    prompt = _build_prompt(topic, context, n, language_name)
    raw = _call_ollama(prompt)
    questions = _parse_quiz(raw)

    if not questions:
        return _error(
            "The quiz could not be generated. Try a more specific topic, or try again.",
            topic, language_name, sources=sources,
        )

    return {
        "topic": topic,
        "language": language_name,
        "grounded": True,
        "questions": questions,
        "sources": sources or [],
        "error": None,
    }


def _error(msg: str, topic: str, language_name: str, sources=None) -> Dict:
    return {
        "topic": topic,
        "language": language_name,
        "grounded": False,
        "questions": [],
        "sources": sources or [],
        "error": msg,
    }