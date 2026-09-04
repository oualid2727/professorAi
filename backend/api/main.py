# backend/api/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect,Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
import asyncio
import base64
import json
import uuid
import os
import re
import httpx

# ── Import local modules at startup so errors surface in logs immediately ─────
from api.memory import load_history, save_turn, clear_history  # noqa: E402
from api.chain import retrieve_context                          # noqa: E402
from api.crag import get_prompt_suffix                          # noqa: E402
from api.stt import transcribe                                  # noqa: E402
from api.tts import synthesize                                  # noqa: E402
from api.analytics import (                                     # noqa: E402
    log_retrieval,
    get_confusion_heatmap,
    get_unanswered_questions,
    get_top_chunks,
    get_activity_summary,
)
from api.trace import trace_query


app = FastAPI(title="AI Professor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── LLM backend toggle ────────────────────────────────────────────────────────
# LLM_BACKEND=ollama (default) uses your local Llama 3 via Ollama.
# LLM_BACKEND=groq  uses Groq's free hosted API (much faster, needs GROQ_API_KEY).

LLM_BACKEND  = os.getenv("LLM_BACKEND", "groq").lower()   # "ollama" | "groq"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# ── Language detection ────────────────────────────────────────────────────────
# The reply language is auto-detected from the student's question text (works
# for both typed text and Whisper transcripts). We then force the LLM to reply
# in that language, because Llama 3 8B does not reliably follow a soft
# "reply in the same language" instruction when the course context is in a
# different language than the question.

try:
    from langdetect import detect as _ld_detect, DetectorFactory
    DetectorFactory.seed = 0   # deterministic detection
    _LANGDETECT_AVAILABLE = True
except Exception as e:
    print(f"[lang] langdetect unavailable ({e}); falling back to default language")
    _LANGDETECT_AVAILABLE = False

# ISO code → (human name for the LLM prompt, TTS language code)
_LANG_MAP = {
    "fr": ("French",  "fr"),
    "en": ("English", "en"),
    "ar": ("Arabic",  "ar"),
    "es": ("Spanish", "es"),
    "de": ("German",  "de"),
    "it": ("Italian", "it"),
    "pt": ("Portuguese", "pt"),
}

DEFAULT_LANG_CODE = os.getenv("DEFAULT_REPLY_LANGUAGE", "fr")


def detect_language(text: str) -> tuple[str, str, str]:
    """
    Detect the language of the student's message.
    Returns (iso_code, human_name, tts_code).
    Falls back to DEFAULT_LANG_CODE when detection is unavailable or unsure.
    """
    fallback_code = DEFAULT_LANG_CODE if DEFAULT_LANG_CODE in _LANG_MAP else "fr"
    fallback = (fallback_code,) + _LANG_MAP.get(fallback_code, ("French", "fr"))

    if not _LANGDETECT_AVAILABLE or not text or len(text.strip()) < 3:
        return fallback

    try:
        code = _ld_detect(text)
    except Exception:
        return fallback

    if code in _LANG_MAP:
        name, tts = _LANG_MAP[code]
        return (code, name, tts)
    # Detected a language we don't explicitly support → fall back
    return fallback


# ── RAG prompt builder ────────────────────────────────────────────────────────

def build_rag_prompt(question: str, context: str, language: str = "", grade: str = "correct") -> str:
    """
    Build the prompt sent to the LLM. The CRAG grade decides whether to
    append a refusal suffix ('incorrect'), a hedge suffix ('ambiguous'),
    or nothing ('correct').
    """
    lang_instruction = f"Reply in {language}." if language else ""
    base = (
        f"Use the following course material to answer the student's question.\n\n"
        f"--- COURSE CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Student question: {question}\n\n"
        f"{lang_instruction}"
    )
    return base + get_prompt_suffix(grade)


def build_refusal_prompt(question: str, language: str = "") -> str:
    """Used when CRAG grade is 'incorrect' — no context available at all."""
    lang_instruction = f"Reply in {language}." if language else ""
    return (
        f"Student question: {question}\n\n"
        f"{lang_instruction}"
        + get_prompt_suffix("incorrect")
    )


def _system_prompt(language_name: str) -> str:
    """Shared system prompt for both backends."""
    return (
        f"You are a specialized Professor speaking aloud to a student. You MUST write your ENTIRE reply in {language_name}.\n\n"
        "You must use:\n"
        "1. The provided course context (if available)\n"
        "2. The conversation history (previous messages)\n\n"
        "If the student asks about something personal mentioned earlier "
        "(like their name or preferences), you MUST use the conversation history.\n\n"
        "If the answer is not in the course context but is in the conversation history, answer using the history.\n\n"
        "Only say you haven't covered the topic if it's neither in the context nor in the conversation.\n\n"
        "IMPORTANT — SPOKEN OUTPUT: your response will be read aloud by a text-to-speech "
        "engine, not displayed as text. Do NOT use any markdown formatting: no asterisks for "
        "bold/italics, no bullet points, no numbered lists, no backticks, no code blocks. "
        "Write in plain, natural spoken sentences, the way a professor would actually talk "
        "in a lecture. If you need to list things, say them in a flowing sentence "
        "(e.g. 'first... then... finally...') instead of using list formatting. "
        "For code examples, describe what the code does in words rather than showing raw syntax.\n\n"
        f"CRITICAL LANGUAGE RULE: The student is communicating in {language_name}. "
        f"Your complete response must be written in {language_name}, from the first word to the last. "
        f"Even if the course material is written in another language, you must translate it and answer in {language_name}. "
        f"Do not switch languages in the middle of your answer."
    )


# ── Ollama streaming ──────────────────────────────────────────────────────────

async def stream_ollama_response(
    prompt: str,
    history: List[Dict[str, str]],
    model: str = "llama3",
    language_name: str = "French",
):
    """Stream tokens from local Ollama with conversation history."""
    url = "http://ollama:11434/api/chat"

    messages = [
        {"role": "system", "content": _system_prompt(language_name)},
        *history,
        {"role": "user", "content": prompt},
    ]

    payload = {"model": model, "stream": True, "messages": messages}

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("done"):
                    break
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content


# ── Groq streaming ─────────────────────────────────────────────────────────────

async def stream_groq_response(
    prompt: str,
    history: List[Dict[str, str]],
    language_name: str = "French",
):
    """Stream tokens from Groq's OpenAI-compatible API. Same contract as
    stream_ollama_response — yields content strings as they arrive."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": _system_prompt(language_name)},
        *history,
        {"role": "user", "content": prompt},
    ]

    payload = {"model": GROQ_MODEL, "stream": True, "messages": messages}

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                body = await response.aread()
                print(f"[groq] error {response.status_code}: {body[:300]}")
                return
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content


def get_llm_stream(prompt: str, history: List[Dict[str, str]], language_name: str):
    """Dispatches to Groq or Ollama depending on LLM_BACKEND. Falls back to
    Ollama automatically if Groq is selected but no API key is set."""
    if LLM_BACKEND == "groq":
        if not GROQ_API_KEY:
            print("[llm] LLM_BACKEND=groq but GROQ_API_KEY is unset — falling back to Ollama")
            return stream_ollama_response(prompt, history, language_name=language_name)
        return stream_groq_response(prompt, history, language_name=language_name)
    return stream_ollama_response(prompt, history, language_name=language_name)


# ── Sentence buffering for TTS streaming ─────────────────────────────────────

_SENTENCE_END = re.compile(r'[.!?;:,*•\-]')
_MIN_CHUNK_LEN = 20

def _is_sentence_end(token: str) -> bool:
    return bool(_SENTENCE_END.search(token))


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/professor")
async def professor_ws(websocket: WebSocket):
    await websocket.accept()

    session_id: str | None = None

    try:
        while True:
            message = await websocket.receive_json()

            # ── Resolve session ───────────────────────────────────────────────
            if session_id is None:
                session_id = message.get("session_id") or str(uuid.uuid4())
                await websocket.send_json({"session_id": session_id, "is_final": False})

            # ── Handle clear_history command ──────────────────────────────────
            if message.get("command") == "clear_history":
                await asyncio.get_event_loop().run_in_executor(
                    None, clear_history, session_id
                )
                await websocket.send_json({"info": "history cleared", "is_final": False})
                continue

            user_text  = message.get("text", "")

            # ── Audio input → Whisper STT ─────────────────────────────────────
            if not user_text and message.get("audio_b64"):
                try:
                    audio_bytes = base64.b64decode(message["audio_b64"])
                    mime_type   = message.get("mime_type", "audio/webm")
                    user_text   = await asyncio.get_event_loop().run_in_executor(
                        None, transcribe, audio_bytes, mime_type
                    )
                    print(f"[stt] transcribed: {user_text!r}")
                    if user_text:
                        await websocket.send_json({
                            "transcript": user_text,
                            "is_final": False,
                        })
                except Exception as e:
                    print(f"[stt] error: {e}")

            if not user_text:
                continue

            # ── Detect reply language from the question (text or transcript) ──
            lang_code, language_name, tts_lang = detect_language(user_text)
            print(f"[lang] detected '{lang_code}' → replying in {language_name}")

            # ── 1. Load conversation history ──────────────────────────────────
            try:
                history = await asyncio.get_event_loop().run_in_executor(
                    None, load_history, session_id
                )
                print(f"[memory] session={session_id[:8]}… loaded {len(history)} turns")
            except Exception as e:
                print(f"[memory] load_history error: {e}")
                history = []

            # ── 2. Multi-stage retrieval with CRAG ────────────────────────────
            try:
                context, sources, grade = await asyncio.get_event_loop().run_in_executor(
                    None, retrieve_context, user_text
                )
            except Exception as e:
                print(f"[retrieval] error: {e}")
                context, sources, grade = "", [], "ambiguous"

            # ── 2b. Log retrieval for analytics dashboard ────────────────────
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, log_retrieval, session_id, user_text, sources
                )
            except Exception as e:
                print(f"[analytics] log_retrieval error: {e}")

            # Send the CRAG grade to the client so the UI can show a badge
            # ("Answered from course material" / "Partial coverage" / "Out of scope")
            await websocket.send_json({
                "crag_grade": grade,
                "is_final":   False,
            })

            # ── 3. Build prompt — varies by CRAG grade ────────────────────────
            if grade == "incorrect":
                # No relevant context — refusal prompt with no course material
                prompt = build_refusal_prompt(user_text, language_name)
            elif context:
                prompt = build_rag_prompt(user_text, context, language_name, grade)
            else:
                # No context but grade isn't 'incorrect' — pass through with hedge
                prompt = user_text + f"\n\nReply in {language_name}."

            # ── 4. Save user turn ─────────────────────────────────────────────
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, save_turn, session_id, "user", user_text
                )
                print(f"[memory] session={session_id[:8]}… saved user turn")
            except Exception as e:
                print(f"[memory] save_turn(user) error: {e}")

            # ── 5. Stream LLM response with sentence-level TTS ────────────────
            accumulated   = ""
            sentence_buf  = ""
            # tts_lang comes from detect_language() above — matches the reply language

            async for token in get_llm_stream(prompt, history, language_name):
                accumulated  += token
                sentence_buf += token

                await websocket.send_json({
                    "text_chunk": token,
                    "audio_b64":  None,
                    "visemes":    [],
                    "is_final":   False,
                })

                if _is_sentence_end(token) and len(sentence_buf.strip()) >= _MIN_CHUNK_LEN:
                    try:
                        wav_bytes, visemes = await asyncio.get_event_loop().run_in_executor(
                            None, synthesize, sentence_buf.strip(), tts_lang
                        )
                        print(f"[ws] sending audio chunk: {len(wav_bytes)} bytes, {len(visemes)} visemes")
                        audio_b64 = base64.b64encode(wav_bytes).decode()
                        await websocket.send_json({
                            "text_chunk": "",
                            "audio_b64":  audio_b64,
                            "visemes":    visemes,
                            "is_final":   False,
                        })
                    except Exception as e:
                        print(f"[tts] error during streaming: {e}")
                    sentence_buf = ""

            if sentence_buf.strip():
                try:
                    wav_bytes, visemes = await asyncio.get_event_loop().run_in_executor(
                        None, synthesize, sentence_buf.strip(), tts_lang
                    )
                    audio_b64 = base64.b64encode(wav_bytes).decode()
                    await websocket.send_json({
                        "text_chunk": "",
                        "audio_b64":  audio_b64,
                        "visemes":    visemes,
                        "is_final":   False,
                    })
                except Exception as e:
                    print(f"[tts] error on final chunk: {e}")

            # ── 6. Save assistant turn ────────────────────────────────────────
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, save_turn, session_id, "assistant", accumulated
                )
                print(f"[memory] session={session_id[:8]}… saved assistant turn")
            except Exception as e:
                print(f"[memory] save_turn(assistant) error: {e}")

            # ── 7. Final message ──────────────────────────────────────────────
            await websocket.send_json({
                "text_chunk": accumulated,
                "audio_b64":  None,
                "visemes":    [],
                "sources":    sources,
                "crag_grade": grade,
                "is_final":   True,
            })

    except WebSocketDisconnect:
        return


# ── Analytics endpoints ──────────────────────────────────────────────────────

@app.get("/analytics/summary")
async def analytics_summary(days: int = 7):
    data = await asyncio.get_event_loop().run_in_executor(
        None, get_activity_summary, days
    )
    return data


@app.get("/analytics/heatmap")
async def analytics_heatmap(days: int = 7):
    data = await asyncio.get_event_loop().run_in_executor(
        None, get_confusion_heatmap, days
    )
    return data


@app.get("/analytics/unanswered")
async def analytics_unanswered(days: int = 7, limit: int = 50):
    data = await asyncio.get_event_loop().run_in_executor(
        None, get_unanswered_questions, days, limit
    )
    return data


@app.get("/analytics/top-chunks")
async def analytics_top_chunks(days: int = 7, limit: int = 20):
    data = await asyncio.get_event_loop().run_in_executor(
        None, get_top_chunks, days, limit
    )
    return data


@app.post("/diagnostic/trace")
async def diagnostic_trace(payload: dict = Body(...)):
    """
    Run a query through the multi-stage RAG + CRAG pipeline with full
    instrumentation. Returns the trace as JSON for the diagnostic UI.
 
    POST /diagnostic/trace
    Body: { "query": "Explain the central limit theorem" }
    """
    query = (payload or {}).get("query", "").strip()
    if not query:
        return {"error": "query is required"}
 
    trace = await asyncio.get_event_loop().run_in_executor(
        None, trace_query, query
    )
    return trace

@app.post("/quiz/generate")
async def quiz_generate(payload: dict = Body(...)):
    """
    Generate a course-grounded multiple-choice quiz.
 
    POST /quiz/generate
    Body: { "topic": "les boucles", "num_questions": 5 }
 
    Returns the quiz dict from quiz.generate_quiz (questions, sources,
    grounded flag, optional error). The reply language is auto-detected
    from the topic text, same as the chat flow.
    """
    from api.quiz import generate_quiz
 
    topic = (payload or {}).get("topic", "").strip()
    num_questions = (payload or {}).get("num_questions", 5)
    if not topic:
        return {"error": "topic is required", "questions": [], "grounded": False}
 
    # Auto-detect the language from the topic (reuses the helper added for the
    # language fix). Falls back to French inside detect_language if unsure.
    _lang_code, language_name, _tts = detect_language(topic)
 
    result = await asyncio.get_event_loop().run_in_executor(
        None, generate_quiz, topic, num_questions, language_name
    )
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}