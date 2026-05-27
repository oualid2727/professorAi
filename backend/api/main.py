# backend/api/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect,Body
from fastapi.middleware.cors import CORSMiddleware
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


# ── RAG prompt builder ────────────────────────────────────────────────────────

def build_rag_prompt(question: str, context: str, language: str = "", grade: str = "correct") -> str:
    """
    Build the prompt sent to Llama 3. The CRAG grade decides whether to
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


# ── Ollama streaming ──────────────────────────────────────────────────────────

async def stream_ollama_response(
    prompt: str,
    history: List[Dict[str, str]],
    model: str = "llama3",
):
    """Stream tokens from local Ollama with conversation history."""
    url = "http://ollama:11434/api/chat"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a specialized Professor.\n\n"
                "You must use:\n"
                "1. The provided course context (if available)\n"
                "2. The conversation history (previous messages)\n\n"
                "If the student asks about something personal mentioned earlier "
                "(like their name or preferences), you MUST use the conversation history.\n\n"
                "If the answer is not in the course context but is in the conversation history, answer using the history.\n\n"
                "Only say you haven't covered the topic if it's neither in the context nor in the conversation.\n\n"
                "Always reply in the same language as the student."
            ),
        },
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
            reply_lang = ""

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
                    configured = os.getenv("WHISPER_LANGUAGE", "")
                    reply_lang = configured if configured else "the same language as the student"
                except Exception as e:
                    print(f"[stt] error: {e}")

            if not user_text:
                continue

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
                prompt = build_refusal_prompt(user_text, reply_lang)
            elif context:
                prompt = build_rag_prompt(user_text, context, reply_lang, grade)
            else:
                # No context but grade isn't 'incorrect' — pass through with hedge
                prompt = user_text + (f"\n\nReply in {reply_lang}." if reply_lang else "")

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
            tts_lang      = reply_lang.replace("the same language as the student", "") or os.getenv("TTS_LANGUAGE", "fr")

            async for token in stream_ollama_response(prompt, history):
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


@app.get("/health")
async def health():
    return {"status": "ok"}