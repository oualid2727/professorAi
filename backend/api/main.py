# backend/api/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
# If any of these fail the server will refuse to start with a clear traceback,
# rather than silently dropping connections at runtime.
from api.memory import load_history, save_turn, clear_history  # noqa: E402
from api.chain import retrieve_context                          # noqa: E402
from api.stt import transcribe                                  # noqa: E402
from api.tts import synthesize                                  # noqa: E402
from api.analytics import log_retrieval, get_confusion_heatmap, get_unanswered_questions, get_top_chunks, get_activity_summary  # noqa: E402


app = FastAPI(title="AI Professor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── RAG prompt builder ────────────────────────────────────────────────────────

def build_rag_prompt(question: str, context: str, language: str = "") -> str:
    lang_instruction = f"Reply in {language}." if language else ""
    return (
        f"Use the following course material to answer the student's question.\n\n"
        f"--- COURSE CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Student question: {question}\n\n"
        f"{lang_instruction}"
    )


# ── Ollama streaming ──────────────────────────────────────────────────────────

async def stream_ollama_response(
    prompt: str,
    history: List[Dict[str, str]],
    model: str = "llama3",
):
    """
    Stream tokens from local Ollama.
    history is injected between the system prompt and the current turn.
    """
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


# ── Sentence buffering ───────────────────────────────────────────────────────
# XTTS v2 synthesizes a full sentence at a time — not per token.
# We buffer tokens until a sentence boundary, then synthesize the whole sentence.
# This gives low latency (audio starts after the first sentence) + good quality.

# Split on sentence endings AND commas/bullets so XTTS gets short inputs.
# Shorter inputs synthesize much faster on CPU — a 10-word phrase takes
# ~3s vs ~30s for a 60-word paragraph.
_SENTENCE_END = re.compile(r'[.!?;:,*•\-]')
_MIN_CHUNK_LEN = 20   # don't synthesize fragments shorter than this

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
            reply_lang = ""   # filled below for audio input, empty for text input

            # ── Audio input → Whisper STT ─────────────────────────────────────
            # If the client sends audio instead of text, transcribe it first.
            # The rest of the pipeline is identical either way.
            if not user_text and message.get("audio_b64"):
                try:
                    audio_bytes = base64.b64decode(message["audio_b64"])
                    mime_type   = message.get("mime_type", "audio/webm")
                    user_text   = await asyncio.get_event_loop().run_in_executor(
                        None, transcribe, audio_bytes, mime_type
                    )
                    print(f"[stt] transcribed: {user_text!r}")
                    # Echo the transcript back so the client can display it
                    if user_text:
                        await websocket.send_json({
                            "transcript": user_text,
                            "is_final": False,
                        })
                    # Tell the LLM which language to reply in — avoids it
                    # drifting to English when the RAG context is in English.
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

            # ── 2. Hybrid retrieval ───────────────────────────────────────────
            try:
                context, sources = await asyncio.get_event_loop().run_in_executor(
                    None, retrieve_context, user_text
                )
            except Exception as e:
                print(f"[retrieval] error: {e}")
                context, sources = "", []

            # ── 3. Build RAG prompt ───────────────────────────────────────────
            prompt = build_rag_prompt(user_text, context, reply_lang) if context else (
                user_text + (f"\n\nReply in {reply_lang}." if reply_lang else "")
            )

            # ── 4. Save user turn ─────────────────────────────────────────────
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, save_turn, session_id, "user", user_text
                )
                print(f"[memory] session={session_id[:8]}… saved user turn")
            except Exception as e:
                print(f"[memory] save_turn(user) error: {e}")

            # ── 5. Stream LLM response with sentence-level TTS ────────────────
            # Tokens stream in one by one. We accumulate them into a sentence
            # buffer and synthesize audio when we hit a sentence boundary.
            # This way audio starts playing after the first sentence rather
            # than waiting for the full response.
            accumulated   = ""
            sentence_buf  = ""
            tts_lang      = reply_lang.replace("the same language as the student", "") or os.getenv("TTS_LANGUAGE", "fr")

            async for token in stream_ollama_response(prompt, history):
                accumulated  += token
                sentence_buf += token

                # Send text token immediately so text renders without waiting for audio
                await websocket.send_json({
                    "text_chunk": token,
                    "audio_b64":  None,
                    "visemes":    [],
                    "is_final":   False,
                })

                # When we hit a sentence boundary, synthesize and send audio
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

            # Synthesize any remaining text that didn't end with punctuation
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
                "audio_b64": None,
                "visemes": [],
                "sources": sources,
                "is_final": True,
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


@app.get("/health")
async def health():
    return {"status": "ok"}