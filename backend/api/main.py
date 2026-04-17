# backend/api/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import asyncio
import base64
import json
import uuid
import httpx
import numpy as np

# ── Import local modules at startup so errors surface in logs immediately ─────
# If any of these fail the server will refuse to start with a clear traceback,
# rather than silently dropping connections at runtime.
from api.memory import load_history, save_turn, clear_history  # noqa: E402
from api.chain import retrieve_context                          # noqa: E402
from api.stt import transcribe                                  # noqa: E402


app = FastAPI(title="AI Professor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── RAG prompt builder ────────────────────────────────────────────────────────

def build_rag_prompt(question: str, context: str) -> str:
    return (
        f"Use the following course material to answer the student's question.\n\n"
        f"--- COURSE CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Student question: {question}"
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


# ── Audio / viseme stubs ──────────────────────────────────────────────────────

def synthesize_dummy_audio(text_chunk: str) -> bytes:
    duration_sec = 0.2
    sample_rate = 16000
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 220.0 * t)
    return (audio * 32767).astype(np.int16).tobytes()


def generate_visemes(text_chunk: str) -> List[Dict[str, Any]]:
    mapping = {"A": "A", "E": "E", "I": "I", "O": "O", "U": "U"}
    visemes, timestamp = [], 0.0
    for ch in text_chunk.upper():
        if ch in mapping:
            visemes.append({"id": mapping[ch], "timestamp": round(timestamp, 3)})
            timestamp += 0.08
    return visemes


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

            user_text = message.get("text", "")

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
            prompt = build_rag_prompt(user_text, context) if context else user_text

            # ── 4. Save user turn ─────────────────────────────────────────────
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, save_turn, session_id, "user", user_text
                )
                print(f"[memory] session={session_id[:8]}… saved user turn")
            except Exception as e:
                print(f"[memory] save_turn(user) error: {e}")

            # ── 5. Stream LLM response ────────────────────────────────────────
            accumulated = ""
            async for token in stream_ollama_response(prompt, history):
                accumulated += token
                audio_b64 = base64.b64encode(synthesize_dummy_audio(token)).decode()
                await websocket.send_json({
                    "text_chunk": token,
                    "audio_b64": audio_b64,
                    "visemes": generate_visemes(token),
                    "is_final": False,
                })

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


@app.get("/health")
async def health():
    return {"status": "ok"}