# backend/api/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import asyncio
import base64
import json
import httpx
import numpy as np


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
    """
    Wraps the student's question with the retrieved course context so the
    LLM answers only from what's in the documents.
    """
    return (
        f"Use the following course material to answer the student's question.\n\n"
        f"--- COURSE CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Student question: {question}"
    )


# ── Ollama streaming ──────────────────────────────────────────────────────────

async def stream_ollama_response(prompt: str, model: str = "llama3"):
    """
    Stream tokens from local Ollama.
    """
    url = "http://ollama:11434/api/chat"
    payload = {
        "model": model,
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a specialized Professor. Answer only based on the "
                    "provided course context. If the answer isn't in the context, "
                    "politely say you haven't covered that topic yet. "
                    "Always reply in the same language the student used in their question."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
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
    """
    Placeholder for local TTS (Piper/Coqui).
    Generates a short dummy waveform per chunk to keep the
    streaming contract intact. Replace with real TTS integration.
    """
    duration_sec = 0.2
    sample_rate = 16000
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    freq = 220.0
    audio = 0.1 * np.sin(2 * np.pi * freq * t)
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


def generate_visemes(text_chunk: str) -> List[Dict[str, Any]]:
    """
    Very simple character-based viseme mapping.
    Replace with phoneme alignment from real TTS.
    """
    mapping = {"A": "A", "E": "E", "I": "I", "O": "O", "U": "U"}
    visemes = []
    timestamp = 0.0
    step = 0.08
    for ch in text_chunk.upper():
        viseme_id = mapping.get(ch)
        if viseme_id:
            visemes.append({"id": viseme_id, "timestamp": round(timestamp, 3)})
            timestamp += step
    return visemes


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/professor")
async def professor_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            user_text = message.get("text", "")

            # ── 1. Hybrid retrieval (BM25 + vector) ──────────────────────────
            # Run in a thread so the async event loop isn't blocked by the
            # synchronous LangChain/ChromaDB calls.
            try:
                from chain import retrieve_context  # noqa: PLC0415
                context, sources = await asyncio.get_event_loop().run_in_executor(
                    None, retrieve_context, user_text
                )
            except Exception as e:
                # If retrieval fails (e.g. Chroma not yet indexed), fall back
                # to answering without context so the API stays responsive.
                print(f"Retrieval error (falling back to no-context): {e}")
                context, sources = "", []

            # ── 2. Build RAG prompt ───────────────────────────────────────────
            if context:
                prompt = build_rag_prompt(user_text, context)
            else:
                prompt = user_text

            # ── 3. Stream LLM response ────────────────────────────────────────
            accumulated = ""
            async for token in stream_ollama_response(prompt):
                accumulated += token
                audio_bytes = synthesize_dummy_audio(token)
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                visemes = generate_visemes(token)

                await websocket.send_json({
                    "text_chunk": token,
                    "audio_b64": audio_b64,
                    "visemes": visemes,
                    "is_final": False,
                })

            # ── 4. Final message with sources ─────────────────────────────────
            await websocket.send_json({
                "text_chunk": accumulated,
                "audio_b64": None,
                "visemes": [],
                "sources": sources,   # list of {filename, subject, chapter, …}
                "is_final": True,
            })

    except WebSocketDisconnect:
        return


@app.get("/health")
async def health():
    return {"status": "ok"}