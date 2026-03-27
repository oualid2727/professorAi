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
                    "politely say you haven't covered that topic yet."
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
    mapping = {
        "A": "A",
        "E": "E",
        "I": "I",
        "O": "O",
        "U": "U",
    }
    visemes = []
    timestamp = 0.0
    step = 0.08
    for ch in text_chunk.upper():
        viseme_id = mapping.get(ch)
        if viseme_id:
            visemes.append({"id": viseme_id, "timestamp": round(timestamp, 3)})
            timestamp += step
    return visemes


@app.websocket("/ws/professor")
async def professor_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            # Supports either text input or prior STT pipeline
            user_text = message.get("text", "")

            accumulated = ""
            async for token in stream_ollama_response(user_text):
                accumulated += token
                audio_bytes = synthesize_dummy_audio(token)
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                visemes = generate_visemes(token)

                chunk_payload = {
                    "text_chunk": token,
                    "audio_b64": audio_b64,
                    "visemes": visemes,
                    "is_final": False,
                }
                await websocket.send_json(chunk_payload)

            final_payload = {
                "text_chunk": accumulated,
                "audio_b64": None,
                "visemes": [],
                "is_final": True,
            }
            await websocket.send_json(final_payload)
    except WebSocketDisconnect:
        return


@app.get("/health")
async def health():
    return {"status": "ok"}

