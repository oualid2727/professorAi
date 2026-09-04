# backend/api/llm_client.py
#
# Shared non-streaming LLM call, used by every module that does a single
# generate-and-return call (query_rewriter, crag llm-mode, quiz).
# The WebSocket chat flow in main.py has its own streaming versions since
# it needs token-by-token output, but both paths respect the same
# LLM_BACKEND env var so switching to Groq is a single toggle everywhere.

import os
import httpx

LLM_BACKEND  = os.getenv("LLM_BACKEND", "groq").lower()   # "ollama" | "groq"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "ollama")
OLLAMA_PORT  = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def _generate_ollama(prompt: str, temperature: float = 0.1, timeout: float = 30,
                      json_mode: bool = False) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"
    resp = httpx.post(
        f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _generate_groq(prompt: str, temperature: float = 0.1, timeout: float = 30,
                    json_mode: bool = False) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def generate(prompt: str, temperature: float = 0.1, timeout: float = 30,
             json_mode: bool = False) -> str:
    """
    Single non-streaming completion. Routes to Groq or Ollama based on
    LLM_BACKEND. Falls back to Ollama automatically if Groq is selected
    but GROQ_API_KEY is missing, or if the Groq call errors out.

    json_mode=True asks the backend to constrain output to valid JSON
    (Ollama's format="json", Groq's response_format={"type":"json_object"}).
    Only use this when the prompt itself also instructs JSON output —
    the flag alone doesn't inject that instruction.
    """
    if LLM_BACKEND == "groq" and GROQ_API_KEY:
        try:
            return _generate_groq(prompt, temperature, timeout, json_mode)
        except Exception as e:
            print(f"[llm_client] groq error, falling back to ollama: {e}")
            return _generate_ollama(prompt, temperature, timeout, json_mode)

    if LLM_BACKEND == "groq" and not GROQ_API_KEY:
        print("[llm_client] LLM_BACKEND=groq but GROQ_API_KEY is unset — using Ollama")

    return _generate_ollama(prompt, temperature, timeout, json_mode)