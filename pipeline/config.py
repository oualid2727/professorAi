# pipeline/config.py

import os

# ── Paths ─────────────────────────────────────────────────────────────────────

LANDING_DIR = "/opt/app/data/landing"
BRONZE_PATH = "/opt/app/data/delta/bronze"
SILVER_PATH = "/opt/app/data/delta/silver"
GOLD_PATH   = "/opt/app/data/delta/gold"

# ── Ollama ────────────────────────────────────────────────────────────────────

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "ollama")
OLLAMA_PORT  = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")