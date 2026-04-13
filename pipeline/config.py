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

# Model used for embeddings — separate from the chat model
EMBED_MODEL  = os.getenv("EMBED_MODEL", "nomic-embed-text")

# ── Chunking ──────────────────────────────────────────────────────────────────

# Target size of each chunk in characters (~500 tokens ≈ 2000 chars for English)
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "2000"))
# How many characters the next chunk re-uses from the previous one
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── ChromaDB ──────────────────────────────────────────────────────────────────

CHROMA_HOST       = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT       = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ai_professor")