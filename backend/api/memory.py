# backend/api/memory.py
#
# Conversation memory backed by PostgreSQL.
#
# Schema (auto-created on first use):
#   chat_history (
#       id         SERIAL PRIMARY KEY,
#       session_id TEXT        NOT NULL,
#       role       TEXT        NOT NULL,   -- 'user' or 'assistant'
#       content    TEXT        NOT NULL,
#       created_at TIMESTAMPTZ DEFAULT NOW()
#   )

import os
from typing import List, Dict

import psycopg2
from psycopg2.extras import RealDictCursor

# ── Config ────────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://ai_prof:ai_prof_pw@postgres:5432/ai_prof_db",
)

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))

# Table is created once at module load — not on every request
_table_ready = False


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_conn():
    dsn = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")
    return psycopg2.connect(dsn)


def _ensure_table() -> None:
    global _table_ready
    if _table_ready:
        return
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id         SERIAL PRIMARY KEY,
                        session_id TEXT        NOT NULL,
                        role       TEXT        NOT NULL,
                        content    TEXT        NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chat_history_session
                    ON chat_history (session_id, created_at)
                """)
            conn.commit()
        _table_ready = True
        print("[memory] chat_history table ready")
    except Exception as e:
        print(f"[memory] _ensure_table error: {e}")
        raise


# ── Public API ────────────────────────────────────────────────────────────────

def load_history(session_id: str) -> List[Dict[str, str]]:
    """
    Return the last MAX_HISTORY_TURNS×2 messages for this session,
    ordered oldest-first so they read naturally in the prompt.
    """
    try:
        _ensure_table()
        with _get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT role, content
                    FROM (
                        SELECT role, content, created_at
                        FROM chat_history
                        WHERE session_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    ) sub
                    ORDER BY created_at ASC
                    """,
                    (session_id, MAX_HISTORY_TURNS * 2),
                )
                rows = [dict(row) for row in cur.fetchall()]
                return rows
    except Exception as e:
        print(f"[memory] load_history error: {e}")
        return []


def save_turn(session_id: str, role: str, content: str) -> int:
    """
    Persist one message turn to PostgreSQL.
    role must be 'user' or 'assistant'.
    Returns the new message's id (or -1 if the insert failed).
    The returned id is used by the pedagogical RAG pipeline to attach
    concept annotations to this specific turn (see concepts_db.set_message_concepts).
    """
    try:
        _ensure_table()
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_history (session_id, role, content)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (session_id, role, content),
                )
                new_id = cur.fetchone()[0]
            conn.commit()
            return int(new_id)
    except Exception as e:
        print(f"[memory] save_turn error: {e}")
        return -1


def clear_history(session_id: str) -> None:
    """Delete all messages for a session."""
    try:
        _ensure_table()
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chat_history WHERE session_id = %s",
                    (session_id,),
                )
            conn.commit()
    except Exception as e:
        print(f"[memory] clear_history error: {e}")