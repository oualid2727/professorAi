# backend/api/concepts_db.py
#
# Pedagogical RAG — Phase 0: schema + data access layer.
#
# This module owns three tables used by the pedagogical scoring system:
#
#   concept_vocabulary
#       The canonical list of concepts in the course material.
#       Populated once during ingestion (Phase 1-2). ~100-200 rows.
#
#   chunk_concepts
#       Many-to-many: which chunks reference which concepts, with role.
#       Populated during gold-layer ingestion (Phase 3).
#
#   message_concepts
#       Many-to-many: which chat messages touched which concepts.
#       Populated at runtime as conversations happen (Phase 4).
#
# All tables auto-create on first use, same pattern as memory.py.
#
# Phase 0 deliverable: this module exists, tables get created on API
# startup, smoke-test CRUD works. No business logic yet — just plumbing.

import json
import os
import struct
from typing import Dict, Iterable, List, Optional, Sequence

import psycopg2
from psycopg2.extras import RealDictCursor


# ── Config ────────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://ai_prof:ai_prof_pw@postgres:5432/ai_prof_db",
)

# Roles for chunk_concepts.role and message_concepts.source.
# Defined as constants so callers don't pass typos.
ROLE_PREREQUISITE = "prerequisite"
ROLE_INTRODUCES   = "introduces"
SOURCE_ASKED      = "asked"   # student question mentioned this concept
SOURCE_TAUGHT     = "taught"  # assistant answer mentioned this concept

_tables_ready = False


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_conn():
    dsn = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")
    return psycopg2.connect(dsn)


def _ensure_tables() -> None:
    """Create the three pedagogical tables on first call. Idempotent.

    Note: message_concepts has a FK to chat_history(id), which lives in
    memory.py. We call memory._ensure_table() first to guarantee the
    target table exists regardless of import order.
    """
    global _tables_ready
    if _tables_ready:
        return

    # Make sure the FK target (chat_history) exists first.
    try:
        from api.memory import _ensure_table as _ensure_chat_history
    except ImportError:
        # When this module is imported outside the api package (e.g. in tests)
        from memory import _ensure_table as _ensure_chat_history
    _ensure_chat_history()

    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                # ── concept_vocabulary ────────────────────────────────────────
                # The canonical concept list. concept_id is a slug like
                # "for_loop" or "eigenvalues" — generated from the canonical
                # label by the normalizer in Phase 2.
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS concept_vocabulary (
                        concept_id      TEXT PRIMARY KEY,
                        canonical_label TEXT        NOT NULL,
                        description     TEXT,
                        embedding       BYTEA,
                        raw_terms       JSONB       NOT NULL DEFAULT '[]'::jsonb,
                        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)

                # ── chunk_concepts ────────────────────────────────────────────
                # Links each course chunk to its prerequisite/introduces concepts.
                # chunk_id is the ChromaDB document ID (a TEXT, not an FK to a
                # local table — chunks live in ChromaDB, not Postgres).
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chunk_concepts (
                        chunk_id   TEXT NOT NULL,
                        concept_id TEXT NOT NULL REFERENCES concept_vocabulary(concept_id)
                                                    ON DELETE CASCADE,
                        role       TEXT NOT NULL CHECK (role IN ('prerequisite','introduces')),
                        PRIMARY KEY (chunk_id, concept_id, role)
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunk_concepts_chunk
                    ON chunk_concepts (chunk_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunk_concepts_concept
                    ON chunk_concepts (concept_id, role)
                """)

                # ── message_concepts ──────────────────────────────────────────
                # Links each chat message to concepts it touched. FK to
                # chat_history.id — if a session is cleared, the cascade also
                # cleans up here.
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS message_concepts (
                        message_id BIGINT NOT NULL REFERENCES chat_history(id)
                                                    ON DELETE CASCADE,
                        concept_id TEXT   NOT NULL REFERENCES concept_vocabulary(concept_id)
                                                    ON DELETE CASCADE,
                        source     TEXT   NOT NULL CHECK (source IN ('asked','taught')),
                        PRIMARY KEY (message_id, concept_id, source)
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_message_concepts_msg
                    ON message_concepts (message_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_message_concepts_concept
                    ON message_concepts (concept_id)
                """)
            conn.commit()
        _tables_ready = True
        print("[concepts_db] pedagogical tables ready")
    except Exception as e:
        print(f"[concepts_db] _ensure_tables error: {e}")
        raise


# ── Embedding serialization ───────────────────────────────────────────────────
# Store embeddings as packed float32 bytes. Compact (~3 KB per 768-dim vector)
# and trivially deserializable. Not as fast as pgvector but no extension needed.

def serialize_embedding(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_embedding(data: bytes) -> List[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


# ── concept_vocabulary CRUD ───────────────────────────────────────────────────

def upsert_concept(
    concept_id: str,
    canonical_label: str,
    description: str = "",
    embedding: Optional[Sequence[float]] = None,
    raw_terms: Optional[List[str]] = None,
) -> None:
    """Insert or update one vocabulary entry. Used by the Phase 2 normalizer."""
    _ensure_tables()
    emb_bytes = serialize_embedding(embedding) if embedding is not None else None
    raw_terms_json = json.dumps(raw_terms or [])
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO concept_vocabulary
                        (concept_id, canonical_label, description, embedding, raw_terms, updated_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                    ON CONFLICT (concept_id) DO UPDATE SET
                        canonical_label = EXCLUDED.canonical_label,
                        description     = EXCLUDED.description,
                        embedding       = COALESCE(EXCLUDED.embedding, concept_vocabulary.embedding),
                        raw_terms       = EXCLUDED.raw_terms,
                        updated_at      = NOW()
                """, (concept_id, canonical_label, description, emb_bytes, raw_terms_json))
            conn.commit()
    except Exception as e:
        print(f"[concepts_db] upsert_concept({concept_id!r}) error: {e}")
        raise


def get_all_concepts(with_embeddings: bool = False) -> List[Dict]:
    """Return the full canonical vocabulary."""
    _ensure_tables()
    try:
        with _get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cols = "concept_id, canonical_label, description, raw_terms"
                if with_embeddings:
                    cols += ", embedding"
                cur.execute(f"SELECT {cols} FROM concept_vocabulary ORDER BY concept_id")
                rows = [dict(r) for r in cur.fetchall()]
                if with_embeddings:
                    for r in rows:
                        r["embedding"] = deserialize_embedding(r["embedding"]) if r["embedding"] else None
                return rows
    except Exception as e:
        print(f"[concepts_db] get_all_concepts error: {e}")
        return []


def get_concept(concept_id: str) -> Optional[Dict]:
    """Return one vocabulary entry by ID, or None if not found."""
    _ensure_tables()
    try:
        with _get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT concept_id, canonical_label, description, raw_terms, embedding
                    FROM concept_vocabulary
                    WHERE concept_id = %s
                """, (concept_id,))
                row = cur.fetchone()
                if not row:
                    return None
                row = dict(row)
                row["embedding"] = deserialize_embedding(row["embedding"]) if row["embedding"] else None
                return row
    except Exception as e:
        print(f"[concepts_db] get_concept({concept_id!r}) error: {e}")
        return None


def count_concepts() -> int:
    """Quick health-check: how many concepts in the vocabulary?"""
    _ensure_tables()
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM concept_vocabulary")
                return cur.fetchone()[0]
    except Exception as e:
        print(f"[concepts_db] count_concepts error: {e}")
        return 0


# ── chunk_concepts CRUD ───────────────────────────────────────────────────────

def set_chunk_concepts(
    chunk_id: str,
    prerequisites: Iterable[str] = (),
    introduces: Iterable[str] = (),
) -> None:
    """
    Replace all concept links for one chunk. Used by Phase 3 (gold integration).
    Concepts that don't exist in the vocabulary are silently skipped (we filter
    via a JOIN on concept_vocabulary, so unknown IDs just don't insert).
    """
    _ensure_tables()
    rows = []
    for cid in prerequisites:
        rows.append((chunk_id, cid, ROLE_PREREQUISITE))
    for cid in introduces:
        rows.append((chunk_id, cid, ROLE_INTRODUCES))

    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                # Clear existing
                cur.execute("DELETE FROM chunk_concepts WHERE chunk_id = %s", (chunk_id,))
                # Insert via SELECT with a JOIN that filters out unknown concept_ids.
                # This is safer than relying on FK errors — keeps the transaction alive.
                if rows:
                    values_sql = ", ".join(["(%s, %s, %s)"] * len(rows))
                    flat = [v for row in rows for v in row]
                    cur.execute(
                        f"""
                        WITH input(chunk_id, concept_id, role) AS (
                            VALUES {values_sql}
                        )
                        INSERT INTO chunk_concepts (chunk_id, concept_id, role)
                        SELECT i.chunk_id, i.concept_id, i.role
                        FROM input i
                        INNER JOIN concept_vocabulary cv ON cv.concept_id = i.concept_id
                        ON CONFLICT DO NOTHING
                        """,
                        flat,
                    )
            conn.commit()
    except Exception as e:
        print(f"[concepts_db] set_chunk_concepts({chunk_id!r}) error: {e}")
        raise


def get_chunks_concepts(chunk_ids: Sequence[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Bulk lookup: for each chunk_id, return its prerequisites and introduces lists.
    Returns: { chunk_id: { "prerequisites": [...], "introduces": [...] } }
    Used at retrieval-time by the Phase 5 pedagogical scorer (one query, no
    per-chunk roundtrips).
    """
    _ensure_tables()
    out: Dict[str, Dict[str, List[str]]] = {
        cid: {"prerequisites": [], "introduces": []} for cid in chunk_ids
    }
    if not chunk_ids:
        return out
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id, concept_id, role
                    FROM chunk_concepts
                    WHERE chunk_id = ANY(%s)
                    """,
                    (list(chunk_ids),),
                )
                for chunk_id, concept_id, role in cur.fetchall():
                    if role == ROLE_PREREQUISITE:
                        out[chunk_id]["prerequisites"].append(concept_id)
                    elif role == ROLE_INTRODUCES:
                        out[chunk_id]["introduces"].append(concept_id)
        return out
    except Exception as e:
        print(f"[concepts_db] get_chunks_concepts error: {e}")
        return out


# ── message_concepts CRUD ─────────────────────────────────────────────────────

def set_message_concepts(
    message_id: int,
    concepts: Iterable[str],
    source: str,
) -> None:
    """
    Tag one chat message with the concepts it touched. Used by Phase 4 at the
    moment the message is saved. source must be 'asked' or 'taught'.
    Concepts not in the vocabulary are silently skipped (JOIN filter).
    """
    if source not in (SOURCE_ASKED, SOURCE_TAUGHT):
        raise ValueError(f"source must be 'asked' or 'taught', got {source!r}")
    _ensure_tables()
    rows = [(message_id, cid, source) for cid in concepts]
    if not rows:
        return
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                values_sql = ", ".join(["(%s, %s, %s)"] * len(rows))
                flat = [v for row in rows for v in row]
                cur.execute(
                    f"""
                    WITH input(message_id, concept_id, source) AS (
                        VALUES {values_sql}
                    )
                    INSERT INTO message_concepts (message_id, concept_id, source)
                    SELECT i.message_id::bigint, i.concept_id, i.source
                    FROM input i
                    INNER JOIN concept_vocabulary cv ON cv.concept_id = i.concept_id
                    ON CONFLICT DO NOTHING
                    """,
                    flat,
                )
            conn.commit()
    except Exception as e:
        print(f"[concepts_db] set_message_concepts({message_id}) error: {e}")
        raise


def get_session_concepts(session_id: str) -> List[Dict]:
    """
    Return every concept mention in a session's history, with timestamp and
    source. Phase 4 (knowledge state builder) consumes this to construct the
    student's confidence map.

    Returns rows shaped like:
      { "concept_id": "for_loop",
        "source":     "asked" | "taught",
        "created_at": datetime,
        "message_id": 42 }
    """
    _ensure_tables()
    try:
        with _get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT mc.concept_id, mc.source, mc.message_id, ch.created_at
                    FROM message_concepts mc
                    JOIN chat_history ch ON ch.id = mc.message_id
                    WHERE ch.session_id = %s
                    ORDER BY ch.created_at ASC
                    """,
                    (session_id,),
                )
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"[concepts_db] get_session_concepts({session_id!r}) error: {e}")
        return []