# backend/api/analytics.py
#
# Analytics data layer for the instructor dashboard.
#
# Tables used:
#   chat_history    (existing) — all conversation turns
#   retrieval_log   (new)      — which chunks were retrieved per query
#
# New schema created on first use:
#   retrieval_log (
#       id          SERIAL PRIMARY KEY,
#       session_id  TEXT        NOT NULL,
#       query       TEXT        NOT NULL,   -- the student's question
#       chunk_id    TEXT        NOT NULL,   -- chunk identifier from ChromaDB
#       chunk_text  TEXT        NOT NULL,   -- first 300 chars of chunk
#       filename    TEXT,                   -- source document
#       subject     TEXT,                   -- course subject
#       chapter     TEXT,
#       score       FLOAT,                  -- relevance score (0-1)
#       retrieved_at TIMESTAMPTZ DEFAULT NOW()
#   )

import os
from typing import List, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://ai_prof:ai_prof_pw@postgres:5432/ai_prof_db",
)

_tables_ready = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_conn():
    dsn = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")
    return psycopg2.connect(dsn)


def _ensure_tables() -> None:
    global _tables_ready
    if _tables_ready:
        return
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS retrieval_log (
                        id           SERIAL PRIMARY KEY,
                        session_id   TEXT        NOT NULL,
                        query        TEXT        NOT NULL,
                        chunk_id     TEXT        NOT NULL,
                        chunk_text   TEXT        NOT NULL,
                        filename     TEXT        DEFAULT '',
                        subject      TEXT        DEFAULT '',
                        chapter      TEXT        DEFAULT '',
                        score        FLOAT       DEFAULT 0.0,
                        retrieved_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_retrieval_log_subject
                    ON retrieval_log (subject, retrieved_at)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_retrieval_log_chunk
                    ON retrieval_log (chunk_id)
                """)
            conn.commit()
        _tables_ready = True
        print("[analytics] retrieval_log table ready")
    except Exception as e:
        print(f"[analytics] _ensure_tables error: {e}")
        raise


# ── Write ─────────────────────────────────────────────────────────────────────

def log_retrieval(session_id: str, query: str, sources: List[Dict]) -> None:
    """
    Persist which chunks were retrieved for a query.
    Called once per WebSocket message after hybrid retrieval completes.
    Silently swallows errors so a DB hiccup never kills the response.
    """
    if not sources:
        return
    try:
        _ensure_tables()
        with _get_conn() as conn:
            with conn.cursor() as cur:
                for src in sources:
                    cur.execute(
                        """
                        INSERT INTO retrieval_log
                            (session_id, query, chunk_id, chunk_text,
                             filename, subject, chapter, score)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            session_id,
                            query[:500],
                            str(src.get("chunk_id", "")),
                            str(src.get("chunk_text", src.get("page_content", "")))[:300],
                            str(src.get("filename",  "")),
                            str(src.get("subject",   "")),
                            str(src.get("chapter",   "")),
                            float(src.get("score", 0.0)),
                        ),
                    )
            conn.commit()
    except Exception as e:
        print(f"[analytics] log_retrieval error: {e}")


# ── Read — dashboard queries ──────────────────────────────────────────────────

def get_confusion_heatmap(days: int = 7) -> List[Dict]:
    """
    Topic confusion heatmap.
    Returns subjects where students asked questions that got a fallback response
    ('not covered', 'not in the context', etc.) — grouped by subject and day.
    """
    try:
        _ensure_tables()
        with _get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        DATE(u.created_at)                          AS day,
                        COALESCE(r.subject, 'Unknown')              AS subject,
                        COUNT(*)                                    AS confusion_count
                    FROM chat_history u
                    JOIN chat_history a
                        ON  a.session_id = u.session_id
                        AND a.role       = 'assistant'
                        AND a.created_at > u.created_at
                        AND a.created_at < u.created_at + INTERVAL '2 minutes'
                    LEFT JOIN retrieval_log r
                        ON  r.session_id = u.session_id
                        AND r.query      = u.content
                    WHERE u.role      = 'user'
                      AND u.created_at > NOW() - INTERVAL '%s days'
                      AND (
                            a.content ILIKE '%%not covered%%'
                         OR a.content ILIKE '%%haven%%t covered%%'
                         OR a.content ILIKE '%%not in the context%%'
                         OR a.content ILIKE '%%pas abordé%%'
                         OR a.content ILIKE '%%pas couvert%%'
                         OR a.content ILIKE '%%je ne sais pas%%'
                      )
                    GROUP BY day, subject
                    ORDER BY day DESC, confusion_count DESC
                    """,
                    (days,),
                )
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"[analytics] get_confusion_heatmap error: {e}")
        return []


def get_unanswered_questions(days: int = 7, limit: int = 50) -> List[Dict]:
    """
    Questions the professor couldn't answer from course material.
    Detects fallback responses and returns the original student questions.
    """
    try:
        _ensure_tables()
        with _get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        u.content                                   AS question,
                        u.created_at                                AS asked_at,
                        u.session_id,
                        COALESCE(r.subject, 'Unknown')              AS subject,
                        a.content                                   AS professor_response
                    FROM chat_history u
                    JOIN chat_history a
                        ON  a.session_id = u.session_id
                        AND a.role       = 'assistant'
                        AND a.created_at > u.created_at
                        AND a.created_at < u.created_at + INTERVAL '2 minutes'
                    LEFT JOIN retrieval_log r
                        ON  r.session_id = u.session_id
                        AND r.query      = u.content
                    WHERE u.role      = 'user'
                      AND u.created_at > NOW() - INTERVAL '%s days'
                      AND (
                            a.content ILIKE '%%not covered%%'
                         OR a.content ILIKE '%%haven%%t covered%%'
                         OR a.content ILIKE '%%not in the context%%'
                         OR a.content ILIKE '%%pas abordé%%'
                         OR a.content ILIKE '%%pas couvert%%'
                         OR a.content ILIKE '%%je ne sais pas%%'
                      )
                    ORDER BY u.created_at DESC
                    LIMIT %s
                    """,
                    (days, limit),
                )
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"[analytics] get_unanswered_questions error: {e}")
        return []


def get_top_chunks(days: int = 7, limit: int = 20) -> List[Dict]:
    """
    Most-retrieved document chunks — shows which parts of the course
    students are engaging with the most.
    """
    try:
        _ensure_tables()
        with _get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        chunk_id,
                        chunk_text,
                        filename,
                        subject,
                        chapter,
                        COUNT(*)                    AS retrieval_count,
                        MAX(retrieved_at)           AS last_retrieved
                    FROM retrieval_log
                    WHERE retrieved_at > NOW() - INTERVAL '%s days'
                    GROUP BY chunk_id, chunk_text, filename, subject, chapter
                    ORDER BY retrieval_count DESC
                    LIMIT %s
                    """,
                    (days, limit),
                )
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"[analytics] get_top_chunks error: {e}")
        return []


def get_activity_summary(days: int = 7) -> Dict[str, Any]:
    """
    High-level activity numbers for the dashboard header.
    """
    try:
        _ensure_tables()
        with _get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE role = 'user')       AS total_questions,
                        COUNT(DISTINCT session_id)                  AS total_sessions,
                        COUNT(*) FILTER (
                            WHERE role = 'user'
                              AND created_at > NOW() - INTERVAL '24 hours'
                        )                                           AS questions_today
                    FROM chat_history
                    WHERE created_at > NOW() - INTERVAL '%s days'
                    """,
                    (days,),
                )
                row = dict(cur.fetchone() or {})

                # Confusion rate
                cur.execute(
                    """
                    SELECT COUNT(*) AS unanswered
                    FROM chat_history u
                    JOIN chat_history a
                        ON  a.session_id = u.session_id
                        AND a.role       = 'assistant'
                        AND a.created_at > u.created_at
                        AND a.created_at < u.created_at + INTERVAL '2 minutes'
                    WHERE u.role      = 'user'
                      AND u.created_at > NOW() - INTERVAL '%s days'
                      AND (
                            a.content ILIKE '%%not covered%%'
                         OR a.content ILIKE '%%haven%%t covered%%'
                         OR a.content ILIKE '%%pas abordé%%'
                         OR a.content ILIKE '%%pas couvert%%'
                      )
                    """,
                    (days,),
                )
                unanswered = (cur.fetchone() or {}).get("unanswered", 0)
                row["unanswered_questions"] = unanswered
                total = row.get("total_questions", 1) or 1
                row["confusion_rate"] = round((unanswered / total) * 100, 1)
                return row
    except Exception as e:
        print(f"[analytics] get_activity_summary error: {e}")
        return {}