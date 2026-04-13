# pipeline/index_to_chroma.py
#
# ChromaDB indexer: reads the Gold Delta table and upserts every chunk
# (text + embedding + metadata) into the "ai_professor" Chroma collection.
#
# Run this AFTER run_pipeline.py:
#
#   docker compose exec spark python /opt/app/pipeline/index_to_chroma.py
#
# Safe to re-run — Chroma upsert is idempotent on chunk_id.

import os
import sys

# ── Pull config from env / config.py ─────────────────────────────────────────
sys.path.insert(0, "/opt/app/pipeline")

from config import (  # noqa: E402
    GOLD_PATH,
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_COLLECTION,
    EMBED_MODEL,
)

BATCH_SIZE = int(os.getenv("CHROMA_BATCH_SIZE", "100"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_chroma_collection():
    import chromadb                          # noqa: PLC0415
    from chromadb.config import Settings    # noqa: PLC0415

    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    # get_or_create so re-runs don't wipe existing data
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def pull_model_if_needed():
    """
    Makes sure nomic-embed-text is pulled in Ollama before we start.
    Ollama will no-op if the model already exists.
    """
    import httpx  # noqa: PLC0415
    host = os.getenv("OLLAMA_HOST", "ollama")
    port = os.getenv("OLLAMA_PORT", "11434")
    try:
        print(f"Ensuring {EMBED_MODEL} is available in Ollama...")
        resp = httpx.post(
            f"http://{host}:{port}/api/pull",
            json={"name": EMBED_MODEL, "stream": False},
            timeout=300,   # pulling can take a while the first time
        )
        resp.raise_for_status()
        print(f"{EMBED_MODEL} ready.")
    except Exception as e:
        print(f"Warning: could not pull {EMBED_MODEL}: {e}")


def read_gold_table():
    """
    Read the Gold Delta table without spinning up a full Spark session.
    Uses delta-rs (pip install deltalake) for a lightweight read.
    """
    try:
        from deltalake import DeltaTable  # noqa: PLC0415
        dt = DeltaTable(GOLD_PATH)
        return dt.to_pandas()
    except ImportError:
        raise ImportError(
            "deltalake is required for the indexer: "
            "pip install deltalake"
        )


def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pull_model_if_needed()

    print(f"Reading Gold table from {GOLD_PATH}...")
    df = read_gold_table()
    print(f"  {len(df)} chunks loaded.")

    if df.empty:
        print("Gold table is empty — run run_pipeline.py first.")
        return

    collection = get_chroma_collection()
    print(f"Upserting into Chroma collection '{CHROMA_COLLECTION}'...")

    total = 0
    for batch_df in batched([df.iloc[i] for i in range(len(df))], BATCH_SIZE):
        ids         = [str(row["chunk_id"]) for row in batch_df]
        documents   = [str(row["chunk_text"]) for row in batch_df]
        embeddings  = [row["embedding"].tolist()
                       if hasattr(row["embedding"], "tolist")
                       else list(row["embedding"])
                       for row in batch_df]
        metadatas   = [
            {
                "filename":  str(row.get("filename",  "")),
                "professor": str(row.get("professor", "")),
                "subject":   str(row.get("subject",   "")),
                "chapter":   str(row.get("chapter",   "")),
                "language":  str(row.get("language",  "")),
                "doc_type":  str(row.get("doc_type",  "")),
            }
            for row in batch_df
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total += len(ids)
        print(f"  Upserted {total}/{len(df)} chunks...")

    print(f"\nIndexing complete — {total} chunks in '{CHROMA_COLLECTION}'.")


if __name__ == "__main__":
    main()