# pipeline/gold.py
#
# Gold layer: semantic chunking + real embeddings.
# Responsibilities:
#   1. Split each document's raw_text into semantically coherent chunks using
#      a recursive character splitter (respects sentence/paragraph boundaries).
#   2. Drop near-empty chunks (< 30 chars after trimming).
#   3. Attach a unique chunk_id and source metadata to each chunk.
#   4. Call Ollama nomic-embed-text to produce a real 768-dim embedding vector
#      for every chunk.

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import ArrayType, FloatType, StringType


# ── Chunker UDF ───────────────────────────────────────────────────────────────
# Runs on the driver+executors. Takes one document's raw_text and returns a
# JSON-encoded list of chunk strings. We use JSON so Spark can handle a
# variable-length list in a single UDF return value.

def _chunk(raw_text: str) -> str:
    """
    Recursive character splitter.
    Tries to split on paragraph breaks first, then sentences, then words,
    falling back to hard character cuts only when necessary.
    Keeps CHUNK_OVERLAP characters of context between consecutive chunks.
    """
    import json   # noqa: PLC0415
    import os     # noqa: PLC0415

    chunk_size    = int(os.getenv("CHUNK_SIZE",    "2000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Priority-ordered separators: paragraph → sentence → word → character
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: list) -> list:
        if not seps:
            # Hard split — last resort
            return [text[i:i + chunk_size]
                    for i in range(0, len(text), chunk_size - chunk_overlap)]

        sep = seps[0]
        parts = text.split(sep) if sep else list(text)

        chunks, current = [], ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If the part itself is too large, recurse with finer separators
                if len(part) > chunk_size:
                    chunks.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(raw_text, separators)

    # Add overlap: prepend the tail of the previous chunk to the current one
    overlapped = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and chunk_overlap > 0:
            prev_tail = raw_chunks[i - 1][-chunk_overlap:]
            chunk = prev_tail + " " + chunk
        overlapped.append(chunk.strip())

    return json.dumps([c for c in overlapped if len(c) > 30])


chunk_udf = F.udf(_chunk, StringType())


# ── Embedding UDF ─────────────────────────────────────────────────────────────
# Calls Ollama's /api/embeddings endpoint for one chunk at a time.
# Returns a list of floats (768 dims for nomic-embed-text).
# Failures return a zero vector so the pipeline doesn't crash on one bad chunk.

# Shared accumulator — incremented by each executor each time a chunk is embedded.
# The driver prints progress every PROGRESS_INTERVAL chunks.
_embed_counter = None
PROGRESS_INTERVAL = 50


def _embed(chunk_text: str) -> list:
    import os       # noqa: PLC0415
    import httpx    # noqa: PLC0415

    host  = os.getenv("OLLAMA_HOST",  "ollama")
    port  = os.getenv("OLLAMA_PORT",  "11434")
    model = os.getenv("EMBED_MODEL",  "nomic-embed-text")

    try:
        resp = httpx.post(
            f"http://{host}:{port}/api/embeddings",
            json={"model": model, "prompt": chunk_text},
            timeout=60,
        )
        resp.raise_for_status()
        result = [float(x) for x in resp.json()["embedding"]]

        # Increment the shared counter if available
        if _embed_counter is not None:
            _embed_counter.add(1)
            current = _embed_counter.value
            if current % PROGRESS_INTERVAL == 0:
                print(f"  Embedded {current} chunks...", flush=True)

        return result
    except Exception:
        return [0.0] * 768


embed_udf = F.udf(_embed, ArrayType(FloatType()))


def init_counter(spark) -> None:
    """Call this from run_pipeline.py before build_gold() to enable progress printing."""
    global _embed_counter
    _embed_counter = spark.sparkContext.accumulator(0)


# ── Public API ────────────────────────────────────────────────────────────────

def build_gold(df: DataFrame) -> DataFrame:
    """
    Transforms the silver DataFrame into one row per semantic chunk with
    a real embedding vector.

    Output schema adds to silver:
        chunk_text  (str)          — the chunk content
        chunk_id    (bigint)       — unique monotonic ID
        embedding   (array<float>) — 768-dim nomic-embed-text vector
    """
    return (
        df
        # 1. Chunk each document → JSON list of strings stored in _chunks
        .withColumn("_chunks", chunk_udf(F.col("raw_text")))

        # 2. Parse the JSON array and explode into one row per chunk
        .withColumn("chunk_text", F.explode(F.from_json(
            F.col("_chunks"),
            ArrayType(StringType())
        )))
        .drop("_chunks")

        # 3. Final length guard after overlap prepending
        .filter(F.length(F.trim(F.col("chunk_text"))) > 30)

        # 4. Stable unique ID
        .withColumn("chunk_id", F.monotonically_increasing_id())

        # 5. Real embeddings — this is the slow step, one HTTP call per chunk
        .withColumn("embedding", embed_udf(F.col("chunk_text")))
    )