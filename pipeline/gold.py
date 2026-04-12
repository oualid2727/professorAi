# pipeline/gold.py
#
# Gold layer: chunking and embeddings.
# Responsibilities:
#   1. Split each document's raw_text into chunks at page/slide boundaries.
#   2. Drop near-empty chunks (< 30 chars after trimming).
#   3. Attach a unique chunk_id per chunk.
#   4. Placeholder embedding column — replaced with real Ollama vectors in step 3.

from pyspark.sql import DataFrame, functions as F


def build_gold(df: DataFrame) -> DataFrame:
    """
    Explode the silver DataFrame into one row per text chunk.
    Output schema adds to silver:
        chunk_text, chunk_id, embedding
    (raw_text is kept so the full document is still accessible per chunk)
    """
    return (
        df
        # Split on double newlines — the extractor marks page/slide boundaries here
        .withColumn("chunks", F.split(F.col("raw_text"), r"\n\n"))
        .withColumn("chunk_text", F.explode(F.col("chunks")))
        .drop("chunks")

        # Drop near-empty chunks that add noise to retrieval
        .filter(F.length(F.trim(F.col("chunk_text"))) > 30)

        # Stable unique ID per chunk
        .withColumn("chunk_id", F.monotonically_increasing_id())

        # Placeholder — replaced by real nomic-embed-text vectors in step 3
        .withColumn("embedding", F.array(F.lit(0.0)))
    )