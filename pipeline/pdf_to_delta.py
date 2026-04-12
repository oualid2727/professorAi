# pipeline/pdf_to_delta.py

import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType


LANDING_DIR = "/opt/app/data/landing"
BRONZE_PATH = "/opt/app/data/delta/bronze"
SILVER_PATH = "/opt/app/data/delta/silver"
GOLD_PATH   = "/opt/app/data/delta/gold"


# ── Spark session ─────────────────────────────────────────────────────────────

def create_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("AIProfessorPDFIngestion")
        .config(
            "spark.jars.packages",
            "io.delta:delta-spark_2.12:3.2.0",
        )
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Ship the extractor module to every executor
        .config("spark.submit.pyFiles", "/opt/app/pipeline/extractor.py")
        .getOrCreate()
    )


# ── Bronze: raw text extraction ───────────────────────────────────────────────

def _extract(path: str) -> str:
    """
    Worker function that runs on each Spark executor.
    Imports extractor locally so it's available after pyFiles distribution.
    """
    from extractor import extract_text_from_file  # noqa: PLC0415
    return extract_text_from_file(path)


extract_udf = F.udf(_extract, StringType())


def read_raw_files(spark: SparkSession):
    files = []
    for root, _, filenames in os.walk(LANDING_DIR):
        for name in filenames:
            if name.lower().endswith((".pdf", ".pptx")):
                files.append(os.path.join(root, name))

    if not files:
        return None

    rows = [(str(Path(f).name), f) for f in files]
    return spark.createDataFrame(rows, ["filename", "path"])


def extract_text(df):
    """
    Bronze layer: apply the real PDF/PPTX extractor UDF to every file path.
    Each row now contains the full raw text of one document.
    """
    return df.withColumn("raw_text", extract_udf(F.col("path")))


# ── Silver: metadata enrichment (stubs — filled in next step) ─────────────────

def enrich_to_silver(df):
    """
    Adds metadata columns derived from filename conventions and text content.
    Replace the F.lit stubs here when you wire up real metadata extraction.

    Expected filename convention (optional):  ProfName_Subject_Chapter.pdf
    e.g.  Smith_LinearAlgebra_Ch3.pdf
    """
    # Split filename on underscores and pull named parts when available
    split_col = F.split(F.regexp_replace(F.col("filename"), r"\.[^.]+$", ""), "_")

    return (
        df
        .withColumn(
            "professor",
            F.when(F.size(split_col) >= 1, split_col.getItem(0)).otherwise(F.lit("Unknown")),
        )
        .withColumn(
            "subject",
            F.when(F.size(split_col) >= 2, split_col.getItem(1)).otherwise(F.lit("Unknown")),
        )
        .withColumn(
            "chapter",
            F.when(F.size(split_col) >= 3, split_col.getItem(2)).otherwise(F.lit("Unknown")),
        )
        # Language detection will be added in the metadata step
        .withColumn("language", F.lit("en"))
        # Word count is cheap and useful for filtering later
        .withColumn("word_count", F.size(F.split(F.col("raw_text"), r"\s+")))
    )


# ── Gold: chunking + dummy embedding (real embeddings added in step 3) ────────

def chunk_and_embed(df):
    """
    Naive fixed-size chunking by splitting on double newlines (slide/page breaks).
    Real semantic chunking and embeddings will replace this in the next step.
    """
    # Explode on page/slide boundaries that the extractor preserved
    chunks_df = (
        df
        .withColumn("chunks", F.split(F.col("raw_text"), r"\n\n"))
        .withColumn("chunk_text", F.explode(F.col("chunks")))
        .filter(F.length(F.trim(F.col("chunk_text"))) > 30)  # drop near-empty chunks
        .drop("chunks")
        .withColumn("chunk_id", F.monotonically_increasing_id())
        # Placeholder — replaced by real Ollama embeddings in step 3
        .withColumn("embedding", F.array(F.lit(0.0)))
    )
    return chunks_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    spark = create_spark()

    raw_df = read_raw_files(spark)
    if raw_df is None or raw_df.rdd.isEmpty():
        print("No PDF/PPTX files found in landing directory:", LANDING_DIR)
        return

    print(f"Found {raw_df.count()} file(s). Starting extraction...")

    bronze_df = extract_text(raw_df)
    bronze_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(BRONZE_PATH)
    print(f"Bronze written → {BRONZE_PATH}")

    silver_df = enrich_to_silver(bronze_df)
    silver_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(SILVER_PATH)
    print(f"Silver written → {SILVER_PATH}")

    gold_df = chunk_and_embed(silver_df)
    gold_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(GOLD_PATH)
    print(f"Gold written   → {GOLD_PATH}  ({gold_df.count()} chunks)")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()