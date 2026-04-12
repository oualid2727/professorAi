# pipeline/silver.py
#
# Silver layer: metadata enrichment.
# Responsibilities:
#   1. Parse professor / subject / chapter from the filename convention.
#   2. Fall back to Ollama for subject when the filename gives nothing.
#   3. Detect document language with langdetect.
#   4. Add cheap derived columns: word_count, doc_type.

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import StringType


# ── UDF ───────────────────────────────────────────────────────────────────────
# Returns a JSON string so all four metadata fields travel in one UDF call.
# F.get_json_object then unpacks each field — avoids four separate UDF calls.

def _enrich(filename: str, raw_text: str) -> str:
    import json                              # noqa: PLC0415
    import os                                # noqa: PLC0415
    from metadata import enrich_metadata    # noqa: PLC0415

    result = enrich_metadata(
        filename,
        raw_text,
        ollama_host=os.getenv("OLLAMA_HOST",  "ollama"),
        ollama_port=int(os.getenv("OLLAMA_PORT", "11434")),
        model=os.getenv("OLLAMA_MODEL", "llama3"),
    )
    return json.dumps(result)


enrich_udf = F.udf(_enrich, StringType())


# ── Public API ────────────────────────────────────────────────────────────────

def build_silver(df: DataFrame) -> DataFrame:
    """
    Enrich the bronze DataFrame with metadata columns.
    Output schema adds to bronze:
        professor, subject, chapter, language, word_count, doc_type
    """
    return (
        df
        .withColumn("_meta", enrich_udf(F.col("filename"), F.col("raw_text")))

        .withColumn("professor", F.get_json_object(F.col("_meta"), "$.professor"))
        .withColumn("subject",   F.get_json_object(F.col("_meta"), "$.subject"))
        .withColumn("chapter",   F.get_json_object(F.col("_meta"), "$.chapter"))
        .withColumn("language",  F.get_json_object(F.col("_meta"), "$.language"))

        .drop("_meta")

        .withColumn("word_count", F.size(F.split(F.col("raw_text"), r"\s+")))
        .withColumn(
            "doc_type",
            F.when(F.lower(F.col("filename")).endswith(F.lit(".pdf")),  F.lit("pdf"))
             .when(F.lower(F.col("filename")).endswith(F.lit(".pptx")), F.lit("pptx"))
             .otherwise(F.lit("unknown"))
        )
    )