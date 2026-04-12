# pipeline/bronze.py
#
# Bronze layer: raw ingestion.
# Responsibilities:
#   1. Walk the landing directory and collect all PDF/PPTX paths.
#   2. Apply the extractor UDF to produce one row per file with raw text.

import os
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import StringType


# ── UDF ───────────────────────────────────────────────────────────────────────

def _extract(path: str) -> str:
    from extractor import extract_text_from_file  # noqa: PLC0415
    return extract_text_from_file(path)


extract_udf = F.udf(_extract, StringType())


# ── Public API ────────────────────────────────────────────────────────────────

def read_landing(spark: SparkSession) -> DataFrame | None:
    from config import LANDING_DIR  # noqa: PLC0415
    files = []
    for root, _, filenames in os.walk(LANDING_DIR):
        for name in filenames:
            if name.lower().endswith((".pdf", ".pptx")):
                files.append(os.path.join(root, name))

    if not files:
        return None

    rows = [(Path(f).name, f) for f in files]
    return spark.createDataFrame(rows, ["filename", "path"])


def build_bronze(df: DataFrame) -> DataFrame:
    """
    Add a raw_text column by running the PDF/PPTX extractor on each file path.
    Output schema:
        filename, path, raw_text
    """
    return df.withColumn("raw_text", extract_udf(F.col("path")))