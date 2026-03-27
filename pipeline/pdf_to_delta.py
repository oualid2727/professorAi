import os
from pathlib import Path

from pyspark.sql import SparkSession, functions as F


LANDING_DIR = "/opt/app/data"
BRONZE_PATH = "/opt/app/data/delta/bronze"
SILVER_PATH = "/opt/app/data/delta/silver"
GOLD_PATH = "/opt/app/data/delta/gold"


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
        .getOrCreate()
    )


def read_raw_files(spark: SparkSession):
    files = []
    for root, _, filenames in os.walk(LANDING_DIR):
        for name in filenames:
            if name.lower().endswith((".pdf", ".pptx")):
                files.append(os.path.join(root, name))
    rows = [(str(Path(f).name), f) for f in files]
    return spark.createDataFrame(rows, ["filename", "path"])


def extract_text(df):
    """
    Very simple text extraction hook.
    In production, call `unstructured` or `pypdf` via pandas UDF.
    """
    extract_udf = F.udf(lambda _: "TEXT_EXTRACTION_NOT_IMPLEMENTED", "string")
    return df.withColumn("raw_text", extract_udf(F.col("path")))


def enrich_to_silver(df):
    return (
        df.withColumn("professor", F.lit("Unknown"))
        .withColumn("subject", F.lit("Unknown"))
        .withColumn("chapter", F.lit("Unknown"))
        .withColumn("language", F.lit("en"))
    )


def chunk_and_embed(df):
    """
    Placeholder for semantic chunking + embedding.
    Here we just create naive chunks and a dummy embedding.
    """
    return (
        df.withColumn("chunk_id", F.monotonically_increasing_id())
        .withColumn("chunk_text", F.col("raw_text"))
        .withColumn("embedding", F.array(F.lit(0.0)))
    )


def main():
    spark = create_spark()

    raw_df = read_raw_files(spark)
    if raw_df.rdd.isEmpty():
        print("No files found in landing directory.")
        return

    bronze_df = extract_text(raw_df)
    bronze_df.write.format("delta").mode("overwrite").save(BRONZE_PATH)

    silver_df = enrich_to_silver(bronze_df)
    silver_df.write.format("delta").mode("overwrite").save(SILVER_PATH)

    gold_df = chunk_and_embed(silver_df)
    gold_df.write.format("delta").mode("overwrite").save(GOLD_PATH)

    print("Pipeline completed: bronze, silver, gold tables written.")


if __name__ == "__main__":
    main()

