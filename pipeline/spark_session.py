# pipeline/spark_session.py

from pyspark.sql import SparkSession


def create_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("AIProfessorPDFIngestion")
        .config(
            "spark.jars.packages",
            "io.delta:delta-spark_2.12:3.2.0",
        )
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Ship all pipeline modules to every executor
        .config("spark.submit.pyFiles",
                "/opt/app/pipeline/extractor.py,"
                "/opt/app/pipeline/metadata.py,"
                "/opt/app/pipeline/config.py,"
                "/opt/app/pipeline/bronze.py,"
                "/opt/app/pipeline/silver.py,"
                "/opt/app/pipeline/gold.py")
        .getOrCreate()
    )