# pipeline/run_pipeline.py
#
# Orchestrator: runs Bronze → Silver → Gold in sequence.
# This is the only file you need to execute:
#
#   docker compose exec spark python /opt/app/pipeline/run_pipeline.py

from spark_session import create_spark
from bronze import read_landing, build_bronze
from silver import build_silver
from gold import build_gold, init_counter
from config import BRONZE_PATH, SILVER_PATH, GOLD_PATH


def _write(df, path: str, label: str) -> None:
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(path)
    )
    print(f"{label} written → {path}")


def main() -> None:
    spark = create_spark()

    # ── Bronze ────────────────────────────────────────────────────────────────
    raw_df = read_landing(spark)
    if raw_df is None or raw_df.rdd.isEmpty():
        print("No PDF/PPTX files found in landing directory.")
        return

    print(f"Found {raw_df.count()} file(s). Starting pipeline...\n")

    bronze_df = build_bronze(raw_df)
    _write(bronze_df, BRONZE_PATH, "Bronze")

    # ── Silver ────────────────────────────────────────────────────────────────
    silver_df = build_silver(bronze_df)
    _write(silver_df, SILVER_PATH, "Silver")

    # ── Gold ──────────────────────────────────────────────────────────────────
    # Init the progress counter before the slow embedding step
    init_counter(spark)
    print("\nStarting Gold layer (chunking + embedding)...")
    print("Progress prints every 50 chunks — this step calls Ollama once per chunk.\n")

    gold_df = build_gold(silver_df)
    _write(gold_df, GOLD_PATH, "Gold")

    total = gold_df.count()
    print(f"\n  Total chunks embedded: {total}")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()