# airflow/dags/professor_pipeline.py
#
# DAG: professor_pipeline
#
# What it does:
#   1. Senses new PDF/PPTX files in data/landing/
#   2. Runs the Spark pipeline  (Bronze → Silver → Gold)
#   3. Indexes the Gold table into ChromaDB
#
# Schedule: checks every 5 minutes for new files.
# If nothing new has arrived since the last successful run, it skips.
#
# How to trigger manually from the Airflow UI:
#   Airflow → DAGs → professor_pipeline → ▶ Trigger DAG

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor


# ── Default task settings ─────────────────────────────────────────────────────
# If a task fails it retries once after 1 minute before marking the run failed.

default_args = {
    "owner": "ai-professor",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="professor_pipeline",
    description="Watches landing/ for new course files and runs the full ingestion pipeline",
    default_args=default_args,
    # Start in the past so Airflow doesn't try to backfill old runs
    start_date=datetime(2024, 1, 1),
    # Check for new files every 5 minutes
    schedule_interval="*/5 * * * *",
    # Never run more than one instance of this DAG at the same time
    max_active_runs=1,
    # Don't backfill all the missed runs since start_date
    catchup=False,
    tags=["professor", "pipeline"],
) as dag:

    # ── Task 1: sense new files ───────────────────────────────────────────────
    # FileSensor checks whether at least one PDF or PPTX exists in landing/.
    # It polls every 30 seconds and gives up after 10 minutes (poke_timeout).
    # mode="reschedule" means it releases the worker slot between polls
    # instead of blocking it — much better for long-running sensors.
    #
    # Note: the Airflow container mounts ./data at /opt/app/data, so the
    # path below resolves correctly inside the container.

    wait_for_files = FileSensor(
        task_id="wait_for_files_in_landing",
        filepath="/opt/app/data/landing",
        # Matches any PDF or PPTX — Airflow FileSensor supports glob patterns
        fs_conn_id="fs_landing",
        poke_interval=30,           # check every 30 seconds
        timeout=60 * 10,            # give up after 10 minutes
        mode="reschedule",          # don't block a worker slot while waiting
        soft_fail=True,             # mark as SKIPPED (not FAILED) if timeout
    )

    # ── Task 2: run the Spark pipeline ───────────────────────────────────────
    # Runs run_pipeline.py inside the already-running Spark container via
    # `docker exec`. The Airflow container and the Spark container share the
    # same Docker socket through the volume mount added in docker-compose.yml.
    #
    # This produces the Bronze, Silver, and Gold Delta tables.

    run_spark_pipeline = BashOperator(
        task_id="run_spark_pipeline",
        bash_command=(
            "docker exec ai-professor-spark "
            "python /opt/app/pipeline/run_pipeline.py"
        ),
        # Give the pipeline up to 2 hours — large document sets take a while
        execution_timeout=timedelta(hours=2),
    )

    # ── Task 3: index into ChromaDB ───────────────────────────────────────────
    # Reads the Gold Delta table and upserts all chunks + vectors into Chroma.
    # Safe to re-run — upsert is idempotent on chunk_id.

    index_to_chroma = BashOperator(
        task_id="index_to_chroma",
        bash_command=(
            "docker exec ai-professor-spark "
            "python /opt/app/pipeline/index_to_chroma.py"
        ),
        execution_timeout=timedelta(hours=1),
    )

    # ── Pipeline order ────────────────────────────────────────────────────────
    # sense new files → run pipeline → index into chroma

    wait_for_files >> run_spark_pipeline >> index_to_chroma