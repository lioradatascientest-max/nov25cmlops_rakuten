"""
DAG Rakuten MLOps Pipeline
Orchestre ingest → train → predict via DockerOperator (socket Docker).
"""

from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# ===========================================================================
# Config partagée
# ===========================================================================

DEFAULT_ARGS = {
    "owner": "Rakuten MLOps Team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

DOCKER_NETWORK = "rakuten-net"

# Variables injectées via env_file: .env dans le service airflow du docker-compose
SHARED_ENV = {
    "EXECUTION_MODE": "docker",
    "DAGSHUB_USER": os.environ.get("DAGSHUB_USER", ""),
    "DAGSHUB_REPO": os.environ.get("DAGSHUB_REPO", ""),
    "DAGSHUB_TOKEN": os.environ.get("DAGSHUB_TOKEN", ""),
    "GIT_AUTHOR_NAME": os.environ.get("GIT_AUTHOR_NAME", ""),
    "GIT_AUTHOR_EMAIL": os.environ.get("GIT_AUTHOR_EMAIL", ""),
}

MLFLOW_ENV = {
    "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
    "MLFLOW_TRACKING_USERNAME": os.environ.get("DAGSHUB_USER", ""),
    "MLFLOW_TRACKING_PASSWORD": os.environ.get("DAGSHUB_TOKEN", ""),
}

# Volumes nommés Docker du docker-compose
SHARED_MOUNTS = [
    Mount(source="rakuten_dvc_cache", target="/app/.dvc/cache", type="volume"),
    Mount(source="rakuten_models",    target="/app/models",     type="volume"),
    Mount(source="rakuten_logs",      target="/app/logs",       type="volume"),
    Mount(source="rakuten_reports",   target="/app/reports",    type="volume"),
]

SSH_MOUNT = Mount(
    source="/root/.ssh",
    target="/root/.ssh",
    type="bind",
)

# ===========================================================================
# DAG
# ===========================================================================

with DAG(
    dag_id="rakuten_ml_pipeline",
    description="Pipeline ML Rakuten : ingest → train → predict",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@weekly",  # None pour déclencher manuellement
    catchup=False,
    tags=["rakuten", "mlops", "docker"],
) as dag:

    # -----------------------------------------------------------------------
    # STEP 1 — Ingest
    # -----------------------------------------------------------------------
    ingest = DockerOperator(
        task_id="ingest",
        image="nov25cmlops_rakuten-api-ingest:latest",
        command=[
            "uvicorn",
            "mlops_rakuten.services.ingest_app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        environment=SHARED_ENV,
        mounts=SHARED_MOUNTS,
        mount_tmp_dir=False,
        auto_remove="success",
        tty=False,
    )

    # -----------------------------------------------------------------------
    # STEP 2 — Train
    # -----------------------------------------------------------------------
    train = DockerOperator(
        task_id="train",
        image="nov25cmlops_rakuten-api-train:latest",
        command=[
            "uvicorn",
            "mlops_rakuten.services.train_app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        environment={
            **SHARED_ENV,
            **MLFLOW_ENV,
            "GIT_SSH_COMMAND": "ssh -i /root/.ssh/id_github -o StrictHostKeyChecking=no -o IdentitiesOnly=yes",
        },
        mounts=SHARED_MOUNTS + [SSH_MOUNT],
        mount_tmp_dir=False,
        auto_remove="success",
        tty=False,
    )

    # -----------------------------------------------------------------------
    # STEP 3 — Predict
    # -----------------------------------------------------------------------
    predict = DockerOperator(
        task_id="predict",
        image="nov25cmlops_rakuten-api-predict:latest",
        command=[
            "uvicorn",
            "mlops_rakuten.services.predict_app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        environment={
            **SHARED_ENV,
            **MLFLOW_ENV,
        },
        mounts=SHARED_MOUNTS,
        mount_tmp_dir=False,
        auto_remove="success",
        tty=False,
    )

    # -----------------------------------------------------------------------
    # Dépendances
    # -----------------------------------------------------------------------
    ingest >> train >> predict