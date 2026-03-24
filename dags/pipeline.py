"""
DAG Rakuten MLOps Pipeline
Orchestre ingest → train → validate → drift_monitor via DockerOperator.

Flow démo :
  1. Déposer manuellement un CSV dans data/uploads/
  2. Déclencher le DAG manuellement depuis l'UI Airflow
  3. ingest        → ingère tous les CSV présents dans data/uploads/
  4. train         → dvc repro + sync DagsHub/MLflow
  5. validate      → vérifie que le modèle @production est opérationnel
  6. drift_monitor → rapport Evidently référence vs dernier batch

En prod : remplacer le trigger manuel par un FileSensor sur data/uploads/
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

DOCKER_NETWORK = "nov25cmlops_rakuten_rakuten-net"

SHARED_ENV = {
    "EXECUTION_MODE": "cli",
    "DAGSHUB_USER":    os.environ.get("DAGSHUB_USER", ""),
    "DAGSHUB_REPO":    os.environ.get("DAGSHUB_REPO", ""),
    "DAGSHUB_TOKEN":   os.environ.get("DAGSHUB_TOKEN", ""),
    "GIT_AUTHOR_NAME": os.environ.get("GIT_AUTHOR_NAME", ""),
    "GIT_AUTHOR_EMAIL":os.environ.get("GIT_AUTHOR_EMAIL", ""),
    "GITHUB_USER":     os.environ.get("GITHUB_USER", ""),
}

MLFLOW_ENV = {
    "MLFLOW_TRACKING_URI":      os.environ.get("MLFLOW_TRACKING_URI", ""),
    "MLFLOW_TRACKING_USERNAME": os.environ.get("DAGSHUB_USER", ""),
    "MLFLOW_TRACKING_PASSWORD": os.environ.get("DAGSHUB_TOKEN", ""),
}

SHARED_MOUNTS = [
    Mount(source="rakuten_dvc_cache", target="/app/.dvc/cache", type="volume"),
    Mount(source="rakuten_models",    target="/app/models",     type="volume"),
    Mount(source="rakuten_logs",      target="/app/logs",       type="volume"),
    Mount(source="rakuten_reports",   target="/app/reports",    type="volume"),
]

APP_MOUNT = Mount(
    source="/home/shiff/datascientest/nov25cmlops_rakuten",
    target="/app",
    type="bind",
)

SSH_MOUNT = Mount(
    source="/home/shiff/.ssh",
    target="/root/.ssh",
    type="bind",
)

# ===========================================================================
# DAG
# ===========================================================================

with DAG(
    dag_id="rakuten_ml_pipeline",
    description="Pipeline ML Rakuten : ingest → train → validate → drift",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # déclenchement manuel pour la démo
    catchup=False,
    tags=["rakuten", "mlops", "docker"],
) as dag:

    # -----------------------------------------------------------------------
    # STEP 1 — Ingest
    # -----------------------------------------------------------------------
    ingest = DockerOperator(
        task_id="ingest",
        image="nov25cmlops_rakuten-api-ingest:latest",
        entrypoint="/entrypoint.sh",
        command=[
            "bash", "-c",
            "FILES=$(ls /app/data/uploads/*.csv 2>/dev/null) && "
            "[ -z \"$FILES\" ] && echo '[ingest] Aucun CSV trouvé — skip' && exit 0 || "
            "for f in $FILES; do "
            "  echo \"[ingest] Traitement de $f\"; "
            "  python -m mlops_rakuten.main ingest \"$f\" || exit 1; "
            "done"
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        environment=SHARED_ENV,
        mounts=SHARED_MOUNTS + [APP_MOUNT, SSH_MOUNT],
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
        entrypoint="/entrypoint.sh",
        command=["python", "-m", "mlops_rakuten.main", "train"],
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        environment={**SHARED_ENV, **MLFLOW_ENV},
        mounts=SHARED_MOUNTS + [APP_MOUNT, SSH_MOUNT],
        mount_tmp_dir=False,
        auto_remove="success",
        tty=False,
    )

    # -----------------------------------------------------------------------
    # STEP 3 — Validate
    # -----------------------------------------------------------------------
    validate = DockerOperator(
        task_id="validate",
        image="nov25cmlops_rakuten-api-predict:latest",
        entrypoint="/entrypoint.sh",
        command=[
            "python", "-m", "mlops_rakuten.main", "predict",
            "Vélo électrique pliable compact", "--top-k", "3"
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        environment={**SHARED_ENV, **MLFLOW_ENV},
        mounts=SHARED_MOUNTS + [APP_MOUNT],
        mount_tmp_dir=False,
        auto_remove="success",
        tty=False,
    )

    # -----------------------------------------------------------------------
    # STEP 4 — Drift Monitor
    # Non bloquant — tourne même si validate échoue
    # Lit mlflow_run_metadata.json pour se rattacher au bon run MLflow
    # -----------------------------------------------------------------------
    drift_monitor = DockerOperator(
        task_id="drift_monitor",
        image="nov25cmlops_rakuten-api-ingest:latest",  # même image = même code
        entrypoint="/entrypoint.sh",
        command=[
            "python", "-c",
            (
                "import os; from pathlib import Path; "
                "from mlops_rakuten.monitoring.drift_report import run_drift_report; "
                "run_drift_report("
                "    reference_path=Path('/app/data/interim/rakuten_train.csv'),"
                "    uploads_dir=Path('/app/data/uploads'),"
                "    seeds_dir=Path('/app/data/raw/rakuten/seeds'),"
                "    report_output_path=Path('/app/reports/drift/drift_report.html'),"
                "    models_dir=Path('/app/models'),"
                "    mlflow_tracking_uri=os.getenv('MLFLOW_TRACKING_URI'),"
                ")"
            )
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        environment={**SHARED_ENV, **MLFLOW_ENV},
        mounts=SHARED_MOUNTS + [APP_MOUNT],
        mount_tmp_dir=False,
        auto_remove="success",
        tty=False,
        # Non bloquant — drift tourne même si validate échoue
        trigger_rule="all_done",
    )

    # -----------------------------------------------------------------------
    # Dépendances
    # -----------------------------------------------------------------------
    ingest >> train >> validate >> drift_monitor