"""
DAG Rakuten Drift Monitoring — indépendant du pipeline principal.

Lance uniquement le rapport Evidently sur le dernier batch disponible.
Permet de monitorer la dérive des données sans re-entraîner le modèle.

Schedule par défaut : toutes les 6h
Peut aussi être déclenché manuellement depuis l'UI Airflow.

Différence avec rakuten_ml_pipeline :
  - Pas d'ingest, pas de train, pas de validate
  - Tourne plus fréquemment que le pipeline complet
  - Utile pour détecter une dérive rapide entre deux entraînements
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from dag_utils import resolve_project_root

# ===========================================================================
# Résolution du chemin hôte
# ===========================================================================

PROJECT_ROOT = resolve_project_root()

if not PROJECT_ROOT:
    raise RuntimeError(
        "DAG drift : impossible de résoudre PROJECT_ROOT via le socket Docker. "
        "Vérifier que /var/run/docker.sock est monté dans le container Airflow "
        "et que ./deployments/airflow/dags est bien monté sur /opt/airflow/dags."
    )

# ===========================================================================
# Config
# ===========================================================================

DEFAULT_ARGS = {
    "owner": "Rakuten MLOps Team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

DOCKER_NETWORK = "nov25cmlops_rakuten_rakuten-net"

SHARED_ENV = {
    "EXECUTION_MODE":   "cli",
    "DAGSHUB_USER":     os.environ.get("DAGSHUB_USER", ""),
    "DAGSHUB_REPO":     os.environ.get("DAGSHUB_REPO", ""),
    "DAGSHUB_TOKEN":    os.environ.get("DAGSHUB_TOKEN", ""),
    "GIT_AUTHOR_NAME":  os.environ.get("GIT_AUTHOR_NAME", ""),
    "GIT_AUTHOR_EMAIL": os.environ.get("GIT_AUTHOR_EMAIL", ""),
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

APP_MOUNT = Mount(source=PROJECT_ROOT, target="/app", type="bind")

# Pas de SSH_MOUNT — ce DAG ne fait pas de git push

# ===========================================================================
# DAG
# ===========================================================================

with DAG(
    dag_id="rakuten_drift_monitoring",
    description="Drift monitoring Rakuten — Evidently (indépendant du pipeline)",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 */6 * * *",  # toutes les 6h
    catchup=False,
    tags=["rakuten", "monitoring", "evidently", "drift"],
) as dag:

    drift_monitor = DockerOperator(
        task_id="drift_monitor",
        image="nov25cmlops_rakuten-api-ingest:latest",
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
            ),
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        environment={**SHARED_ENV, **MLFLOW_ENV},
        mounts=SHARED_MOUNTS + [APP_MOUNT],
        mount_tmp_dir=False,
        auto_remove="success",
        tty=False,
    )

    drift_monitor