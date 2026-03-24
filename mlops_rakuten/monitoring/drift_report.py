"""
Drift monitoring Rakuten — Evidently.

Colonnes : designation (text), prdtypecode (target)
Référence : data/interim/rakuten_train.csv (74k lignes)
Courant   : dernier batch dans data/uploads/ ou data/raw/rakuten/seeds/

MLflow : se rattache à l'experiment du train via mlflow_run_metadata.json
"""

import glob
import json
import logging
import os
from pathlib import Path
from loguru import logger

import mlflow
import pandas as pd

# Evidently — compatibilité versions 0.4.x et 0.5+
try:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import ColumnDistributionMetric
    
    from evidently.report import Report
except ImportError:
    from evidently.pipeline.column_mapping import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import ColumnDistributionMetric
    from evidently.report import Report


COLUMN_MAPPING = ColumnMapping(
    target="prdtypecode",
    text_features=["designation"],
)

FEATURE_COLS = ["designation", "prdtypecode"]


def get_latest_batch(uploads_dir: Path, seeds_dir: Path) -> Path:
    """
    Retourne le dernier batch disponible.
    Priorité : uploads/ (ingest récent) > seeds/ (batches de base)
    """
    uploads = sorted(glob.glob(str(uploads_dir / "rakuten_batch_*.csv")))
    if uploads:
        latest = Path(uploads[-1])
        logger.info("Batch trouvé dans uploads/ : %s", latest.name)
        return latest

    seeds = sorted(glob.glob(str(seeds_dir / "rakuten_batch_*.csv")))
    if seeds:
        latest = Path(seeds[-1])
        logger.info("Batch trouvé dans seeds/ : %s", latest.name)
        return latest

    raise FileNotFoundError(
        f"Aucun batch trouvé dans {uploads_dir} ni {seeds_dir}"
    )

def load_category_mapping(categories_path: Path) -> dict:
    """Lit data/raw/product_categories.csv → {code: nom}"""
    if not categories_path.exists():
        logger.warning(f"product_categories.csv introuvable : {categories_path}")
        return {}
    df = pd.read_csv(categories_path)
    return dict(zip(df["prdtypecode"], df["category_name"]))

def load_run_metadata(models_dir: Path) -> dict:
    """
    Lit mlflow_run_metadata.json généré par ModelTrainer.
    Contient : run_id, model_version, experiment

    Retourne {} si le fichier n'existe pas.
    """
    metadata_path = models_dir / "mlflow_run_metadata.json"
    if not metadata_path.exists():
        logger.warning("mlflow_run_metadata.json introuvable : %s", metadata_path)
        return {}
    with open(metadata_path) as f:
        data = json.load(f)
    logger.info(
        "Run metadata chargé — run_id=%s, model_version=%s, experiment=%s",
        data.get("run_id"), data.get("model_version"), data.get("experiment"),
    )
    return data


def load_csv(path: Path, category_mapping: dict = {}) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[cols].dropna(subset=["designation"])
    if "prdtypecode" in df.columns and category_mapping:
        df["prdtypecode"] = (
            df["prdtypecode"]
            .map(category_mapping)
            .fillna(df["prdtypecode"].astype(str))
        )
    return df


def run_drift_report(
    reference_path: Path,
    uploads_dir: Path,
    seeds_dir: Path,
    report_output_path: Path,
    models_dir: Path | None = None,
    mlflow_tracking_uri: str | None = None,
) -> dict:
    """
    Génère le rapport Evidently et log les métriques dans MLflow.

    Si models_dir est fourni, lit mlflow_run_metadata.json pour :
    - se rattacher au même experiment que le train
    - taguer le run drift avec run_id et model_version du train

    Returns:
        dict avec drift_share, drift_count, dataset_drift, current_batch
    """
    current_path = get_latest_batch(uploads_dir, seeds_dir)

    categories_path = Path("/app/data/raw/product_categories.csv")
    category_mapping = load_category_mapping(categories_path)

    reference = load_csv(reference_path, category_mapping)
    logger.info(f"Référence : {reference_path.name}")

    current = load_csv(current_path, category_mapping)
    logger.info(f"Courant : {current_path.name}")

    logger.info(
        f"Comparaison : {len(reference)} lignes référence vs {len(current)} lignes courant"
    )

    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        ColumnDistributionMetric(column_name="prdtypecode"),
    ])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=COLUMN_MAPPING,
    )

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(report_output_path))
    logger.info("Rapport HTML sauvegardé : %s", report_output_path)

    metrics = _extract_metrics(report.as_dict())
    metrics["current_batch"] = current_path.name
    logger.info("Métriques drift : %s", metrics)

    # -------------------------------------------------------------------------
    # Log MLflow — même experiment que le train, lié au run_id du train
    # -------------------------------------------------------------------------
    if mlflow_tracking_uri:
        try:
            # Credentials DagsHub — même pattern que ModelTrainer
            dagshub_token = os.getenv("DAGSHUB_TOKEN")
            mlflow_user   = os.getenv("MLFLOW_TRACKING_USERNAME")
            if dagshub_token and mlflow_user:
                os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_user
                os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

            mlflow.set_tracking_uri(mlflow_tracking_uri)

            # Récupère les infos du dernier train
            run_metadata  = load_run_metadata(models_dir) if models_dir else {}
            experiment_name = run_metadata.get("experiment", "train_rakuten_model_mlflow")
            train_run_id    = run_metadata.get("run_id")
            model_version   = run_metadata.get("model_version")

            # Même experiment que le train → les runs drift apparaissent
            # dans la même vue MLflow que les runs d'entraînement
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="drift_monitoring"):
                # Métriques scalaires drift
                mlflow.log_metrics({
                    k: v for k, v in metrics.items()
                    if isinstance(v, (int, float))
                })
                # Tags de traçabilité
                mlflow.set_tag("run_type",     "drift_monitoring")
                mlflow.set_tag("current_batch", current_path.name)
                if train_run_id:
                    mlflow.set_tag("train_run_id",  train_run_id)
                if model_version:
                    mlflow.set_tag("model_version", model_version)

                # Rapport HTML Evidently comme artifact
                mlflow.log_artifact(
                    str(report_output_path),
                    artifact_path="drift_reports",
                )

            logger.info(
                "Drift loggé dans MLflow — experiment=%s, lié au train run_id=%s",
                experiment_name, train_run_id,
            )

        except Exception as mlflow_err:
            # Non bloquant — rapport HTML déjà sauvegardé
            logger.warning("MLflow log échoué (non bloquant) : %s", mlflow_err)

    return metrics


def _extract_metrics(report_dict: dict) -> dict:
    metrics = {}
    metric_list = report_dict.get("metrics", [])
    
    if not metric_list:
        return metrics
    
    # Metric 0 = DataDriftPreset — c'est celle qui a les bonnes valeurs dataset
    result = metric_list[0].get("result", {})
    
    metrics["drift_share"]  = result.get("share_of_drifted_columns", 0.0)
    metrics["drift_count"]  = float(result.get("number_of_drifted_columns", 0))
    metrics["dataset_drift"] = float(result.get("dataset_drift", 0))
    
    return metrics