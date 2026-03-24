"""
api-monitor — Service FastAPI drift monitoring Rakuten.

Endpoints :
  GET  /health
  POST /drift          → lance le rapport Evidently
  GET  /drift/report   → retourne le HTML pour Streamlit
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from mlops_rakuten.monitoring.drift_report import run_drift_report

app = FastAPI(title="Rakuten Monitor API", version="1.0.0")

# ============================================================================
# Chemins (injectés via env ou valeurs par défaut)
# ============================================================================
REFERENCE_PATH = Path(os.getenv(
    "REFERENCE_CSV",
    "/app/data/interim/rakuten_train.csv"
))
UPLOADS_DIR = Path(os.getenv(
    "UPLOADS_DIR",
    "/app/data/uploads"
))
SEEDS_DIR = Path(os.getenv(
    "SEEDS_DIR",
    "/app/data/raw/rakuten/seeds"
))
REPORT_PATH = Path(os.getenv(
    "REPORT_PATH",
    "/app/reports/drift/drift_report.html"
))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")


# ============================================================================
# Schemas
# ============================================================================
class DriftResponse(BaseModel):
    drift_share: float
    drift_count: float
    dataset_drift: float
    current_batch: str
    report_url: str


# ============================================================================
# Endpoints
# ============================================================================
@app.get("/health")
def health():
    return {"status": "ok", "service": "api-monitor"}


@app.post("/drift", response_model=DriftResponse)
def run_drift():
    """
    Détecte automatiquement le dernier batch (uploads/ prioritaire sur seeds/)
    et compare contre rakuten_train.csv (référence).
    """
    if not REFERENCE_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Référence introuvable : {REFERENCE_PATH}"
        )

    try:
        metrics = run_drift_report(
            reference_path=REFERENCE_PATH,
            uploads_dir=UPLOADS_DIR,
            seeds_dir=SEEDS_DIR,
            report_output_path=REPORT_PATH,
            mlflow_tracking_uri=MLFLOW_URI,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return DriftResponse(
        drift_share=metrics.get("drift_share", 0.0),
        drift_count=metrics.get("drift_count", 0.0),
        dataset_drift=metrics.get("dataset_drift", 0.0),
        current_batch=metrics.get("current_batch", "unknown"),
        report_url="/drift/report",
    )


@app.get("/drift/report", response_class=HTMLResponse)
def get_report():
    """Retourne le rapport HTML Evidently — à afficher dans Streamlit."""
    if not REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Rapport non généré — lance POST /drift d'abord"
        )
    return HTMLResponse(content=REPORT_PATH.read_text(encoding="utf-8"))