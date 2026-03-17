from __future__ import annotations
from loguru import logger
from pathlib import Path
import shutil
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile, status, Query
from mlops_rakuten.config.constants import UPLOADS_DATA_DIR
from mlops_rakuten.pipelines.data_ingestion import DataIngestionPipeline
from mlops_rakuten.utils.utils import create_directories
import os

EXECUTION_MODE = os.getenv("EXECUTION_MODE", "cli")

if EXECUTION_MODE == "docker":
    from mlops_rakuten.utils.docker import _dvc, sync_ingest_data, sync_init
    SYNC_MODE = None  # docker_utils hardcode "Docker-DID"
    logger.info("Transport : docker exec (Docker-DID)")
else:
    from mlops_rakuten.utils.cli import _dvc, sync_ingest_data, sync_init
    SYNC_MODE = "Docker-CLI"  # subprocess dans le container API
    logger.info("Transport : subprocess (Docker-CLI)")

app = FastAPI(
    title="Rakuten Ingest API",
    version="1.0.0",
    description=f"Mode d'exécution actuel : **{EXECUTION_MODE}**",
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "execution_mode": EXECUTION_MODE}


@app.post("/init")
def init_dataset(
    force: bool = Query(False, description="Force la régénération complète", title="Force Rebuild")
) -> Dict[str, Any]:
    try:
        mode_label = "FORCE REBUILD" if force else "Normal Init"
        logger.info(f"{mode_label} - Initialisation du dataset rakuten")

        _dvc("dvc pull 2>&1 || true")
        _dvc("dvc repro seed --force" if force else "dvc repro seed")

        sync_results = sync_init(force=force, mode=SYNC_MODE)

        if not sync_results["summary"]["success"]:
            raise HTTPException(status_code=500, detail=f"Git+DVC sync failed: {sync_results['errors']}")

        return {
            "status": "initialisation_complete",
            "mode": "force_rebuild" if force else "normal",
            "message": "Seed CSV ingéré, tracké et synchro avec Git+DVC+DagsHub",
            "sync_details": sync_results["summary"],
            "pipeline_ready": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ingest")
async def ingest_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un .csv")

    create_directories([UPLOADS_DATA_DIR])
    uploads_path = Path(UPLOADS_DATA_DIR) / file.filename
    with uploads_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        ingested_dataset_path = DataIngestionPipeline().run(uploaded_csv_path=uploads_path)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Ingestion failed: {e}") from e

    try:
        _dvc("dvc repro preprocess")
        sync_results = sync_ingest_data(file.filename, mode=SYNC_MODE)

        if not sync_results["summary"]["success"]:
            raise HTTPException(status_code=500, detail=f"Git+DVC sync failed: {sync_results['errors']}")

        return {
            "status": "ingested_and_synced",
            "dataset_path": str(ingested_dataset_path),
            "filename": file.filename,
            "message": "Dataset ingéré, pipeline updaté, synchro Git+DVC ✓",
            "sync_details": sync_results["summary"],
            "train_ready": True,
            "next_step": "Ready for POST /train",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Git+DVC operation failed: {str(e)}") from e