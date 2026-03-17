from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from loguru import logger

from mlops_rakuten.pipelines.prediction import PredictionPipeline
from mlops_rakuten.services.schemas import (
    CategoryScore,
    PredictionRequest,
    PredictionResponse,
    ModelReloadResponse,
)


# State container
class AppState:
    pipeline: Optional[PredictionPipeline] = None


# Lifespan: charge au startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: charge le modèle au démarrage du container"""
    try:
        AppState.pipeline = PredictionPipeline()
        logger.success("Modèle prêt en mémoire - Container opérationnel")
    except Exception as e:
        logger.error(f"Erreur au startup: {e}")
        raise
    
    yield  #  Serveur tourne ici
    
    logger.info("Arrêt du container predict_app")
    AppState.pipeline = None


# FastAPI app avec lifespan
app = FastAPI(
    title="Rakuten Predict API",
    version="1.0.0",
    lifespan=lifespan
)


def get_pipeline() -> PredictionPipeline:
    """Récupère le pipeline depuis l'état de l'app"""
    if AppState.pipeline is None:
        raise RuntimeError("Pipeline non initialisé")
    return AppState.pipeline


@app.get("/health")
def health() -> Dict[str, str]:
    """Simple health check"""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """
    Prédiction avec modèle déjà chargé en mémoire.
    """
    pipeline = get_pipeline()
    info = pipeline.get_model_info()
    results_per_text = pipeline.run(
        texts=[payload.designation],
        top_k=payload.top_k
    )
    preds_raw = results_per_text[0]

    preds = [
        CategoryScore(
            prdtypecode=p["prdtypecode"],
            category_name=p.get("category_name"),
            proba=p["proba"],
        )
        for p in preds_raw
    ]

    return PredictionResponse(
        designation=payload.designation,
        predictions=preds,
        model_version=info.get("version", "unknown"),
        model_name=info.get("name", "unknown"),
    )


@app.get("/info")
def model_info() -> Dict[str, Any]:
    """
    Infos du modèle en production.
    
    Instantané car le modèle est déjà chargé.
    """
    pipeline = get_pipeline()
    info = pipeline.get_model_info()
    
    if info and info.get("status") != "error":
        return {
            "status": "ok",
            "model": info,
            "loaded_at_startup": True,
            "message": "Modèle chargé au démarrage du container"
        }
    else:
        return {
            "status": "error",
            "message": "Pas de modèle en production",
            "error_details": info.get("error") if info else None
        }


@app.post("/reload", response_model=ModelReloadResponse)
async def reload_model() -> ModelReloadResponse:
    """
    Recharge le modèle MLflow.
    
    Appelé par la gateway après /train ou manuellement si besoin de recharger sans faire un training complet.
    
    """
    try:
        start = time.time()
        logger.info("Rechargement du modèle depuis MLflow...")
        
        AppState.pipeline = PredictionPipeline()
        
        duration = time.time() - start
        
        logger.success(f"Modèle rechargé en {duration:.1f}s")
        
        return ModelReloadResponse(
            status="ok",
            message="Modèle rechargé avec succès",
            reload_time_seconds=duration
        )
    
    except Exception as e:
        logger.error(f"Erreur rechargement: {e}")
        raise HTTPException(status_code=500, detail=str(e))