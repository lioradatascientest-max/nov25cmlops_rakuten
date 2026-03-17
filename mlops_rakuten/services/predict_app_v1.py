from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, status

from mlops_rakuten.config.constants import MODELS_DIR, PROCESSED_DATA_DIR
from mlops_rakuten.pipelines.prediction import PredictionPipeline
from mlops_rakuten.services.schemas import (
    CategoryScore,
    PredictionRequest,
    PredictionResponse,
)


app = FastAPI(title="Rakuten Predict API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    pipe = PredictionPipeline()
    results_per_text = pipe.run(
        texts=[payload.designation], top_k=payload.top_k)
    preds_raw = results_per_text[0]

    preds = [
        CategoryScore(
            prdtypecode=p["prdtypecode"],
            category_name=p.get("category_name"),
            proba=p["proba"],
        )
        for p in preds_raw
    ]

    return PredictionResponse(designation=payload.designation, predictions=preds)


@app.get("/info")
def model_info() -> Dict[str, Any]:
    """Affiche les infos du modèle en production"""
    pipe = PredictionPipeline()
    info = pipe.get_model_info()
    
    if info:
        return {"status": "ok", "model": info}
    else:
        return {"status": "error", "message": "Pas de modèle en production"}