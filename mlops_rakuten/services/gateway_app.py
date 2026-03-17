from __future__ import annotations

from typing import Any, Dict

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordRequestForm
from loguru import logger
import time
import httpx

from mlops_rakuten.auth.auth_simple import (
    authenticate_user,
    create_access_token,
    require_admin,
    require_user,
)
from mlops_rakuten.services.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Rakuten Gateway", version="1.0.0")

PREDICT_URL = "http://api-predict:8000"
INGEST_URL = "http://api-ingest:8000"
TRAIN_URL = "http://api-train:8000"


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/token")
async def token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants invalides",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        username=user["username"], role=user["role"])
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/init")
async def proxy_init(force: bool = False, _=Depends(require_admin)) -> Any:
    async with httpx.AsyncClient(timeout=600) as client:
        r  = await client.post(f"{INGEST_URL}/init", params={"force": force})
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@app.post("/ingest")
async def proxy_ingest(file: UploadFile = File(...), _=Depends(require_admin)) -> Any:
    files = {"file": (file.filename, await file.read(), file.content_type or "text/csv")}
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{INGEST_URL}/ingest", files=files)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()


@app.post("/train")
async def proxy_train(_=Depends(require_admin)) -> Any:
    """
    Lance le training et recharge automatiquement le modèle.
    
    **Workflow:**
    Training
    Reload du modèle (5-10 sec)
    Nouveau modèle prêt pour les prédictions
    """
    train_start = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=3600) as client:
            train_r = await client.post(f"{TRAIN_URL}/train")
        
        if train_r.status_code >= 400:
            logger.error(f"Training échoué: {train_r.status_code}")
            raise HTTPException(status_code=train_r.status_code, detail=train_r.text)
        
        train_result = train_r.json()
        train_duration = time.time() - train_start
        
        logger.success(f"Training terminé en {train_duration:.1f}s")
        
        logger.info("Rechargement du modèle MLflow...")
        
        reload_start = time.time()
        
        async with httpx.AsyncClient(timeout=60) as client:
            reload_r = await client.post(f"{PREDICT_URL}/reload")
        
        reload_success = reload_r.status_code < 400
        reload_result = reload_r.json() if reload_success else {"error": reload_r.text}
        reload_duration = time.time() - reload_start
        
        if not reload_success:
            logger.warning(f"Reload échoué: {reload_r.status_code}")
        else:
            logger.success(f"Modèle rechargé en {reload_duration:.1f}s")
        
        return {
            "status": "complete",
            "stages": ["train", "reload"],
            "message": "Training et reload du modèle terminés",
            "training": {
                "status": "success",
                "duration_seconds": train_duration,
                "details": train_result
            },
            "model_reload": {
                "status": "success" if reload_success else "failed",
                "duration_seconds": reload_duration,
                "details": reload_result
            },
            "total_duration_seconds": train_duration + reload_duration,
            "ready_for_predictions": reload_success,
            "next_step": "Modèle prêt pour /predict" if reload_success else "Vérifier les logs"
        }
    
    except Exception as e:
        logger.error(f"Erreur pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
async def reload_model(_=Depends(require_admin)) -> Dict[str, Any]:
    """
    Recharge manuellement le modèle MLflow.
    
    Utile si besoin de recharger sans faire un training complet.
    """
    logger.info("Rechargement manuel du modèle...")
    
    reload_start = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{PREDICT_URL}/reload")
        
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        
        reload_duration = time.time() - reload_start
        result = r.json()
        result["total_duration_seconds"] = reload_duration
        
        logger.success(f"Modèle rechargé en {reload_duration:.1f}s")
        return result
    
    except Exception as e:
        logger.error(f"Erreur reload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def proxy_info(_=Depends(require_user)):
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(f"{PREDICT_URL}/info")
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()


@app.post("/predict", response_model=PredictionResponse)
async def proxy_predict(payload: PredictionRequest, _=Depends(require_user)) -> PredictionResponse:
    data = payload.model_dump() if hasattr(
        payload, "model_dump") else payload.dict()

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{PREDICT_URL}/predict", json=data)

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    return r.json()
