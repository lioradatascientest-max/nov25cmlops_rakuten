from __future__ import annotations
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from loguru import logger
import os

EXECUTION_MODE = os.getenv("EXECUTION_MODE", "cli")

if EXECUTION_MODE == "docker":
    from mlops_rakuten.utils.docker import _dvc, sync_training_results
    SYNC_MODE = None  # docker_utils hardcode "Docker-DID"
    logger.info("Transport : docker exec (Docker-DID)")
else:
    from mlops_rakuten.utils.cli import _dvc, sync_training_results
    SYNC_MODE = "Docker-CLI"
    logger.info("Transport : subprocess (Docker-CLI)")

app = FastAPI(
    title="Rakuten Train API",
    version="1.0.0",
    description=f"Mode d'exécution actuel : **{EXECUTION_MODE}**",
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "execution_mode": EXECUTION_MODE}


@app.post("/train")
def train() -> Dict[str, Any]:
    try:
        logger.info("=" * 60)
        logger.info("Starting Training Pipeline...")
        
        # --- AJOUT : Configuration du Token DagsHub ---
        # On utilise --local pour ne pas modifier le fichier dvc/config qui est versionné par Git
        #dagshub_token = os.getenv("DAGSHUB_TOKEN")
        #logger.info("Configuring DVC authentication...")
        #_dvc("dvc remote modify --local storage --unset auth") 
        #_dvc(f"dvc remote modify --local storage user samuel.beau")
        #_dvc(f"dvc remote modify --local storage password {dagshub_token}")
        #_dvc("dvc remote modify --local storage auth basic")
        # ----------------------------------------------


        # ici le true permet de ne pas échouer si il y a deja des elements dans le cache distant, on veut juste s'assurer d'avoir la derniere version avant de lancer le repro et ecraser les changements locaux.
        _dvc("dvc pull 2>&1 || true")
        _dvc("dvc repro")

        # sync_training_results lit la config en interne pour model_version/f1/run_id
        sync_results = sync_training_results(mode=SYNC_MODE)

        if not sync_results["summary"]["success"]:
            return {
                "status": "training_complete_sync_failed",
                "message": "Training succeeded but git+dvc sync failed",
                "sync_errors": sync_results["errors"],
                "manual_steps": [
                    "1. git add dvc.lock models/ reports/",
                    "2. git commit -m 'training: pipeline complete'",
                    "3. git push origin <branch>",
                ],
            }

        logger.success("Training Pipeline Complete!")
        return {
            "status": "complete",
            "stages": ["transform", "train", "evaluate"],
            "message": "All stages executed and synced successfully",
            "sync_summary": sync_results["summary"],
        }

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))