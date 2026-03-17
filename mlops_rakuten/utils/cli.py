from __future__ import annotations

"""
Version CLI de la synchronisation DVC + Git.

Transport : subprocess (commandes locales).
Utilise sync_core.sync_git_dvc() pour la logique commune.
"""

import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from mlops_rakuten.utils.sync_core import sync_git_dvc,read_training_artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Transport local
# ─────────────────────────────────────────────────────────────────────────────

def _dvc(cmd: str) -> str:
    """Exécute une commande DVC en local."""
    logger.info(f"[DVC] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            logger.debug(f"   {line}")
    if result.returncode != 0:
        logger.error(result.stderr.strip())
        raise RuntimeError(f"DVC failed: {result.stderr.strip()}")
    return result.stdout


def _git(cmd: str) -> str:
    """Exécute une commande Git en local."""
    logger.info(f"[Git] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(result.stderr.strip())
        raise RuntimeError(f"Git failed: {result.stderr.strip()}")
    return result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# Commandes spécialisées — même interface que docker_utils
# ─────────────────────────────────────────────────────────────────────────────

def sync_ingest_data(
    uploaded_filename: str,
    mode: str | None = None,
) -> dict[str, Any]:
    prefix = mode or "CLI-local"
    logger.info(f"[Ingest] Sync post-ingestion : {uploaded_filename} [{prefix}]")

    return sync_git_dvc(
        run_dvc=_dvc,
        run_git=_git,
        commit_prefix=f"{prefix}:ingest",
        commit_message=f"batch={Path(uploaded_filename).stem}",
        git_paths=[
            "data/interim/rakuten_train.csv.dvc",
            "dvc.lock",
            ".dvc/",
        ],
        dvc_files=["data/interim/rakuten_train.csv"],
        push=True,
    )


def sync_training_results(mode: str | None = None) -> dict[str, Any]:
    
    artifacts = read_training_artifacts()
    prefix = mode or "CLI-local"  

    result= sync_git_dvc(
        run_dvc=_dvc,
        run_git=_git,
        commit_prefix=f"{prefix}:train",
        commit_message=f"model v{artifacts['version']}, f1_macro={artifacts['f1']}, run_id={artifacts['run_id']}",
        git_paths=["mlops_rakuten/"],
        dvc_files=None,
        push=True,
    )
    result["artifacts"] = artifacts  # pour le logger dans main.py
    return result


def sync_init(
    force: bool = False,
    mode: str | None = None,
) -> dict[str, Any]:
    prefix = mode or "CLI-local"
    label = "force-rebuild" if force else "normal"
    logger.info(f"[Init] Sync post-seed [{label}] [{prefix}]")

    return sync_git_dvc(
        run_dvc=_dvc,
        run_git=_git,
        commit_prefix=f"{prefix}:init",
        commit_message=f"seed dataset [{label}]",
        git_paths=[
            "data/interim/rakuten_train.csv.dvc",
            "dvc.lock",
            ".dvc/",
        ],
        dvc_files=["data/interim/rakuten_train.csv"],
        push=True,
    )