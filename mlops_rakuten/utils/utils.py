# See: https://dagshub.com/licence.pedago/overview_mlops_wine_quality_student/src/main/src/common_utils.py

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

from box import ConfigBox
from box.exceptions import BoxValueError
from loguru import logger
import pandas as pd
import yaml


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Lit un fichier YAML et renvoie un ConfigBox (accès par attributs).
    """
    if not path_to_yaml.exists():
        raise FileNotFoundError(f"Le fichier YAML n'existe pas : {path_to_yaml}")
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
        if content is None:
            raise BoxValueError("empty yaml")
        logger.info(f"YAML chargé : {path_to_yaml}")
        return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"Le fichier YAML est vide : {path_to_yaml}")
    except Exception as e:
        logger.error(f"Erreur lecture {path_to_yaml}: {e}")
        raise


def create_directories(directories: Iterable[Path], verbose: bool = True) -> None:
    """Crée une liste de répertoires si ils n'existent pas."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        if verbose:
            logger.info(f"Répertoire créé ou existant : {directory}")


def check_file_exists(file_path: Path, check_readable: bool = True) -> bool:
    """Vérifie qu'un fichier existe et est lisible."""
    if not file_path.exists():
        return False
    if check_readable:
        try:
            df = pd.read_csv(file_path, nrows=5)
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.debug(f"{file_path.name} ({size_mb:.1f} MB, {df.shape[1]} colonnes)")
            return True
        except Exception as e:
            logger.warning(f"{file_path.name} : {e}")
            return False
    return True


def check_required_data_files(
    required_files: dict[str, Path],
    show_instructions: bool = True,
) -> bool:
    """Vérifie la présence de fichiers de données requis."""
    missing_files = [
        file_path.name
        for file_path in required_files.values()
        if not check_file_exists(file_path)
    ]
    if missing_files:
        logger.error("Fichiers manquants dans data/raw/ :")
        for filename in missing_files:
            logger.error(f"  {filename}")
        if show_instructions:
            logger.info("Consultez le README (section 'Configuration des données')")
        return False
    return True


def get_latest_run_dir(parent_dir: Path) -> Path:
    """Retourne le sous-répertoire le plus récent (tri lexical ISO-8601)."""
    if not parent_dir.exists():
        raise FileNotFoundError(f"{parent_dir} n'existe pas")
    run_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"Aucun sous-répertoire dans {parent_dir}")
    latest_dir = sorted(run_dirs)[-1]
    logger.info(f"Dernier run : {latest_dir.name}")
    return latest_dir


# ─────────────────────────────────────────────────────────────────────────────
# Git helpers — partagés par CLI (sync_utils) et Docker (docker_utils)
# ─────────────────────────────────────────────────────────────────────────────

def shell(cmd: list[str], cwd: Path | None = None) -> str:
    """
    Exécute une commande shell locale et retourne stdout.
    Logue un warning si returncode != 0 mais ne lève pas d'exception
    (laisser l'appelant décider).
    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd or Path.cwd(),
    )
    if result.returncode != 0:
        logger.warning(f"shell warning [{' '.join(cmd)}]: {result.stderr.strip()}")
    return result.stdout.strip()


def git_commit(prefix: str, message: str, paths: list[str] | None = None) -> None:
    """
    Stage les fichiers indiqués et effectue un commit préfixé.

    Args:
        prefix:  Préfixe standardisé, ex. "CLI:train", "CLI:ingest"
        message: Corps du message, ex. "model v3, f1_macro=0.821, run_id=abc123"
        paths:   Fichiers/dossiers à stager. Si None → git add -A.
    """
    author_name = os.getenv("GIT_AUTHOR_NAME", "Rakuten MLOps")
    author_email = os.getenv("GIT_AUTHOR_EMAIL", "mlops@rakuten.local")

    if paths:
        shell(["git", "add"] + paths)
    else:
        shell(["git", "add", "-A"])

    # Vérifier qu'il y a quelque chose à commiter
    status = shell(["git", "status"])
    if not status:
        logger.info("git_commit: rien à commiter (working tree propre)")
        return

    full_message = f"{prefix} | {message}"
    shell([
        "git", "commit",
        "--author", f"{author_name} <{author_email}>",
        "-m", full_message,
    ])
    logger.success(f"Git commit : {full_message}")