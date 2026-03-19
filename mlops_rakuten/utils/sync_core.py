from __future__ import annotations

"""
Logique de synchronisation DVC + Git, sans transport.

Ce module est agnostique : il reçoit des callables `run_dvc` et `run_git`
(subprocess pour le CLI, container.exec_run pour Docker) et orchestre
les opérations dans le même ordre dans les deux cas.

Importé par :
  - mlops_rakuten/utils/sync_utils.py   (CLI, subprocess)
  - mlops_rakuten/utils/docker_utils.py (Docker, container.exec_run)
"""

from typing import Any, Callable
from loguru import logger

from mlops_rakuten.utils.utils import git_commit


RunFn = Callable[[str], str]  # (cmd) -> stdout


def sync_git_dvc(
    run_dvc: RunFn,
    run_git: RunFn,
    commit_prefix: str,
    commit_message: str,
    git_paths: list[str],
    dvc_files: list[str] | None = None,
    push: bool = True,
) -> dict[str, Any]:
    """
    Synchronise DVC + Git et pousse vers DagsHub/GitHub.

    Args:
        run_dvc:        Callable qui exécute une commande DVC (CLI ou Docker)
        run_git:        Callable qui exécute une commande Git (CLI ou Docker)
        commit_prefix:  Ex. "CLI:ingest", "CLI:train"
        commit_message: Corps du message de commit
        git_paths:      Fichiers à stager dans Git
        dvc_files:      Fichiers à tracker avec `dvc add` (optionnel)
        push:           Si True, pousse vers DVC remote et Git remote

    Returns:
        dict avec dvc_operations, git_operations, errors, summary
    """
    results: dict[str, Any] = {
        "dvc_operations": [],
        "git_operations": [],
        "errors": [],
    }

    try:
        # ── Étape 1 : dvc add ────────────────────────────────────────────────
        if dvc_files:
            logger.info(f"[Sync] dvc add ({len(dvc_files)} fichier(s))")
            for f in dvc_files:
                try:
                    run_dvc(f"dvc add {f}")
                    results["dvc_operations"].append({"file": f, "status": "success"})
                except Exception as e:
                    msg = f"dvc add {f} failed: {e}"
                    logger.error(msg)
                    results["errors"].append(msg)

        # ── Étape 2 : git commit ─────────────────────────────────────────────
        # Les .dvc et dvc.lock générés par dvc add sont ajoutés aux paths
        all_git_paths = git_paths.copy()
        if dvc_files:
            all_git_paths += ["*.dvc", "dvc.lock", ".gitignore"]

        logger.info(f"[Sync] git commit — {commit_prefix} | {commit_message}")
        try:
            git_commit(prefix=commit_prefix, message=commit_message, paths=all_git_paths)
            results["git_operations"].append({"operation": "commit", "status": "success"})
        except Exception as e:
            msg = str(e)
            if "nothing to commit" in msg.lower() or "working tree clean" in msg.lower():
                logger.info("[Sync] Rien à commiter — working tree propre")
                results["git_operations"].append({
                    "operation": "commit",
                    "status": "skipped",
                    "reason": "working tree clean",
                })
            else:
                logger.error(f"git commit failed: {msg}")
                results["errors"].append(f"git commit failed: {msg}")

        # ── Étape 3 : push ───────────────────────────────────────────────────
        if push:
            # dvc push toujours — pas seulement si dvc_files
            logger.info("[Sync] dvc push")
            try:
                run_dvc("dvc push")
                results["dvc_operations"].append({"operation": "push", "status": "success"})
            except Exception as e:
                msg = f"dvc push failed: {e}"
                logger.error(msg)
                results["errors"].append(msg)

            has_commit = any(
                op.get("operation") == "commit" and op.get("status") == "success"
                for op in results["git_operations"]
            )
            if has_commit:
                logger.info("[Sync] git push")
                try:
                    branch = run_git("git rev-parse --abbrev-ref HEAD").strip()
                    run_git(f"git push myfork {branch}")
                    results["git_operations"].append({
                        "operation": "push",
                        "branch": branch,
                        "status": "success",
                    })
                except Exception as e:
                    msg = f"git push failed: {e}"
                    logger.error(msg)
                    results["errors"].append(msg)
            else:
                logger.info("[Sync] Pas de commit à pousser — git push ignoré")
                results["git_operations"].append({
                    "operation": "push",
                    "status": "skipped",
                    "reason": "no commits to push",
                })

    except Exception as e:
        logger.error(f"[Sync] Erreur inattendue : {e}")
        results["errors"].append(str(e))

    results["summary"] = {
        "total_dvc_ops": len(results["dvc_operations"]),
        "total_git_ops": len(results["git_operations"]),
        "total_errors": len(results["errors"]),
        "success": len(results["errors"]) == 0,
    }

    if results["summary"]["success"]:
        logger.success("[Sync] Synchronisation complète")
    else:
        logger.warning(f"[Sync] Terminé avec {results['summary']['total_errors']} erreur(s)")

    return results


def read_training_artifacts() -> dict:
    """Lit les artefacts post-training via ConfigurationManager."""
    from mlops_rakuten.config.config_manager import ConfigurationManager
    import json
    from pathlib import Path

    config       = ConfigurationManager()
    model_dir    = Path(config.get_model_trainer_config().model_dir)
    metrics_path = Path(config.get_model_evaluation_config().metrics_path)

    metadata_path = model_dir / "mlflow_run_metadata.json"
    meta = {"run_id": "unknown", "version": "?"}
    if metadata_path.exists():
        with open(metadata_path) as f:
            data = json.load(f)
        meta = {
            "run_id": data.get("run_id", "unknown")[:7],
            "version": data.get("model_version", "?"),
        }

    f1 = "?"
    if metrics_path.exists():
        with open(metrics_path) as f:
            f1 = str(round(json.load(f).get("val_f1_macro", 0), 4))

    return {"version": meta["version"], "f1": f1, "run_id": meta["run_id"]}