"""
Utilitaires partagés entre les DAGs Rakuten.

Résolution des chemins hôte via le socket Docker — sans variable d'environnement.
"""

from __future__ import annotations

import socket
from pathlib import Path

import docker as docker_sdk


def resolve_project_root() -> str:
    """
    Retourne le chemin absolu hôte de la racine du projet.

    Le container Airflow monte ./deployments/airflow/dags → /opt/airflow/dags.
    En remontant deux niveaux depuis le chemin hôte de ce mount :
      Source hôte : .../nov25cmlops_rakuten/deployments/airflow/dags
      .parent      : .../nov25cmlops_rakuten/deployments/airflow
      .parent.parent : .../nov25cmlops_rakuten  ← racine projet
    """
    try:
        client = docker_sdk.from_env()
        container = client.containers.get(socket.gethostname())
        for mount in container.attrs.get("Mounts", []):
            if mount.get("Destination") == "/opt/airflow/dags":
                return str(Path(mount["Source"]).parent.parent)
    except Exception:
        pass
    return ""


def resolve_ssh_dir() -> str:
    """
    Retourne le chemin absolu hôte du dossier .ssh.
    Cherche le mount dont la target est /root/.ssh dans le container courant.
    """
    try:
        client = docker_sdk.from_env()
        container = client.containers.get(socket.gethostname())
        for mount in container.attrs.get("Mounts", []):
            if mount.get("Destination") == "/root/.ssh":
                return mount["Source"]
    except Exception:
        pass
    return ""