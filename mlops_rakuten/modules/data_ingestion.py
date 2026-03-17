from pathlib import Path

from loguru import logger
import pandas as pd

from mlops_rakuten.config.entities import DataIngestionConfig
from mlops_rakuten.utils.utils import check_required_data_files, create_directories


class DataIngestion:
    """
    Ingestion incrémentale:
    - prend un CSV uploadé (designation, prdtypecode)
    - append dans data/interim/rakuten_train.csv
    """

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config
        self._check_data_availability()

    def _check_data_availability(self) -> None:
        """Vérifie que les fichiers de données d'entrée sont disponibles."""

        required_files = {
            "Fichier de données mergé": self.config.train_path,
        }

        logger.info("Vérification de la disponibilité des fichiers de données requis...")
        all_ok = check_required_data_files(required_files)

        if not all_ok:
            raise FileNotFoundError(
                "Les fichiers de données requis sont manquants ou invalides. "
                "Veuillez suivre les instructions du README pour les obtenir."
            )
        logger.info("Tous les fichiers de données requis sont disponibles.")

    def run(self, uploaded_csv_path: Path) -> Path:
        cfg = self.config
        create_directories([cfg.train_path.parent])

        logger.info(f"Lecture batch uploadé : {uploaded_csv_path}")
        batch = pd.read_csv(uploaded_csv_path)

        required = [cfg.text_column, cfg.target_column]
        missing = set(required) - set(batch.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans upload: {sorted(missing)}")
        batch = batch[required]

        if cfg.train_path.exists():
            logger.info(f"Lecture dataset courant : {cfg.train_path}")
            current = pd.read_csv(cfg.train_path)
            current = current[required]
            merged = pd.concat([current, batch], ignore_index=True)
        else:
            logger.info("Aucun dataset courant trouvé, initialisation avec le batch")
            merged = batch

        # Optionnel: drop_duplicates
        merged = merged.drop_duplicates(subset=[cfg.text_column, cfg.target_column])

        logger.info(f"Sauvegarde dataset courant : {cfg.train_path}")
        merged.to_csv(cfg.train_path, index=False)

        logger.success("DataIngestion incrémentale terminée")
        return cfg.train_path
