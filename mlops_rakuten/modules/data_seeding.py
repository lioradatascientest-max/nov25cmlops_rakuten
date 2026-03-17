from pathlib import Path

from loguru import logger
import pandas as pd

from mlops_rakuten.config.entities import DataSeedingConfig
from mlops_rakuten.utils.utils import create_directories


class DataSeeding:
    """
    Étape de seeding (simulation d'uploads admin) à partir des données brutes Rakuten.

    - Charge X_train_update.csv (features)
    - Charge Y_train_CVw08PX.csv (labels)
    - Vérifie l'alignement des index
    - Fusionne X et y
    - Conserve uniquement: (designation, prdtypecode)
    - Génère 10 fichiers batch de taille batch_size dans data/raw/uploads_seed/
    - Sauvegarde le reste dans rakuten_dataset_remainder.csv
    - Sauvegarde le dataset complet dans rakuten_dataset_full.csv
    - Initialise data/interim/rakuten_train.csv avec le remainder
    """

    def __init__(self, config: DataSeedingConfig) -> None:
        self.config = config

    def run(self) -> Path:
        """
        Exécute le seeding et retourne le chemin du fichier
        data/interim/rakuten_train.csv initialisé (remainder).
        """
        logger.info("Démarrage de l'étape DataSeeding")

        cfg = self.config

        # 1. Charger X
        logger.info(f"Lecture de X_train depuis : {cfg.x_train_path}")
        X = pd.read_csv(cfg.x_train_path, index_col=0)
        logger.debug(f"X shape: {X.shape}")

        # 2. Charger y
        logger.info(f"Lecture de Y_train depuis : {cfg.y_train_path}")
        y = pd.read_csv(cfg.y_train_path, index_col=0)
        logger.debug(f"y shape: {y.shape}")

        # 3. Vérifier l'alignement des index
        logger.info("Vérification de l'alignement des index entre X et y")
        if not X.index.equals(y.index):
            logger.error("Les index de X et y ne correspondent pas")
            raise ValueError(
                "Les index de X_train et Y_train ne correspondent pas. "
                "Impossible de faire un merge sûr."
            )

        # 4. Fusionner
        logger.info("Fusion de X et y")
        df = X.join(y)
        logger.debug(f"Dataset fusionné shape: {df.shape}")

        # 5. Conserver uniquement les colonnes utiles au modèle
        logger.info("Sélection des colonnes utiles: designation + prdtypecode")
        required_cols = {cfg.text_column, cfg.target_column}
        missing = required_cols.difference(df.columns)
        if missing:
            logger.error(f"Colonnes manquantes après merge: {missing}")
            raise ValueError(
                f"Colonnes manquantes après merge: {missing}. "
                "Vérifie text_col/target_col et les sources X/Y."
            )

        df = df[[cfg.text_column, cfg.target_column]]
        logger.debug(f"Dataset (2 colonnes) shape: {df.shape}")

        # 6. Créer les dossiers de sortie
        create_directories([cfg.seeds_dir, cfg.output_dataset_path.parent])

        # 7. Sauvegarder dataset complet
        full_path = cfg.output_full_path
        logger.info(f"Sauvegarde du dataset complet vers : {full_path}")
        df.to_csv(full_path, index=False)

        # 8. Générer les batches
        logger.info(
            f"Génération de {cfg.n_batches} batches de taille {cfg.batch_size} dans : {cfg.seeds_dir}"
        )
        total_batch_rows = cfg.n_batches * cfg.batch_size

        if len(df) < total_batch_rows:
            logger.error(
                f"Dataset trop petit pour générer {cfg.n_batches} batches "
                f"de {cfg.batch_size} lignes (total={total_batch_rows}). "
                f"Rows disponibles: {len(df)}"
            )
            raise ValueError(
                "Dataset trop petit pour le seeding demandé "
                f"(need={total_batch_rows}, got={len(df)})."
            )

        for i in range(cfg.n_batches):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            batch_df = df.iloc[start:end]

            batch_path = cfg.seeds_dir / f"rakuten_batch_{i + 1:04d}.csv"
            logger.info(f"Sauvegarde du batch {i + 1}/{cfg.n_batches} vers : {batch_path}")
            batch_df.to_csv(batch_path, index=False)

        # 9. Remainder
        remainder_df = df.iloc[total_batch_rows:]
        remainder_path = cfg.output_remainder_path
        logger.info(f"Sauvegarde du remainder vers : {remainder_path}")
        remainder_df.to_csv(remainder_path, index=False)
        logger.debug(f"Remainder rows: {len(remainder_df)}")

        # 10. Initialiser data/interim/rakuten_train.csv avec le remainder
        dataset_path = cfg.output_dataset_path
        logger.info(f"Initialisation de rakuten_dataset vers : {dataset_path}")
        remainder_df.to_csv(dataset_path, index=False)

        logger.success("DataSeeding terminé avec succès")
        return dataset_path
