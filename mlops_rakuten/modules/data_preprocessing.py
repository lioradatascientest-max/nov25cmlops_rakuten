from pathlib import Path

from loguru import logger
import pandas as pd

from mlops_rakuten.config.entities import DataPreprocessingConfig
from mlops_rakuten.utils.utils import create_directories


class DataPreprocessing:
    """
    Étape de prétraitement des données Rakuten.

    - Charge le dataset fusionné (issu de DataIngestion)
    - Nettoie la colonne texte (NA, vides, longueur, ratio lettres)
    - Nettoie la cible (NA éventuels)
    - Supprime les doublons si demandé
    - Sauvegarde un dataset prétraité dans data/interim/
    """

    def __init__(self, config: DataPreprocessingConfig) -> None:
        self.config = config

    @staticmethod
    def _alpha_ratio(text: str) -> float:
        """Calcule la proportion de caractères alphabétiques dans une chaîne."""
        if not isinstance(text, str) or not text:
            return 0.0
        nb_alpha = sum(ch.isalpha() for ch in text)
        return nb_alpha / len(text)

    def run(self) -> Path:
        """
        Exécute le prétraitement des données et retourne le chemin
        du fichier dataset prétraité.
        """
        logger.info("Démarrage de l'étape DataPreprocessing")

        cfg = self.config

        # 1. Charger le dataset d'entrée
        logger.info(f"Lecture du dataset depuis : {cfg.input_dataset_path}")
        df = pd.read_csv(cfg.input_dataset_path)
        logger.debug(f"Dataset initial shape: {df.shape}")

        # 2. Nettoyage de la colonne texte
        if cfg.text_column not in df.columns:
            raise KeyError(
                f"Colonne texte '{cfg.text_column}' absente du dataset "
                f"(colonnes disponibles: {list(df.columns)})"
            )
        if cfg.target_column not in df.columns:
            raise KeyError(
                f"Colonne cible '{cfg.target_column}' absente du dataset "
                f"(colonnes disponibles: {list(df.columns)})"
            )

        # Conversion en str + strip
        logger.info(f"Normalisation de la colonne texte '{cfg.text_column}'")
        df[cfg.text_column] = df[cfg.text_column].astype(str).str.strip()

        # 3. Gestion des valeurs manquantes / vides
        n_before = len(df)

        if cfg.drop_na_text:
            logger.info("Suppression des lignes avec texte manquant")
            df = df[df[cfg.text_column].notna()]

        # texte vide après strip
        logger.info("Suppression des lignes avec texte vide après strip")
        df = df[df[cfg.text_column] != ""]

        if cfg.drop_na_target:
            logger.info("Suppression des lignes avec cible manquante")
            df = df[df[cfg.target_column].notna()]

        n_after_na = len(df)
        logger.debug(f"Lignes supprimées (NA / texte vide): {n_before - n_after_na}")

        # 4. Longueur de texte
        logger.info("Filtrage sur la longueur des textes")
        df["__char_len__"] = df[cfg.text_column].str.len()

        mask_len = (df["__char_len__"] >= cfg.min_char_length) & (
            df["__char_len__"] <= cfg.max_char_length
        )
        n_before_len = len(df)
        df = df[mask_len]
        logger.debug(f"Lignes supprimées (longueur texte): {n_before_len - len(df)}")

        # 5. Ratio de caractères alphabétiques
        logger.info("Filtrage sur le ratio de caractères alphabétiques")
        df["__alpha_ratio__"] = df[cfg.text_column].apply(self._alpha_ratio)

        n_before_alpha = len(df)
        df = df[df["__alpha_ratio__"] >= cfg.min_alpha_ratio]
        logger.debug(f"Lignes supprimées (alpha_ratio): {n_before_alpha - len(df)}")

        # 6. Suppression des colonnes techniques temporaires
        df = df.drop(columns=["__char_len__", "__alpha_ratio__"], errors="ignore")

        # 7. Suppression des doublons
        if cfg.drop_duplicates:
            logger.info("Suppression des doublons (texte + cible)")
            n_before_dups = len(df)
            df = df.drop_duplicates(subset=[cfg.text_column, cfg.target_column])
            logger.debug(f"Lignes supprimées (doublons): {n_before_dups - len(df)}")

        logger.info(f"Shape final du dataset prétraité: {df.shape}")

        # 8. Création du dossier de sortie
        output_path = cfg.output_dataset_path
        create_directories([output_path.parent])

        # 9. Sauvegarde
        logger.info(f"Sauvegarde du dataset prétraité vers : {output_path}")
        df.to_csv(output_path, index=False)

        logger.success("DataPreprocessing terminée avec succès")
        return output_path
