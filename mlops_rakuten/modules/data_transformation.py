import json
from pathlib import Path
import pickle

from loguru import logger
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mlops_rakuten.config.entities import DataTransformationConfig
from mlops_rakuten.utils.utils import create_directories


class DataTransformation:
    """
    Étape de transformation des données Rakuten.

    À partir du dataset prétraité, cette étape :
    - charge le dataset prétraité (issu de DataPreprocessing)
    - vérifie la présence des colonnes texte et cible
    - effectue le split train / validation
    - encode la cible via un LabelEncoder
    - vectorise le texte avec un TfidfVectorizer
    - sauvegarde tous les artefacts nécessaires à l'entraînement :
        - X_train / X_val (sparse .npz)
        - y_train / y_val (.npy)
        - tfidf_vectorizer.pkl
        - label_encoder.pkl
        - class_mapping.json
    """

    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config

    def run(self) -> Path:
        """
        Exécute la transformation des données et retourne le dossier
        contenant les artefacts transformés (data/processed).
        """
        logger.info("Démarrage de l'étape DataTransformation")

        cfg = self.config

        # 1. Charger le dataset prétraité
        logger.info(f"Lecture du dataset prétraité depuis : {cfg.input_dataset_path}")
        df = pd.read_csv(cfg.input_dataset_path)
        logger.debug(f"Dataset prétraité shape: {df.shape}")

        # 2. Vérifier la présence des colonnes texte et cible
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

        # 3. Extraire X (texte) et y (cible)
        X_text = df[cfg.text_column].astype(str)
        y = df[cfg.target_column]

        logger.info("Split train / validation")
        stratify_arg = y if cfg.stratify else None

        X_train_text, X_val_text, y_train, y_val = train_test_split(
            X_text,
            y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=stratify_arg if cfg.stratify else None,
        )

        logger.debug(f"X_train_text shape: {X_train_text.shape}")
        logger.debug(f"X_val_text shape: {X_val_text.shape}")
        logger.debug(f"y_train length: {len(y_train)}")
        logger.debug(f"y_val length: {len(y_val)}")

        # 4. Label encoding de la cible
        logger.info("Encodage de la cible (LabelEncoder)")
        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train)
        y_val_enc = label_encoder.transform(y_val)

        # Mapping lisible : index encodé -> label original
        class_mapping = {int(idx): int(label) for idx, label in enumerate(label_encoder.classes_)}

        # 5. Vectorisation TF-IDF
        logger.info("Vectorisation TF-IDF du texte")

        stop_words = cfg.stop_words if cfg.stop_words is not None else None

        vectorizer = TfidfVectorizer(
            max_features=cfg.max_features,
            ngram_range=(cfg.ngram_min, cfg.ngram_max),
            lowercase=cfg.lowercase,
            stop_words=stop_words,
        )

        X_train_vec = vectorizer.fit_transform(X_train_text)
        X_val_vec = vectorizer.transform(X_val_text)

        logger.debug(f"X_train_vec shape: {X_train_vec.shape}")
        logger.debug(f"X_val_vec shape: {X_val_vec.shape}")

        # 6. Création du dossier de sortie
        create_directories([cfg.output_dir])

        # 7. Sauvegarde des artefacts
        logger.info(f"Sauvegarde du vectorizer vers : {cfg.vectorizer_path}")
        with open(cfg.vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

        logger.info(f"Sauvegarde du LabelEncoder vers : {cfg.label_encoder_path}")
        with open(cfg.label_encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)

        logger.info(f"Sauvegarde du mapping des classes vers : {cfg.class_mapping_path}")
        with open(cfg.class_mapping_path, "w") as f:
            json.dump(class_mapping, f, indent=2)

        logger.info(f"Sauvegarde de X_train (sparse) vers : {cfg.X_train_path}")
        sparse.save_npz(cfg.X_train_path, X_train_vec)

        logger.info(f"Sauvegarde de X_val (sparse) vers : {cfg.X_val_path}")
        sparse.save_npz(cfg.X_val_path, X_val_vec)

        logger.info(f"Sauvegarde de y_train vers : {cfg.y_train_path}")
        np.save(cfg.y_train_path, y_train_enc)

        logger.info(f"Sauvegarde de y_val vers : {cfg.y_val_path}")
        np.save(cfg.y_val_path, y_val_enc)

        logger.success("DataTransformation terminée avec succès")
        return cfg.output_dir
