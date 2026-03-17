import pickle
import os
import tempfile
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import mlflow

from mlops_rakuten.config.entities import PredictionConfig
from mlops_rakuten.utils.utils import check_required_data_files


class Prediction:
    """
    Étape d'inférence pour le modèle Rakuten avec artifacts depuis MLflow.

    Architecture:
    - Vectorizer TF-IDF: chargé depuis MLflow artifacts
    - LabelEncoder: chargé depuis MLflow artifacts  
    - Modèle entraîné: chargé depuis MLflow registre (alias "production")
    - Categories mapping: chargé depuis MLflow artifacts

    Avantages:
    - Single source of truth dans MLflow
    - Versioning cohérent
    - Déploiement facile (pas besoin DVC en prod)
    """

    MODEL_REGISTRY_NAME = "text_classifier_tfidf_m"
    
    # Noms des artifacts attendus dans MLflow
    VECTORIZER_ARTIFACT_NAME = "tfidf_vectorizer.pkl"
    LABEL_ENCODER_ARTIFACT_NAME = "label_encoder.pkl"
    CATEGORIES_ARTIFACT_NAME = "class_mapping.json"

    def __init__(self, config: PredictionConfig, use_local_fallback: bool = True) -> None:
        """
        Args:
            config: PredictionConfig avec chemins locaux (fallback)
            use_local_fallback: Si True et MLflow indisponible, charge depuis local
        """
        self.config = config
        self.use_local_fallback = use_local_fallback
        self._setup_mlflow()
        self._load_artifacts()

    def _setup_mlflow(self) -> None:
        """Configure la connexion à DagsHub MLflow"""
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        dagshub_user = os.getenv("DAGSHUB_USER")
        dagshub_repo = os.getenv("DAGSHUB_REPO")

        if not all([dagshub_token, dagshub_user, dagshub_repo]):
            logger.warning("Variables d'env MLflow manquantes")
            self.mlflow_enabled = False
            return

        self.mlflow_enabled = True
        mlflow.set_tracking_uri(
            f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
        )
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        logger.info("MLflow configuré")

    def _load_artifacts(self) -> None:
        """
        Charge les artifacts:
        1. Essaie MLflow (si activé)
        2. Fallback: fichiers locaux (si use_local_fallback=True)
        """
        self._load_vectorizer()
        self._load_label_encoder()
        self._load_model()
        self._load_categories()
        
        logger.success("Initialisation de Prediction terminée")

    # ========== VECTORIZER ==========

    def _load_vectorizer(self) -> None:
        """Charge le vectorizer TF-IDF depuis MLflow ou local"""
        if self.mlflow_enabled:
            try:
                self._load_vectorizer_from_mlflow()
                return
            except Exception as e:
                logger.warning(f"Vectorizer MLflow indisponible: {e}")
                if not self.use_local_fallback:
                    raise

        # Fallback local
        self._load_vectorizer_local()

    def _load_vectorizer_from_mlflow(self) -> None:
        """
        Charge vectorizer depuis MLflow artifacts (alias "production").
        
        Les artifacts sont loggés dans ModelTrainer avec artifact_path="preprocessing"
        donc le chemin complet est: preprocessing/vectorizer_path (le nom du fichier)
        """
        try:
            client = mlflow.tracking.MlflowClient()
            version = client.get_model_version_by_alias(
                name=self.MODEL_REGISTRY_NAME, alias="production"
            )
            run_id = version.run_id

            logger.info(f"Téléchargement vectorizer depuis MLflow (run: {run_id})")

            # Créer un dossier temporaire pour les artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                # Télécharger l'artifact depuis le sous-dossier "preprocessing"
                # où ModelTrainer l'a loggé
                local_path = mlflow.artifacts.download_artifacts(
                    artifact_path="preprocessing/tfidf_vectorizer.pkl",
                    run_id=run_id,
                    dst_path=tmpdir
                )

                # Charger en mémoire
                with open(local_path, "rb") as f:
                    self.vectorizer = pickle.load(f)

            logger.success(
                f"Vectorizer chargé depuis MLflow (run: {run_id})"
            )

        except Exception as e:
            logger.error(f"Erreur chargement vectorizer MLflow: {e}")
            raise

    def _load_vectorizer_local(self) -> None:
        """Fallback: charge vectorizer depuis fichier local"""
        cfg = self.config
        logger.info(f"Chargement vectorizer local: {cfg.vectorizer_path}")
        
        if not Path(cfg.vectorizer_path).exists():
            raise FileNotFoundError(
                f"Vectorizer local introuvable: {cfg.vectorizer_path}"
            )

        with open(cfg.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        logger.success("Vectorizer chargé (local)")

    # ========== LABEL ENCODER ==========

    def _load_label_encoder(self) -> None:
        """Charge le label encoder depuis MLflow ou local"""
        if self.mlflow_enabled:
            try:
                self._load_label_encoder_from_mlflow()
                return
            except Exception as e:
                logger.warning(f"LabelEncoder MLflow indisponible: {e}")
                if not self.use_local_fallback:
                    raise

        self._load_label_encoder_local()

    def _load_label_encoder_from_mlflow(self) -> None:
        """
        Charge label encoder depuis MLflow artifacts (alias "production").
        
        L'artifact est loggé dans ModelTrainer avec artifact_path="preprocessing"
        """
        try:
            client = mlflow.tracking.MlflowClient()
            version = client.get_model_version_by_alias(
                name=self.MODEL_REGISTRY_NAME, alias="production"
            )
            run_id = version.run_id

            logger.info(f"Téléchargement label_encoder depuis MLflow (run: {run_id})")

            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = mlflow.artifacts.download_artifacts(
                    artifact_path="preprocessing/label_encoder.pkl",
                    run_id=run_id,
                    dst_path=tmpdir
                )

                with open(local_path, "rb") as f:
                    self.label_encoder = pickle.load(f)

            logger.success(f"LabelEncoder chargé depuis MLflow (run: {run_id})")

        except Exception as e:
            logger.error(f"Erreur chargement label_encoder MLflow: {e}")
            raise

    def _load_label_encoder_local(self) -> None:
        """Fallback: charge label encoder depuis fichier local"""
        cfg = self.config
        logger.info(f"Chargement label_encoder local: {cfg.label_encoder_path}")

        if not Path(cfg.label_encoder_path).exists():
            raise FileNotFoundError(
                f"LabelEncoder local introuvable: {cfg.label_encoder_path}"
            )

        with open(cfg.label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        logger.success("LabelEncoder chargé (local)")

    # ========== MODÈLE ==========

    def _load_model(self) -> None:
        """Charge le modèle depuis MLflow ou local"""
        if self.mlflow_enabled:
            try:
                self._load_model_from_mlflow()
                return
            except Exception as e:
                logger.warning(f"Modèle MLflow indisponible: {e}")
                if not self.use_local_fallback:
                    raise

        self._load_model_local()

    def _load_model_from_mlflow(self) -> None:
        """Charge le modèle depuis MLflow registre (alias "production")"""
        try:
            model_uri = f"models:/{self.MODEL_REGISTRY_NAME}@production"
            logger.info(f"Chargement modèle depuis MLflow: {model_uri}")

            self.model = mlflow.sklearn.load_model(model_uri)
            logger.success("Modèle chargé depuis MLflow (alias 'production')")

        except Exception as e:
            logger.error(f"Erreur chargement modèle MLflow: {e}")
            raise

    def _load_model_local(self) -> None:
        """Fallback: charge modèle depuis fichier local"""
        cfg = self.config
        logger.info(f"Chargement modèle local: {cfg.model_path}")

        if not Path(cfg.model_path).exists():
            raise FileNotFoundError(
                f"Modèle local introuvable: {cfg.model_path}"
            )

        with open(cfg.model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.success("Modèle chargé (local)")

    # ========== CATEGORIES ==========

    def _load_categories(self) -> None:
        """Charge le mapping catégories depuis MLflow ou local"""
        self.category_mapping: dict[int, str] | None = None

        if self.mlflow_enabled:
            try:
                self._load_categories_from_mlflow()
                return
            except Exception as e:
                logger.warning(f"Categories MLflow indisponible: {e}")
                if not self.use_local_fallback:
                    raise

        self._load_categories_local()

    def _load_categories_from_mlflow(self) -> None:
        """Charge categories.csv depuis MLflow artifacts"""
        try:
            client = mlflow.tracking.MlflowClient()
            version = client.get_model_version_by_alias(
                name=self.MODEL_REGISTRY_NAME, alias="production"
            )
            run_id = version.run_id

            logger.info(f"Téléchargement categories depuis MLflow (run: {run_id})")

            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = mlflow.artifacts.download_artifacts(
                    artifact_path=('preprocessing/class_mapping.json'),
                    run_id=run_id,
                    dst_path=tmpdir
                )

                df_cat = pd.read_json(local_path)
                self._process_categories(df_cat)

            logger.success(
                f"Categories chargées depuis MLflow ({len(self.category_mapping)} entrées)"
            )

        except Exception as e:
            logger.error(f"Erreur chargement categories MLflow: {e}")
            raise

    def _load_categories_local(self) -> None:
        """Fallback: charge categories.csv depuis local"""
        cfg = self.config

        if cfg.categories_path is None:
            logger.warning("Pas de fichier categories configuré")
            return

        logger.info(f"Chargement categories local: {cfg.categories_path}")

        if not Path(cfg.categories_path).exists():
            raise FileNotFoundError(
                f"Categories local introuvable: {cfg.categories_path}"
            )

        df_cat = pd.read_csv(cfg.categories_path)
        self._process_categories(df_cat)
        logger.success(
            f"Categories chargées (local) - {len(self.category_mapping)} entrées"
        )

    def _process_categories(self, df_cat: pd.DataFrame) -> None:
        """Traite le DataFrame categories et construit le mapping"""
        cfg = self.config

        if cfg.category_code_column not in df_cat.columns:
            raise KeyError(
                f"Colonne code '{cfg.category_code_column}' absente"
            )
        if cfg.category_name_column not in df_cat.columns:
            raise KeyError(
                f"Colonne nom '{cfg.category_name_column}' absente"
            )

        self.category_mapping = dict(
            zip(
                df_cat[cfg.category_code_column],
                df_cat[cfg.category_name_column],
            )
        )

    def get_model_info(self) -> dict:
        """Retourne des informations sur le modèle et artifacts chargés"""
        if not self.mlflow_enabled:
            return {"status": "offline", "note": "MLflow non disponible"}

        try:
            client = mlflow.tracking.MlflowClient()
            version = client.get_model_version_by_alias(
                name=self.MODEL_REGISTRY_NAME, alias="production"
            )
            run = client.get_run(run_id=version.run_id)

            metrics = {
                k: v for k, v in run.data.metrics.items()
                if k.startswith("val_")
            }

            return {
                "version": version.version,
                "status": version.status,
                "metrics": metrics,
                "created_at": version.creation_timestamp,
                "alias": "production",
                "run_id": version.run_id,
            }
        except Exception as e:
            logger.warning(f"Impossible de récupérer info modèle: {e}")
            return {"status": "error", "error": str(e)}

    def predict(self, texts, top_k: int | None = None):
        """
        Prend une liste de textes (designations produits)
        et renvoie un tableau de prdtypecode (int) prédits.

        Retourne, pour chaque texte, une liste de dicts:
        [
          {"prdtypecode": 10, "category_name": "Vêtements", "proba": 0.72},
          {"prdtypecode": 20, "category_name": "Smartphones", "proba": 0.18},
          ...
        ]

        Args:
            texts: str ou list[str]
            top_k: nombre de top prédictions par texte (optionnel)
        """
        if isinstance(texts, str):
            texts = [texts]

        logger.info(f"Inférence (avec probabilités) sur {len(texts)} texte(s)")

        # 1. Vectorisation
        X_vec = self.vectorizer.transform(texts)

        # 2. Probabilités par classe
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                "Le modèle courant ne supporte pas predict_proba. "
                "Utilise 'logistic_regression' dans model_trainer.model_type."
            )

        proba = self.model.predict_proba(X_vec)

        # 3. Mapping indices -> prdtypecode
        prdtypecodes = self.label_encoder.inverse_transform(
            np.arange(len(self.label_encoder.classes_))
        )

        results_all_texts = []

        for i in range(proba.shape[0]):
            proba_i = proba[i]
            sorted_idx = np.argsort(proba_i)[::-1]

            if top_k is not None:
                sorted_idx = sorted_idx[:top_k]

            results_one_text = []
            for idx in sorted_idx:
                code = int(prdtypecodes[idx])
                p = float(proba_i[idx])

                name = None
                if self.category_mapping is not None:
                    name = self.category_mapping.get(code)

                results_one_text.append(
                    {
                        "prdtypecode": code,
                        "category_name": name,
                        "proba": p,
                    }
                )

            results_all_texts.append(results_one_text)

        return results_all_texts