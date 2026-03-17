import json
from pathlib import Path
import pickle
from typing import Optional

from loguru import logger
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC

from mlops_rakuten.config.entities import ModelTrainerConfig
from mlops_rakuten.utils.utils import create_directories
import mlflow
import os


class ModelTrainer:
    """
    Étape d'entraînement du modèle Rakuten. Le modèle est couplé a MLflow pour le suivi d'expérimentations et le registre de modèles.

    - Charge X_train et y_train transformés (TF-IDF + label encoding)
    - Instancie le modèle sklearn (par défaut LinearSVC)
    - Entraîne le modèle
    - Calcule quelques métriques sur le jeu d'entraînement
    - Sauvegarde :
        - le modèle entraîné (text_classifier.pkl)
        - un fichier JSON décrivant la configuration d'entraînement
        - un fichier JSON contenant les métriques d'entraînement
        - un rapport texte de classification sur le train
    """

    MODEL_REGISTRY_NAME = "text_classifier_tfidf_m"

    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Configure la connexion à DagsHub MLflow"""
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        dagshub_user = os.getenv("DAGSHUB_USER")
        dagshub_repo = os.getenv("DAGSHUB_REPO")

        mlflow_user = os.getenv("MLFLOW_TRACKING_USERNAME")

        if not all([dagshub_token, dagshub_user, dagshub_repo]):
            logger.warning("Variables d'env MLflow manquantes")
            self.mlflow_enabled = False
            return

        self.mlflow_enabled = True
        mlflow.set_tracking_uri(
            f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
        )

        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        logger.info("MLflow configuré")


    def _build_model(self):
        """
        Construit le modèle sklearn en fonction de la configuration.
        """
        cfg = self.config

        if cfg.model_type == "linear_svc":
            class_weight = "balanced" if cfg.use_class_weight else None
            logger.info(
                f"Instanciation d'un LinearSVC (C={cfg.C}, "
                f"max_iter={cfg.max_iter}, class_weight={class_weight})"
            )
            return LinearSVC(
                C=cfg.C,
                max_iter=cfg.max_iter,
                class_weight=class_weight,
            )

        if cfg.model_type == "logistic_regression":
            class_weight = "balanced" if cfg.use_class_weight else None
            logger.info(
                "Instanciation d'une LogisticRegression "
                f"(C={cfg.C}, max_iter={cfg.max_iter}, "
                f"class_weight={class_weight}, multi_class='multinomial')"
            )
            return LogisticRegression(
                C=cfg.C,
                max_iter=cfg.max_iter,
                class_weight=class_weight,
            )

        raise ValueError(
            f"Type de modèle non supporté : '{cfg.model_type}'. "
            "Actuellement 'linear_svc' et 'logistic_regression' sont gérés."
        )

    def run(self) -> Path:
        """Entraîne le modèle et l'enregistre en statut "pending" dans MLflow. Le stage Evaluation permettra de l'évaluer et de changer son statut."""
        logger.info("Démarrage de l'étape ModelTrainer")
        
        # Nom de l'expérience MLflow
        mlflow.set_experiment("train_rakuten_model_mlflow")
        
        with mlflow.start_run():
            cfg = self.config

            # 1. Charger les données d'entraînement
            logger.info(f"Chargement de X_train depuis : {cfg.X_train_path}")
            X_train = sparse.load_npz(cfg.X_train_path)
            
            logger.info(f"Chargement de y_train depuis : {cfg.y_train_path}")
            y_train = np.load(cfg.y_train_path)

            logger.debug(f"X_train shape: {X_train.shape}")
            logger.debug(f"y_train shape: {y_train.shape}")

            # Log params dans MLflow
            mlflow.log_param("model_type", cfg.model_type)
            mlflow.log_param("C", cfg.C)
            mlflow.log_param("max_iter", cfg.max_iter)
            mlflow.log_param("use_class_weight", cfg.use_class_weight)

            # 2. Construire et entraîner
            model = self._build_model()
            
            logger.info("Entraînement du modèle")
            model.fit(X_train, y_train)
            logger.success("Modèle entraîné avec succès")

            # 3. Évaluer
            logger.info("Évaluation du modèle sur le jeu d'entraînement")
            y_pred_train = model.predict(X_train)

            train_accuracy = accuracy_score(y_train, y_pred_train)
            train_f1_macro = f1_score(y_train, y_pred_train, average="macro")

            logger.info(f"Train accuracy: {train_accuracy:.4f}")
            logger.info(f"Train F1 macro: {train_f1_macro:.4f}")

            # Log metrics dans MLflow
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("train_f1_macro", train_f1_macro)

            # Rapport de classification
            cls_report = classification_report(y_train, y_pred_train)

            # 4. Créer répertoire et sauvegarder
            create_directories([cfg.model_dir])

            logger.info(f"Sauvegarde du modèle vers : {cfg.model_path}")
            with open(cfg.model_path, "wb") as f:
                pickle.dump(model, f)

            # Log artifact du vectorizer et label encoder 
            logger.info("Logging des artifacts (vectorizer, label_encoder, categories) dans MLflow...")
            
            # Log vectorizer
            mlflow.log_artifact(str(cfg.vectorizer_path), artifact_path="preprocessing")
            # Log label encoder
            mlflow.log_artifact(str(cfg.label_encoder_path), artifact_path="preprocessing")

            mlflow.log_artifact(str(cfg.class_mapping_path), artifact_path="preprocessing")
    
            
            logger.success("Artifacts loggés dans MLflow")

            # Log + Register en une seule étape avec registered_model_name
            logger.info("Logging et enregistrement du modèle dans MLflow...")
            try:
                # Log le modèle et l'enregistrer dans le registre MLflow
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=self.MODEL_REGISTRY_NAME
                )
                logger.success("Modèle loggé et enregistré dans MLflow")

                # Récupérer la version qui vient d'être créée pour permettre la transition de statut et d'incrémenter les versions de modèle
                client = mlflow.tracking.MlflowClient()
                latest_versions = client.get_latest_versions(self.MODEL_REGISTRY_NAME)
                
                if latest_versions:
                    version = latest_versions[0].version
                    logger.info(f"Version créée : v{version}")
                    
                    # Transitionner vers "pending" pour ensuite l'évaluation sur les donénes de test
                    self._set_pending_alias(version)
                else:
                    logger.warning("Impossible de récupérer la version créée")

            except mlflow.exceptions.MlflowException as e:
                logger.error(f"Erreur MLflow : {e}")
                raise
            except Exception as e:
                logger.error(f"Erreur lors du logging : {e}")
                raise

            # 7. Sauvegarder config
            model_config_path = cfg.model_dir / "model_config.json"
            logger.info(f"Sauvegarde de la configuration vers : {model_config_path}")

            model_config = {
                "model_type": cfg.model_type,
                "params": {
                    "C": cfg.C,
                    "max_iter": cfg.max_iter,
                    "use_class_weight": cfg.use_class_weight,
                },
                "training_data": {
                    "X_train_path": str(cfg.X_train_path),
                    "y_train_path": str(cfg.y_train_path),
                },
            }

            with open(model_config_path, "w") as f:
                json.dump(model_config, f, indent=2)

            # 8. Sauvegarder métriques
            metrics_path = cfg.model_dir / "metrics_train.json"
            logger.info(f"Sauvegarde des métriques vers : {metrics_path}")

            metrics = {
                "train_accuracy": train_accuracy,
                "train_f1_macro": train_f1_macro,
            }

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # 9. Sauvegarder rapport classification
            cls_report_path = cfg.model_dir / "classification_report_train.txt"
            logger.info(f"Sauvegarde du rapport vers : {cls_report_path}")


            with open(cls_report_path, "w") as f:
                f.write(cls_report)

            # Log artifacts dans MLflow
            mlflow.log_artifact(str(model_config_path))
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(cls_report_path))

            # Sauvegarder les métadata du run MLflow afin de pouvoir les réutiliser dans l'étapes Evaluation pour retrouver l'ID du run et ajouter les infos d'évaluation. Ce fichier est aussi versionné via DVC.
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Run ID: {run_id}")
            
            run_metadata = {
                "run_id": run_id,
                "model_version": str(version),
                "experiment": "train_rakuten_model_mlflow",
            }
            
            run_metadata_path = cfg.model_dir / "mlflow_run_metadata.json"
            with open(run_metadata_path, "w") as f:
                json.dump(run_metadata, f, indent=2)
            
            logger.success(f"Métadata du run sauvegardée: {run_metadata_path}")

            logger.success("ModelTrainer terminé avec succès")
            return cfg.model_path

    
    def _set_pending_alias(self, version: str) -> None:
        """
        Positione le modèle enregistré en statut "pending" dans le registre MLflow.
        """
        try:
            client = mlflow.tracking.MlflowClient()

            logger.info(f"Ajout de l'alias 'pending' à v{version}...")

            client.set_registered_model_alias(
                name=self.MODEL_REGISTRY_NAME,
                alias="pending",
                version=version
            )

            logger.success(
                f"Modèle v{version} placé en 'Pending', prêt pour l'évaluation."
            )

            # Log les infos de registre et ajout du statut dans MLflow
            mlflow.log_param("registered_version", version)
            mlflow.log_param("registered_stage", "Pending")

        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Erreur lors de la transition : {e}")
            raise