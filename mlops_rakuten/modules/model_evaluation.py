import json
from pathlib import Path
import time
import pickle
from typing import Optional
from loguru import logger
import numpy as np
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from mlops_rakuten.config.entities import ModelEvaluationConfig
from mlops_rakuten.utils.utils import create_directories
import os
import mlflow
import traceback


class ModelEvaluation:
    """
    Étape d'évaluation du modèle Rakuten sur le jeu de validation. 
    Le modèle est couplé a MLflow pour le suivi d'expérimentations et le registre de modèles.
    """

    MODEL_REGISTRY_NAME = "text_classifier_tfidf_m"
    F1_THRESHOLD_IMPROVEMENT = 0.01

    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config
        self._setup_mlflow()

    def _setup_mlflow(self):
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

    def run(self) -> Path:
        """
        Évalue le modèle sur le jeu de validation et sauvegarde les métriques 
        dans des fichiers et dans MLflow.
        """
        logger.info("Démarrage de l'étape ModelEvaluation")

        cfg = self.config

        #charger les données de validation
        logger.info(f"Chargement de X_val depuis : {cfg.X_val_path}")
        X_val = sparse.load_npz(cfg.X_val_path)

        logger.info(f"Chargement de y_val depuis : {cfg.y_val_path}")
        y_val = np.load(cfg.y_val_path)

        logger.debug(f"X_val shape: {X_val.shape}")
        logger.debug(f"y_val shape: {y_val.shape}")

        #charger le modèle
        model = self._load_model()

        # Prédiction sur le jeu de validation
        logger.info("Prédiction sur le jeu de validation")
        y_pred = model.predict(X_val)

        # Calcul des métriques
        val_accuracy = accuracy_score(y_val, y_pred)
        val_f1_macro = f1_score(y_val, y_pred, average="macro")
        val_f1_weighted = f1_score(y_val, y_pred, average="weighted")

        logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation F1 macro: {val_f1_macro:.4f}")
        logger.info(f"Validation F1 weighted: {val_f1_weighted:.4f}")

        cls_report = classification_report(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        create_directories([cfg.metrics_dir])

        # sauvegarder les métriques dans des fichiers
        metrics = {
            "val_accuracy": val_accuracy,
            "val_f1_macro": val_f1_macro,
            "val_f1_weighted": val_f1_weighted,
        }

        logger.info(f"Sauvegarde des métriques de validation vers : {cfg.metrics_path}")
        with open(cfg.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(
            "Sauvegarde du rapport de classification (validation) "
            f"vers : {cfg.classification_report_path}"
        )
        with open(cfg.classification_report_path, "w") as f:
            f.write(cls_report)

        logger.info(
            f"Sauvegarde de la matrice de confusion (validation) vers : {cfg.confusion_matrix_path}"
        )
        with open(cfg.confusion_matrix_path, "w") as f:
            f.write(str(cm))

        # metriques dans MLflow
        if self.mlflow_enabled:
            # Récupérer le run_id du training
            train_run_id = self._get_train_run_id(cfg.model_path.parent)
            
            if train_run_id:
                logger.info(f"Logging des métriques dans le run du training: {train_run_id}")
                try:
                    # Log dans le même run que le training pour garder l'historique complet
                    with mlflow.start_run(run_id=train_run_id):
                        mlflow.log_metric("val_accuracy", val_accuracy)
                        mlflow.log_metric("val_f1_macro", val_f1_macro)
                        mlflow.log_metric("val_f1_weighted", val_f1_weighted)
                        
                        mlflow.log_artifact(str(cfg.metrics_path))
                        mlflow.log_artifact(str(cfg.classification_report_path))
                        
                        logger.success(f"Métriques loggées dans le run {train_run_id}")
                except Exception as e:
                    logger.warning(f"Impossible de continuer le run: {e}")
                    self._log_eval_in_new_run(val_accuracy, val_f1_macro, val_f1_weighted, cfg)
            else:
                logger.warning("Run_id du training non trouvé, création d'un nouveau run")
                self._log_eval_in_new_run(val_accuracy, val_f1_macro, val_f1_weighted, cfg)

            # Gérer la promotion d'alias
            self._manage_alias_promotion(val_f1_macro)
        else:
            logger.warning("MLflow non configuré, vérifier les variables d'environnement")

        logger.success("ModelEvaluation terminée avec succès")
        return cfg.metrics_path

    def _load_model(self):
        """
        Charge UNIQUEMENT depuis MLflow avec alias 'pending'.
        Lève une exception si ça fail.
        """
        if not self.mlflow_enabled:
            raise RuntimeError(
                "❌ MLflow n'est pas configuré."
            )
        
        try:
            logger.info(f"Chargement du modèle depuis MLflow (alias: 'pending')...")
            import time
            time.sleep(10)  # Attendre la propagation
            
            model_uri = f"models:/{self.MODEL_REGISTRY_NAME}@pending"
            model = mlflow.sklearn.load_model(model_uri)
            logger.success(f"✅ Modèle chargé")
            return model
        
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            raise RuntimeError(
                f"Modèle non trouvé. Assure-toi que ModelTrainer a roulé d'abord."
            ) from e


    def _get_train_run_id(self, model_dir: Path) -> Optional[str]:
        """Récupère le run_id depuis le fichier métadata du training"""
        metadata_path = model_dir / "mlflow_run_metadata.json"
        
        if not metadata_path.exists():
            logger.debug(f"Fichier métadata non trouvé: {metadata_path}")
            return None
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                run_id = metadata.get("run_id")
                if run_id:
                    logger.info(f"Run ID du training trouvé: {run_id}")
                return run_id
        except Exception as e:
            logger.warning(f"Erreur lecture métadata: {e}")
            return None

    def _log_eval_in_new_run(
        self, 
        val_accuracy: float,
        val_f1_macro: float,
        val_f1_weighted: float,
        cfg
    ) -> None:
        """
        Log les métriques d'évaluation dans un nouveau run MLflow 
        si le run du training n'est pas accessible.
        """
        try:
            mlflow.set_experiment("model_evaluation")
            with mlflow.start_run():
                mlflow.log_metric("val_accuracy", val_accuracy)
                mlflow.log_metric("val_f1_macro", val_f1_macro)
                mlflow.log_metric("val_f1_weighted", val_f1_weighted)
                
                mlflow.log_artifact(str(cfg.metrics_path))
                mlflow.log_artifact(str(cfg.classification_report_path))
                
                logger.success("Métriques loggées dans un nouvel experiment")
        except Exception as e:
            logger.warning(f"Impossible de logger: {e}")

    def _manage_alias_promotion(self, val_f1_macro: float) -> None:
        """
        Gère la promotion des alias dans le registre MLflow Model Registry
        Aliases utilisées:
        - "pending" = Nouvellement entraîné, en attente d'évaluation
        - "production" = En production, meilleur modèle
        - "archived" = Retiré, ancien production
        """
        try:
            client = mlflow.tracking.MlflowClient()

            # ═══════════════════════════════════════════════════════════════
            # ÉTAPE 1: Récupérer la version avec alias "pending"
            # ═══════════════════════════════════════════════════════════════
            logger.info("Recherche version avec alias 'pending'...")
            pending_version = None
            
            try:
                pending_version = client.get_model_version_by_alias(
                    name=self.MODEL_REGISTRY_NAME,
                    alias="pending"
                )
                logger.info(f"Version en attente trouvée : v{pending_version.version}")
            except mlflow.exceptions.MlflowException as e:
                logger.warning(f"Aucune version avec alias 'pending'")
                logger.debug(f"Détail erreur: {e}")
                return
            except Exception as e:
                logger.error(f"Erreur inattendue lors de la recherche 'pending': {e}")
                return

            if pending_version is None:
                logger.warning("Impossible de récupérer la version 'pending'")
                return

            pending_version_num = pending_version.version

            # ═══════════════════════════════════════════════════════════════
            # ÉTAPE 2: Récupérer la version avec alias "production"
            # ═══════════════════════════════════════════════════════════════
            logger.info("Recherche version avec alias 'production'...")
            production_version = None
            
            try:
                production_version = client.get_model_version_by_alias(
                    name=self.MODEL_REGISTRY_NAME,
                    alias="production"
                )
                logger.info(f"Version production trouvée : v{production_version.version}")
            except mlflow.exceptions.MlflowException as e:
                logger.info(f"Aucune version avec alias 'production' (attendu au premier run)")
                logger.debug(f"Détail: {e}")
                production_version = None
            except Exception as e:
                logger.warning(f"Erreur inattendue lors de la recherche 'production': {e}")
                production_version = None

            # ═══════════════════════════════════════════════════════════════
            # CAS 1: Pas de version en production
            # ═══════════════════════════════════════════════════════════════
            if production_version is None:
                logger.info(
                    f"Aucune version en production (alias 'production')\n"
                    f"Promotion de v{pending_version_num} : alias 'production'"
                )

                try:
                    # Ajouter alias "production"
                    client.set_registered_model_alias(
                        name=self.MODEL_REGISTRY_NAME,
                        alias="production",
                        version=str(pending_version_num)
                    )
                    logger.success(
                        f"v{pending_version_num} : alias 'production' "
                        f"(F1: {val_f1_macro:.4f})"
                    )

                    # Supprimer l'alias "pending" pour éviter les confusions
                    try:
                        client.delete_registered_model_alias(
                            name=self.MODEL_REGISTRY_NAME,
                            alias="pending"
                        )
                        logger.success(f"Suppression de l'alias 'pending'")
                    except Exception as e:
                        logger.warning(f"Impossible de supprimer alias 'pending': {e}")
                
                except mlflow.exceptions.MlflowException as e:
                    logger.error(f"Erreur MLflow lors de la promotion : {e}")
                    return
                except Exception as e:
                    logger.error(f"Erreur inattendue lors de la promotion : {e}")
                    return

                return

            # ═══════════════════════════════════════════════════════════════
            # CAS 2: Une version existe en production → Comparer les métriques
            # ═══════════════════════════════════════════════════════════════
            production_version_num = production_version.version
            logger.info(f"Version production existante : v{production_version_num}")

            # Charger les métriques de la version en production
            prod_metrics = self._load_metrics_from_registry(production_version)

            if prod_metrics is None:
                logger.warning(
                    f"Impossible de charger métriques de v{production_version_num}\n"
                    f"Skipping promotion"
                )
                return

            prod_f1_macro = prod_metrics.get("val_f1_macro", 0.0)

            logger.info(
                f"Comparaison F1:\n"
                f"  Production (v{production_version_num}): {prod_f1_macro:.4f}\n"
                f"  Pending (v{pending_version_num}):    {val_f1_macro:.4f}"
            )

            # Décision sur un seuil d'amélioration
            f1_improvement = val_f1_macro - prod_f1_macro

            # ═══════════════════════════════════════════════════════════════
            # DÉCISION: Amélioration suffisante?
            # ═══════════════════════════════════════════════════════════════
            if f1_improvement >= self.F1_THRESHOLD_IMPROVEMENT:
                logger.info(
                    f"Amélioration détectée (+{f1_improvement:.4f})\n"
                    f"Promotion v{pending_version_num} : alias 'production'"
                )

                try:
                    # Promouvoir pending vers production
                    client.set_registered_model_alias(
                        name=self.MODEL_REGISTRY_NAME,
                        alias="production",
                        version=str(pending_version_num)
                    )
                    logger.success(f"v{pending_version_num} : alias 'production'")

                    # Supprimer l'alias "pending"
                    try:
                        client.delete_registered_model_alias(
                            name=self.MODEL_REGISTRY_NAME,
                            alias="pending"
                        )
                        logger.success(f"Suppression de l'alias 'pending'")
                    except Exception as e:
                        logger.warning(f"Impossible de supprimer alias 'pending': {e}")
                    
                    # Ajouter alias "archived" à l'ancienne version
                    try:
                        client.set_registered_model_alias(
                            name=self.MODEL_REGISTRY_NAME,
                            alias="archived",
                            version=str(production_version_num)
                        )
                        logger.success(f"v{production_version_num} : alias 'archived'")
                    except Exception as e:
                        logger.warning(
                            f"Impossible d'ajouter alias 'archived' à v{production_version_num}: {e}"
                        )
                
                except mlflow.exceptions.MlflowException as e:
                    logger.error(f"Erreur MLflow lors de la promotion : {e}")
                    return
                except Exception as e:
                    logger.error(f"Erreur inattendue lors de la promotion : {e}")
                    return

            else:
                # Pas d'amélioration suffisante
                logger.warning(
                    f"Pas d'amélioration suffisante ({f1_improvement:.4f} < {self.F1_THRESHOLD_IMPROVEMENT:.4f})\n"
                    f"v{pending_version_num} reste avec alias 'pending'\n"
                    f"v{production_version_num} reste avec alias 'production'"
                )

        except Exception as e:
            logger.error(f"Erreur gestion aliases : {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")


    def _load_metrics_from_registry(self, model_version) -> Optional[dict]:
        """Charge les métriques depuis le run de training du modèle"""
        try:
            client = mlflow.tracking.MlflowClient()

            # Récupérer le run qui a créé cette version
            run_id = model_version.run_id
            logger.debug(f"Récupération des métriques depuis le run: {run_id}")

            try:
                run = client.get_run(run_id)
            except Exception as e:
                logger.warning(f"Impossible de charger le run {run_id}: {e}")
                return None

            # Les métriques sont dans run.data.metrics
            metrics = {}

            # Chercher les métriques de validation commençant par "val_"
            for key, value in run.data.metrics.items():
                if key.startswith("val_"):
                    try:
                        metrics[key] = float(value)
                    except (ValueError, TypeError):
                        pass

            if metrics:
                logger.info(f"Métriques chargées depuis le run: {metrics}")
                return metrics

            logger.warning(
                f"Pas de métriques 'val_*' trouvées pour run {run_id}\n"
                f"Métriques disponibles: {list(run.data.metrics.keys())}"
            )
            return None

        except Exception as e:
            logger.warning(f"Erreur chargement métriques : {e}")
            return None