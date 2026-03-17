# mlops_rakuten/config_manager.py
from datetime import datetime
from pathlib import Path

from loguru import logger
import yaml

from mlops_rakuten.config.constants import (
    CONFIG_FILE_PATH,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAKUTEN_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SEEDS_DATA_DIR,
)
from mlops_rakuten.config.entities import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    DataSeedingConfig,
    DataTransformationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    PredictionConfig,
)
from mlops_rakuten.utils.utils import create_directories


class ConfigurationManager:
    """
    Centralise la lecture du fichier config.yml
    et expose des méthodes pour construire les objets de config.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = CONFIG_FILE_PATH

        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)
            

    def get_data_seeding_config(self) -> DataSeedingConfig:
        """
        Construit un DataSeedingConfig à partir de la section
        'data_seeding' du YAML, en utilisant avec RAKUTEN_DATA_DIR.
        """
        c = self._config["data_seeding"]

        x_path = RAKUTEN_DATA_DIR / c["x_train_filename"]
        y_path = RAKUTEN_DATA_DIR / c["y_train_filename"]
        full_path = SEEDS_DATA_DIR / c["output_full_dataset_filename"]
        remainder_path = SEEDS_DATA_DIR / c["output_remainder_dataset_filename"]
        dataset_path = INTERIM_DATA_DIR / c["output_dataset_filename"]

        logger.debug(f"x_train_path resolved to: {x_path}")
        logger.debug(f"y_train_path resolved to: {y_path}")
        logger.debug(f"output_full_path resolved to: {full_path}")
        logger.debug(f"output_remainder_path resolved to: {remainder_path}")
        logger.debug(f"output_dataset_path resolved to: {dataset_path}")

        return DataSeedingConfig(
            x_train_path=x_path,
            y_train_path=y_path,
            output_full_path=full_path,
            output_remainder_path=remainder_path,
            output_dataset_path=dataset_path,
            seeds_dir=SEEDS_DATA_DIR,
            text_column=c["text_column"],
            target_column=c["target_column"],
            batch_size=c["batch_size"],
            n_batches=c["n_batches"],
        )

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Construit un DataIngestionConfig à partir de la section
        'data_ingestion' du YAML, en combinant avec RAW_DATA_DIR / INTERIM_DATA_DIR.
        """
        c = self._config["data_ingestion"]

        train_path = INTERIM_DATA_DIR / c["train_filename"]

        logger.debug(f"train_path resolved to: {train_path}")

        return DataIngestionConfig(
            train_path=train_path,
            text_column=c["text_column"],
            target_column=c["target_column"],
        )

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        """
        Construit un DataPreprocessingConfig à partir de la section
        'data_preprocessing' du YAML, en combinant avec INTERIM_DATA_DIR.
        """
        c = self._config["data_preprocessing"]

        input_path = INTERIM_DATA_DIR / c["input_dataset_filename"]

        output_dir = INTERIM_DATA_DIR 
        create_directories([output_dir])
        output_path = output_dir / c["output_dataset_filename"]

        logger.debug(f"input_dataset_path resolved to: {input_path}")
        logger.debug(f"output_dataset_path resolved to: {output_path}")

        return DataPreprocessingConfig(
            input_dataset_path=input_path,
            output_dataset_path=output_path,
            text_column=c["text_column"],
            target_column=c["target_column"],
            drop_na_text=c.get("drop_na_text", True),
            drop_na_target=c.get("drop_na_target", True),
            drop_duplicates=c.get("drop_duplicates", True),
            min_char_length=c.get("min_char_length", 10),
            max_char_length=c.get("max_char_length", 1000),
            min_alpha_ratio=c.get("min_alpha_ratio", 0.2),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Construit un DataTransformationConfig à partir de la section
        'data_transformation' du YAML, en combinant avec INTERIM_DATA_DIR /
        PROCESSED_DATA_DIR.

        Cette configuration permet de :
        - charger le dataset prétraité (CSV)
        - appliquer le split train/validation
        - vectoriser le texte avec TF-IDF
        - encoder la cible avec un LabelEncoder
        - sauvegarder tous les artefacts dans data/processed/
        """
        c = self._config["data_transformation"]

        # Trouver le dernier run de preprocessing
        logger.info(f"Recherche du dernier preprocessing dans : {INTERIM_DATA_DIR}")
        latest_preproc_dir =INTERIM_DATA_DIR

        input_path = latest_preproc_dir / c["input_dataset_filename"]
        if not input_path.exists():
            logger.error(f"Fichier manquant : {input_path}")
            raise FileNotFoundError(f"{input_path} introuvable")
        output_dir = PROCESSED_DATA_DIR 
        create_directories([output_dir])

        return DataTransformationConfig(
            input_dataset_path=input_path,
            output_dir=output_dir,
            test_size=c["test_size"],
            random_state=c["random_state"],
            stratify=c["stratify"],
            max_features=c["max_features"],
            ngram_min=c["ngram_min"],
            ngram_max=c["ngram_max"],
            lowercase=c["lowercase"],
            stop_words=c["stop_words"],
            vectorizer_path=output_dir / c["vectorizer_filename"],
            label_encoder_path=output_dir / c["label_encoder_filename"],
            class_mapping_path=output_dir / c["class_mapping_filename"],
            X_train_path=output_dir / c["X_train_filename"],
            X_val_path=output_dir / c["X_val_filename"],
            y_train_path=output_dir / c["y_train_filename"],
            y_val_path=output_dir / c["y_val_filename"],
            text_column=c["text_column"],
            target_column=c["target_column"],
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Construit un ModelTrainerConfig à partir de la section
        'model_trainer' du YAML, en combinant avec PROCESSED_DATA_DIR / MODELS_DIR.

        Cette configuration permet de :
        - charger X_train / y_train transformés
        - instancier le modèle sklearn (LinearSVC, etc.)
        - sauvegarder le modèle entraîné et ses hyperparamètres.
        """
        c = self._config["model_trainer"]

        # Trouver le dernier run de transformation
        logger.info(f"Recherche de la dernière transformation dans : {PROCESSED_DATA_DIR}")
        latest_transformation_dir = PROCESSED_DATA_DIR

        X_train_path = latest_transformation_dir / c["X_train_filename"]
        y_train_path = latest_transformation_dir / c["y_train_filename"]
        data_transform_cfg = self.get_data_transformation_config()
        vectorizer_path = data_transform_cfg.vectorizer_path
        label_encoder_path = data_transform_cfg.label_encoder_path
        class_mapping_path = data_transform_cfg.class_mapping_path

        # Créer le répertoire dans modèle
        model_dir = MODELS_DIR 
        create_directories([model_dir])
        model_path = model_dir / c["model_filename"]

        logger.debug(f"X_train_path resolved to: {X_train_path}")
        logger.debug(f"y_train_path resolved to: {y_train_path}")
        logger.debug(f"vectorizer_path resolved to: {vectorizer_path}")
        logger.debug(f"label_encoder_path resolved to: {label_encoder_path}")
        logger.debug(f"class_mapping_path resolved to: {class_mapping_path}")
        logger.debug(f"model_path resolved to: {model_path}")
        logger.debug(f"model_dir resolved to: {model_dir}")

        return ModelTrainerConfig(
            model_path=model_path,
            model_dir=model_dir,
            X_train_path=X_train_path,
            y_train_path=y_train_path,
            model_type=c["model_type"],
            vectorizer_path=vectorizer_path,
            label_encoder_path=label_encoder_path,
            class_mapping_path=class_mapping_path,
            C=c["C"],
            max_iter=c["max_iter"],
            use_class_weight=c["use_class_weight"],
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Construit un ModelEvaluationConfig à partir de la section
        'model_evaluation' du YAML, en combinant avec PROCESSED_DATA_DIR / MODELS_DIR.

        Cette configuration permet de :
        - charger X_val / y_val transformés
        - charger le modèle entraîné
        - sauvegarder les métriques de validation et les rapports associés.
        """
        c = self._config["model_evaluation"]

        # Trouver le dernier run de transformation
        logger.info(f"Recherche de la dernière transformation dans : {PROCESSED_DATA_DIR}")
        latest_transformation_dir = PROCESSED_DATA_DIR

        X_val_path = latest_transformation_dir / c["X_val_filename"]
        y_val_path = latest_transformation_dir / c["y_val_filename"]

        # Trouver le dernier modèle
        logger.info(f"Recherche du dernier modèle dans : {MODELS_DIR}")
        latest_model_dir = MODELS_DIR

        model_path = latest_model_dir / c["model_filename"]

        # Créer le répertoire dans modèle
        metrics_dir = REPORTS_DIR 
        create_directories([metrics_dir])
        metrics_path = metrics_dir / c["metrics_filename"]
        classification_report_path = metrics_dir / c["classification_report_filename"]
        confusion_matrix_path = metrics_dir / c["confusion_matrix_filename"]

        logger.debug(f"X_val_path resolved to: {X_val_path}")
        logger.debug(f"y_val_path resolved to: {y_val_path}")
        logger.debug(f"model_path resolved to: {model_path}")
        logger.debug(f"metrics_path resolved to: {metrics_path}")
        logger.debug(f"classification_report_path resolved to: {classification_report_path}")
        logger.debug(f"confusion_matrix_path resolved to: {confusion_matrix_path}")

        return ModelEvaluationConfig(
            model_path=model_path,
            X_val_path=X_val_path,
            y_val_path=y_val_path,
            metrics_path=metrics_path,
            metrics_dir=metrics_dir,
            classification_report_path=classification_report_path,
            confusion_matrix_path=confusion_matrix_path,
        )

    def get_prediction_config(self) -> PredictionConfig:
        """
        Construit un PredictionConfig à partir de la section
        'prediction' du YAML, en combinant avec PROCESSED_DATA_DIR / MODELS_DIR / RAW_DATA_DIR.

        Cette configuration permet de :
        - recharger le vectorizer TF-IDF
        - recharger le LabelEncoder
        - recharger le modèle entraîné
        """
        c = self._config["prediction"]

        # Trouver le dernier run de transformation
        logger.info(f"Recherche de la dernière transformation dans : {PROCESSED_DATA_DIR}")
        latest_transformation_dir = PROCESSED_DATA_DIR

        vectorizer_path = latest_transformation_dir / c["vectorizer_filename"]
        label_encoder_path = latest_transformation_dir / c["label_encoder_filename"]

        # Trouver le dernier modèle
        logger.info(f"Recherche du dernier modèle dans : {MODELS_DIR}")
        latest_model_dir = MODELS_DIR

        model_path = latest_model_dir / c["model_filename"]

        categories_path = None
        code_col = None
        name_col = None
        if "categories_filename" in c:
            categories_path = RAW_DATA_DIR / c["categories_filename"]
            code_col = c.get("category_code_column", "prdtypecode")
            name_col = c.get("category_name_column", "category_name")

        logger.debug(f"vectorizer_path resolved to: {vectorizer_path}")
        logger.debug(f"label_encoder_path resolved to: {label_encoder_path}")
        logger.debug(f"model_path resolved to: {model_path}")
        logger.debug(f"categories_path resolved to: {categories_path}")

        return PredictionConfig(
            vectorizer_path=vectorizer_path,
            label_encoder_path=label_encoder_path,
            model_path=model_path,
            text_column=c["text_column"],
            categories_path=categories_path,
            category_code_column=code_col,
            category_name_column=name_col,
        )
