import json
from pathlib import Path
import pickle

import numpy as np
from scipy import sparse

from mlops_rakuten.config.entities import ModelTrainerConfig
from mlops_rakuten.modules.model_trainer import ModelTrainer


def test_model_trainer_trains_and_saves_artifacts(tmp_path):
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    models_dir.mkdir()

    X_train_path = processed_dir / "X_train_tfidf.npz"
    y_train_path = processed_dir / "y_train.npy"
    model_path = models_dir / "text_classifier.pkl"
    
    vectorizer_path = processed_dir / "tfidf_vectorizer.pkl"
    label_encoder_path = processed_dir / "label_encoder.pkl"
    class_mapping_path = processed_dir / "class_mapping.json"

    X_data = np.array(
        [
            [1.0, 0.0, 0.5, 0.0],
            [0.9, 0.1, 0.4, 0.0],
            [0.0, 1.0, 0.2, 0.3],
            [0.0, 0.9, 0.1, 0.4],
            [0.8, 0.0, 0.6, 0.0],
            [0.0, 1.1, 0.0, 0.5],
        ]
    )
    y_data = np.array([0, 0, 1, 1, 0, 1])

    X_train = sparse.csr_matrix(X_data)
    sparse.save_npz(X_train_path, X_train)
    np.save(y_train_path, y_data)

    cfg = ModelTrainerConfig(
        X_train_path=X_train_path,
        y_train_path=y_train_path,
        model_dir=models_dir,
        model_path=model_path,
        vectorizer_path=vectorizer_path,
        label_encoder_path=label_encoder_path,
        class_mapping_path=class_mapping_path,
        model_type="logistic_regression",
        C=1.0,
        max_iter=1000,
        use_class_weight=False,
    )
    
    # Créer des fichiers factices pour satisfaire le logging MLflow
    with open(vectorizer_path, "wb") as f:
        pickle.dump("dummy_vectorizer", f)
    with open(label_encoder_path, "wb") as f:
        pickle.dump("dummy_le", f)
    with open(class_mapping_path, "w") as f:
        json.dump({"dummy": 0}, f)

    step = ModelTrainer(config=cfg)
    output_model_path = step.run()

    assert output_model_path == model_path
    assert model_path.exists()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    assert hasattr(model, "predict")

    # Bonus utile si ton API dépend des proba
    assert hasattr(model, "predict_proba")

    model_config_path = models_dir / "model_config.json"
    assert model_config_path.exists()
    model_config = json.loads(model_config_path.read_text())

    assert model_config["model_type"] == "logistic_regression"
    assert model_config["params"]["C"] == 1.0
    assert model_config["params"]["use_class_weight"] is False

    metrics_path = models_dir / "metrics_train.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())

    assert "train_accuracy" in metrics
    assert "train_f1_macro" in metrics
    assert 0.0 <= metrics["train_accuracy"] <= 1.0
    assert 0.0 <= metrics["train_f1_macro"] <= 1.0

    cls_report_path = models_dir / "classification_report_train.txt"
    assert cls_report_path.exists()
    assert "precision" in cls_report_path.read_text()