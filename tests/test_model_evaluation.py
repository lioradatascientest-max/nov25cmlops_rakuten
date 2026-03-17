import json
import pickle

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from mlops_rakuten.config.entities import ModelEvaluationConfig
from mlops_rakuten.modules.model_evaluation import ModelEvaluation
from mlops_rakuten.utils.utils import create_directories


def test_model_evaluation_computes_and_saves_metrics(tmp_path):
    # 1. Préparer un modèle entraîné + données de validation factices
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    create_directories([processed_dir, models_dir, metrics_dir])

    X_val_path = processed_dir / "X_val_tfidf.npz"
    y_val_path = processed_dir / "y_val.npy"
    model_path = models_dir / "text_classifier.pkl"

    # Mini dataset : 4 échantillons, 3 features
    X_data = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.9, 0.1, 0.4],
            [0.0, 1.0, 0.2],
            [0.0, 0.9, 0.1],
        ]
    )
    y_data = np.array([0, 0, 1, 1])

    X_val = sparse.csr_matrix(X_data)
    sparse.save_npz(X_val_path, X_val)
    np.save(y_val_path, y_data)

    # Entraîner un modèle simple pour le test
    model = LogisticRegression(max_iter=1000)
    model.fit(X_val, y_data)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Fichiers de sortie
    metrics_path = metrics_dir / "metrics_val.json"
    cls_report_path = metrics_dir / "classification_report_val.txt"
    cm_path = metrics_dir / "confusion_matrix_val.npy"

    cfg = ModelEvaluationConfig(
        model_path=model_path,
        X_val_path=X_val_path,
        y_val_path=y_val_path,
        metrics_path=metrics_path,
        metrics_dir=metrics_dir,
        classification_report_path=cls_report_path,
        confusion_matrix_path=cm_path,
    )

    step = ModelEvaluation(config=cfg)

    # 2. Run
    output_metrics_path = step.run()

    # 3. Vérifications
    assert output_metrics_path == metrics_path
    assert metrics_path.exists()
    assert cls_report_path.exists()
    assert cm_path.exists()

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert "val_accuracy" in metrics
    assert 0.0 <= metrics["val_accuracy"] <= 1.0

    # Matrice de confusion
    cm = np.load(cm_path)
    assert cm.shape == (2, 2)  # 2 classes
