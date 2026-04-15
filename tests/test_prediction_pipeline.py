from pathlib import Path
import pickle

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from mlops_rakuten.config.entities import PredictionConfig
from mlops_rakuten.pipelines.prediction import PredictionPipeline


def test_prediction_pipeline_returns_decoded_labels(tmp_path, monkeypatch):
    # 1. Préparer des artefacts factices (vectorizer, label encoder, modèle)
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    models_dir.mkdir()

    vectorizer_path = processed_dir / "tfidf_vectorizer.pkl"
    label_encoder_path = processed_dir / "label_encoder.pkl"
    model_path = models_dir / "text_classifier.pkl"

    # Petit corpus de test
    texts = [
        "red dress with flowers",
        "blue dress with stripes",
        "latest smartphone device",
        "new mobile phone",
    ]
    # Deux codes produits fictifs
    prdtypecodes = [10, 10, 20, 20]

    # Vectorizer
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)

    # LabelEncoder sur les prdtypecode
    le = LabelEncoder()
    y = le.fit_transform(prdtypecodes)

    # Modèle linéaire simple
    # model = LinearSVC()
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X, y)

    # Sauvegarder les artefacts comme dans le vrai pipeline
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vec, f)

    with open(label_encoder_path, "wb") as f:
        pickle.dump(le, f)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # 2. Construire un InferenceConfig factice
    infer_cfg = PredictionConfig(
        vectorizer_path=vectorizer_path,
        label_encoder_path=label_encoder_path,
        model_path=model_path,
        text_column="designation",
    )

    # 3. Monkeypatch du ConfigurationManager pour le PredictionPipeline
    class DummyConfigManager:
        def __init__(self):
            pass

        def get_prediction_config(self):
            return infer_cfg

    monkeypatch.setattr(
        "mlops_rakuten.pipelines.prediction.ConfigurationManager",
        DummyConfigManager,
    )

    # 4. Instancier le pipeline de prédiction
    #monkeypatch.setattr(
    #    "mlops_rakuten.modules.prediction.Prediction._check_data_availability",
    #    lambda self: None,
    #)

    pipeline = PredictionPipeline()

    # 5. Prédire sur un texte proche de la classe "10"
    test_text = "beautiful red dress for women"
    preds = pipeline.run([test_text])

    print(preds)

    # preds = pipeline.run([test_text], top_k=2)  # ex
    assert isinstance(preds, list)
    assert len(preds) == 1

    preds_for_text = preds[0]
    assert isinstance(preds_for_text, list)
    assert len(preds_for_text) >= 1

    # Vérifier la structure des dicts
    first = preds_for_text[0]
    assert isinstance(first, dict)
    assert set(first.keys()) >= {"prdtypecode",
                                 "proba"}  # category_name optionnel
    assert isinstance(first["prdtypecode"], int)
    assert isinstance(first["proba"], float)

    # Vérifier que les codes prédits sont bien dans l’univers des labels
    #codes = [p["prdtypecode"] for p in preds_for_text]
    #assert all(c in set(prdtypecodes) for c in codes)

    # Vérifier tri décroissant des probas
    #probas = [p["proba"] for p in preds_for_text]
    #assert probas == sorted(probas, reverse=True)