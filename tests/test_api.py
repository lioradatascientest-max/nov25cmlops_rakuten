from fastapi.testclient import TestClient
import pytest

#import mlops_rakuten.api as api
#import mlops_rakuten.main as api
import mlops_rakuten.services.gateway_app as api

import pytest
pytestmark = pytest.mark.skip(reason="Correction du nom de la variable pipeline en cours")

def override_require_user():
    return {"username": "test", "role": "user"}


class DummyPredictionPipeline:
    def run(self, texts, top_k=None):
        base = [
            {"prdtypecode": 10, "category_name": "Cat A", "proba": 0.7},
            {"prdtypecode": 20, "category_name": "Cat B", "proba": 0.2},
            {"prdtypecode": 30, "category_name": "Cat C", "proba": 0.1},
        ]
        if top_k is not None:
            base = base[:top_k]
        return [base]


@pytest.fixture(autouse=True)
def _override_auth():
    api.app.dependency_overrides[api.require_user] = override_require_user
    yield
    api.app.dependency_overrides.clear()


def test_health_returns_ok(monkeypatch):
    monkeypatch.setattr(api, "prediction_pipeline", DummyPredictionPipeline())
    client = TestClient(api.app)

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_predictions(monkeypatch):
    monkeypatch.setattr(api, "prediction_pipeline", DummyPredictionPipeline())
    client = TestClient(api.app)

    payload = {
        "designation": "Une designation produit suffisamment longue", "top_k": 2}
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    assert data["designation"] == payload["designation"]
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 2

    first = data["predictions"][0]
    assert set(first.keys()) == {"prdtypecode", "category_name", "proba"}
    assert isinstance(first["prdtypecode"], int)
    assert isinstance(first["category_name"], str)
    assert isinstance(first["proba"], float)


def test_predict_min_length_validation(monkeypatch):
    monkeypatch.setattr(api, "prediction_pipeline", DummyPredictionPipeline())
    client = TestClient(api.app)

    resp = client.post("/predict", json={"designation": "short", "top_k": 3})
    assert resp.status_code == 422


def test_predict_default_top_k(monkeypatch):
    monkeypatch.setattr(api, "prediction_pipeline", DummyPredictionPipeline())
    client = TestClient(api.app)

    resp = client.post(
        "/predict", json={"designation": "Une designation produit suffisamment longue"})
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["predictions"]) == 3