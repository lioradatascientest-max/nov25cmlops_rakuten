from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlops_rakuten.monitoring.prometheus_metrics import (
    configure_metrics,
    observe_prediction_confidence,
)


def create_test_app() -> FastAPI:
    app = FastAPI()
    configure_metrics(app)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/predict")
    def predict():
        observe_prediction_confidence(0.87, "42")
        return {"status": "ok"}

    return app


def test_metrics_endpoint_exposes_http_metrics():
    client = TestClient(create_test_app())

    response = client.get("/health")
    assert response.status_code == 200

    metrics_response = client.get("/metrics")
    body = metrics_response.text

    assert metrics_response.status_code == 200
    assert 'api_requests_total{endpoint="/health",method="GET",status_code="200"}' in body
    assert 'api_request_duration_seconds_count{endpoint="/health",method="GET",status_code="200"}' in body


def test_metrics_endpoint_exposes_model_confidence_metric():
    client = TestClient(create_test_app())

    response = client.get("/predict")
    assert response.status_code == 200

    metrics_response = client.get("/metrics")
    body = metrics_response.text

    assert metrics_response.status_code == 200
    assert 'model_prediction_confidence_bucket{le="0.9",model_version="42"}' in body
