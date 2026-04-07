from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, REGISTRY, make_asgi_app


def _get_or_create_counter(name: str, documentation: str, labelnames: tuple[str, ...]) -> Counter:
    collector = REGISTRY._names_to_collectors.get(name)
    if collector is not None:
        return collector
    return Counter(name, documentation, labelnames=labelnames)


def _get_or_create_histogram(
    name: str,
    documentation: str,
    labelnames: tuple[str, ...],
    buckets: tuple[float, ...] | None = None,
) -> Histogram:
    collector = REGISTRY._names_to_collectors.get(name)
    if collector is not None:
        return collector
    kwargs: dict[str, Any] = {"labelnames": labelnames}
    if buckets is not None:
        kwargs["buckets"] = buckets
    return Histogram(name, documentation, **kwargs)


API_REQUESTS_TOTAL = _get_or_create_counter(
    "api_requests_total",
    "Total number of HTTP requests handled by the API.",
    ("endpoint", "method", "status_code"),
)

API_REQUEST_DURATION_SECONDS = _get_or_create_histogram(
    "api_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("endpoint", "method", "status_code"),
)

MODEL_PREDICTION_CONFIDENCE = _get_or_create_histogram(
    "model_prediction_confidence",
    "Top-1 confidence score returned by the prediction model.",
    ("model_version",),
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)


def _normalize_endpoint(request: Request) -> str:
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if route_path:
        return route_path
    return request.url.path


def configure_metrics(app: FastAPI) -> None:
    if getattr(app.state, "metrics_enabled", False):
        return

    @app.middleware("http")
    async def prometheus_http_metrics(request: Request, call_next):
        start = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            endpoint = _normalize_endpoint(request)
            if endpoint != "/metrics":
                labels = {
                    "endpoint": endpoint,
                    "method": request.method,
                    "status_code": str(status_code),
                }
                elapsed = time.perf_counter() - start
                API_REQUESTS_TOTAL.labels(**labels).inc()
                API_REQUEST_DURATION_SECONDS.labels(**labels).observe(elapsed)

    app.mount("/metrics", make_asgi_app())
    app.state.metrics_enabled = True


def observe_prediction_confidence(confidence: float, model_version: str | None) -> None:
    MODEL_PREDICTION_CONFIDENCE.labels(
        model_version=str(model_version or "unknown")
    ).observe(confidence)
