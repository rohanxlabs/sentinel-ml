from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
import time
import logging

logger = logging.getLogger(__name__)

# --- Prometheus Metrics ---

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions made",
    ["result"]   # fraud or legitimate
)

FRAUD_PROBABILITY = Histogram(
    "fraud_probability_score",
    "Distribution of fraud probability scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

MODEL_VERSION = Gauge(
    "model_version_info",
    "Current model version in serving",
    ["version"]
)

ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of requests currently being processed"
)


async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    ACTIVE_REQUESTS.inc()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        raise e
    finally:
        duration = time.time() - start_time
        endpoint = request.url.path

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()

        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        ACTIVE_REQUESTS.dec()

    return response


def track_prediction(prediction: int, probability: float):
    label = "fraud" if prediction == 1 else "legitimate"
    PREDICTION_COUNT.labels(result=label).inc()
    FRAUD_PROBABILITY.observe(probability)


def set_model_version(version: str):
    MODEL_VERSION.labels(version=version).set(1)


def get_metrics_response():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )