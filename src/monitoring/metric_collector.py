import requests
import yaml
import logging
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def query_prometheus(query: str, prometheus_url: str) -> Optional[float]:
    """Run a PromQL query and return scalar result"""
    try:
        response = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=5
        )
        data = response.json()
        results = data.get("data", {}).get("result", [])
        if results:
            return float(results[0]["value"][1])
        return None
    except Exception as e:
        logger.warning(f"Prometheus query failed: {query} | Error: {e}")
        return None


def collect_metrics(prometheus_url: str = "http://localhost:9090") -> dict:
    metrics = {}

    queries = {
        # Request rate (last 5 min)
        "request_rate":
            "rate(api_requests_total[5m])",

        # Error rate
        "error_rate":
            "rate(api_requests_total{status_code=~'5..'}[5m])",

        # P95 latency
        "latency_p95":
            "histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[5m]))",

        # P50 latency
        "latency_p50":
            "histogram_quantile(0.50, rate(api_request_latency_seconds_bucket[5m]))",

        # Total predictions
        "total_predictions":
            "predictions_total",

        # Fraud rate (last 5 min)
        "fraud_rate":
            "rate(predictions_total{result='fraud'}[5m]) / rate(predictions_total[5m])",

        # Active requests
        "active_requests":
            "active_requests",

        # Avg fraud probability
        "avg_fraud_probability":
            "histogram_quantile(0.50, rate(fraud_probability_score_bucket[5m]))",
    }

    for name, query in queries.items():
        value = query_prometheus(query, prometheus_url)
        metrics[name] = value

    metrics["timestamp"] = datetime.utcnow().isoformat()
    metrics["prometheus_url"] = prometheus_url

    logger.info(f"Metrics collected: {len([v for v in metrics.values() if v is not None])} values")
    return metrics


def get_serving_health(serving_url: str = "http://localhost:8000") -> dict:
    """Direct health check from FastAPI — no Prometheus needed"""
    try:
        response = requests.get(f"{serving_url}/health", timeout=5)
        return response.json()
    except Exception as e:
        logger.error(f"Serving health check failed: {e}")
        return {
            "status": "unreachable",
            "model_loaded": False,
            "model_version": "unknown",
            "uptime_seconds": 0
        }


def get_full_snapshot(
    prometheus_url: str = "http://localhost:9090",
    serving_url: str = "http://localhost:8000"
) -> dict:
    """Single snapshot of all platform metrics"""
    return {
        "serving_health": get_serving_health(serving_url),
        "prometheus_metrics": collect_metrics(prometheus_url),
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import json
    snapshot = get_full_snapshot()
    print(json.dumps(snapshot, indent=2))