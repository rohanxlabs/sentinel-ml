import numpy as np
import json
import logging
from collections import deque
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricWindow:
    """Rolling window for a single metric"""

    def __init__(self, name: str, window_size: int = 100):
        self.name = name
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def add(self, value: float):
        if value is not None:
            self.values.append(value)

    def mean(self) -> Optional[float]:
        if len(self.values) < 2:
            return None
        return float(np.mean(self.values))

    def std(self) -> Optional[float]:
        if len(self.values) < 2:
            return None
        return float(np.std(self.values))

    def is_anomaly(self, value: float, z_threshold: float = 3.0) -> bool:
        """Z-score based anomaly detection"""
        if len(self.values) < 10:
            return False
        mean = self.mean()
        std = self.std()
        if std == 0:
            return False
        z_score = abs((value - mean) / std)
        return z_score > z_threshold


class AnomalyDetector:

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.windows = {
            "latency_p95": MetricWindow("latency_p95", window_size),
            "error_rate": MetricWindow("error_rate", window_size),
            "fraud_rate": MetricWindow("fraud_rate", window_size),
            "request_rate": MetricWindow("request_rate", window_size),
        }

        # Hard thresholds (regardless of history)
        self.hard_thresholds = {
            "latency_p95": 2.0,       # 2 seconds max
            "error_rate": 0.05,        # 5% error rate max
            "fraud_rate": 0.10,        # 10% fraud rate spike
        }

    def update(self, metrics: dict):
        """Feed new metrics into rolling windows"""
        for key, window in self.windows.items():
            if key in metrics and metrics[key] is not None:
                window.add(metrics[key])

    def detect(self, metrics: dict) -> list:
        """Returns list of anomaly alerts"""
        alerts = []

        for metric_name, value in metrics.items():
            if value is None or metric_name not in self.windows:
                continue

            alert = None

            # Check hard threshold breach
            if metric_name in self.hard_thresholds:
                threshold = self.hard_thresholds[metric_name]
                if value > threshold:
                    alert = {
                        "type": "threshold_breach",
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "severity": "critical",
                        "timestamp": datetime.utcnow().isoformat()
                    }

            # Check statistical anomaly
            elif self.windows[metric_name].is_anomaly(value):
                window = self.windows[metric_name]
                alert = {
                    "type": "statistical_anomaly",
                    "metric": metric_name,
                    "value": value,
                    "mean": window.mean(),
                    "std": window.std(),
                    "severity": "warning",
                    "timestamp": datetime.utcnow().isoformat()
                }

            if alert:
                logger.warning(f"ANOMALY | {alert['type']} | {metric_name}: {value}")
                alerts.append(alert)

        return alerts

    def check_model_degradation(
        self,
        current_f1: float,
        baseline_f1: float,
        drop_threshold: float = 0.05
    ) -> Optional[dict]:
        """Detect model performance degradation"""
        drop = baseline_f1 - current_f1
        if drop > drop_threshold:
            alert = {
                "type": "model_degradation",
                "metric": "f1_score",
                "current": current_f1,
                "baseline": baseline_f1,
                "drop": drop,
                "threshold": drop_threshold,
                "severity": "critical",
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.warning(f"MODEL DEGRADATION | F1 dropped {drop:.4f}")
            return alert
        return None


if __name__ == "__main__":
    detector = AnomalyDetector(window_size=50)

    # Simulate normal traffic
    for i in range(30):
        detector.update({
            "latency_p95": np.random.normal(0.1, 0.01),
            "error_rate": np.random.normal(0.01, 0.002),
            "fraud_rate": np.random.normal(0.02, 0.003),
        })

    # Inject anomaly
    anomalies = detector.detect({
        "latency_p95": 3.5,    # spike above hard threshold
        "error_rate": 0.001,
        "fraud_rate": 0.15,    # fraud spike
    })

    print(json.dumps(anomalies, indent=2))