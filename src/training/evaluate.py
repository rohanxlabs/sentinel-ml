import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def compute_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    logger.info("=" * 40)
    logger.info("EVALUATION RESULTS")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 40)
    logger.info("\n" + classification_report(y_test, y_pred))

    return metrics


def check_thresholds(metrics: dict, config: dict) -> bool:
    thresholds = config["evaluation"]["thresholds"]
    passed = True

    for metric, threshold in thresholds.items():
        value = metrics.get(metric, 0)
        status = "PASS" if value >= threshold else "FAIL"
        logger.info(f"  [{status}] {metric}: {value:.4f} (min: {threshold})")
        if value < threshold:
            passed = False

    return passed


def log_metrics_to_mlflow(metrics: dict, run_id: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
    logger.info(f"Metrics logged to MLflow run: {run_id}")


def evaluate(model, X_test, y_test, run_id: str, config: dict) -> bool:
    metrics = compute_metrics(model, X_test, y_test)
    log_metrics_to_mlflow(metrics, run_id, config["mlflow"]["tracking_uri"])
    passed = check_thresholds(metrics, config)

    if passed:
        logger.info("All thresholds passed — model is ready for registration")
    else:
        logger.warning("Threshold check FAILED — model will NOT be registered")

    return passed, metrics


if __name__ == "__main__":
    import pickle

    config = load_config()

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_csv(f"{config['data']['processed_path']}/X_test.csv")
    y_test = pd.read_csv(f"{config['data']['processed_path']}/y_test.csv").squeeze()

    with open("latest_run.txt","r") as f:
        run_id = f.read().strip()

    passed, metrics = evaluate(model, X_test, y_test,run_id, config)