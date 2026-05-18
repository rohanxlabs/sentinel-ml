import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric
import yaml
import logging
import json
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_reference_data(config: dict) -> pd.DataFrame:
    """Training data is our reference (baseline)"""
    path = f"{config['data']['processed_path']}/X_train.csv"
    df = pd.read_csv(path)
    logger.info(f"Reference data loaded | shape: {df.shape}")
    return df


def load_current_data(path: str) -> pd.DataFrame:
    """Current production data or drift sample"""
    df = pd.read_csv(path)
    logger.info(f"Current data loaded | shape: {df.shape}")
    return df


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, config: dict) -> dict:
    threshold = config["monitoring"]["drift_threshold"]

    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftPreset(),
        DataQualityPreset()
    ])

    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()

    # Extract key drift info
    drift_summary = result["metrics"][0]["result"]
    drifted_features = []

    # Find which features drifted
    for metric in result["metrics"]:
        if "result" in metric and "drift_by_columns" in metric["result"]:
            for col, info in metric["result"]["drift_by_columns"].items():
                if info.get("drift_detected", False):
                    drifted_features.append({
                        "feature": col,
                        "drift_score": info.get("stattest_threshold", 0),
                        "p_value": info.get("p_value", None)
                    })

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "drift_detected": drift_summary.get("dataset_drift", False),
        "drift_share": drift_summary.get("share_of_drifted_columns", 0.0),
        "drifted_features_count": drift_summary.get("number_of_drifted_columns", 0),
        "total_features": drift_summary.get("number_of_columns", 0),
        "drifted_features": drifted_features,
        "threshold": threshold
    }

    logger.info(f"Drift detected: {output['drift_detected']} | "
                f"Drifted features: {output['drifted_features_count']}/{output['total_features']}")

    # Save report
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report.save_html(f"reports/drift_report_{timestamp}.html")

    return output


def run_drift_check(current_data_path: str = None) -> dict:
    config = load_config()
    reference = load_reference_data(config)

    if current_data_path is None:
        # Default to amount drift sample for testing
        current_data_path = f"{config['data']['drift_samples_path']}/drift_amount.csv"
        current = pd.read_csv(current_data_path)
        # Align columns with reference
        current = current.drop(columns=["Class"], errors="ignore")
        current = current[[c for c in reference.columns if c in current.columns]]
    else:
        current = load_current_data(current_data_path)

    # Align columns
    common_cols = [c for c in reference.columns if c in current.columns]
    reference = reference[common_cols]
    current = current[common_cols]

    return detect_drift(reference, current, config)


if __name__ == "__main__":
    result = run_drift_check()
    print(json.dumps(result, indent=2))