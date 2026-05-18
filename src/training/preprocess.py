import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import yaml
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def validate_data(df: pd.DataFrame):
    required_cols = ["Time", "Amount", "Class"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    assert df.isnull().sum().sum() == 0, "Dataset has null values"
    assert df["Class"].nunique() == 2, "Class column must be binary"
    logger.info(f"Validation passed | Fraud ratio: {df['Class'].mean():.4f}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # V1-V28 are already PCA transformed, only normalize Time and Amount
    df = df.copy()
    df["Time_norm"] = (df["Time"] - df["Time"].mean()) / df["Time"].std()
    df["Amount_norm"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    df = df.drop(columns=["Time", "Amount"])
    logger.info(f"Features after engineering: {df.shape[1]}")
    return df


def split_and_save(df: pd.DataFrame, config: dict, output_path: str):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["preprocessing"]["test_size"],
        random_state=config["preprocessing"]["random_state"],
        stratify=y
    )

    os.makedirs(output_path, exist_ok=True)

    X_train.to_csv(f"{output_path}/X_train.csv", index=False)
    X_test.to_csv(f"{output_path}/X_test.csv", index=False)
    y_train.to_csv(f"{output_path}/y_train.csv", index=False)
    y_test.to_csv(f"{output_path}/y_test.csv", index=False)

    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    logger.info(f"Saved splits to {output_path}")

    return X_train, X_test, y_train, y_test


def generate_drift_samples(df: pd.DataFrame, output_path: str):
    os.makedirs(output_path, exist_ok=True)

    # Drift 1: Amount spike (simulates new spending pattern)
    drift_amount = df.copy()
    drift_amount["Amount"] = drift_amount["Amount"] * 5
    drift_amount.to_csv(f"{output_path}/drift_amount.csv", index=False)

    # Drift 2: Time shift (simulates off-hours traffic)
    drift_time = df.copy()
    drift_time["Time"] = drift_time["Time"] + 86400
    drift_time.to_csv(f"{output_path}/drift_time.csv", index=False)

    # Drift 3: Fraud ratio spike (simulates new attack wave)
    fraud = df[df["Class"] == 1]
    normal = df[df["Class"] == 0].sample(len(fraud) * 5, random_state=42)
    drift_fraud = pd.concat([fraud, normal]).sample(frac=1, random_state=42)
    drift_fraud.to_csv(f"{output_path}/drift_fraud_ratio.csv", index=False)

    logger.info("Drift samples generated")


if __name__ == "__main__":
    config = load_config()

    df = load_data(config["data"]["raw_path"])
    validate_data(df)
    df = engineer_features(df)

    split_and_save(df, config, config["data"]["processed_path"])
    generate_drift_samples(
        pd.read_csv(config["data"]["raw_path"]),
        config["data"]["drift_samples_path"]
    )

    logger.info("Preprocessing complete")