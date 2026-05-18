import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import yaml
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_processed_data(path: str):
    X_train = pd.read_csv(f"{path}/X_train.csv")
    X_test = pd.read_csv(f"{path}/X_test.csv")
    y_train = pd.read_csv(f"{path}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{path}/y_test.csv").squeeze()
    logger.info(f"Loaded processed data | X_train: {X_train.shape}")
    return X_train, X_test, y_train, y_test


def build_model(config: dict):
    cfg = config["training"]
    if cfg["model_type"] == "random_forest":
        model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            class_weight=cfg["class_weight"],
            random_state=cfg["random_state"],
            n_jobs=-1
        )
    elif cfg["model_type"] == "logistic_regression":
        model = LogisticRegression(
            class_weight=cfg["class_weight"],
            random_state=cfg["random_state"],
            max_iter=1000
        )
    else:
        raise ValueError(f"Unknown model type: {cfg['model_type']}")

    logger.info(f"Model built: {cfg['model_type']}")
    return model


def train(config: dict):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    X_train, X_test, y_train, y_test = load_processed_data(
        config["data"]["processed_path"]
    )

    with mlflow.start_run() as run:
        logger.info(f"MLflow run started: {run.info.run_id}")

        # Log config params
        mlflow.log_params({
            "model_type": config["training"]["model_type"],
            "n_estimators": config["training"]["n_estimators"],
            "max_depth": config["training"]["max_depth"],
            "test_size": config["preprocessing"]["test_size"],
            "random_state": config["training"]["random_state"],
        })

        # Train
        model = build_model(config)
        logger.info("Training started...")
        model.fit(X_train, y_train)
        logger.info("Training complete")

        # Save model artifact
        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=config["mlflow"]["model_name"]
        )

        with open("latest_run.txt","w") as f:
            f.write(run.info.run_id)

        logger.info(f"Model logged to MLflow | Run ID: {run.info.run_id}")
        return model, run.info.run_id, X_test, y_test


if __name__ == "__main__":
    config = load_config()
    model, run_id, X_test, y_test = train(config)
    logger.info(f"Training pipeline complete | run_id: {run_id}")