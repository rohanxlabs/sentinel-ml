import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model = None
_model_version = None
_load_time = None


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_model():
    global _model, _model_version, _load_time

    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    model_name = config["mlflow"]["model_name"]
    stage = config["serving"]["model_stage"]

    model_uri = f"models:/{model_name}/{stage}"
    logger.info(f"Loading model from: {model_uri}")

    _model = mlflow.sklearn.load_model(model_uri)

    # Get version info
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])
    _model_version = versions[0].version if versions else "unknown"
    _load_time = time.time()

    logger.info(f"Model loaded | version: {_model_version} | stage: {stage}")
    return _model


def get_model():
    global _model
    if _model is None:
        load_model()
    return _model


def get_model_version() -> str:
    return _model_version or "unknown"


def get_uptime() -> float:
    if _load_time is None:
        return 0.0
    return time.time() - _load_time


def reload_model():
    global _model, _model_version, _load_time
    _model = None
    _model_version = None
    _load_time = None
    logger.info("Model cache cleared — reloading...")
    return load_model()


import mlflow