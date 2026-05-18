import mlflow
from mlflow.tracking import MlflowClient
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def get_latest_run_id(client: MlflowClient, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        raise ValueError("No runs found in experiment")
    run_id = runs[0].info.run_id
    logger.info(f"Latest run ID: {run_id}")
    return run_id


def promote_to_production(client: MlflowClient, model_name: str, run_id: str):
    # Get all versions for this run
    versions = client.search_model_versions(f"name='{model_name}'")
    target_version = None

    for v in versions:
        if v.run_id == run_id:
            target_version = v.version
            break

    if not target_version:
        raise ValueError(f"No model version found for run_id: {run_id}")

    # Archive existing Production model
    for v in versions:
        if v.current_stage == "Production":
            client.transition_model_version_stage(
                name=model_name,
                version=v.version,
                stage="Archived"
            )
            logger.info(f"Archived old Production version: {v.version}")

    # Promote new version
    client.transition_model_version_stage(
        name=model_name,
        version=target_version,
        stage="Production"
    )
    logger.info(f"Promoted version {target_version} to Production")
    return target_version


def add_model_tags(client: MlflowClient, model_name: str, version: str, metrics: dict):
    for key, value in metrics.items():
        client.set_model_version_tag(model_name, version, key, str(round(value, 4)))
    logger.info("Model tags set")


def register(run_id: str, metrics: dict, config: dict):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()
    model_name = config["mlflow"]["model_name"]

    version = promote_to_production(client, model_name, run_id)
    add_model_tags(client, model_name, version, metrics)

    logger.info(f"Model '{model_name}' v{version} is now in Production")
    return version


if __name__ == "__main__":
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    run_id = get_latest_run_id(client, config["mlflow"]["experiment_name"])
    # For standalone run, pass dummy metrics
    register(run_id, {}, config)