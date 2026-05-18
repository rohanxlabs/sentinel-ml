import subprocess
import requests
import yaml
import logging
import os
from datetime import datetime
from mlflow.tracking import MlflowClient
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def action_retrain() -> dict:
    """Trigger retraining pipeline"""
    logger.info("ACTION: Triggering retraining pipeline...")
    try:
        result = subprocess.run(
            ["python", "src/training/train.py"],
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode == 0:
            # Run evaluate + register after training
            subprocess.run(["python", "src/training/evaluate.py"], timeout=120)
            subprocess.run(["python", "src/training/register_model.py"], timeout=60)

            # Hot reload serving layer
            action_reload_model()

            logger.info("ACTION: Retraining complete + model reloaded")
            return {
                "action": "retrain",
                "status": "success",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            logger.error(f"Retraining failed: {result.stderr}")
            return {
                "action": "retrain",
                "status": "failed",
                "error": result.stderr,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Retrain exception: {e}")
        return {"action": "retrain", "status": "error", "error": str(e)}


def action_rollback() -> dict:
    """Roll back to previous Production model version"""
    logger.info("ACTION: Rolling back model...")
    config = load_config()

    try:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        client = MlflowClient()
        model_name = config["mlflow"]["model_name"]

        # Get all versions
        versions = client.search_model_versions(f"name='{model_name}'")
        production = [v for v in versions if v.current_stage == "Production"]
        archived = sorted(
            [v for v in versions if v.current_stage == "Archived"],
            key=lambda v: int(v.version),
            reverse=True
        )

        if not archived:
            logger.warning("No archived versions to rollback to")
            return {"action": "rollback", "status": "no_archived_version"}

        # Archive current production
        if production:
            client.transition_model_version_stage(
                name=model_name,
                version=production[0].version,
                stage="Archived"
            )

        # Restore latest archived as production
        rollback_version = archived[0].version
        client.transition_model_version_stage(
            name=model_name,
            version=rollback_version,
            stage="Production"
        )

        # Reload serving
        action_reload_model()

        logger.info(f"ACTION: Rolled back to version {rollback_version}")
        return {
            "action": "rollback",
            "status": "success",
            "version": rollback_version,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        return {"action": "rollback", "status": "error", "error": str(e)}


def action_reload_model(serving_url: str = "http://localhost:8000") -> dict:
    """Hot reload model in serving layer"""
    logger.info("ACTION: Reloading model in serving layer...")
    try:
        response = requests.post(f"{serving_url}/reload", timeout=30)
        if response.status_code == 200:
            logger.info("ACTION: Model reloaded successfully")
            return {"action": "reload", "status": "success"}
        else:
            return {"action": "reload", "status": "failed", "code": response.status_code}
    except Exception as e:
        logger.error(f"Reload failed: {e}")
        return {"action": "reload", "status": "error", "error": str(e)}


def action_alert(message: str, severity: str = "warning") -> dict:
    """Log alert — extend this to Slack/email"""
    timestamp = datetime.utcnow().isoformat()
    alert = {
        "action": "alert",
        "severity": severity,
        "message": message,
        "timestamp": timestamp
    }

    # Log to file
    os.makedirs("logs", exist_ok=True)
    with open("logs/alerts.log", "a") as f:
        f.write(f"[{timestamp}] [{severity.upper()}] {message}\n")

    logger.warning(f"ALERT [{severity.upper()}]: {message}")
    return alert


def action_restart_serving() -> dict:
    """Restart the serving process"""
    logger.info("ACTION: Restarting serving layer...")
    try:
        # In production this would restart the k8s pod
        # Locally we just attempt a health check + reload
        result = action_reload_model()
        return {
            "action": "restart",
            "status": result["status"],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"action": "restart", "status": "error", "error": str(e)}


def dispatch_action(action_name: str, context: dict = {}) -> dict:
    """Central action dispatcher"""
    actions = {
        "retrain":  action_retrain,
        "rollback": action_rollback,
        "reload":   action_reload_model,
        "restart":  action_restart_serving,
        "alert":    lambda: action_alert(
            context.get("message", "Platform alert"),
            context.get("severity", "warning")
        )
    }

    if action_name not in actions:
        logger.error(f"Unknown action: {action_name}")
        return {"action": action_name, "status": "unknown"}

    return actions[action_name]()