import time
import yaml
import json
import logging
import operator
from datetime import datetime, timedelta
from typing import Dict

from monitoring.metric_collector import get_full_snapshot
from monitoring.drift_detector import run_drift_check
from monitoring.anomaly_detector import AnomalyDetector
from sre_agent.state_machine import StateMachine, PlatformState
from sre_agent.actions import dispatch_action, action_alert
from sre_agent.llm_reporter import generate_incident_report, save_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_rules():
    with open("src/sre_agent/rules.yaml", "r") as f:
        return yaml.safe_load(f)["rules"]


OPS = {
    ">":  operator.gt,
    "<":  operator.lt,
    "==": operator.eq,
    ">=": operator.ge,
    "<=": operator.le,
}


class SREAgent:

    def __init__(self):
        self.config = load_config()
        self.rules = load_rules()
        self.state_machine = StateMachine()
        self.anomaly_detector = AnomalyDetector(
            window_size=self.config["monitoring"]["anomaly_window"]
        )
        self.cooldowns: Dict[str, datetime] = {}
        self.retrain_count_today = 0
        self.last_retrain_date = datetime.utcnow().date()
        self.poll_interval = self.config["sre_agent"]["poll_interval_seconds"]

        logger.info("SRE Agent initialized")

    def is_on_cooldown(self, rule_name: str) -> bool:
        if rule_name not in self.cooldowns:
            return False
        return datetime.utcnow() < self.cooldowns[rule_name]

    def set_cooldown(self, rule_name: str, seconds: int):
        self.cooldowns[rule_name] = datetime.utcnow() + timedelta(seconds=seconds)

    def reset_daily_retrain_count(self):
        today = datetime.utcnow().date()
        if today != self.last_retrain_date:
            self.retrain_count_today = 0
            self.last_retrain_date = today

    def evaluate_rules(self, metrics: dict, drift_result: dict) -> list:
        """Check all rules against current metrics"""
        triggered = []

        # Merge drift into metrics for rule evaluation
        flat_metrics = {**metrics}
        if drift_result:
            flat_metrics["drift_share"] = drift_result.get("drift_share", 0)

        # Check serving status
        health = metrics.get("serving_health", {})
        flat_metrics["serving_status"] = 1 if health.get("status") == "healthy" else 0

        for rule in self.rules:
            metric_name = rule["metric"]
            value = flat_metrics.get(metric_name)

            if value is None:
                continue

            condition_fn = OPS.get(rule["condition"])
            if not condition_fn:
                continue

            if condition_fn(value, rule["threshold"]):
                if not self.is_on_cooldown(rule["name"]):
                    triggered.append({
                        "rule": rule["name"],
                        "metric": metric_name,
                        "value": value,
                        "threshold": rule["threshold"],
                        "severity": rule["severity"],
                        "action": rule["action"],
                        "cooldown_seconds": rule["cooldown_seconds"]
                    })
                    logger.warning(
                        f"RULE TRIGGERED | {rule['name']} | "
                        f"{metric_name}={value} {rule['condition']} {rule['threshold']}"
                    )

        return triggered

    def execute_actions(self, triggered_rules: list) -> list:
        """Execute actions for triggered rules"""
        actions_taken = []
        self.reset_daily_retrain_count()
        max_retrain = self.config["sre_agent"]["max_retrain_per_day"]

        for rule in triggered_rules:
            action = rule["action"]

            # Guard: retrain limit
            if action == "retrain":
                if self.retrain_count_today >= max_retrain:
                    logger.warning(f"Retrain limit reached ({max_retrain}/day) — skipping")
                    action_alert(
                        f"Retrain limit reached. Manual intervention may be needed.",
                        severity="critical"
                    )
                    continue
                self.retrain_count_today += 1

            # Guard: don't act if already busy
            if self.state_machine.is_busy() and action in ["retrain", "rollback"]:
                logger.info(f"Agent busy ({self.state_machine.current()}) — skipping {action}")
                continue

            # Transition state
            if action == "retrain":
                self.state_machine.transition(
                    PlatformState.RETRAINING, reason=rule["rule"]
                )
            elif action == "rollback":
                self.state_machine.transition(
                    PlatformState.ROLLING_BACK, reason=rule["rule"]
                )

            # Execute
            result = dispatch_action(action, context={
                "message": f"Rule triggered: {rule['rule']} | {rule['metric']}={rule['value']}",
                "severity": rule["severity"]
            })

            actions_taken.append(result)
            self.set_cooldown(rule["rule"], rule["cooldown_seconds"])

            # Transition back to healthy if action succeeded
            if result.get("status") == "success":
                self.state_machine.transition(
                    PlatformState.HEALTHY, reason=f"{action} succeeded"
                )
            else:
                self.state_machine.transition(
                    PlatformState.CRITICAL, reason=f"{action} failed"
                )

        return actions_taken

    def run_cycle(self):
        """Single agent poll cycle"""
        logger.info(f"--- Agent cycle | State: {self.state_machine.current()} ---")

        # 1. Collect metrics
        snapshot = get_full_snapshot()
        metrics = snapshot.get("prometheus_metrics", {})
        serving_health = snapshot.get("serving_health", {})
        metrics["serving_health"] = serving_health

        # 2. Update anomaly detector windows
        self.anomaly_detector.update(metrics)

        # 3. Run drift check (every cycle for demo; in prod do every N cycles)
        try:
            drift_result = run_drift_check()
        except Exception as e:
            logger.warning(f"Drift check failed: {e}")
            drift_result = {}

        # 4. Detect anomalies
        anomalies = self.anomaly_detector.detect(metrics)

        # 5. Evaluate rules
        triggered = self.evaluate_rules(metrics, drift_result)

        # 6. Execute actions
        actions_taken = []
        if triggered:
            actions_taken = self.execute_actions(triggered)

            # 7. Generate incident report
            all_alerts = triggered + anomalies
            report = generate_incident_report(
                alerts=all_alerts,
                actions_taken=actions_taken,
                platform_state=self.state_machine.current(),
                metrics_snapshot=metrics
            )
            save_report(report)
            print("\n" + report + "\n")

        logger.info(
            f"Cycle complete | "
            f"Rules triggered: {len(triggered)} | "
            f"Actions taken: {len(actions_taken)} | "
            f"State: {self.state_machine.current()}"
        )

    def run(self):
        """Main agent loop"""
        logger.info(f"SRE Agent started | Poll interval: {self.poll_interval}s")

        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("SRE Agent stopped by user")
                break
            except Exception as e:
                logger.error(f"Agent cycle error: {e}", exc_info=True)

            time.sleep(self.poll_interval)


if __name__ == "__main__":
    agent = SREAgent()
    agent.run()