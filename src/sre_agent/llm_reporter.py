import requests
import json
import logging
import os
from datetime import datetime
import groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
GROQ_API_KEY ="your api key"

def generate_incident_report(
    alerts: list,
    actions_taken: list,
    platform_state: str,
    metrics_snapshot: dict
) -> str:
    """Use Claude to generate a human-readable incident report"""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping LLM report")
        return _fallback_report(alerts, actions_taken, platform_state)

    prompt = f"""
You are an ML Platform SRE. Generate a concise incident report based on the following data.

Platform State: {platform_state}
Timestamp: {datetime.utcnow().isoformat()}

Alerts Triggered:
{json.dumps(alerts, indent=2)}

Actions Taken:
{json.dumps(actions_taken, indent=2)}

Key Metrics:
{json.dumps(metrics_snapshot, indent=2)}

Write a short incident report with:
1. Summary (1-2 sentences)
2. Root cause (likely cause)
3. Actions taken
4. Current status
5. Recommended next steps

Be concise and technical. Use bullet points.
"""

    try:
        response = requests.post(
            GROQ_API_KEY,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )

        if response.status_code == 200:
            report = response.json()["content"][0]["text"]
            logger.info("LLM incident report generated")
            return report
        else:
            logger.error(f"LLM API error: {response.status_code}")
            return _fallback_report(alerts, actions_taken, platform_state)

    except Exception as e:
        logger.error(f"LLM report failed: {e}")
        return _fallback_report(alerts, actions_taken, platform_state)


def _fallback_report(alerts: list, actions_taken: list, platform_state: str) -> str:
    """Plain text report when LLM is unavailable"""
    lines = [
        f"=== INCIDENT REPORT ===",
        f"Timestamp: {datetime.utcnow().isoformat()}",
        f"Platform State: {platform_state}",
        f"Alerts: {len(alerts)}",
        f"Actions taken: {[a.get('action') for a in actions_taken]}",
    ]
    for alert in alerts:
        lines.append(f"  - [{alert.get('severity','?').upper()}] {alert.get('metric')}: {alert.get('value')}")
    return "\n".join(lines)


def save_report(report: str):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"logs/incident_{timestamp}.txt"
    with open(path, "w") as f:
        f.write(report)
    logger.info(f"Incident report saved: {path}")
    return path