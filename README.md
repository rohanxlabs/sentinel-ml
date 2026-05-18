# ⬡ Autonomous ML Platform

> A self-healing MLOps platform with autonomous drift detection, auto-retraining, and LLM-generated incident reports.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-2.x-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square)

---

## What This Is

Most MLOps projects stop at "train a model and deploy it." This platform goes further — it watches itself, detects problems, and fixes them without human intervention.

When data drift or model degradation is detected, the SRE Agent automatically:
- Triggers retraining
- Evaluates the new model against quality thresholds
- Promotes it to Production in MLflow
- Hot-reloads the serving layer without downtime
- Generates a Claude-powered natural language incident report

**Domain:** Credit card fraud detection (284k transactions, Kaggle dataset)

---

## Architecture

```
Streamlit Dashboard (localhost:8501)
         │
FastAPI Serving Layer (localhost:8000)
/predict · /health · /metrics · /reload
         │
    ┌────┴──────────────────────┐
MLflow Registry    Evidently AI    Anomaly Detector
(model versions)  (drift checks)  (Z-score + thresholds)
    │                  │                 │
    └──────────────────┴─────────────────┘
                       │
                   SRE Agent
       poll → detect → rule check → action → LLM report
                       │
              Anthropic Claude API
              (incident reports)
```

---

## Project Structure

```
autonomous-ml-platform/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv           # Kaggle dataset (117MB)
│   ├── processed/
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── drift_samples/
│       ├── drift_amount.csv         # Amount distribution shifted x5
│       ├── drift_time.csv           # Time distribution shifted +1 day
│       └── drift_fraud_ratio.csv    # Fraud ratio spike simulation
│
├── src/
│   ├── training/
│   │   ├── preprocess.py            # Feature engineering + drift samples
│   │   ├── train.py                 # MLflow-logged training pipeline
│   │   ├── evaluate.py              # Metrics + threshold checks
│   │   └── register_model.py        # Promote model to Production
│   │
│   ├── serving/
│   │   ├── app.py                   # FastAPI server
│   │   ├── model_loader.py          # Load champion model from MLflow
│   │   ├── schemas.py               # Pydantic request/response schemas
│   │   └── middleware.py            # Prometheus instrumentation
│   │
│   ├── monitoring/
│   │   ├── drift_detector.py        # Evidently AI drift detection
│   │   ├── metric_collector.py      # Prometheus + health queries
│   │   └── anomaly_detector.py      # Rolling window anomaly detection
│   │
│   ├── sre_agent/
│   │   ├── agent.py                 # Main autonomous loop
│   │   ├── actions.py               # Retrain, rollback, reload, alert
│   │   ├── llm_reporter.py          # Claude API incident reports
│   │   ├── state_machine.py         # Platform health state transitions
│   │   └── rules.yaml               # Threshold rules + remediation policies
│   │
│   └── dashboard/
│       └── app.py                   # Streamlit control room
│
├── logs/                            # Agent alerts + incident reports
├── reports/                         # Evidently HTML drift reports
├── models/                          # Local model artifacts
├── config.yaml                      # Global configuration
├── Makefile                         # Command shortcuts
└── README.md
```

---

## Tech Stack

| Layer          | Technology                       |
|----------------|----------------------------------|
| Training       | scikit-learn, MLflow, DVC        |
| Serving        | FastAPI, Uvicorn, Pydantic       |
| Monitoring     | Evidently AI, Prometheus         |
| Autonomous Ops | Custom SRE Agent + state machine |
| LLM Reports    | Anthropic Claude API             |
| Dashboard      | Streamlit                        |
| Dataset        | Kaggle Credit Card Fraud (284k)  |

---

## Quickstart

### Step 1 — Clone and install

```bash
git clone https://github.com/rohanxlabs/autonomous-ml-platform
cd autonomous-ml-platform

python -m venv venv
venv\Scripts\activate

pip install pandas numpy scikit-learn mlflow fastapi uvicorn pydantic prometheus-client pyyaml requests evidently streamlit
```

### Step 2 — Add dataset

Download from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the file at:
```
data/raw/creditcard.csv
```

### Step 3 — Create folder structure

```bash
mkdir -p data/raw data/processed data/drift_samples
mkdir -p src/training src/serving src/monitoring src/sre_agent src/dashboard
mkdir -p logs reports models

type nul > src/__init__.py
type nul > src/training/__init__.py
type nul > src/serving/__init__.py
type nul > src/monitoring/__init__.py
type nul > src/sre_agent/__init__.py
```

### Step 4 — Set environment variables

```bash
set PYTHONPATH=src
set ANTHROPIC_API_KEY=your_api_key_here
```

---

## Running the Platform

You need 3 terminals open simultaneously.

---

### Terminal 1 — MLflow Tracking Server

```bash
cd autonomous-ml-platform
venv\Scripts\activate
mlflow server --host 0.0.0.0 --port 5000
```

Verify at: http://localhost:5000

---

### Terminal 2 — Train and Register Model (run once)

```bash
cd autonomous-ml-platform
venv\Scripts\activate
set PYTHONPATH=src

python src/training/preprocess.py
python src/training/train.py
python src/training/evaluate.py
python src/training/register_model.py
```

Expected output:
```
INFO - Loaded 284807 rows
INFO - Validation passed | Fraud ratio: 0.0017
INFO - Train size: 227845 | Test size: 56962
INFO - Drift samples generated
INFO - MLflow run started
INFO - Training complete
INFO - f1_score: 0.xxxx
INFO - All thresholds passed
INFO - Promoted version 1 to Production
```

Verify at: http://localhost:5000 → Models → fraud-classifier → Production

Then start the serving layer:

```bash
cd src
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
```

Expected:
```
INFO - Model loaded | version: 1 | stage: Production
INFO - Uvicorn running on http://0.0.0.0:8000
```

---

### Terminal 3 — SRE Agent

```bash
cd autonomous-ml-platform
venv\Scripts\activate
set PYTHONPATH=src
set ANTHROPIC_API_KEY=your_key_here
python src/sre_agent/agent.py
```

Expected:
```
INFO - SRE Agent initialized | poll_interval=60s
INFO - Agent cycle | State: healthy
INFO - Cycle complete | rules_triggered=0 | actions=0 | state=healthy
```

---

### Dashboard

```bash
cd autonomous-ml-platform
venv\Scripts\activate
streamlit run src/dashboard/app.py
```

Open: http://localhost:8501

---

## API Reference

| Endpoint | Method | Description                    |
|----------|--------|--------------------------------|
| /health  | GET    | Model status, version, uptime  |
| /predict | POST   | Run fraud inference            |
| /metrics | GET    | Prometheus scrape endpoint     |
| /reload  | POST   | Hot-swap model without restart |
| /docs    | GET    | Swagger UI (auto-generated)    |

### Sample Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d "{\"V1\":-3.04,\"V2\":1.17,\"V3\":-4.34,\"V4\":1.17,\"V5\":-1.21,\"V6\":-1.04,\"V7\":-1.22,\"V8\":0.19,\"V9\":-1.14,\"V10\":-2.83,\"V11\":1.96,\"V12\":-1.03,\"V13\":-0.28,\"V14\":-2.60,\"V15\":0.79,\"V16\":-0.98,\"V17\":-1.54,\"V18\":0.05,\"V19\":-0.59,\"V20\":0.04,\"V21\":0.32,\"V22\":0.51,\"V23\":-0.03,\"V24\":0.24,\"V25\":0.54,\"V26\":0.17,\"V27\":0.16,\"V28\":0.05,\"Amount_norm\":1.8,\"Time_norm\":-1.2}"
```

Response:
```json
{
  "prediction": 1,
  "probability": 0.8741,
  "model_version": "1",
  "status": "ok"
}
```

---

## Dashboard Pages

| Page            | What you see                                        |
|-----------------|-----------------------------------------------------|
| Overview        | Live metrics, throughput chart, system status, logs |
| Live Prediction | Send real requests, see fraud probability gauge     |
| Drift Monitor   | Feature-level drift bars, drift share score         |
| Agent Logs      | Full event stream with level filtering              |
| Model Registry  | Version history, F1/AUC trend chart                |

---

## SRE Agent — How It Works

The agent runs a continuous loop:

```
1. Collect metrics     → FastAPI health + Prometheus
2. Run drift check     → Evidently AI (every 10 cycles)
3. Detect anomalies    → Z-score + hard thresholds
4. Evaluate rules      → rules.yaml
5. Execute actions     → retrain / rollback / reload / alert
6. Generate report     → Claude API incident summary
7. Sleep               → poll_interval_seconds (default: 60s)
```

### Remediation Rules

| Rule              | Trigger               | Action   |
|-------------------|-----------------------|----------|
| high_latency      | P95 > 2.0s            | alert    |
| high_error_rate   | error rate > 5%       | rollback |
| fraud_rate_spike  | fraud rate > 10%      | retrain  |
| model_degradation | F1 drops > 0.05       | rollback |
| data_drift        | drift share > 30%     | retrain  |
| service_down      | serving unreachable   | restart  |

---

## Memory Optimization (Low RAM Devices)

If RAM usage exceeds 12GB, apply these fixes:

**Reduce model size in config.yaml:**
```yaml
training:
  n_estimators: 20
  max_depth: 6
```

**Use data sample in preprocess.py (after load_data):**
```python
df = df.sample(n=50000, random_state=42)
```

**Increase poll interval in config.yaml:**
```yaml
sre_agent:
  poll_interval_seconds: 120
monitoring:
  poll_interval_seconds: 60
```

**Run drift check less frequently in agent.py:**
```python
# Add in __init__
self.cycle_count = 0

# Replace drift block in run_cycle
self.cycle_count += 1
drift_result = {}
if self.cycle_count % 10 == 0:
    drift_result = run_drift_check()
```

---

## Common Errors

| Error                        | Fix                                  |
|------------------------------|--------------------------------------|
| ModuleNotFoundError: serving | Run: set PYTHONPATH=src              |
| MLflow connection refused    | Start Terminal 1 first               |
| Model not found in registry  | Run register_model.py before serving |
| Port already in use          | netstat -ano then kill the PID       |
| Serving layer offline        | Check Terminal 2 for errors          |
| Evidently import error       | pip install evidently                |

---

## Roadmap

- [ ] Layer 5 — Docker + docker-compose + Kubernetes
- [ ] Layer 6 — GitHub Actions CI/CD pipeline
- [ ] Layer 7 — Unit, integration, and chaos tests
- [ ] Grafana dashboard integration
- [ ] Slack alerting via webhook
- [ ] Multi-model serving support

---

## Built By

**Rohan** — [@rohanxlabs](https://github.com/rohanxlabs)

Pre-college builder focused on MLOps, robotics, and agentic AI systems.
Starting BTech CSE AI/ML at Delhi Technical Campus (AKTU) — August 2026.

---

## License

MIT License — free to use, modify, and distribute.
