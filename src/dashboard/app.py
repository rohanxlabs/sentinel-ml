import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime, timedelta
import random

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="ML Platform Control Room",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0a0f;
    color: #e2e8f0;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #00ff88; border-radius: 2px; }

/* ── HEADER ── */
.ctrl-header {
    background: linear-gradient(135deg, #0d1117 0%, #111827 100%);
    border: 1px solid #1e293b;
    border-left: 3px solid #00ff88;
    border-radius: 4px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.ctrl-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #00ff88;
    letter-spacing: 0.05em;
    margin: 0;
}
.ctrl-subtitle {
    font-size: 0.78rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.live-badge {
    background: #00ff8820;
    border: 1px solid #00ff88;
    color: #00ff88;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 10px;
    border-radius: 2px;
    letter-spacing: 0.12em;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── STATE BANNER ── */
.state-banner {
    border-radius: 4px;
    padding: 0.8rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
    border-left: 4px solid;
}
.state-healthy  { background: #00ff8812; border-color: #00ff88; color: #00ff88; }
.state-degraded { background: #f59e0b12; border-color: #f59e0b; color: #f59e0b; }
.state-critical { background: #ef444412; border-color: #ef4444; color: #ef4444; }
.state-retraining { background: #3b82f612; border-color: #3b82f6; color: #3b82f6; }

/* ── METRIC CARDS ── */
.metric-card {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #334155; }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.green::before  { background: #00ff88; }
.metric-card.blue::before   { background: #3b82f6; }
.metric-card.yellow::before { background: #f59e0b; }
.metric-card.red::before    { background: #ef4444; }
.metric-card.purple::before { background: #a855f7; }

.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1;
}
.metric-value.green  { color: #00ff88; }
.metric-value.blue   { color: #3b82f6; }
.metric-value.yellow { color: #f59e0b; }
.metric-value.red    { color: #ef4444; }
.metric-value.purple { color: #a855f7; }
.metric-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    margin-top: 0.3rem;
}

/* ── SECTION TITLES ── */
.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e293b;
}

/* ── LOG TERMINAL ── */
.terminal {
    background: #050508;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    line-height: 1.8;
    max-height: 280px;
    overflow-y: auto;
    color: #94a3b8;
}
.log-time    { color: #334155; }
.log-info    { color: #00ff88; }
.log-warning { color: #f59e0b; }
.log-error   { color: #ef4444; }
.log-agent   { color: #3b82f6; }

/* ── PREDICTION PANEL ── */
.pred-result {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 1.5rem;
    text-align: center;
}
.pred-fraud {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #ef4444;
}
.pred-legit {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00ff88;
}
.pred-prob {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.5rem;
}

/* ── ACTION BUTTONS ── */
.stButton > button {
    background: #0d1117 !important;
    border: 1px solid #1e293b !important;
    color: #94a3b8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 3px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    border-color: #00ff88 !important;
    color: #00ff88 !important;
    background: #00ff8808 !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── DRIFT BAR ── */
.drift-bar-container {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    margin: 0.3rem 0;
}
.drift-feature-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #64748b;
    margin-bottom: 0.3rem;
}
.drift-bar-track {
    background: #1e293b;
    border-radius: 2px;
    height: 6px;
    position: relative;
    overflow: hidden;
}
.drift-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.5s ease;
}

/* ── ALERT ITEM ── */
.alert-item {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 4px;
    padding: 0.7rem 1rem;
    margin: 0.3rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.alert-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)

# ── Config ─────────────────────────────────────────────
SERVING_URL = "http://localhost:8000"
POLL_INTERVAL = 3

# ── Session state ──────────────────────────────────────
if "logs" not in st.session_state:
    st.session_state.logs = []
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "platform_state" not in st.session_state:
    st.session_state.platform_state = "healthy"
if "retrain_count" not in st.session_state:
    st.session_state.retrain_count = 0
if "total_predictions" not in st.session_state:
    st.session_state.total_predictions = 0


# ── Helpers ────────────────────────────────────────────
def add_log(level: str, message: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, {
        "time": ts, "level": level, "message": message
    })
    st.session_state.logs = st.session_state.logs[:60]


def get_health():
    try:
        r = requests.get(f"{SERVING_URL}/health", timeout=3)
        return r.json()
    except:
        return {"status": "unreachable", "model_loaded": False,
                "model_version": "—", "uptime_seconds": 0}


def make_prediction(payload: dict):
    try:
        r = requests.post(f"{SERVING_URL}/predict",
                          json=payload, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def hot_reload():
    try:
        r = requests.post(f"{SERVING_URL}/reload", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def simulate_metrics():
    """Simulate live metrics when Prometheus not connected"""
    base_latency = 0.08 + random.gauss(0, 0.01)
    return {
        "latency_p95": round(max(0.02, base_latency), 3),
        "error_rate": round(max(0, random.gauss(0.008, 0.002)), 4),
        "request_rate": round(max(0, random.gauss(12, 2)), 1),
        "fraud_rate": round(max(0, random.gauss(0.017, 0.003)), 4),
    }


def render_metric_card(label, value, color, unit="", delta=""):
    return f"""
    <div class="metric-card {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color}">{value}<span style="font-size:0.9rem;opacity:0.6">{unit}</span></div>
        <div class="metric-delta">{delta}</div>
    </div>"""


def render_log_terminal(logs):
    html = '<div class="terminal">'
    for log in logs[:25]:
        level_class = {
            "INFO": "log-info", "WARNING": "log-warning",
            "ERROR": "log-error", "AGENT": "log-agent"
        }.get(log["level"], "log-info")
        html += (f'<div><span class="log-time">[{log["time"]}]</span> '
                 f'<span class="{level_class}">[{log["level"]}]</span> '
                 f'{log["message"]}</div>')
    html += "</div>"
    return html


def state_banner(state: str):
    icons = {
        "healthy": "● SYSTEM HEALTHY",
        "degraded": "▲ SYSTEM DEGRADED",
        "critical": "✕ CRITICAL — INTERVENTION ACTIVE",
        "retraining": "⟳ RETRAINING IN PROGRESS",
        "drifting": "~ DATA DRIFT DETECTED",
    }
    label = icons.get(state, f"● {state.upper()}")
    return f'<div class="state-banner state-{state}">{label}</div>'


# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
    color:#475569;letter-spacing:0.15em;text-transform:uppercase;
    margin-bottom:1rem;padding-bottom:0.5rem;border-bottom:1px solid #1e293b;">
    ⬡ Control Panel
    </div>""", unsafe_allow_html=True)

    page = st.selectbox(
        "Navigation",
        ["Overview", "Live Prediction", "Drift Monitor", "Agent Logs", "Model Registry"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="section-title">Agent Config</div>', unsafe_allow_html=True)
    poll_interval = st.slider("Poll interval (s)", 5, 120, 60)
    drift_threshold = st.slider("Drift threshold", 0.01, 0.5, 0.05)
    auto_retrain = st.toggle("Auto retrain", value=True)
    auto_rollback = st.toggle("Auto rollback", value=True)

    st.markdown('<div class="section-title">Actions</div>', unsafe_allow_html=True)

    if st.button("⟳  Reload Model"):
        result = hot_reload()
        if result.get("status") == "reloaded":
            add_log("INFO", f"Model hot-reloaded → v{result.get('version','?')}")
            st.success("Reloaded")
        else:
            add_log("ERROR", f"Reload failed: {result}")
            st.error("Failed")

    if st.button("▶  Run Agent Cycle"):
        add_log("AGENT", "Manual agent cycle triggered")
        st.info("Cycle queued")

    if st.button("⚠  Inject Drift"):
        add_log("WARNING", "Drift injected — amount distribution shifted ×10")
        st.session_state.platform_state = "drifting"
        st.warning("Drift injected")

    if st.button("✕  Simulate Failure"):
        add_log("ERROR", "Simulated serving failure — rollback initiated")
        st.session_state.platform_state = "critical"
        st.error("Failure injected")

    st.markdown(f"""
    <div style="margin-top:2rem;font-family:'JetBrains Mono',monospace;
    font-size:0.62rem;color:#1e293b;text-align:center;">
    autonomous-ml-platform v1.0<br>rohanxlabs
    </div>""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────
health = get_health()
serving_up = health.get("status") == "healthy"
metrics = simulate_metrics()

st.markdown(f"""
<div class="ctrl-header">
    <div>
        <div class="ctrl-title">⬡ ML PLATFORM CONTROL ROOM</div>
        <div class="ctrl-subtitle">Autonomous Fraud Detection System · rohanxlabs</div>
    </div>
    <div style="margin-left:auto">
        <span class="live-badge">{'● LIVE' if serving_up else '○ OFFLINE'}</span>
    </div>
</div>
{state_banner(st.session_state.platform_state)}
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════
if page == "Overview":

    # Top metrics row
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(render_metric_card(
            "Model Version",
            health.get("model_version", "—"),
            "green", "", "Production stage"
        ), unsafe_allow_html=True)

    with c2:
        uptime = health.get("uptime_seconds", 0)
        uptime_str = f"{int(uptime//3600)}h {int((uptime%3600)//60)}m" if uptime > 0 else "—"
        st.markdown(render_metric_card(
            "Uptime", uptime_str, "blue", "", "Since last reload"
        ), unsafe_allow_html=True)

    with c3:
        st.markdown(render_metric_card(
            "P95 Latency",
            f"{metrics['latency_p95']*1000:.0f}",
            "green" if metrics['latency_p95'] < 0.5 else "yellow",
            "ms", "< 500ms threshold"
        ), unsafe_allow_html=True)

    with c4:
        fraud_pct = metrics['fraud_rate'] * 100
        st.markdown(render_metric_card(
            "Fraud Rate",
            f"{fraud_pct:.2f}",
            "green" if fraud_pct < 5 else "red",
            "%", "Last 5 min"
        ), unsafe_allow_html=True)

    with c5:
        err_pct = metrics['error_rate'] * 100
        st.markdown(render_metric_card(
            "Error Rate",
            f"{err_pct:.2f}",
            "green" if err_pct < 2 else "red",
            "%", "5xx responses"
        ), unsafe_allow_html=True)

    st.markdown("")

    # Charts row
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-title">Request Throughput & Latency</div>',
                    unsafe_allow_html=True)

        # Simulate time series
        now = datetime.now()
        times = [now - timedelta(seconds=i*10) for i in range(30, 0, -1)]
        req_rates = [max(0, random.gauss(12, 2)) for _ in times]
        latencies = [max(0.02, random.gauss(80, 8)) for _ in times]

        chart_df = pd.DataFrame({
            "Time": [t.strftime("%H:%M:%S") for t in times],
            "Requests/s": req_rates,
            "Latency (ms)": latencies
        }).set_index("Time")

        st.line_chart(chart_df, color=["#00ff88", "#3b82f6"], height=200)

    with col_right:
        st.markdown('<div class="section-title">Prediction Distribution</div>',
                    unsafe_allow_html=True)

        fraud_count = int(st.session_state.total_predictions * 0.017)
        legit_count = st.session_state.total_predictions - fraud_count

        dist_df = pd.DataFrame({
            "Type": ["Legitimate", "Fraud"],
            "Count": [max(legit_count, 1), max(fraud_count, 1)]
        }).set_index("Type")

        st.bar_chart(dist_df, color=["#00ff88"], height=200)

    # Agent activity + System status
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown('<div class="section-title">Agent Activity Log</div>',
                    unsafe_allow_html=True)
        if not st.session_state.logs:
            add_log("INFO", "SRE Agent initialized — monitoring started")
            add_log("INFO", f"Model v{health.get('model_version','?')} loaded from Production stage")
            add_log("AGENT", "Drift check scheduled — next in 10 cycles")
        st.markdown(render_log_terminal(st.session_state.logs), unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-title">System Status</div>',
                    unsafe_allow_html=True)

        components = [
            ("FastAPI Server",  serving_up,  "8000"),
            ("MLflow Tracking", True,         "5000"),
            ("SRE Agent",       True,         "active"),
            ("Drift Detector",  True,         "idle"),
            ("Prometheus",      False,        "9090"),
        ]

        for name, status, port in components:
            dot_color = "#00ff88" if status else "#ef4444"
            status_text = "online" if status else "offline"
            st.markdown(f"""
            <div class="alert-item">
                <div class="alert-dot" style="background:{dot_color}"></div>
                <div style="flex:1;color:#94a3b8">{name}</div>
                <div style="font-size:0.65rem;color:#334155">:{port}</div>
                <div style="color:{dot_color};font-size:0.65rem">{status_text}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTION
# ════════════════════════════════════════════════════════
elif page == "Live Prediction":

    st.markdown('<div class="section-title">Live Inference Panel</div>',
                unsafe_allow_html=True)

    col_form, col_result = st.columns([3, 2])

    with col_form:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
        color:#475569;margin-bottom:1rem;">
        Enter transaction features or use random sample
        </div>""", unsafe_allow_html=True)

        use_random = st.toggle("Use random transaction", value=True)

        if use_random:
            # Generate random transaction
            is_fraud = random.random() < 0.05
            if is_fraud:
                v_values = {f"V{i}": round(random.gauss(-1.5, 2), 5) for i in range(1, 29)}
            else:
                v_values = {f"V{i}": round(random.gauss(0, 1), 5) for i in range(1, 29)}

            payload = {
                **v_values,
                "Amount_norm": round(random.gauss(0, 1), 5),
                "Time_norm": round(random.gauss(0, 1), 5)
            }
            st.json(payload, expanded=False)
        else:
            col1, col2 = st.columns(2)
            amount = col1.number_input("Amount_norm", value=0.24, format="%.5f")
            time_n = col2.number_input("Time_norm", value=-0.99, format="%.5f")

            st.markdown("""<div style="font-family:'JetBrains Mono',monospace;
            font-size:0.68rem;color:#334155;margin:0.5rem 0;">
            V1–V28 features (PCA transformed)</div>""", unsafe_allow_html=True)

            v_cols = st.columns(4)
            v_values = {}
            for i in range(1, 29):
                col_idx = (i - 1) % 4
                v_values[f"V{i}"] = v_cols[col_idx].number_input(
                    f"V{i}", value=round(random.gauss(0, 1), 3), format="%.3f",
                    label_visibility="visible"
                )
            payload = {**v_values, "Amount_norm": amount, "Time_norm": time_n}

        predict_btn = st.button("▶  Run Inference", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

        if predict_btn:
            if not serving_up:
                st.error("Serving layer offline")
            else:
                with st.spinner(""):
                    result = make_prediction(payload)

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                    add_log("ERROR", f"Prediction failed: {result['error']}")
                else:
                    pred = result.get("prediction", 0)
                    prob = result.get("probability", 0)
                    ver = result.get("model_version", "?")

                    st.session_state.total_predictions += 1
                    st.session_state.prediction_history.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "prediction": pred,
                        "probability": prob
                    })

                    add_log("INFO",
                        f"Prediction: {'FRAUD' if pred else 'LEGIT'} | "
                        f"prob={prob:.4f} | v{ver}"
                    )

                    label_class = "pred-fraud" if pred == 1 else "pred-legit"
                    label_text = "⚠ FRAUD" if pred == 1 else "✓ LEGITIMATE"

                    st.markdown(f"""
                    <div class="pred-result">
                        <div class="{label_class}">{label_text}</div>
                        <div class="pred-prob">
                            Fraud probability: {prob:.4f}<br>
                            Model version: v{ver}<br>
                            Threshold: 0.50
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # Probability gauge
                    st.markdown("")
                    bar_color = "#ef4444" if prob > 0.5 else "#00ff88"
                    st.markdown(f"""
                    <div style="margin-top:0.5rem">
                        <div style="font-family:'JetBrains Mono',monospace;
                        font-size:0.65rem;color:#475569;margin-bottom:0.3rem;">
                        FRAUD PROBABILITY
                        </div>
                        <div style="background:#1e293b;border-radius:2px;height:8px">
                            <div style="width:{prob*100:.1f}%;background:{bar_color};
                            height:100%;border-radius:2px;transition:width 0.5s"></div>
                        </div>
                        <div style="font-family:'JetBrains Mono',monospace;
                        font-size:0.7rem;color:{bar_color};margin-top:0.3rem">
                        {prob*100:.1f}%
                        </div>
                    </div>""", unsafe_allow_html=True)

        # Prediction history
        if st.session_state.prediction_history:
            st.markdown('<div class="section-title">Recent</div>', unsafe_allow_html=True)
            for p in reversed(st.session_state.prediction_history[-8:]):
                color = "#ef4444" if p["prediction"] else "#00ff88"
                label = "FRAUD" if p["prediction"] else "LEGIT"
                st.markdown(f"""
                <div class="alert-item">
                    <div class="alert-dot" style="background:{color}"></div>
                    <div style="color:#64748b">{p['time']}</div>
                    <div style="color:{color};flex:1;text-align:right">{label}</div>
                    <div style="color:#334155">{p['probability']:.4f}</div>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PAGE: DRIFT MONITOR
# ════════════════════════════════════════════════════════
elif page == "Drift Monitor":

    st.markdown('<div class="section-title">Data Drift Analysis</div>',
                unsafe_allow_html=True)

    col_summary, col_action = st.columns([3, 1])

    with col_summary:
        drift_triggered = st.session_state.platform_state == "drifting"
        overall_drift = 0.48 if drift_triggered else 0.06

        c1, c2, c3 = st.columns(3)
        with c1:
            color = "red" if overall_drift > 0.3 else "green"
            st.markdown(render_metric_card(
                "Drift Share", f"{overall_drift:.0%}", color,
                "", f"Threshold: {drift_threshold:.0%}"
            ), unsafe_allow_html=True)
        with c2:
            drifted_count = int(overall_drift * 29)
            st.markdown(render_metric_card(
                "Drifted Features", str(drifted_count), color,
                "", "of 29 total"
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(render_metric_card(
                "Last Check",
                datetime.now().strftime("%H:%M:%S"),
                "blue", "", "Reference: X_train"
            ), unsafe_allow_html=True)

    with col_action:
        st.markdown("")
        if st.button("Run Drift Check", use_container_width=True):
            add_log("AGENT", "Manual drift check triggered")
            st.rerun()
        if st.button("Reset to Healthy", use_container_width=True):
            st.session_state.platform_state = "healthy"
            add_log("INFO", "Platform state reset to healthy")
            st.rerun()

    st.markdown('<div class="section-title">Feature Drift Scores</div>',
                unsafe_allow_html=True)

    features = [f"V{i}" for i in range(1, 15)] + ["Amount_norm", "Time_norm"]
    drifted_features = random.sample(features, k=int(overall_drift * len(features)))

    col1, col2 = st.columns(2)
    for i, feature in enumerate(features):
        drifted = feature in drifted_features
        score = random.uniform(0.35, 0.85) if drifted else random.uniform(0.01, 0.08)
        bar_color = "#ef4444" if score > 0.3 else "#00ff88"
        bar_width = min(100, score * 100)

        html = f"""
        <div class="drift-bar-container">
            <div class="drift-feature-name">{feature}
                {'<span style="color:#ef4444;margin-left:0.5rem">● drift</span>' if drifted else ''}
            </div>
            <div class="drift-bar-track">
                <div class="drift-bar-fill"
                     style="width:{bar_width:.0f}%;background:{bar_color}"></div>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
            color:#334155;margin-top:0.2rem">{score:.3f}</div>
        </div>"""

        if i % 2 == 0:
            col1.markdown(html, unsafe_allow_html=True)
        else:
            col2.markdown(html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PAGE: AGENT LOGS
# ════════════════════════════════════════════════════════
elif page == "Agent Logs":

    st.markdown('<div class="section-title">SRE Agent Event Stream</div>',
                unsafe_allow_html=True)

    col_filter, col_clear = st.columns([4, 1])
    level_filter = col_filter.selectbox(
        "Filter", ["ALL", "AGENT", "WARNING", "ERROR", "INFO"],
        label_visibility="collapsed"
    )
    if col_clear.button("Clear"):
        st.session_state.logs = []
        st.rerun()

    # Add some sample logs if empty
    if not st.session_state.logs:
        sample_logs = [
            ("INFO",    "SRE Agent v1.0 started | poll_interval=60s"),
            ("INFO",    "Model fraud-classifier v1 loaded from Production"),
            ("AGENT",   "Cycle 1 complete | rules_triggered=0 | state=healthy"),
            ("INFO",    "Drift check scheduled — every 10 cycles"),
            ("AGENT",   "Cycle 2 complete | rules_triggered=0 | state=healthy"),
            ("WARNING", "Latency spike detected | p95=1.2s > baseline 0.08s"),
            ("AGENT",   "Rule: high_latency triggered | action=alert"),
            ("INFO",    "Alert logged to logs/alerts.log"),
            ("INFO",    "Cooldown set: high_latency → 300s"),
            ("AGENT",   "Cycle 3 complete | rules_triggered=1 | state=healthy"),
        ]
        for level, msg in reversed(sample_logs):
            add_log(level, msg)

    # Filtered logs
    filtered = st.session_state.logs
    if level_filter != "ALL":
        filtered = [l for l in filtered if l["level"] == level_filter]

    st.markdown(render_log_terminal(filtered), unsafe_allow_html=True)

    # Stats
    st.markdown('<div class="section-title">Event Summary</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    counts = {l: sum(1 for x in st.session_state.logs if x["level"] == l)
              for l in ["INFO", "WARNING", "ERROR", "AGENT"]}

    with c1:
        st.markdown(render_metric_card("Info", str(counts["INFO"]), "green"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card("Warnings", str(counts["WARNING"]), "yellow"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card("Errors", str(counts["ERROR"]), "red"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_metric_card("Agent Events", str(counts["AGENT"]), "blue"), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PAGE: MODEL REGISTRY
# ════════════════════════════════════════════════════════
elif page == "Model Registry":

    st.markdown('<div class="section-title">MLflow Model Registry</div>',
                unsafe_allow_html=True)

    # Mock registry data
    versions = [
        {"version": "3", "stage": "Production", "f1": 0.9142, "roc_auc": 0.9821,
         "created": "2026-05-18 09:12", "run_id": "a3f9b2c1"},
        {"version": "2", "stage": "Archived",   "f1": 0.8934, "roc_auc": 0.9743,
         "created": "2026-05-17 14:30", "run_id": "b7d1e4f2"},
        {"version": "1", "stage": "Archived",   "f1": 0.8721, "roc_auc": 0.9612,
         "created": "2026-05-16 11:05", "run_id": "c2a8d6e3"},
    ]

    for v in versions:
        stage_color = {
            "Production": "#00ff88",
            "Staging": "#3b82f6",
            "Archived": "#334155"
        }.get(v["stage"], "#475569")

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:0.6rem;padding:1rem 1.4rem">
            <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.8rem">
                <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;
                font-weight:700;color:#e2e8f0">v{v['version']}</div>
                <div style="background:{stage_color}20;border:1px solid {stage_color};
                color:{stage_color};font-family:'JetBrains Mono',monospace;
                font-size:0.62rem;padding:2px 8px;border-radius:2px;
                letter-spacing:0.1em">{v['stage'].upper()}</div>
                <div style="margin-left:auto;font-family:'JetBrains Mono',monospace;
                font-size:0.65rem;color:#334155">{v['created']}</div>
            </div>
            <div style="display:flex;gap:2rem">
                <div>
                    <div class="metric-label">F1 Score</div>
                    <div style="font-family:'JetBrains Mono',monospace;
                    font-size:1.1rem;color:{stage_color}">{v['f1']:.4f}</div>
                </div>
                <div>
                    <div class="metric-label">ROC AUC</div>
                    <div style="font-family:'JetBrains Mono',monospace;
                    font-size:1.1rem;color:{stage_color}">{v['roc_auc']:.4f}</div>
                </div>
                <div>
                    <div class="metric-label">Run ID</div>
                    <div style="font-family:'JetBrains Mono',monospace;
                    font-size:0.8rem;color:#475569">{v['run_id']}...</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Performance Trend</div>',
                unsafe_allow_html=True)

    trend_df = pd.DataFrame({
        "Version": ["v1", "v2", "v3"],
        "F1 Score": [0.8721, 0.8934, 0.9142],
        "ROC AUC": [0.9612, 0.9743, 0.9821]
    }).set_index("Version")

    st.line_chart(trend_df, color=["#00ff88", "#3b82f6"], height=180)