import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import yaml
import logging
import time

from serving.schemas import PredictRequest, PredictResponse, HealthResponse
from serving.model_loader import get_model, get_model_version, get_uptime, reload_model
from serving.middleware import (
    prometheus_middleware,
    track_prediction,
    set_model_version,
    get_metrics_response
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up — loading model...")
    model = get_model()
    set_model_version(get_model_version())
    logger.info("Model ready")
    yield
    # Shutdown
    logger.info("Shutting down serving layer")


app = FastAPI(
    title="Fraud Detection API",
    description="Autonomous ML Platform — Serving Layer",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.middleware("http")(prometheus_middleware)


# ── Routes ──────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    model = get_model()
    return HealthResponse(
        status="healthy" if model else "degraded",
        model_loaded=model is not None,
        model_version=get_model_version(),
        uptime_seconds=get_uptime()
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_df = pd.DataFrame([request.model_dump()])
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        track_prediction(prediction, probability)

        logger.info(f"Prediction: {prediction} | Probability: {probability:.4f}")

        return PredictResponse(
            prediction=prediction,
            probability=round(probability, 4),
            model_version=get_model_version()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return get_metrics_response()


@app.post("/reload")
def reload():
    """Hot reload model from MLflow registry — called by SRE agent after retraining"""
    try:
        reload_model()
        set_model_version(get_model_version())
        logger.info("Model hot-reloaded successfully")
        return {"status": "reloaded", "version": get_model_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "service": "fraud-detection-api",
        "version": get_model_version(),
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    config = load_config()
    uvicorn.run(
        "serving.app:app",
        host=config["serving"]["host"],
        port=config["serving"]["port"],
        reload=False
    )