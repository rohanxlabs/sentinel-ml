from pydantic import BaseModel, Field
from typing import List


class PredictRequest(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount_norm: float
    Time_norm: float

    class Config:
        json_schema_extra = {
            "example": {
                "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37,
                "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09,
                "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61,
                "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47,
                "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
                "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06,
                "V25": 0.12, "V26": -0.18, "V27": 0.13, "V28": -0.02,
                "Amount_norm": 0.24, "Time_norm": -0.99
            }
        }


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="0 = legitimate, 1 = fraud")
    probability: float = Field(..., description="Fraud probability score")
    model_version: str
    status: str = "ok"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float