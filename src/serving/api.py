"""
src/serving/api.py
FastAPI inference endpoint for PharmIQ Price Tier Classifier.
"""

import os
import math
import logging
import joblib
import json as _json
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "price_tier_classifier_v1.pkl"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = None

TIER_MAP = {0: "Budget", 1: "Mid", 2: "Premium", 3: "Luxury"}
DOSAGE_FORMS = [
    "tablet", "capsule", "liquid", "injection", "topical",
    "drops", "spray", "inhaler", "powder", "patch", "suppository", "other",
]

PREDICTIONS_LOG = PROJECT_ROOT / "logs" / "predictions.jsonl"

def log_prediction(payload: dict, prediction: str, confidence: float) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": payload,
        "prediction": prediction,
        "confidence": round(confidence, 4),
    }
    PREDICTIONS_LOG.parent.mkdir(exist_ok=True)
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(_json.dumps(record) + "\n")

@asynccontextmanager
async def lifespan(app):
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield


app = FastAPI(
    title="PharmIQ — Medicine Price Tier Classifier",
    description=(
        "Predicts medicine price tier (Budget/Mid/Premium/Luxury) relative to its dosage-form category. "
        "No MRP fed to model — prediction is purely from formulation attributes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MedicineInput(BaseModel):
    salt_composition: str = Field(..., description="Full salt composition e.g. 'Paracetamol (500mg)'")
    dosage_form: str = Field(..., description="One of the supported dosage forms")
    pack_size_units: float = Field(..., gt=0, description="Pack size (units/ml/gm)")
    salt_count: int = Field(..., ge=1, le=10, description="Number of active ingredients")
    manufacturer_tier: int = Field(..., ge=0, le=2, description="0=generic, 1=mid, 2=top-tier")
    max_dose_mg: float = Field(0.0, ge=0, description="Largest active ingredient dose in mg")

    @field_validator("dosage_form")
    @classmethod
    def validate_form(cls, v):
        if v not in DOSAGE_FORMS:
            raise ValueError(f"dosage_form must be one of {DOSAGE_FORMS}")
        return v


class PredictionResponse(BaseModel):
    price_tier: str
    tier_code: int
    probabilities: dict
    interpretation: str
    model_version: str = "v1"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


@app.get("/", include_in_schema=False)
def root():
    return {"message": "PharmIQ API — visit /docs"}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=pipeline is not None, version="1.0.0")


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(payload: MedicineInput):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import pandas as pd

    features = pd.DataFrame([{
        "salt_count": payload.salt_count,
        "pack_size_units": payload.pack_size_units,
        "manufacturer_tier": payload.manufacturer_tier,
        "dosage_form": payload.dosage_form,
        "max_dose_mg": payload.max_dose_mg,
        "log_max_dose": math.log1p(payload.max_dose_mg),
        "Salt_Composition": payload.salt_composition,
    }])

    try:
        pred = int(pipeline.predict(features)[0])
        proba = pipeline.predict_proba(features)[0]
        proba_dict = {TIER_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    interpretations = {
        0: "Budget tier — priced in the bottom 25% for this dosage form category.",
        1: "Mid tier — priced between the 25th and 50th percentile for this dosage form.",
        2: "Premium tier — priced between the 50th and 75th percentile for this dosage form.",
        3: "Luxury tier — priced in the top 25% for this dosage form category.",
    }

    log_prediction(
        payload=payload.model_dump(),
        prediction=TIER_MAP[pred],
        confidence=float(proba[pred]),
    )

    return PredictionResponse(
        price_tier=TIER_MAP[pred],
        tier_code=pred,
        probabilities=proba_dict,
        interpretation=interpretations[pred],
    )


@app.get("/api/v1/tiers", tags=["metadata"])
def get_tiers():
    return {
        "tiers": [
            {"code": 0, "label": "Budget", "description": "Bottom 25% within dosage form group"},
            {"code": 1, "label": "Mid", "description": "25th-50th percentile within dosage form group"},
            {"code": 2, "label": "Premium", "description": "50th-75th percentile within dosage form group"},
            {"code": 3, "label": "Luxury", "description": "Top 25% within dosage form group"},
        ],
        "dosage_forms": DOSAGE_FORMS,
        "manufacturer_tier_guide": {
            "0": "Generic / small manufacturer",
            "1": "Mid-tier pharma (Micro Labs, Macleods, FDC, etc.)",
            "2": "Top-tier pharma (Sun Pharma, Cipla, Lupin, Abbott, Pfizer, etc.)",
        },
    }
