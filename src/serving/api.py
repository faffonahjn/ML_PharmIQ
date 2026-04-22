"""
src/serving/api.py
FastAPI inference endpoint for PharmIQ Price Tier Classifier.
"""

import os
import sys
import math
import logging
import joblib
from contextlib import asynccontextmanager
from pathlib import Path

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


@asynccontextmanager
async def lifespan(app):
    global pipeline, recommender, category_pipeline

    # System 1: Price Tier Classifier
    try:
        pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load price tier model: {e}")

    # System 2: Generic Alternative Recommender
    try:
        from src.recommender.engine import GenericRecommender
        with open(RECOMMENDER_PATH, "rb") as f:
            index = pickle.load(f)
        recommender = GenericRecommender(index)
        logger.info(f"Recommender index loaded from {RECOMMENDER_PATH}")
    except Exception as e:
        logger.error(f"Failed to load recommender: {e}")

    # System 3: Therapeutic Category Classifier
    try:
        category_pipeline = joblib.load(CATEGORY_MODEL_PATH)
        logger.info(f"Category classifier loaded from {CATEGORY_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load category classifier: {e}")

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
    is_branded: int = Field(0, ge=0, le=1, description="1 if branded name, 0 if generic")

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
        "is_branded": payload.is_branded,
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


# ─── System 2: Generic Alternative Recommender ───────────────────────────────

import pickle

RECOMMENDER_PATH = os.getenv(
    "RECOMMENDER_PATH",
    str(PROJECT_ROOT / "models" / "recommender_index.pkl")
)

recommender = None



class RecommendRequest(BaseModel):
    salt_composition: str = Field(..., description="Salt composition string")
    query_name: str = Field("", description="Medicine name (excluded from results)")
    query_mrp: float = Field(0.0, ge=0, description="MRP of the query medicine for savings calculation")
    top_n: int = Field(10, ge=1, le=50)
    mode: str = Field("exact", description="'exact' = same dose | 'ingredient' = any dose")
    dosage_form_filter: str = Field("", description="Optional: filter by dosage form")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v not in ("exact", "ingredient"):
            raise ValueError("mode must be 'exact' or 'ingredient'")
        return v


class NameSearchRequest(BaseModel):
    medicine_name: str = Field(..., description="Brand or generic medicine name")
    top_n: int = Field(10, ge=1, le=50)
    mode: str = Field("exact", description="'exact' | 'ingredient'")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v not in ("exact", "ingredient"):
            raise ValueError("mode must be 'exact' or 'ingredient'")
        return v


class AlternativeItem(BaseModel):
    name: str
    salt_composition: str
    mrp: float
    unit_price: float
    pack_qty: float
    quantity: str
    manufacturer: str
    manufacturer_tier: int
    manufacturer_tier_label: str
    mrp_savings_pct: float
    unit_price_savings_pct: float


class RecommendationResponse(BaseModel):
    query_name: str
    query_salt_composition: str
    query_mrp: float
    query_unit_price: float
    query_manufacturer: str
    mode: str
    alternatives: list
    total_found: int
    cheapest_savings_pct: float
    message: str


@app.post("/api/v1/recommend/by-salt", response_model=RecommendationResponse, tags=["recommender"])
def recommend_by_salt(payload: RecommendRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")
    try:
        result = recommender.recommend_by_salt(
            salt_composition=payload.salt_composition,
            query_name=payload.query_name,
            query_mrp=payload.query_mrp,
            top_n=payload.top_n,
            mode=payload.mode,
            dosage_form_filter=payload.dosage_form_filter or None,
        )
        return RecommendationResponse(
            query_name=result.query_name,
            query_salt_composition=result.query_salt_composition,
            query_mrp=result.query_mrp,
            query_unit_price=result.query_unit_price,
            query_manufacturer=result.query_manufacturer,
            mode=result.mode,
            alternatives=[a.__dict__ for a in result.alternatives],
            total_found=result.total_found,
            cheapest_savings_pct=result.cheapest_savings_pct,
            message=result.message,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/recommend/by-name", response_model=RecommendationResponse, tags=["recommender"])
def recommend_by_name(payload: NameSearchRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")
    try:
        result = recommender.search_by_name(
            medicine_name=payload.medicine_name,
            top_n=payload.top_n,
            mode=payload.mode,
        )
        return RecommendationResponse(
            query_name=result.query_name,
            query_salt_composition=result.query_salt_composition,
            query_mrp=result.query_mrp,
            query_unit_price=result.query_unit_price,
            query_manufacturer=result.query_manufacturer,
            mode=result.mode,
            alternatives=[a.__dict__ for a in result.alternatives],
            total_found=result.total_found,
            cheapest_savings_pct=result.cheapest_savings_pct,
            message=result.message,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── System 3: Therapeutic Category Classifier ───────────────────────────────

CATEGORY_MODEL_PATH = os.getenv(
    "CATEGORY_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "category_classifier_v1.pkl")
)

category_pipeline = None


class CategoryInput(BaseModel):
    salt_composition: str = Field(..., description="Full salt composition string")
    quantity: str = Field("strip of 10 tablets", description="Pack quantity string for dosage form extraction")


class CategoryResponse(BaseModel):
    predicted_category: str
    category_code: int
    probabilities: dict
    rule_label: str
    model_version: str = "v1"


@app.post("/api/v1/classify/category", response_model=CategoryResponse, tags=["category-classifier"])
def classify_category(payload: CategoryInput):
    if category_pipeline is None:
        raise HTTPException(status_code=503, detail="Category classifier not loaded")

    import pandas as pd
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.features.engineer import extract_dosage_form, extract_salt_count
    from src.classifier.label_engine import assign_label, CODE_TO_CATEGORY, CATEGORY_CODES

    dosage_form = extract_dosage_form(payload.quantity)
    salt_count = extract_salt_count(payload.salt_composition)

    features = pd.DataFrame([{
        "Salt_Composition": payload.salt_composition,
        "dosage_form": dosage_form,
        "salt_count": salt_count,
    }])

    try:
        pred_code = int(category_pipeline.predict(features)[0])
        proba = category_pipeline.predict_proba(features)[0]
        classes = category_pipeline.classes_
        proba_dict = {CODE_TO_CATEGORY.get(int(c), str(c)): round(float(p), 4)
                      for c, p in zip(classes, proba)}
        predicted_category = CODE_TO_CATEGORY.get(pred_code, "Other")
        rule_label = assign_label(payload.salt_composition)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return CategoryResponse(
        predicted_category=predicted_category,
        category_code=pred_code,
        probabilities=proba_dict,
        rule_label=rule_label,
    )
