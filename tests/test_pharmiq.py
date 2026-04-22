"""
tests/test_pharmiq.py
Production test suite for PharmIQ Price Tier Classifier.
Target: 30+ tests covering data, features, model, API.
"""

import pytest
import sys
import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingest import load_raw, clean, validate
from src.features.engineer import (
    extract_dosage_form, extract_pack_size, extract_salt_count,
    extract_unit_price, extract_max_dose_mg, is_branded_name,
    manufacturer_tier, engineer, FEATURE_COLS, TARGET_COL,
    TOP_TIER_MANUFACTURERS, MID_TIER_MANUFACTURERS,
)
from src.serving.api import app

client = TestClient(app)
MODEL_PATH = PROJECT_ROOT / "models" / "price_tier_classifier_v1.pkl"

# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def raw_df():
    return load_raw(PROJECT_ROOT / "tata_1mg_Medicine_data.csv")

@pytest.fixture(scope="session")
def clean_df(raw_df):
    return clean(raw_df)

@pytest.fixture(scope="session")
def engineered_df(clean_df):
    return engineer(clean_df)

@pytest.fixture(scope="session")
def model():
    assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"
    return joblib.load(MODEL_PATH)

@pytest.fixture
def valid_payload():
    return {
        "salt_composition": "Paracetamol (500mg)",
        "dosage_form": "tablet",
        "pack_size_units": 10.0,
        "salt_count": 1,
        "manufacturer_tier": 1,
        "max_dose_mg": 500.0,
        "is_branded": 0,
    }

# ─────────────────────────────────────────────
# 1. DATA INGESTION TESTS
# ─────────────────────────────────────────────

def test_raw_load_shape(raw_df):
    assert raw_df.shape[0] > 200_000
    assert raw_df.shape[1] == 7

def test_raw_columns(raw_df):
    expected = {"Name", "MRP", "Quantity", "Manufacturer", "Salt_Composition", "Image_URL"}
    assert expected.issubset(set(raw_df.columns))

def test_no_null_mrp(raw_df):
    assert raw_df["MRP"].isna().sum() == 0

def test_no_null_salt(raw_df):
    assert raw_df["Salt_Composition"].isna().sum() == 0

def test_clean_removes_outliers(clean_df, raw_df):
    upper = raw_df["MRP"].quantile(0.999)
    assert clean_df["MRP"].max() <= upper

def test_clean_no_zero_mrp(clean_df):
    assert (clean_df["MRP"] > 0).all()

def test_validation_passes(clean_df):
    validate(clean_df)  # Should not raise

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING TESTS
# ─────────────────────────────────────────────

def test_extract_dosage_form_tablet():
    assert extract_dosage_form("strip of 10 tablets") == "tablet"

def test_extract_dosage_form_injection():
    assert extract_dosage_form("vial of 1 Injection") == "injection"

def test_extract_dosage_form_liquid():
    assert extract_dosage_form("bottle of 100 ml Syrup") == "liquid"

def test_extract_dosage_form_topical():
    assert extract_dosage_form("tube of 15 gm Cream") == "topical"

def test_extract_dosage_form_unknown():
    assert extract_dosage_form("something unknown") == "other"

def test_extract_pack_size_tablet():
    assert extract_pack_size("strip of 10 tablets") == 10.0

def test_extract_pack_size_bottle():
    assert extract_pack_size("bottle of 100 ml Syrup") == 100.0

def test_extract_pack_size_fallback():
    assert extract_pack_size("no number here") == 1.0

def test_extract_salt_count_single():
    assert extract_salt_count("Paracetamol (500mg)") == 1

def test_extract_salt_count_multi():
    assert extract_salt_count("Paracetamol (500mg) + Ibuprofen (400mg) + Caffeine (65mg)") == 3

def test_extract_unit_price():
    assert extract_unit_price(100.0, 10.0) == pytest.approx(10.0)

def test_extract_unit_price_zero_pack():
    assert extract_unit_price(100.0, 0.0) == 100.0

def test_extract_max_dose_mg_single():
    result = extract_max_dose_mg("Paracetamol (500mg)")
    assert result == pytest.approx(500.0)

def test_extract_max_dose_mg_multi():
    result = extract_max_dose_mg("Paracetamol (500mg) + Ibuprofen (400mg)")
    assert result == pytest.approx(500.0)

def test_extract_max_dose_mcg_conversion():
    result = extract_max_dose_mg("Fluticasone (500mcg)")
    assert result == pytest.approx(0.5)

def test_is_branded_short_name():
    assert is_branded_name("Crocin") == 1

def test_is_branded_generic_name():
    assert is_branded_name("Paracetamol 500mg Tablet") == 0

def test_manufacturer_tier_top():
    assert manufacturer_tier("Cipla Ltd") == 2

def test_manufacturer_tier_mid():
    assert manufacturer_tier("Micro Labs Ltd") == 1

def test_manufacturer_tier_generic():
    assert manufacturer_tier("Unknown Small Pharma Co") == 0

def test_engineered_df_has_feature_cols(engineered_df):
    for col in FEATURE_COLS:
        assert col in engineered_df.columns, f"Missing feature: {col}"

def test_target_col_exists(engineered_df):
    assert TARGET_COL in engineered_df.columns

def test_target_four_classes(engineered_df):
    assert set(engineered_df[TARGET_COL].unique()) == {0, 1, 2, 3}

def test_target_balanced(engineered_df):
    counts = engineered_df[TARGET_COL].value_counts()
    ratio = counts.max() / counts.min()
    assert ratio < 1.15, f"Target imbalanced: {counts.to_dict()}"

def test_no_leaky_features():
    leaky = ["log_mrp", "log_unit_price"]
    for f in leaky:
        assert f not in FEATURE_COLS, f"Leaky feature {f} in FEATURE_COLS"

# ─────────────────────────────────────────────
# 3. MODEL TESTS
# ─────────────────────────────────────────────

def test_model_loads(model):
    assert model is not None

def test_model_has_pipeline_steps(model):
    assert hasattr(model, "named_steps")
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps

def test_model_predict_shape(model, engineered_df):
    X = engineered_df[FEATURE_COLS].head(100)
    preds = model.predict(X)
    assert preds.shape == (100,)

def test_model_predict_valid_classes(model, engineered_df):
    X = engineered_df[FEATURE_COLS].head(100)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1, 2, 3})

def test_model_predict_proba_sums_to_one(model, engineered_df):
    X = engineered_df[FEATURE_COLS].head(50)
    probas = model.predict_proba(X)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

def test_model_auc_above_threshold(model, engineered_df):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    X = engineered_df[FEATURE_COLS]
    y = engineered_df[TARGET_COL]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_proba = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    assert auc >= 0.80, f"AUC below threshold: {auc:.4f}"

# ─────────────────────────────────────────────
# 4. API TESTS
# ─────────────────────────────────────────────

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_tiers_endpoint():
    resp = client.get("/api/v1/tiers")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["tiers"]) == 4
    assert "dosage_forms" in data

def test_predict_valid_payload(valid_payload):
    resp = client.post("/api/v1/predict", json=valid_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["price_tier"] in {"Budget", "Mid", "Premium", "Luxury"}
    assert 0 <= data["tier_code"] <= 3
    assert len(data["probabilities"]) == 4
    assert "interpretation" in data

def test_predict_probabilities_sum_to_one(valid_payload):
    resp = client.post("/api/v1/predict", json=valid_payload)
    probs = resp.json()["probabilities"]
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-3

def test_predict_invalid_dosage_form():
    payload = {
        "salt_composition": "Paracetamol (500mg)",
        "dosage_form": "INVALID_FORM",
        "pack_size_units": 10.0,
        "salt_count": 1,
        "manufacturer_tier": 1,
        "max_dose_mg": 500.0,
        "is_branded": 0,
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422

def test_predict_zero_pack_size(valid_payload):
    payload = {**valid_payload, "pack_size_units": 0.0}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 422

def test_predict_injection_high_tier():
    """Injections + top manufacturer should lean Premium/Luxury."""
    payload = {
        "salt_composition": "Vancomycin (500mg)",
        "dosage_form": "injection",
        "pack_size_units": 1.0,
        "salt_count": 1,
        "manufacturer_tier": 2,
        "max_dose_mg": 500.0,
        "is_branded": 1,
    }
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["tier_code"] >= 1  # At least Mid
