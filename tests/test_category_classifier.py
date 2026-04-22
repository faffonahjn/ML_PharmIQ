"""
tests/test_category_classifier.py
Production test suite for PharmIQ Therapeutic Category Classifier (System 3).
25 tests: label engine, model, API endpoint.
"""

import sys
import pytest
import joblib
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.label_engine import (
    assign_label, assign_label_code, coverage_report,
    CATEGORY_CODES, CODE_TO_CATEGORY, PRIORITY_ORDER,
    _parse_salt_names,
)
from src.serving.api import app

client = TestClient(app)
MODEL_PATH = PROJECT_ROOT / "models" / "category_classifier_v1.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="session")
def sample_data():
    import warnings
    warnings.filterwarnings("ignore")
    from src.data.ingest import load_raw, clean
    from src.features.engineer import extract_dosage_form, extract_salt_count
    df = clean(load_raw(PROJECT_ROOT / "data" / "raw" / "tata_1mg_Medicine_data.csv"))
    df["dosage_form"] = df["Quantity"].apply(extract_dosage_form)
    df["salt_count"] = df["Salt_Composition"].apply(extract_salt_count)
    df["category"] = df["Salt_Composition"].apply(assign_label)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. LABEL ENGINE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_antibiotic_label():
    assert assign_label("Azithromycin (500mg)") == "Antibiotic"


def test_analgesic_label():
    assert assign_label("Paracetamol (500mg) + Ibuprofen (400mg)") == "Analgesic"


def test_antidiabetic_label():
    assert assign_label("Metformin (500mg) + Glimepiride (1mg)") == "Anti-diabetic"


def test_cardiac_label():
    assert assign_label("Amlodipine (5mg) + Telmisartan (40mg)") == "Cardiac"


def test_respiratory_label():
    assert assign_label("Salbutamol (2mg) + Guaifenesin (100mg)") == "Respiratory"


def test_gi_label():
    assert assign_label("Pantoprazole (40mg) + Domperidone (30mg)") == "Gastrointestinal"


def test_neurological_label():
    assert assign_label("Pregabalin (75mg) + Methylcobalamin (500mcg)") == "Neurological"


def test_vitamin_label():
    assert assign_label("Methylcobalamin (1500mcg) + Pyridoxine (20mg)") == "Vitamin/Supplement"


def test_hormonal_label():
    assert assign_label("Prednisolone (5mg)") == "Hormonal"


def test_dermatology_label():
    assert assign_label("Clobetasol (0.05%) + Ketoconazole (2%)") == "Dermatology"


def test_musculoskeletal_label():
    assert assign_label("Thiocolchicoside (4mg) + Diclofenac (50mg)") == "Analgesic"  # Analgesic takes priority


def test_other_label_unknown():
    assert assign_label("Completely Unknown Compound (999mg)") == "Other"


def test_priority_antidiabetic_over_analgesic():
    """Anti-diabetic checked before Analgesic in priority order."""
    assert PRIORITY_ORDER.index("Anti-diabetic") < PRIORITY_ORDER.index("Analgesic")


def test_label_code_valid():
    code = assign_label_code("Azithromycin (500mg)")
    assert code in CATEGORY_CODES.values()


def test_coverage_above_80pct(sample_data):
    report = coverage_report(sample_data["Salt_Composition"].tolist())
    assert report["coverage_pct"] >= 80.0, f"Coverage {report['coverage_pct']}% < 80%"


def test_coverage_report_structure(sample_data):
    report = coverage_report(sample_data["Salt_Composition"][:1000].tolist())
    assert "total" in report
    assert "labelled" in report
    assert "coverage_pct" in report
    assert "distribution" in report


def test_parse_salt_names_multi():
    names = _parse_salt_names("Paracetamol (500mg) + Ibuprofen (400mg)")
    assert "paracetamol" in names
    assert "ibuprofen" in names


def test_category_codes_complete():
    for cat in PRIORITY_ORDER:
        assert cat in CATEGORY_CODES


def test_code_to_category_invertible():
    for cat, code in CATEGORY_CODES.items():
        if cat != "Other":
            assert CODE_TO_CATEGORY[code] == cat


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_model_loads(model):
    assert model is not None


def test_model_pipeline_steps(model):
    assert hasattr(model, "named_steps")
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_model_accuracy_above_threshold(model, sample_data):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    df_lab = sample_data[sample_data["category"] != "Other"].copy()
    df_lab["label_code"] = df_lab["category"].map(CATEGORY_CODES)
    X = df_lab[["Salt_Composition", "dosage_form", "salt_count"]]
    y = df_lab["label_code"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    assert acc >= 0.90, f"Accuracy {acc:.4f} below threshold"


def test_model_auc_above_threshold(model, sample_data):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    df_lab = sample_data[sample_data["category"] != "Other"].copy()
    df_lab["label_code"] = df_lab["category"].map(CATEGORY_CODES)
    X = df_lab[["Salt_Composition", "dosage_form", "salt_count"]]
    y = df_lab["label_code"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    proba = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
    assert auc >= 0.97, f"AUC {auc:.4f} below threshold"


def test_model_predict_proba_sums_to_one(model, sample_data):
    import numpy as np
    X = sample_data[["Salt_Composition", "dosage_form", "salt_count"]].head(50)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 3. API TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_category_endpoint_antibiotic():
    resp = client.post("/api/v1/classify/category", json={
        "salt_composition": "Azithromycin (500mg)",
        "quantity": "strip of 3 tablets",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_category"] == "Antibiotic"
    assert data["rule_label"] == "Antibiotic"


def test_category_endpoint_cardiac():
    resp = client.post("/api/v1/classify/category", json={
        "salt_composition": "Amlodipine (5mg) + Atenolol (50mg)",
        "quantity": "strip of 10 tablets",
    })
    assert resp.status_code == 200
    assert resp.json()["predicted_category"] == "Cardiac"


def test_category_endpoint_probabilities_sum_to_one():
    resp = client.post("/api/v1/classify/category", json={
        "salt_composition": "Metformin (500mg)",
        "quantity": "strip of 10 tablets",
    })
    assert resp.status_code == 200
    probs = resp.json()["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_category_endpoint_response_structure():
    resp = client.post("/api/v1/classify/category", json={
        "salt_composition": "Pantoprazole (40mg)",
        "quantity": "strip of 10 tablets",
    })
    assert resp.status_code == 200
    data = resp.json()
    for key in ["predicted_category", "category_code", "probabilities", "rule_label", "model_version"]:
        assert key in data
