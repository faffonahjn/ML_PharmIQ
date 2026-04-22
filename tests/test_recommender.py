"""
tests/test_recommender.py
Production test suite for PharmIQ Generic Alternative Recommender (System 2).
20 tests covering: salt normalization, index building, engine logic, API endpoints.
"""

import sys
import pickle
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender.salt_normalizer import canonical_key, ingredient_key, normalize_for_display, _parse_salts
from src.recommender.index_builder import load_index, INDEX_PATH
from src.recommender.engine import GenericRecommender
from src.serving.api import app

client = TestClient(app)

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def index():
    assert INDEX_PATH.exists(), f"Index not found at {INDEX_PATH}. Run index_builder first."
    return load_index(INDEX_PATH)


@pytest.fixture(scope="session")
def recommender(index):
    return GenericRecommender(index)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SALT NORMALIZER TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_salts_single():
    result = _parse_salts("Azithromycin (500mg)")
    assert len(result) == 1
    assert result[0][0] == "azithromycin"
    assert result[0][1] == "500mg"


def test_parse_salts_multi():
    result = _parse_salts("Paracetamol (500mg) + Ibuprofen (400mg)")
    assert len(result) == 2
    names = {r[0] for r in result}
    assert "paracetamol" in names
    assert "ibuprofen" in names


def test_parse_salts_extra_whitespace():
    result = _parse_salts("Amoxycillin  (400mg/5ml) +  Clavulanic Acid (57mg/5ml)")
    assert len(result) == 2
    assert result[0][0] == "amoxycillin"


def test_canonical_key_sorted():
    k1 = canonical_key("Paracetamol (500mg) + Ibuprofen (400mg)")
    k2 = canonical_key("Ibuprofen (400mg) + Paracetamol (500mg)")
    assert k1 == k2


def test_canonical_key_single():
    k = canonical_key("Azithromycin (500mg)")
    assert k == "azithromycin_500mg"


def test_canonical_key_dose_sensitive():
    k1 = canonical_key("Azithromycin (250mg)")
    k2 = canonical_key("Azithromycin (500mg)")
    assert k1 != k2


def test_ingredient_key_dose_insensitive():
    k1 = ingredient_key("Azithromycin (250mg)")
    k2 = ingredient_key("Azithromycin (500mg)")
    assert k1 == k2


def test_ingredient_key_sorted():
    k1 = ingredient_key("Paracetamol (500mg) + Ibuprofen (400mg)")
    k2 = ingredient_key("Ibuprofen (200mg) + Paracetamol (650mg)")
    assert k1 == k2


def test_normalize_for_display():
    raw = "Amoxycillin  (400mg/5ml) +  Clavulanic Acid (57mg/5ml)"
    result = normalize_for_display(raw)
    assert "  " not in result
    assert "+" in result


# ─────────────────────────────────────────────────────────────────────────────
# 2. INDEX TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_index_loads(index):
    assert "exact_index" in index
    assert "ingredient_index" in index
    assert "metadata" in index


def test_index_exact_coverage(index):
    meta = index["metadata"]
    assert meta["unique_canonical"] > 10_000


def test_index_ingredient_coverage(index):
    meta = index["metadata"]
    assert meta["unique_ingredients"] > 3_000


def test_index_records_sorted_by_unit_price(index):
    key = "azithromycin_500mg"
    if key in index["exact_index"]:
        records = index["exact_index"][key]
        prices = [r["unit_price"] for r in records]
        assert prices == sorted(prices)


def test_index_total_medicines(index):
    assert index["metadata"]["total_medicines"] > 250_000


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENGINE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_recommend_by_salt_exact(recommender):
    result = recommender.recommend_by_salt(
        salt_composition="Azithromycin (500mg)",
        query_name="Azithral 500 Tablet",
        query_mrp=116.70,
        top_n=10,
        mode="exact",
    )
    assert result.total_found > 100
    assert len(result.alternatives) <= 10
    assert result.mode == "exact"


def test_recommend_returns_cheaper_first(recommender):
    result = recommender.recommend_by_salt(
        salt_composition="Azithromycin (500mg)",
        query_name="Azithral 500 Tablet",
        query_mrp=116.70,
        top_n=10,
        mode="exact",
    )
    prices = [a.unit_price for a in result.alternatives]
    assert prices == sorted(prices)


def test_recommend_excludes_query_medicine(recommender):
    result = recommender.recommend_by_salt(
        salt_composition="Pantoprazole (40mg)",
        query_name="Pan 40 Tablet",
        query_mrp=50.0,
        top_n=20,
        mode="exact",
    )
    names = [a.name for a in result.alternatives]
    assert "Pan 40 Tablet" not in names


def test_recommend_ingredient_mode_finds_more(recommender):
    exact_result = recommender.recommend_by_salt(
        salt_composition="Metformin (500mg)",
        query_name="Glycomet 500 Tablet",
        query_mrp=45.0,
        top_n=50,
        mode="exact",
    )
    ingredient_result = recommender.recommend_by_salt(
        salt_composition="Metformin (500mg)",
        query_name="Glycomet 500 Tablet",
        query_mrp=45.0,
        top_n=50,
        mode="ingredient",
    )
    assert ingredient_result.total_found >= exact_result.total_found


def test_search_by_name_resolves(recommender):
    result = recommender.search_by_name("Pan 40 Tablet", top_n=5)
    assert result.query_salt_composition != ""
    assert "pantoprazole" in result.query_salt_composition.lower()
    assert len(result.alternatives) > 0


def test_search_by_name_not_found(recommender):
    result = recommender.search_by_name("ZZZZNOTAMEDICINE999")
    assert result.total_found == 0
    assert "not found" in result.message.lower()


def test_savings_pct_positive_for_cheaper(recommender):
    result = recommender.recommend_by_salt(
        salt_composition="Azithromycin (500mg)",
        query_name="Azithral 500 Tablet",
        query_mrp=116.70,
        top_n=5,
        mode="exact",
    )
    for alt in result.alternatives:
        assert alt.unit_price_savings_pct > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. API ENDPOINT TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_recommend_by_salt_endpoint():
    resp = client.post("/api/v1/recommend/by-salt", json={
        "salt_composition": "Azithromycin (500mg)",
        "query_name": "Azithral 500 Tablet",
        "query_mrp": 116.70,
        "top_n": 5,
        "mode": "exact",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_found"] > 100
    assert len(data["alternatives"]) == 5
    assert data["mode"] == "exact"


def test_recommend_by_name_endpoint():
    resp = client.post("/api/v1/recommend/by-name", json={
        "medicine_name": "Pan 40 Tablet",
        "top_n": 5,
        "mode": "exact",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_found"] > 0
    assert len(data["alternatives"]) > 0


def test_recommend_invalid_mode():
    resp = client.post("/api/v1/recommend/by-salt", json={
        "salt_composition": "Paracetamol (500mg)",
        "mode": "INVALID",
    })
    assert resp.status_code == 422


def test_recommend_response_structure():
    resp = client.post("/api/v1/recommend/by-salt", json={
        "salt_composition": "Pantoprazole (40mg)",
        "query_name": "Pan 40 Tablet",
        "query_mrp": 50.0,
        "top_n": 3,
        "mode": "exact",
    })
    assert resp.status_code == 200
    data = resp.json()
    required_keys = {"query_name", "query_salt_composition", "mode",
                     "alternatives", "total_found", "cheapest_savings_pct", "message"}
    assert required_keys.issubset(data.keys())
    alt = data["alternatives"][0]
    alt_keys = {"name", "mrp", "unit_price", "manufacturer",
                "manufacturer_tier_label", "unit_price_savings_pct"}
    assert alt_keys.issubset(alt.keys())
