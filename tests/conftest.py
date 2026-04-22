"""
tests/conftest.py
Injects trained models and recommender index into API globals before tests run.
"""
import sys
import pickle
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.serving.api as api_module

def pytest_configure(config):
    # System 1: Price Tier Classifier
    model_path = PROJECT_ROOT / "models" / "price_tier_classifier_v1.pkl"
    if model_path.exists():
        api_module.pipeline = joblib.load(model_path)

    # System 2: Generic Recommender
    index_path = PROJECT_ROOT / "models" / "recommender_index.pkl"
    if index_path.exists():
        from src.recommender.engine import GenericRecommender
        with open(index_path, "rb") as f:
            index = pickle.load(f)
        api_module.recommender = GenericRecommender(index)

    # System 3: Category Classifier
    cat_path = PROJECT_ROOT / "models" / "category_classifier_v1.pkl"
    if cat_path.exists():
        api_module.category_pipeline = joblib.load(cat_path)
