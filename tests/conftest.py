"""
tests/conftest.py
Injects the trained model into the API pipeline global before API tests run.
"""
import sys
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.serving.api as api_module

def pytest_configure(config):
    model_path = PROJECT_ROOT / "models" / "price_tier_classifier_v1.pkl"
    if model_path.exists():
        api_module.pipeline = joblib.load(model_path)
