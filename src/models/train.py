"""
src/models/train.py
Training pipeline: sklearn Pipeline + XGBoost + MLflow tracking.
"""

import os
import sys
import logging
import joblib
from sklearn import pipeline
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingest import load_raw, clean, validate
from src.features.engineer import engineer, FEATURE_COLS, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["salt_count", "pack_size_units", "manufacturer_tier",
                        "max_dose_mg", "log_max_dose"]
    categorical_features = ["dosage_form"]
    text_feature = "Salt_Composition"

    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
        ("tfidf", TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=10), text_feature),
    ])


def build_pipeline(config: dict) -> Pipeline:
    params = config["model"]["params"]
    xgb = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        eval_metric=params["eval_metric"],
        random_state=params["random_state"],
        n_jobs=-1,
        verbosity=0,
    )
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", xgb),
    ])


def train():
    config = load_config()
    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment("PharmIQ_PriceTier")

    # --- Data ---
    df_raw = load_raw(PROJECT_ROOT / config["data"]["raw_path"])
    df_clean = clean(df_raw)
    validate(df_clean)
    df = engineer(df_clean)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y,
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # --- CV ---
    pipeline = build_pipeline(config)
    cv = StratifiedKFold(n_splits=config["training"]["cv_folds"], shuffle=True,
                     random_state=config["training"]["random_state"])
    cv_acc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc_ovr_weighted", n_jobs=-1)

    logger.info(f"CV Accuracy : {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
    logger.info(f"CV AUC OvR  : {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

    # --- Final fit ---
    with mlflow.start_run(run_name="xgb_price_tier_v1"):
        pipeline.fit(X_train, y_train)

        from src.evaluation.evaluate import evaluate_model
        metrics = evaluate_model(pipeline, X_test, y_test)

        mlflow.log_params(config["model"]["params"])
        mlflow.log_metrics({
            "cv_accuracy_mean": float(cv_acc.mean()),
            "cv_accuracy_std": float(cv_acc.std()),
            "cv_auc_mean": float(cv_auc.mean()),
            "cv_auc_std": float(cv_auc.std()),
            **metrics,
        })
        mlflow.sklearn.log_model(pipeline, "model")

        # Persist
        model_path = PROJECT_ROOT / "models" / "price_tier_classifier_v1.pkl"
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved → {model_path}")

    return pipeline, metrics


if __name__ == "__main__":
    train()
