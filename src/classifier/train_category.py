"""
src/classifier/train_category.py
System 3: Therapeutic Category Classifier.

Label source   : rule-seeded labels from label_engine.py (82.5% coverage)
Training data  : labelled subset only (225K+ medicines)
Features       : TF-IDF on Salt_Composition (bigrams) + salt_count + dosage_form
Algorithm      : LightGBM (fast, handles 13-class imbalance well)
Target         : 13 therapeutic categories

Run: python src/classifier/train_category.py
"""

import os
import sys
import logging
import joblib
import json
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingest import load_raw, clean
from src.features.engineer import extract_dosage_form, extract_salt_count
from src.classifier.label_engine import (
    assign_label, PRIORITY_ORDER, CATEGORY_CODES, CODE_TO_CATEGORY, N_CLASSES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = PROJECT_ROOT / "models" / "category_classifier_v1.pkl"
FEATURE_COLS = ["Salt_Composition", "dosage_form", "salt_count"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    import re
    df = df.copy()
    df["dosage_form"] = df["Quantity"].apply(extract_dosage_form)
    df["salt_count"] = df["Salt_Composition"].apply(extract_salt_count)
    return df


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("tfidf", TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=5,
            sublinear_tf=True,
        ), "Salt_Composition"),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["dosage_form"]),
        ("num", "passthrough", ["salt_count"]),
    ])


def train():
    try:
        import lightgbm as lgb
        Classifier = lgb.LGBMClassifier
        clf_params = dict(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf_name = "LightGBM"
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        from xgboost import XGBClassifier
        Classifier = XGBClassifier
        clf_params = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        clf_name = "XGBoost"

    logger.info(f"Classifier: {clf_name}")

    # Data
    df_raw = load_raw(PROJECT_ROOT / "data" / "raw" / "tata_1mg_Medicine_data.csv")
    df = clean(df_raw)
    df = build_features(df)

    # Labels
    df["category"] = df["Salt_Composition"].apply(assign_label)
    df["label_code"] = df["category"].map(CATEGORY_CODES)

    # Train on labelled subset only
    df_labelled = df[df["category"] != "Other"].copy()
    logger.info(f"Labelled subset: {len(df_labelled):,} / {len(df):,} ({len(df_labelled)/len(df)*100:.1f}%)")
    logger.info(f"Class distribution:\n{df_labelled['category'].value_counts().to_string()}")

    X = df_labelled[FEATURE_COLS]
    y = df_labelled["label_code"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", Classifier(**clf_params)),
    ])

    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    logger.info(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit + evaluate
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment("PharmIQ_CategoryClassifier")

    with mlflow.start_run(run_name=f"{clf_name}_category_v1"):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

        class_names = [CODE_TO_CATEGORY[i] for i in sorted(CODE_TO_CATEGORY.keys()) if CODE_TO_CATEGORY[i] != "Other"]
        # Only include classes present in test set
        present_codes = sorted(y_test.unique())
        present_names = [CODE_TO_CATEGORY[c] for c in present_codes]

        report = classification_report(y_test, y_pred, labels=present_codes, target_names=present_names)

        logger.info(f"\n{'='*60}")
        logger.info(f"Test Accuracy : {acc:.4f}")
        logger.info(f"AUC OvR Macro : {auc:.4f}")
        logger.info(f"\nClassification Report:\n{report}")
        logger.info(f"{'='*60}")

        metrics = {
            "test_accuracy": round(acc, 4),
            "auc_ovr_macro": round(auc, 4),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "n_classes": N_CLASSES,
            "labelled_samples": len(df_labelled),
        }
        mlflow.log_metrics(metrics)

        # Save model
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        # Save metrics
        metrics_path = PROJECT_ROOT / "models" / "category_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Model saved → {MODEL_PATH}")

    return pipeline, metrics


if __name__ == "__main__":
    train()
