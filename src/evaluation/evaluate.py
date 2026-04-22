"""
src/evaluation/evaluate.py
Model evaluation: multiclass metrics, confusion matrix, ROC-AUC OvR.
"""

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Budget", "Mid", "Premium", "Luxury"]


def evaluate_model(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc_ovr = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    auc_ovo = roc_auc_score(y_test, y_proba, multi_class="ovo", average="macro")

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "test_accuracy": round(acc, 4),
        "auc_ovr_macro": round(auc_ovr, 4),
        "auc_ovo_macro": round(auc_ovo, 4),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Test Accuracy : {acc:.4f}")
    logger.info(f"AUC OvR Macro : {auc_ovr:.4f}")
    logger.info(f"AUC OvO Macro : {auc_ovo:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"{'='*50}")

    # Persist metrics
    out = Path(__file__).resolve().parents[2] / "models" / "metrics.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
