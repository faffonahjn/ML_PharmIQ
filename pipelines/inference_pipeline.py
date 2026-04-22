"""
pipelines/inference_pipeline.py
Batch inference pipeline: load model → process input CSV → predict → save results.
Run: python pipelines/inference_pipeline.py --input data/external/sample_1000.csv
"""

import sys
import argparse
import logging
import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(),
                               logging.FileHandler(PROJECT_ROOT / "logs" / "inference.log", mode="a")])
logger = logging.getLogger("inference")

TIER_MAP = {0: "Budget", 1: "Mid", 2: "Premium", 3: "Luxury"}


def run_inference(input_path: str, output_path: str = None):
    from src.features.engineer import (
        extract_dosage_form, extract_pack_size, extract_salt_count,
        extract_max_dose_mg, manufacturer_tier, FEATURE_COLS,
    )

    logger.info(f"Loading model...")
    pipeline = joblib.load(PROJECT_ROOT / "models" / "price_tier_classifier_v1.pkl")

    logger.info(f"Loading input: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"  Records: {len(df):,}")

    # Feature engineering
    df["dosage_form"] = df["Quantity"].apply(extract_dosage_form)
    df["pack_size_units"] = df["Quantity"].apply(extract_pack_size)
    df["salt_count"] = df["Salt_Composition"].apply(extract_salt_count)
    df["manufacturer_tier"] = df["Manufacturer"].apply(manufacturer_tier)
    df["max_dose_mg"] = df["Salt_Composition"].apply(extract_max_dose_mg)
    df["log_max_dose"] = np.log1p(df["max_dose_mg"])

    X = df[FEATURE_COLS]
    preds = pipeline.predict(X)
    probas = pipeline.predict_proba(X)

    df["Predicted_Tier"] = [TIER_MAP[p] for p in preds]
    df["Confidence"] = [round(probas[i][p], 4) for i, p in enumerate(preds)]
    df["Budget_Prob"] = probas[:, 0].round(4)
    df["Mid_Prob"] = probas[:, 1].round(4)
    df["Premium_Prob"] = probas[:, 2].round(4)
    df["Luxury_Prob"] = probas[:, 3].round(4)

    if output_path is None:
        output_path = PROJECT_ROOT / "artifacts" / "batch_predictions.csv"

    df.to_csv(output_path, index=False)
    logger.info(f"Results saved → {output_path}")

    # Summary
    tier_dist = df["Predicted_Tier"].value_counts()
    logger.info(f"\nTier distribution:\n{tier_dist.to_string()}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PharmIQ Batch Inference Pipeline")
    parser.add_argument("--input", default="data/external/sample_1000.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_inference(args.input, args.output)
