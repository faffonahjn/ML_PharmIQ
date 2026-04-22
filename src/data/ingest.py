"""
src/data/ingest.py
Raw data loading, cleaning, and validation for PharmIQ Price Tier Classifier.
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} records from {path}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["Name", "Salt_Composition", "MRP"])
    logger.info(f"Dropped {before - len(df):,} exact duplicates")

    # Drop extreme MRP outliers (top 0.1% — data entry errors)
    upper = df["MRP"].quantile(0.999)
    df = df[df["MRP"] <= upper]
    df = df[df["MRP"] > 0]
    logger.info(f"After MRP filter: {len(df):,} records")

    # Strip whitespace in text columns
    for col in ["Name", "Manufacturer", "Salt_Composition", "Quantity"]:
        df[col] = df[col].astype(str).str.strip()

    df = df.reset_index(drop=True)
    return df


def validate(df: pd.DataFrame) -> None:
    assert df["MRP"].isna().sum() == 0, "MRP has nulls"
    assert (df["MRP"] > 0).all(), "Non-positive MRP detected"
    assert df["Salt_Composition"].isna().sum() == 0, "Salt_Composition has nulls"
    logger.info("Validation passed")
