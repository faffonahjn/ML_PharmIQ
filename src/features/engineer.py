"""
src/features/engineer.py
Feature engineering pipeline for PharmIQ Price Tier Classifier.
"""

import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

# Manufacturer tier mapping — top Indian pharma by revenue/market cap
TOP_TIER_MANUFACTURERS = {
    "Sun Pharmaceutical Industries Ltd", "Cipla Ltd", "Dr. Reddy's Laboratories Ltd",
    "Lupin Ltd", "Aurobindo Pharma Ltd", "Alkem Laboratories Ltd",
    "Torrent Pharmaceuticals Ltd", "Abbott", "GlaxoSmithKline", "Pfizer Ltd",
    "Novartis India Ltd", "Sanofi India Ltd", "Mankind Pharma Ltd",
    "Intas Pharmaceuticals Ltd", "Zydus Lifesciences Ltd",
}

MID_TIER_MANUFACTURERS = {
    "Micro Labs Ltd", "Leeford Healthcare Ltd", "Macleods Pharmaceuticals Ltd",
    "Cadila Pharmaceuticals Ltd", "Elder Pharmaceuticals Ltd",
    "FDC Ltd", "Wockhardt Ltd", "IPCA Laboratories Ltd",
}

DOSAGE_FORM_MAP = {
    "tablet": "tablet", "capsule": "capsule", "syrup": "liquid",
    "suspension": "liquid", "injection": "injection", "vial": "injection",
    "cream": "topical", "ointment": "topical", "gel": "topical",
    "drops": "drops", "spray": "spray", "inhaler": "inhaler",
    "lotion": "topical", "solution": "liquid", "dry syrup": "liquid",
    "powder": "powder", "patch": "patch", "suppository": "suppository",
}


def extract_dosage_form(quantity: str) -> str:
    q = quantity.lower()
    for key, form in DOSAGE_FORM_MAP.items():
        if key in q:
            return form
    return "other"


def extract_pack_size(quantity: str) -> float:
    """Extract numeric pack size (units/ml/gm)."""
    match = re.search(r"of\s+([\d.]+)", quantity)
    if match:
        return float(match.group(1))
    return 1.0


def extract_salt_count(salt_composition: str) -> int:
    """Count number of active ingredients."""
    return len([s for s in salt_composition.split("+") if s.strip()])


def extract_unit_price(mrp: float, pack_size: float) -> float:
    """Price per unit/ml."""
    if pack_size <= 0:
        return mrp
    return mrp / pack_size


def extract_max_dose_mg(salt_composition: str) -> float:
    """Extract the largest dosage value (mg/mcg/%) from salt composition."""
    # Convert mcg → mg (/1000), % → treat as 0
    parts = re.findall(r"([\d.]+)\s*(mg|mcg|%|g\b|iu|units?)", salt_composition.lower())
    values = []
    for val, unit in parts:
        v = float(val)
        if unit == "mcg":
            v /= 1000
        elif unit in ("%", "iu", "units", "unit"):
            v = 0
        elif unit == "g":
            v *= 1000
        values.append(v)
    return max(values) if values else 0.0


def is_branded_name(name: str) -> int:
    """Heuristic: branded names are shorter, title-cased, without generic suffixes."""
    generic_suffixes = ["tablet", "capsule", "syrup", "injection", "cream", "gel",
                        "solution", "suspension", "drops", "spray", "powder"]
    lower = name.lower()
    has_suffix = any(s in lower for s in generic_suffixes)
    # Short names without generic suffix words tend to be brand names
    word_count = len(name.split())
    return int(word_count <= 3 and not has_suffix)


def manufacturer_tier(manufacturer: str) -> int:
    """0=generic, 1=mid, 2=top"""
    if manufacturer in TOP_TIER_MANUFACTURERS:
        return 2
    if manufacturer in MID_TIER_MANUFACTURERS:
        return 1
    return 0


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Category-relative price tier — is this medicine expensive FOR ITS DOSAGE FORM?

    Tier labels (4-class):
      0 = Budget   : MRP <= 25th pctile of its dosage_form group
      1 = Mid      : 25th < MRP <= 50th pctile of its dosage_form group
      2 = Premium  : 50th < MRP <= 75th pctile of its dosage_form group
      3 = Luxury   : MRP > 75th pctile of its dosage_form group

    Uses vectorized groupby transform — no column drop side effects.
    """
    def _assign(group):
        q25, q50, q75 = group.quantile([0.25, 0.50, 0.75])
        conditions = [
            group <= q25,
            (group > q25) & (group <= q50),
            (group > q50) & (group <= q75),
        ]
        return pd.Series(np.select(conditions, [0, 1, 2], default=3), index=group.index)

    df["Price_Tier"] = (
        df.groupby("dosage_form")["MRP"]
        .transform(_assign)
        .astype(int)
    )
    logger.info(f"Target distribution:\n{df['Price_Tier'].value_counts().sort_index()}")
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["dosage_form"] = df["Quantity"].apply(extract_dosage_form)
    df["pack_size_units"] = df["Quantity"].apply(extract_pack_size)
    df["salt_count"] = df["Salt_Composition"].apply(extract_salt_count)
    df["unit_price"] = df.apply(lambda r: extract_unit_price(r["MRP"], r["pack_size_units"]), axis=1)
    df["manufacturer_tier"] = df["Manufacturer"].apply(manufacturer_tier)
    df["max_dose_mg"] = df["Salt_Composition"].apply(extract_max_dose_mg)
    df["log_max_dose"] = np.log1p(df["max_dose_mg"])

    # Log-transform skewed numerics
    df["log_mrp"] = np.log1p(df["MRP"])
    df["log_unit_price"] = np.log1p(df["unit_price"])

    df = build_target(df)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


FEATURE_COLS = [
    "salt_count", "pack_size_units", "manufacturer_tier",
    "dosage_form", "max_dose_mg", "log_max_dose",
    "Salt_Composition"  # TF-IDF text feature
]
# EXCLUDED (leakage): log_mrp, log_unit_price — derived from MRP which defines the target
# EXCLUDED (low signal): is_branded — word-count heuristic (Pearson r=0.01); see LIMITATIONS.md

TARGET_COL = "Price_Tier"
