"""
src/recommender/index_builder.py
Builds and persists the in-memory lookup indices for the Generic Alternative Recommender.

Two indices:
  1. exact_index   : canonical_key → list of medicine records (same drug, same dose)
  2. ingredient_index : ingredient_key → list of medicine records (same drug, any dose)

Both are pre-sorted by unit_price ascending at build time — recommender just slices top-N.
"""

import re
import logging
import pickle
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

from src.recommender.salt_normalizer import canonical_key, ingredient_key

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "models" / "recommender_index.pkl"


def _extract_pack_qty(quantity: str) -> float:
    """Extract numeric pack size from Quantity string."""
    m = re.search(r"of\s+([\d.]+)", quantity)
    return float(m.group(1)) if m else 1.0


def build_index(df: pd.DataFrame) -> Dict:
    """
    Build dual lookup index from cleaned dataframe.
    Returns dict with keys: exact_index, ingredient_index, metadata.
    """
    logger.info("Building recommender indices...")

    records = []
    for _, row in df.iterrows():
        pack_qty = _extract_pack_qty(str(row["Quantity"]))
        unit_price = round(float(row["MRP"]) / pack_qty, 4) if pack_qty > 0 else float(row["MRP"])
        comp = str(row["Salt_Composition"])
        records.append({
            "name": str(row["Name"]),
            "salt_composition": comp,
            "mrp": float(row["MRP"]),
            "unit_price": unit_price,
            "pack_qty": pack_qty,
            "quantity": str(row["Quantity"]),
            "manufacturer": str(row["Manufacturer"]),
            "canonical_key": canonical_key(comp),
            "ingredient_key": ingredient_key(comp),
        })

    df_idx = pd.DataFrame(records)

    # Build exact index: same salt + same dose
    exact_index: Dict[str, List[dict]] = {}
    for key, group in df_idx.groupby("canonical_key"):
        sorted_group = group.sort_values("unit_price").to_dict(orient="records")
        exact_index[key] = sorted_group

    # Build ingredient index: same active ingredient(s), any dose
    ingredient_index: Dict[str, List[dict]] = {}
    for key, group in df_idx.groupby("ingredient_key"):
        sorted_group = group.sort_values("unit_price").to_dict(orient="records")
        ingredient_index[key] = sorted_group

    index = {
        "exact_index": exact_index,
        "ingredient_index": ingredient_index,
        "metadata": {
            "total_medicines": len(df_idx),
            "unique_canonical": len(exact_index),
            "unique_ingredients": len(ingredient_index),
        },
    }

    logger.info(f"  Total medicines      : {len(df_idx):,}")
    logger.info(f"  Unique exact keys    : {len(exact_index):,}")
    logger.info(f"  Unique ingredient keys: {len(ingredient_index):,}")

    return index


def save_index(index: Dict, path: Path = INDEX_PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Index saved → {path} ({size_mb:.1f} MB)")


def load_index(path: Path = INDEX_PATH) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_and_save(df: pd.DataFrame, path: Path = INDEX_PATH) -> Dict:
    index = build_index(df)
    save_index(index, path)
    return index
