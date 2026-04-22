"""
src/recommender/engine.py
Generic Alternative Recommender — core lookup engine.

Three recommendation modes:
  1. EXACT      — same salt composition + same dose strength → strict substitution
  2. INGREDIENT — same active ingredient(s), any dose → broader alternatives
  3. SEARCH     — free-text medicine name → resolve to salt → recommend

Ranking logic (applied after index lookup):
  Primary   : unit_price ascending (cheapest per unit first)
  Secondary : manufacturer_tier descending (prefer quality)
  Filter    : exclude query medicine itself; optionally filter by dosage_form
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path

from src.recommender.salt_normalizer import canonical_key, ingredient_key, normalize_for_display
from src.features.engineer import manufacturer_tier as get_mfr_tier

logger = logging.getLogger(__name__)

TIER_LABELS = {0: "Generic", 1: "Mid-tier", 2: "Top-tier"}

TOP_TIER_MANUFACTURERS = {
    "Sun Pharmaceutical Industries Ltd", "Cipla Ltd", "Dr. Reddy's Laboratories Ltd",
    "Lupin Ltd", "Aurobindo Pharma Ltd", "Alkem Laboratories Ltd",
    "Torrent Pharmaceuticals Ltd", "Abbott", "GlaxoSmithKline", "Pfizer Ltd",
    "Novartis India Ltd", "Sanofi India Ltd", "Mankind Pharma Ltd",
    "Intas Pharmaceuticals Ltd", "Zydus Lifesciences Ltd",
}


@dataclass
class Alternative:
    name: str
    salt_composition: str
    mrp: float
    unit_price: float
    pack_qty: float
    quantity: str
    manufacturer: str
    manufacturer_tier: int
    manufacturer_tier_label: str
    mrp_savings_pct: float          # % cheaper than query medicine MRP
    unit_price_savings_pct: float   # % cheaper per unit than query medicine


@dataclass
class RecommendationResult:
    query_name: str
    query_salt_composition: str
    query_mrp: float
    query_unit_price: float
    query_manufacturer: str
    mode: str                        # "exact" | "ingredient"
    alternatives: List[Alternative]
    total_found: int
    cheapest_savings_pct: float      # Best-case savings vs query
    message: str = ""


class GenericRecommender:
    def __init__(self, index: Dict):
        self.exact_index = index["exact_index"]
        self.ingredient_index = index["ingredient_index"]
        self.metadata = index["metadata"]
        logger.info(
            f"GenericRecommender loaded — "
            f"{self.metadata['total_medicines']:,} medicines, "
            f"{self.metadata['unique_canonical']:,} exact keys, "
            f"{self.metadata['unique_ingredients']:,} ingredient keys"
        )

    def _enrich_alternative(
        self, record: dict, query_mrp: float, query_unit_price: float
    ) -> Alternative:
        mfr_tier = get_mfr_tier(record["manufacturer"])
        mrp_savings = round((query_mrp - record["mrp"]) / query_mrp * 100, 1) if query_mrp > 0 else 0.0
        up_savings = round((query_unit_price - record["unit_price"]) / query_unit_price * 100, 1) if query_unit_price > 0 else 0.0
        return Alternative(
            name=record["name"],
            salt_composition=normalize_for_display(record["salt_composition"]),
            mrp=record["mrp"],
            unit_price=record["unit_price"],
            pack_qty=record["pack_qty"],
            quantity=record["quantity"],
            manufacturer=record["manufacturer"],
            manufacturer_tier=mfr_tier,
            manufacturer_tier_label=TIER_LABELS[mfr_tier],
            mrp_savings_pct=mrp_savings,
            unit_price_savings_pct=up_savings,
        )

    def _rank(
        self,
        candidates: List[dict],
        query_name: str,
        query_unit_price: float,
        dosage_form_filter: Optional[str],
        top_n: int,
    ) -> List[dict]:
        # Exclude query medicine itself
        candidates = [c for c in candidates if c["name"].lower() != query_name.lower()]

        # Optional dosage form filter
        if dosage_form_filter:
            from src.features.engineer import extract_dosage_form
            candidates = [
                c for c in candidates
                if extract_dosage_form(c["quantity"]) == dosage_form_filter
            ]

        # Sort: unit_price ASC (primary), manufacturer tier DESC (secondary)
        candidates.sort(
            key=lambda c: (
                c["unit_price"],
                -get_mfr_tier(c["manufacturer"]),
            )
        )
        return candidates[:top_n]

    def recommend_by_salt(
        self,
        salt_composition: str,
        query_name: str = "",
        query_mrp: float = 0.0,
        top_n: int = 10,
        mode: str = "exact",
        dosage_form_filter: Optional[str] = None,
    ) -> RecommendationResult:
        """
        Core recommendation method.
        mode: 'exact' (same dose) | 'ingredient' (any dose of same drug)
        """
        if mode == "exact":
            key = canonical_key(salt_composition)
            candidates = self.exact_index.get(key, [])
        else:
            key = ingredient_key(salt_composition)
            candidates = self.ingredient_index.get(key, [])

        # Compute query unit price
        query_unit_price = query_mrp  # fallback if no pack info
        # Try to find exact query record for accurate unit price
        for c in candidates:
            if c["name"].lower() == query_name.lower():
                query_unit_price = c["unit_price"]
                break

        ranked = self._rank(candidates, query_name, query_unit_price, dosage_form_filter, top_n)
        alternatives = [self._enrich_alternative(r, query_mrp, query_unit_price) for r in ranked]

        cheapest_savings = alternatives[0].unit_price_savings_pct if alternatives else 0.0
        total_found = len([c for c in candidates if c["name"].lower() != query_name.lower()])

        return RecommendationResult(
            query_name=query_name or "Unknown",
            query_salt_composition=normalize_for_display(salt_composition),
            query_mrp=query_mrp,
            query_unit_price=query_unit_price,
            query_manufacturer="",
            mode=mode,
            alternatives=alternatives,
            total_found=total_found,
            cheapest_savings_pct=cheapest_savings,
            message=f"Found {total_found} alternatives ({mode} match). Showing top {len(alternatives)}.",
        )

    def search_by_name(
        self,
        medicine_name: str,
        top_n: int = 10,
        mode: str = "exact",
        dosage_form_filter: Optional[str] = None,
    ) -> RecommendationResult:
        """
        Resolve medicine name → salt composition → recommend alternatives.
        Falls back to ingredient mode if exact yields < 3 results.
        """
        # Search across all records
        name_lower = medicine_name.lower().strip()
        matched = None

        # Try exact index first
        for key, records in self.exact_index.items():
            for r in records:
                if r["name"].lower() == name_lower:
                    matched = r
                    break
            if matched:
                break

        if not matched:
            # Partial match fallback
            for key, records in self.exact_index.items():
                for r in records:
                    if name_lower in r["name"].lower():
                        matched = r
                        break
                if matched:
                    break

        if not matched:
            return RecommendationResult(
                query_name=medicine_name,
                query_salt_composition="",
                query_mrp=0.0,
                query_unit_price=0.0,
                query_manufacturer="",
                mode=mode,
                alternatives=[],
                total_found=0,
                cheapest_savings_pct=0.0,
                message=f"Medicine '{medicine_name}' not found in database.",
            )

        result = self.recommend_by_salt(
            salt_composition=matched["salt_composition"],
            query_name=matched["name"],
            query_mrp=matched["mrp"],
            top_n=top_n,
            mode=mode,
            dosage_form_filter=dosage_form_filter,
        )
        result.query_manufacturer = matched["manufacturer"]
        result.query_unit_price = matched["unit_price"]
        return result
