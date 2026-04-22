"""
src/recommender/salt_normalizer.py
Normalizes salt composition strings into canonical keys for exact and fuzzy matching.

Design decisions:
- Canonical key: sorted, lowercased, whitespace-collapsed salt names WITHOUT doses
  → allows matching Azithromycin (250mg) vs Azithromycin (500mg) as same active ingredient
- Exact key: sorted, lowercased WITH doses
  → strict equivalent substitution (same drug, same dose)
- Fuzzy key: salt names only, sorted alphabetically
  → catches spelling variants (Amoxycillin vs Amoxicillin)
"""

import re
from typing import List, Tuple


def _parse_salts(composition: str) -> List[Tuple[str, str]]:
    """
    Parse 'Salt1 (dose1) + Salt2 (dose2)' into [(name, dose), ...]
    Returns list of (salt_name, dose_string) tuples.
    """
    parts = [p.strip() for p in re.split(r'\+', composition) if p.strip()]
    result = []
    for part in parts:
        # Extract dose inside parentheses
        dose_match = re.search(r'\(([^)]+)\)', part)
        dose = dose_match.group(1).strip() if dose_match else ""
        # Strip dose from name
        name = re.sub(r'\s*\([^)]*\)', '', part).strip().lower()
        # Collapse whitespace
        name = re.sub(r'\s+', ' ', name)
        result.append((name, dose))
    return result


def canonical_key(composition: str) -> str:
    """
    Exact match key: sorted salt+dose pairs.
    Paracetamol (500mg) + Ibuprofen (400mg) → 'ibuprofen_400mg|paracetamol_500mg'
    """
    salts = _parse_salts(composition)
    parts = []
    for name, dose in salts:
        dose_norm = re.sub(r'\s+', '', dose.lower())
        parts.append(f"{name}_{dose_norm}")
    return "|".join(sorted(parts))


def ingredient_key(composition: str) -> str:
    """
    Active ingredient key: sorted names WITHOUT doses.
    Matches across different dosages of the same drug(s).
    Paracetamol (500mg) + Ibuprofen (400mg) → 'ibuprofen|paracetamol'
    """
    salts = _parse_salts(composition)
    return "|".join(sorted(name for name, _ in salts))


def normalize_for_display(composition: str) -> str:
    """Clean up whitespace for display purposes."""
    parts = [re.sub(r'\s+', ' ', p.strip()) for p in re.split(r'\+', composition) if p.strip()]
    return " + ".join(parts)
