"""
scripts/monitor_drift.py
Reads predictions.jsonl and flags input distribution drift.
Run: python scripts/monitor_drift.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"

# Training baseline stats (from EDA)
BASELINES = {
    "salt_count":       {"mean": 1.72,  "std": 0.85},
    "pack_size_units":  {"mean": 18.4,  "std": 22.1},
    "max_dose_mg":      {"mean": 187.3, "std": 312.4},
    "confidence":       {"mean": 0.52,  "std": 0.14},
}

DRIFT_THRESHOLD = 2.0  # standard deviations


def load_logs() -> list:
    if not LOG_PATH.exists():
        print("No predictions log found. Run the API and make some predictions first.")
        sys.exit(0)
    records = []
    with open(LOG_PATH) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def check_drift(records: list) -> None:
    print(f"Prediction Log Monitor")
    print(f"=" * 50)
    print(f"Total predictions logged : {len(records)}")
    print(f"First prediction         : {records[0]['timestamp']}")
    print(f"Latest prediction        : {records[-1]['timestamp']}")
    print()

    # Extract numeric fields
    data = {k: [] for k in BASELINES}
    tier_counts = {}

    for r in records:
        inp = r["input"]
        for k in ["salt_count", "pack_size_units", "max_dose_mg"]:
            if k in inp:
                data[k].append(float(inp[k]))
        data["confidence"].append(float(r["confidence"]))
        tier = r["prediction"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    # Drift check
    print("Drift Analysis (vs training baseline):")
    print(f"{'Feature':<22} {'Baseline':>10} {'Current':>10} {'Z-score':>10} {'Status':>10}")
    print("-" * 65)

    alerts = []
    for feature, baseline in BASELINES.items():
        values = data[feature]
        if len(values) < 10:
            print(f"{feature:<22} {'(need 10+ samples)':>42}")
            continue
        current_mean = np.mean(values)
        z = abs(current_mean - baseline["mean"]) / baseline["std"]
        status = "DRIFT" if z > DRIFT_THRESHOLD else "OK"
        if status == "DRIFT":
            alerts.append(feature)
        print(f"{feature:<22} {baseline['mean']:>10.2f} {current_mean:>10.2f} {z:>10.2f} {status:>10}")

    print()
    print("Prediction Distribution:")
    for tier in ["Budget", "Mid", "Premium", "Luxury"]:
        count = tier_counts.get(tier, 0)
        pct = count / len(records) * 100
        print(f"  {tier:<10}: {count:>5} ({pct:.1f}%)")

    print()
    if alerts:
        print(f"ALERT: Drift detected in: {', '.join(alerts)}")
        print("Recommendation: inspect recent inputs and consider retraining.")
    else:
        print("No drift detected. Model inputs within expected distribution.")


if __name__ == "__main__":
    records = load_logs()
    check_drift(records)