"""
notebooks/04_tuning.py
Optuna hyperparameter search for PharmIQ XGBoost classifier.
Run: python notebooks/04_tuning.py
"""

import sys
import yaml
import optuna
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from src.data.ingest import load_raw, clean
from src.features.engineer import engineer, FEATURE_COLS, TARGET_COL

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Data ---
print("Loading data...")
df = engineer(clean(load_raw(PROJECT_ROOT / "data" / "raw" / "tata_1mg_Medicine_data.csv")))
X = df[FEATURE_COLS]
y = df[TARGET_COL]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

numeric_features = ["salt_count", "pack_size_units", "manufacturer_tier",
                    "max_dose_mg", "log_max_dose"]
categorical_features = ["dosage_form"]
text_feature = "Salt_Composition"

def build_pipeline(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
        ("tfidf", TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=10), text_feature),
    ])

    xgb = XGBClassifier(
        **params,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", xgb)])


def objective(trial):
    pipeline = build_pipeline(trial)
    scores = cross_val_score(
        pipeline, X, y, cv=cv,
        scoring="roc_auc_ovr_weighted",
        n_jobs=-1,
    )
    return scores.mean()


print("Starting Optuna search (50 trials)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest AUC  : {study.best_value:.4f}")
print(f"Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# --- Write best params back to config.yaml ---
config_path = PROJECT_ROOT / "configs" / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

config["model"]["params"].update({
    "n_estimators": study.best_params["n_estimators"],
    "max_depth": study.best_params["max_depth"],
    "learning_rate": round(study.best_params["learning_rate"], 6),
    "subsample": round(study.best_params["subsample"], 4),
    "colsample_bytree": round(study.best_params["colsample_bytree"], 4),
    "min_child_weight": study.best_params["min_child_weight"],
})

with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"\nBest params written to configs/config.yaml")
print("Run python pipelines/training_pipeline.py to retrain with tuned params.")