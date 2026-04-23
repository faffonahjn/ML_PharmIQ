# PharmIQ — Pharmaceutical Intelligence Platform

> A three-system production ML platform for Indian pharmaceutical price intelligence, generic substitution, and therapeutic classification across 273,000+ medicines.

[![Tests](https://img.shields.io/badge/tests-98%2F98%20passing-brightgreen)]()
[![AUC](https://img.shields.io/badge/AUC%20OvR-0.8701-blue)]()
[![Python](https://img.shields.io/badge/python-3.13-blue)]()
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)]()
[![Azure](https://img.shields.io/badge/deployed-Azure%20Container%20Apps-0078D4)]()
[![LightGBM](https://img.shields.io/badge/System%203-LightGBM-orange)]()

---

## Platform Overview

PharmIQ is a production ML platform built on the Tata 1mg medicine catalogue (273,655 SKUs). It exposes three distinct ML systems through a single FastAPI service, all deployed on Azure Container Apps.

| System | Description | Endpoint |
|--------|-------------|----------|
| **System 1** — Price Tier Classifier | Predicts Budget/Mid/Premium/Luxury tier from formulation alone — no MRP required | `POST /api/v1/predict` |
| **System 2** — Generic Alternative Recommender | Finds cheaper alternatives to any medicine by salt composition or name | `POST /api/v1/recommend/by-name` |
| **System 3** — Therapeutic Category Classifier | Classifies medicine into 13 therapeutic categories from salt composition | `POST /api/v1/classify/category` |

---

## Live Demo

| Service | URL |
|---------|-----|
| REST API (Swagger) | https://pharmiq-api.redsand-37d94e81.eastus.azurecontainerapps.io/docs |
| Streamlit Dashboard | https://pharmiq-ui.redsand-37d94e81.eastus.azurecontainerapps.io |

---

## System 1 — Price Tier Classifier

Classifies Indian pharmaceutical products into four price tiers relative to their dosage-form category using only formulation attributes. No MRP at inference time.

**The key design decision:** tiers are assigned *within* dosage-form group, not globally. A Rs.200 injection is cheap. A Rs.200 tablet is expensive. Global quantile cuts conflate these. PharmIQ asks: *is this medicine expensive for its type?*

### Performance

| Metric | Value |
|--------|-------|
| AUC OvR Macro (test) | **0.8701** |
| CV AUC OvR (5-fold) | **0.8690 +/- 0.0013** |
| Test Accuracy | 64.5% |
| Training samples | 218,924 |
| Classes | 4 (balanced, ratio 1.009) |

### Per-Class Performance

| Class | AUC | F1 |
|-------|-----|----|
| Budget | 0.870 | 0.69 |
| Mid | 0.812 | 0.60 |
| Premium | 0.796 | 0.56 |
| Luxury | 0.884 | 0.72 |

### Example Request

```bash
curl -X POST https://pharmiq-api.redsand-37d94e81.eastus.azurecontainerapps.io/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "salt_composition": "Paracetamol (500mg)",
    "dosage_form": "tablet",
    "pack_size_units": 10,
    "salt_count": 1,
    "manufacturer_tier": 1,
    "max_dose_mg": 500
  }'
```

```json
{
  "price_tier": "Budget",
  "tier_code": 0,
  "probabilities": {"Budget": 0.7231, "Mid": 0.1893, "Premium": 0.0614, "Luxury": 0.0262},
  "interpretation": "Budget tier — priced in the bottom 25% for this dosage form category.",
  "model_version": "v1"
}
```

---

## System 2 — Generic Alternative Recommender

Index-based lookup engine that finds cheaper alternatives to any medicine. Two recommendation modes:

- **Exact mode** — same active ingredient(s) + same dose strength (strict substitution)
- **Ingredient mode** — same active ingredient(s), any dose (broader alternatives)

Ranked by unit price ascending, then manufacturer tier descending.

### Example — 3,489 alternatives found, 97.3% savings

```bash
curl -X POST https://pharmiq-api.redsand-37d94e81.eastus.azurecontainerapps.io/api/v1/recommend/by-name \
  -H "Content-Type: application/json" \
  -d '{"medicine_name": "Pan 40 Tablet", "top_n": 3, "mode": "exact"}'
```

```json
{
  "query_name": "PAN 40 Tablet",
  "query_salt_composition": "Pantoprazole (40mg)",
  "query_mrp": 187.0,
  "total_found": 3489,
  "cheapest_savings_pct": 97.3,
  "alternatives": [
    {
      "name": "Proplus 40mg Tablet",
      "mrp": 5.0,
      "unit_price": 0.5,
      "manufacturer": "Lexus Organics",
      "manufacturer_tier_label": "Generic",
      "unit_price_savings_pct": 96.0
    }
  ]
}
```

### Index Coverage

| Metric | Value |
|--------|-------|
| Total medicines indexed | 273,655 |
| Unique exact keys (salt + dose) | 15,042 |
| Unique ingredient keys (salt only) | 3,709 |

---

## System 3 — Therapeutic Category Classifier

Rule-seeded LightGBM classifier that assigns medicines to 13 therapeutic categories from salt composition text.

**Architecture:** keyword rules generate labels (82.5% catalogue coverage) -> LightGBM trained on labelled subset -> model generalises to unlabelled records.

### Categories

Antibiotic · Analgesic · Anti-diabetic · Cardiac · Respiratory · Gastrointestinal · Neurological · Vitamin/Supplement · Hormonal · Dermatology · Musculoskeletal · Anti-parasitic · Ophthalmic

### Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | >90% |
| AUC OvR Macro | >0.97 |
| Label coverage | 82.5% of 273k medicines |
| CV (5-fold) | Stable |

### Example

```bash
curl -X POST https://pharmiq-api.redsand-37d94e81.eastus.azurecontainerapps.io/api/v1/classify/category \
  -H "Content-Type: application/json" \
  -d '{"salt_composition": "Azithromycin (500mg)", "quantity": "strip of 3 tablets"}'
```

```json
{
  "predicted_category": "Antibiotic",
  "category_code": 1,
  "probabilities": {"Antibiotic": 0.9994, "Cardiac": 0.0, "Analgesic": 0.0},
  "rule_label": "Antibiotic",
  "model_version": "v1"
}
```

---

## Project Structure

```
ML_PharmIQ/
├── src/
│   ├── data/ingest.py              # Load, clean, validate
│   ├── features/engineer.py        # Feature extraction + target design
│   ├── models/train.py             # System 1 training pipeline
│   ├── recommender/
│   │   ├── salt_normalizer.py      # Canonical + ingredient key generation
│   │   ├── index_builder.py        # Dual lookup index builder
│   │   └── engine.py               # Recommendation engine
│   ├── classifier/
│   │   ├── label_engine.py         # Rule-seeded therapeutic labeller
│   │   └── train_category.py       # System 3 LightGBM training
│   ├── evaluation/evaluate.py      # Metrics + metrics.json
│   └── serving/api.py              # FastAPI — all three systems
├── pipelines/
│   ├── training_pipeline.py        # System 1 end-to-end pipeline
│   └── inference_pipeline.py       # Batch inference
├── notebooks/
│   ├── exploratory/                # EDA notebooks
│   ├── modeling/                   # Training experiments
│   ├── evaluation/                 # Evaluation report
│   └── tuning/04_tuning.py         # Optuna 50-trial search
├── streamlit_app/app.py            # 3-tab Streamlit dashboard
├── scripts/
│   ├── deploy_azure.sh             # Azure deployment script
│   └── monitor_drift.py            # Input drift detection
├── docker/
│   ├── Dockerfile                  # API container
│   └── Dockerfile.streamlit        # UI container
├── tests/
│   ├── test_pharmiq.py             # System 1: 45 tests
│   ├── test_recommender.py         # System 2: 20 tests
│   └── test_category_classifier.py # System 3: 28 tests
├── models/
│   ├── price_tier_classifier_v1.pkl
│   ├── recommender_index.pkl
│   └── category_classifier_v1.pkl
├── MODEL_CARD.md
└── LIMITATIONS.md
```

---

## ML Pipeline — System 1

```
ColumnTransformer
  ├── StandardScaler      -> numeric (5 features)
  ├── OrdinalEncoder      -> dosage_form
  └── TF-IDF (200 terms)  -> Salt_Composition (bigrams, min_df=10)

XGBClassifier (Optuna-tuned, 50 trials)
  ├── n_estimators    : 499
  ├── max_depth       : 8
  ├── learning_rate   : 0.138
  ├── subsample       : 0.734
  ├── colsample_bytree: 0.613
  └── min_child_weight: 6
```

---

## Quickstart

### Prerequisites

```bash
git clone https://github.com/faffonahjn/ML_PharmIQ.git
cd ML_PharmIQ
pip install -r requirements.txt
# Place tata_1mg_Medicine_data.csv in data/raw/
```

### Train all systems

```bash
# System 1: Price Tier Classifier
python pipelines/training_pipeline.py

# System 2: Build recommender index
python -c "
from src.data.ingest import load_raw, clean
from src.recommender.index_builder import build_and_save
df = clean(load_raw('data/raw/tata_1mg_Medicine_data.csv'))
build_and_save(df)
"

# System 3: Therapeutic Category Classifier
python src/classifier/train_category.py
```

### Serve locally

```bash
uvicorn src.serving.api:app --reload --port 8000
# Visit http://localhost:8000/docs
```

### Run tests

```bash
python -m pytest tests/ -v
# Expected: 98 passed
```

### Docker full stack

```bash
docker build -t pharmiq-api:latest -f docker/Dockerfile .
docker build -t pharmiq-ui:latest -f docker/Dockerfile.streamlit .
docker run -p 8000:8000 pharmiq-api:latest
docker run -p 8501:8501 pharmiq-ui:latest
```

---

## Dataset

Tata 1mg medicine catalogue — 273,929 raw records, 273,655 after cleaning.

| Stat | Value |
|------|-------|
| MRP range | Rs.0.80 - Rs.32,650 |
| MRP median | Rs.90 |
| Dosage forms | 12 |
| Unique manufacturers | 11,000+ |
| Multi-salt medicines | 47.8% |
| Dominant form | Tablet (60.6%) |

Raw data not included in repo. Place `tata_1mg_Medicine_data.csv` in `data/raw/` before training.

---

## Key Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| Category-relative target design | A Rs.200 injection is cheap; a Rs.200 tablet is expensive. Group quantile cuts produce balanced classes (ratio 1.009) without resampling. |
| TF-IDF on Salt_Composition | Molecule identity lifted AUC from 0.77 to 0.87 in ablation. Top feature: tfidf_proxetil 200mg (importance 0.042). |
| Dual index (exact + ingredient) | Exact mode for strict substitution; ingredient mode for broader alternatives across dose strengths. |
| Rule-seeded labels for System 3 | 82.5% label coverage without manual annotation. Rules encode clinical domain knowledge; LightGBM generalises. |
| Unified lifespan for all 3 models | Single FastAPI lifespan loads all models once at startup — no per-request disk reads, no deprecated on_event handlers. |
| Optuna tuning (50 trials) | Lifted AUC from 0.8394 to 0.8701 (+307 bps). Deeper trees capture salt composition interactions missed by default params. |

---

## Production Features

- **FastAPI** REST API — all 3 systems, single service, Swagger docs at `/docs`
- **Prediction logging** — every inference written to `logs/predictions.jsonl`
- **Drift detection** — `scripts/monitor_drift.py` flags 2-sigma input shifts
- **MLflow tracking** — experiment metrics and model artifacts logged
- **Azure Container Apps** — API and UI deployed separately, auto-scaling
- **98 pytest tests** — data, features, model, recommender, category, and API layers

---

## Version History

| Version | Systems | AUC | Notes |
|---------|---------|-----|-------|
| v1.0 | System 1 only | 0.8404 | Initial release |
| v1.1 | System 1 | 0.8701 | Optuna tuning, is_branded removed, build_target() refactored, prediction logging added |
| v2.0 | Systems 1 + 2 + 3 | 0.8701 | Generic Recommender + Therapeutic Classifier added, unified lifespan, 98/98 tests |

---

## Author

**Francis Affonah** — Registered Nurse to Clinical ML Engineer  
Accra, Ghana

[LinkedIn](https://linkedin.com/in/francis-affonah-23745a205) · [GitHub](https://github.com/faffonahjn)

---

## Related Projects

- [Medical Insurance Risk Classifier](https://insurance-risk-api.redsand-37d94e81.eastus.azurecontainerapps.io/docs) — XGBoost, AUC 0.899, live on Azure
- [Fertility Outcome Classifier](https://fertility-api.redsand-37d94e81.eastus.azurecontainerapps.io/docs) — XGBoost, AUC 0.9504, live on Azure
