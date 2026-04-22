# Model Card — PharmIQ Price Tier Classifier v1.1

## Model Details

| Field | Value |
|-------|-------|
| **Name** | PharmIQ Price Tier Classifier |
| **Version** | v1.1 |
| **Type** | Multiclass Classification (4-class) |
| **Algorithm** | XGBoost inside sklearn Pipeline |
| **Serialisation** | joblib (`price_tier_classifier_v1.pkl`) |
| **Author** | Francis Affonah (RN -> Clinical ML Engineer) |
| **Date** | April 2026 |

---

## Intended Use

Classify Indian pharmaceutical products into price tiers (Budget / Mid / Premium / Luxury)
**relative to their dosage-form category**, based solely on formulation attributes — no MRP input required at inference time.

**Primary use cases:**
- Generic substitution recommendation (same formulation, lower tier)
- Pharmacy formulary management and tiering
- Drug affordability analytics
- Insurance reimbursement tier assignment
- Procurement anomaly detection (flag over-priced SKUs)

**Out-of-scope:**
- Absolute price prediction (regression)
- Non-Indian pharmaceutical markets (training data is India-specific)
- Regulatory or clinical pricing decisions
- Medicines with no salt composition information

---

## Training Data

| Field | Detail |
|-------|--------|
| Source | Tata 1mg Medicine Database |
| Raw records | 273,929 |
| After cleaning | 273,655 |
| Dropped | 274 duplicates + top 0.1% MRP outliers |
| Train / Test split | 80% / 20%, stratified |
| Geography | India |
| Coverage | Analgesics, antibiotics, anti-diabetics, cardiac, vitamins, supplements, and more |

---

## Features

| Feature | Type | Description |
|---------|------|-------------|
| `Salt_Composition` | Text (TF-IDF, 200 bigram terms) | Active ingredients string |
| `dosage_form` | Categorical (OrdinalEncoder) | tablet / capsule / injection / liquid / topical / etc. |
| `salt_count` | Numeric | Number of active ingredients |
| `pack_size_units` | Numeric | Pack size (units / ml / gm) |
| `manufacturer_tier` | Ordinal (0/1/2) | 0=generic, 1=mid, 2=top-tier |
| `max_dose_mg` | Numeric | Largest dose in mg (mcg converted) |
| `log_max_dose` | Numeric | log(1 + max_dose_mg) |

**Excluded (leakage):** `log_mrp`, `log_unit_price` — direct MRP derivatives. Initial AUC was 1.0 with these features; removal confirmed leakage.

**Excluded (low signal):** `is_branded` — word-count heuristic showed Pearson r=0.01 with Price_Tier and zero AUC impact on removal. See LIMITATIONS.md.

---

## Target Design

Price tier assigned **within dosage-form group** at Q25/Q50/Q75 MRP thresholds.

| Label | Code | MRP Range |
|-------|------|-----------|
| Budget | 0 | <= 25th percentile within dosage form |
| Mid | 1 | 25th-50th percentile within dosage form |
| Premium | 2 | 50th-75th percentile within dosage form |
| Luxury | 3 | > 75th percentile within dosage form |

**Rationale:** Category-relative tiers are clinically meaningful. A Rs.200 injection is cheap; a Rs.200 tablet is expensive. Global quantile cuts conflate these. Group-level quantiles ask: *is this medicine expensive for its type?* Target class ratio: 1.009 (near-perfectly balanced).

---

## Hyperparameter Tuning

Optuna search over 50 trials, 5-fold stratified CV, scoring=roc_auc_ovr_weighted.

| Parameter | Default | Tuned |
|-----------|---------|-------|
| `n_estimators` | 300 | 499 |
| `max_depth` | 6 | 8 |
| `learning_rate` | 0.05 | 0.1381 |
| `subsample` | 0.80 | 0.7335 |
| `colsample_bytree` | 0.80 | 0.6131 |
| `min_child_weight` | — | 6 |

AUC lift from tuning: 0.8394 -> 0.8701 (+307 bps).

---

## Performance

### Cross-Validation (5-fold Stratified, training set)

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.6450 | 0.0020 |
| AUC OvR (weighted) | 0.8690 | 0.0013 |

### Hold-out Test Set (54,731 samples)

| Metric | Value |
|--------|-------|
| AUC OvR Macro | **0.8701** |
| AUC OvO Macro | 0.8701 |
| Test Accuracy | 64.5% |

### Per-Class Performance

| Class | AUC | Precision | Recall | F1 |
|-------|-----|-----------|--------|----|
| Budget | 0.870 | 0.68 | 0.70 | 0.69 |
| Mid | 0.812 | 0.62 | 0.58 | 0.60 |
| Premium | 0.796 | 0.58 | 0.55 | 0.56 |
| Luxury | 0.884 | 0.69 | 0.75 | 0.72 |

### Confidence Calibration

| Cohort | Median Confidence |
|--------|------------------|
| Correct predictions | 0.568 |
| Incorrect predictions | 0.430 |

Recommended uncertainty threshold: **0.45**. Predictions below this are flagged as low-confidence in the API response.

### Key Ablation Finding

Salt composition TF-IDF lifted AUC from 0.77 -> 0.84 in feature iteration experiment. Molecule identity is the dominant price signal — structural features alone are insufficient. Top feature by XGBoost importance: `tfidf_proxetil 200mg` (score 0.042).

---

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for full details.

1. **Premium/Mid boundary ambiguity** — F1 ~0.56-0.60. Two medicines with identical formulation attributes can sit on opposite sides of a quantile cut. Data property, not model failure.
2. **Multivitamin misclassification** — high salt_count supplements occasionally predicted as Luxury due to TF-IDF weighting of long ingredient lists.
3. **India-centric** — manufacturer tier lists and MRP ranges reflect the Indian market only.
4. **Static manufacturer tiers** — top/mid-tier lists are hardcoded and require periodic updates.
5. **TF-IDF vocabulary fixed at training time** — novel salt names not in vocabulary receive zero contribution.
6. **Data vintage** — recommend quarterly retraining against refreshed catalogue snapshot.

---

## Ethical Considerations

- Model does not use patient data — only publicly available pharmaceutical product data.
- Price tier is relative (within dosage form), not absolute — avoids false equivalences across drug categories.
- Not intended to replace clinical pharmacist judgement on drug substitution.
- Should not be used to set or justify actual drug prices.
- Low-confidence predictions should be routed to human review in production deployments.

---

## Infrastructure

| Component | Detail |
|-----------|--------|
| Serving | FastAPI on Azure Container Apps |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Tests | pytest (45/45 passing) |
| Container | Docker |
| Prediction Logging | JSONL (logs/predictions.jsonl) |
| Drift Detection | scripts/monitor_drift.py (2-sigma threshold) |

---

## Version History

| Version | Date | AUC | Notes |
|---------|------|-----|-------|
| v1.0 | April 2026 | 0.8404 | Initial release, default XGBoost params |
| v1.1 | April 2026 | 0.8701 | Optuna tuning (+297 bps), `is_branded` removed, `build_target()` refactored, prediction logging + drift detection added |