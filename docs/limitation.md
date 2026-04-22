# PharmIQ — Known Limitations

## 1. `is_branded` Feature Removed (v1.1)
Word-count heuristic (≤3 words, no generic suffix) showed Pearson r=0.01
with Price_Tier and zero AUC impact on removal (AUC unchanged at 0.8407).
Proper brand detection requires external drug database lookup (e.g., CDSCO
registry or OpenFDA). Scheduled for v2.0.

## 2. Premium/Mid Boundary Ambiguity
Premium class F1=0.56, Mid F1=0.60 — weakest diagonal in confusion matrix.
Two medicines with identical salt composition, dosage form, and pack size
can sit on opposite sides of a quantile cut. This is a data property, not
a model failure. The category-relative quantile target is intentionally
strict — a ₹91 tablet and a ₹89 tablet in the same form group will be
assigned different tiers.

## 3. Multivitamin Misclassification
High-ingredient-count supplements (salt_count ≥ 6) are occasionally
misclassified as Luxury due to TF-IDF weighting of long ingredient lists.
The active molecules (vitamins, minerals) are cheap commodities but the
model reads ingredient complexity as a price signal. Mitigation: add a
supplement category flag in v2.0.

## 4. Indian Formulary Scope
Model trained exclusively on Tata 1mg catalogue (273k SKUs, Indian market).
Pricing tiers are relative to Indian pharma pricing norms. Not applicable
to non-Indian markets without retraining on local formulary data.

## 5. Data Vintage
Training data sourced from Tata 1mg at a fixed point in time. Drug pricing
changes with regulatory updates, patent expirations, and market dynamics.
Model should be retrained quarterly against a refreshed catalogue snapshot.

## 6. Confidence Calibration
Median confidence on correct predictions: 0.568. Median on incorrect: 0.430.
Predictions below 0.45 confidence should be treated as uncertain and flagged
for human review. The API returns a confidence score for this purpose.