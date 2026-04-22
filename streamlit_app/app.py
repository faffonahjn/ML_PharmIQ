"""
streamlit_app/app.py
PharmIQ — Medicine Price Tier Classifier Dashboard
"""

import os
import sys
import math
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.engineer import (
    extract_dosage_form, extract_pack_size, extract_salt_count,
    extract_max_dose_mg, manufacturer_tier, FEATURE_COLS,
)

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmIQ | Price Tier Classifier",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "price_tier_classifier_v1.pkl"))
API_URL = os.getenv("API_URL", "http://localhost:8000")

TIER_MAP = {0: "Budget", 1: "Mid", 2: "Premium", 3: "Luxury"}
TIER_COLORS = {
    "Budget": "#2ECC71",
    "Mid": "#3498DB",
    "Premium": "#9B59B6",
    "Luxury": "#E74C3C",
}
DOSAGE_FORMS = [
    "tablet", "capsule", "liquid", "injection", "topical",
    "drops", "spray", "inhaler", "powder", "patch", "suppository", "other",
]

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    pipeline = load_model()
    model_loaded = True
except Exception as e:
    pipeline = None
    model_loaded = False

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {font-size:2.4rem; font-weight:700; color:#0D4A4A; margin-bottom:0;}
    .sub-title  {font-size:1.1rem; color:#666; margin-bottom:1.5rem;}
    .tier-badge {
        display:inline-block; padding:0.4rem 1.2rem;
        border-radius:20px; font-size:1.3rem; font-weight:700;
        color:white; margin:0.5rem 0;
    }
    .metric-card {
        background:#f8f9fa; border-radius:10px;
        padding:1rem 1.2rem; margin:0.3rem 0;
        border-left:4px solid #0D4A4A;
    }
    .stAlert {border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💊 PharmIQ")
    st.caption("Medicine Price Tier Classifier v1.0")
    st.divider()

    st.markdown("#### Model Status")
    if model_loaded:
        st.success("Model loaded ✓")
    else:
        st.error("Model not found")

    st.divider()
    st.markdown("#### About")
    st.markdown("""
    Predicts whether a medicine is **Budget / Mid / Premium / Luxury**
    relative to others in the **same dosage-form category**.

    **Features used:**
    - Salt composition (TF-IDF)
    - Dosage form
    - Manufacturer tier
    - Pack size
    - Salt count
    - Max dose strength

    **No MRP fed to model** — formulation-only prediction.
    """)
    st.divider()
    st.markdown(f"**AUC OvR:** 0.8404 | **Tests:** 45/45 ✓")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">💊 PharmIQ</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Medicine Price Tier Intelligence — Formulation-based classification across 273K+ Indian medicines</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Batch Analysis", "📈 Model Insights"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Single Prediction
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### Medicine Details")

        salt_composition = st.text_input(
            "Salt Composition",
            value="Paracetamol (500mg)",
            help="e.g. Paracetamol (500mg) + Ibuprofen (400mg)",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            dosage_form = st.selectbox("Dosage Form", DOSAGE_FORMS)
            pack_size = st.number_input("Pack Size (units/ml/gm)", min_value=0.1, value=10.0, step=1.0)
        with col_b:
            salt_count = st.number_input("Number of Active Salts", min_value=1, max_value=10, value=1)
            mfr_tier = st.selectbox(
                "Manufacturer Tier",
                options=[0, 1, 2],
                format_func=lambda x: {0: "0 — Generic", 1: "1 — Mid-tier", 2: "2 — Top-tier"}[x],
            )

        max_dose = st.number_input("Max Dose Strength (mg, 0 if unknown)", min_value=0.0, value=500.0, step=50.0)
        is_branded = st.radio("Name Type", ["Generic", "Branded"], horizontal=True)
        is_branded_int = 1 if is_branded == "Branded" else 0

        predict_btn = st.button("Predict Price Tier", type="primary", use_container_width=True)

    with col_result:
        st.markdown("### Prediction")

        if predict_btn:
            if not model_loaded:
                st.error("Model not loaded. Check MODEL_PATH.")
            else:
                log_max_dose = math.log1p(max_dose)
                features = pd.DataFrame([{
                    "salt_count": salt_count,
                    "pack_size_units": pack_size,
                    "manufacturer_tier": mfr_tier,
                    "dosage_form": dosage_form,
                    "max_dose_mg": max_dose,
                    "is_branded": is_branded_int,
                    "log_max_dose": log_max_dose,
                    "Salt_Composition": salt_composition,
                }])

                pred = int(pipeline.predict(features)[0])
                proba = pipeline.predict_proba(features)[0]
                tier_label = TIER_MAP[pred]
                color = TIER_COLORS[tier_label]

                st.markdown(
                    f'<div class="tier-badge" style="background:{color};">{tier_label} Tier</div>',
                    unsafe_allow_html=True,
                )

                interpretations = {
                    0: "Priced in the bottom 25% for this dosage form.",
                    1: "Priced between the 25th–50th percentile for this dosage form.",
                    2: "Priced between the 50th–75th percentile for this dosage form.",
                    3: "Priced in the top 25% for this dosage form.",
                }
                st.info(interpretations[pred])

                # Probability bar chart
                fig = go.Figure(go.Bar(
                    x=[TIER_MAP[i] for i in range(4)],
                    y=[round(float(p) * 100, 1) for p in proba],
                    marker_color=[TIER_COLORS[TIER_MAP[i]] for i in range(4)],
                    text=[f"{round(float(p)*100,1)}%" for p in proba],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Class Probabilities",
                    yaxis_title="Probability (%)",
                    yaxis_range=[0, 110],
                    height=320,
                    margin=dict(t=40, b=20, l=20, r=20),
                    plot_bgcolor="white",
                    showlegend=False,
                )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
                st.plotly_chart(fig, use_container_width=True)

                # Confidence
                confidence = float(proba[pred]) * 100
                st.metric("Model Confidence", f"{confidence:.1f}%")
        else:
            st.markdown("""
            <div style="text-align:center; color:#aaa; padding:3rem 0;">
                <div style="font-size:3rem;">💊</div>
                <div>Fill in medicine details and click Predict</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Batch Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Batch Prediction from CSV")
    st.markdown("Upload a CSV with columns: `Salt_Composition`, `Quantity`, `Manufacturer`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.dataframe(df_upload.head(5), use_container_width=True)

        if st.button("Run Batch Predictions", type="primary"):
            if not model_loaded:
                st.error("Model not loaded.")
            else:
                with st.spinner("Running predictions..."):
                    df_upload["dosage_form"] = df_upload["Quantity"].apply(extract_dosage_form)
                    df_upload["pack_size_units"] = df_upload["Quantity"].apply(extract_pack_size)
                    df_upload["salt_count"] = df_upload["Salt_Composition"].apply(extract_salt_count)
                    df_upload["manufacturer_tier"] = df_upload["Manufacturer"].apply(manufacturer_tier)
                    df_upload["max_dose_mg"] = df_upload["Salt_Composition"].apply(extract_max_dose_mg)
                    df_upload["log_max_dose"] = np.log1p(df_upload["max_dose_mg"])

                    X = df_upload[FEATURE_COLS]
                    preds = pipeline.predict(X)
                    probas = pipeline.predict_proba(X)

                    df_upload["Predicted_Tier"] = [TIER_MAP[p] for p in preds]
                    df_upload["Confidence"] = [f"{probas[i][p]*100:.1f}%" for i, p in enumerate(preds)]

                st.success(f"Predictions complete for {len(df_upload):,} medicines.")
                st.dataframe(
                    df_upload[["Salt_Composition", "dosage_form", "manufacturer_tier", "Predicted_Tier", "Confidence"]],
                    use_container_width=True,
                )

                # Distribution chart
                tier_counts = df_upload["Predicted_Tier"].value_counts().reindex(["Budget", "Mid", "Premium", "Luxury"], fill_value=0)
                fig = go.Figure(go.Pie(
                    labels=tier_counts.index.tolist(),
                    values=tier_counts.values.tolist(),
                    marker_colors=[TIER_COLORS[t] for t in tier_counts.index],
                    hole=0.4,
                ))
                fig.update_layout(title="Tier Distribution", height=350)
                st.plotly_chart(fig, use_container_width=True)

                csv_out = df_upload.to_csv(index=False)
                st.download_button("Download Results CSV", csv_out, "pharmiq_predictions.csv", "text/csv")
    else:
        st.info("Upload a CSV to run batch predictions.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Model Insights
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Model Card")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC OvR Macro", "0.8404")
    col2.metric("Test Accuracy", "60.0%")
    col3.metric("Training Samples", "218,924")
    col4.metric("Test Suite", "45/45 ✓")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Architecture")
        st.code("""
Pipeline:
  ColumnTransformer
    ├── StandardScaler      → numeric (6 features)
    ├── OrdinalEncoder      → dosage_form
    └── TF-IDF (200 terms)  → Salt_Composition

  XGBClassifier
    ├── n_estimators : 300
    ├── max_depth    : 6
    ├── learning_rate: 0.05
    └── objective    : softmax (4-class)
        """, language="text")

    with col_right:
        st.markdown("#### Leakage Audit")
        st.markdown("""
        | Feature | Status |
        |---------|--------|
        | `log_mrp` | ❌ **Removed** — direct MRP derivative |
        | `log_unit_price` | ❌ **Removed** — MRP / pack_size |
        | `Salt_Composition` (TF-IDF) | ✅ Clean |
        | `dosage_form` | ✅ Clean |
        | `manufacturer_tier` | ✅ Clean |
        | `salt_count` | ✅ Clean |
        | `max_dose_mg` | ✅ Clean |
        | `is_branded` | ✅ Clean |
        """)

    st.divider()
    st.markdown("#### Target Design")
    st.markdown("""
    Price tier is assigned **within dosage-form group** (not globally), making the label
    clinically meaningful: *"Is this tablet expensive for a tablet?"* rather than an
    arbitrary global quantile cut. Each group is split at its own Q25/Q50/Q75 MRP thresholds.
    """)
