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
    page_title="PharmIQ Platform v2.0",
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Price Predict", "💊 Generic Finder",
    "🏷️ Category Classifier", "📊 Batch Analysis", "📈 Model Insights"
])

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
# TAB 2: Generic Alternative Finder
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 💊 Generic Alternative Finder")
    st.markdown("Find cheaper medicines with the **same active ingredient(s)** from across 273K+ Indian pharmaceutical products.")

    @st.cache_resource
    def load_recommender():
        import pickle
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.recommender.engine import GenericRecommender
        idx_path = PROJECT_ROOT / "models" / "recommender_index.pkl"
        with open(idx_path, "rb") as f:
            index = pickle.load(f)
        return GenericRecommender(index)

    try:
        rec_engine = load_recommender()
        rec_loaded = True
    except Exception as e:
        rec_engine = None
        rec_loaded = False
        st.error(f"Recommender index not found: {e}")

    # ── Search mode toggle
    search_mode = st.radio(
        "Search by", ["Medicine Name", "Salt Composition"],
        horizontal=True,
        help="Name search resolves to salt composition automatically.",
    )

    col_inp, col_opts = st.columns([2, 1])

    with col_inp:
        if search_mode == "Medicine Name":
            query_input = st.text_input(
                "Medicine Name",
                value="Azithral 500 Tablet",
                placeholder="e.g. Crocin 500 Tablet, Pan 40 Tablet",
            )
        else:
            query_input = st.text_input(
                "Salt Composition",
                value="Azithromycin (500mg)",
                placeholder="e.g. Paracetamol (500mg) + Ibuprofen (400mg)",
            )
            query_mrp = st.number_input("Your medicine's MRP (₹)", min_value=0.0, value=0.0, step=5.0,
                                         help="Used to calculate savings %. Leave 0 if unknown.")

    with col_opts:
        match_mode = st.selectbox(
            "Match Mode",
            options=["exact", "ingredient"],
            format_func=lambda x: {
                "exact": "Exact (same dose)",
                "ingredient": "Ingredient (any dose)",
            }[x],
            help="Exact: same salt + same dose. Ingredient: same drug, any dose.",
        )
        top_n = st.slider("Results", min_value=5, max_value=30, value=10, step=5)

    search_btn = st.button("Find Alternatives", type="primary", use_container_width=True)

    if search_btn:
        if not rec_loaded:
            st.error("Recommender index not loaded.")
        else:
            with st.spinner("Searching 273K medicines..."):
                try:
                    if search_mode == "Medicine Name":
                        result = rec_engine.search_by_name(
                            medicine_name=query_input,
                            top_n=top_n,
                            mode=match_mode,
                        )
                    else:
                        result = rec_engine.recommend_by_salt(
                            salt_composition=query_input,
                            query_name="",
                            query_mrp=query_mrp if "query_mrp" in dir() else 0.0,
                            top_n=top_n,
                            mode=match_mode,
                        )
                except Exception as e:
                    st.error(f"Recommendation error: {e}")
                    result = None

            if result:
                if not result.alternatives:
                    st.warning(result.message)
                else:
                    # ── Query summary
                    st.divider()
                    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
                    qcol1.metric("Query Medicine", result.query_name)
                    qcol2.metric("Salt Composition", result.query_salt_composition[:40] + ("..." if len(result.query_salt_composition) > 40 else ""))
                    qcol3.metric("Total Alternatives Found", f"{result.total_found:,}")
                    qcol4.metric("Best Savings", f"{result.cheapest_savings_pct:.1f}%", delta=f"{result.cheapest_savings_pct:.1f}% cheaper/unit")

                    st.markdown(f"**{result.message}** | Match mode: `{result.mode}`")
                    st.divider()

                    # ── Results table
                    rows = []
                    for i, alt in enumerate(result.alternatives, 1):
                        savings_color = "🟢" if alt.unit_price_savings_pct > 50 else ("🟡" if alt.unit_price_savings_pct > 0 else "🔴")
                        rows.append({
                            "Rank": i,
                            "Medicine": alt.name,
                            "Salt": alt.salt_composition[:50] + ("..." if len(alt.salt_composition) > 50 else ""),
                            "Pack": alt.quantity,
                            "MRP (₹)": f"₹{alt.mrp:.2f}",
                            "Unit Price (₹)": f"₹{alt.unit_price:.3f}",
                            "Savings/Unit": f"{savings_color} {alt.unit_price_savings_pct:.1f}%",
                            "Manufacturer": alt.manufacturer[:30],
                            "Tier": alt.manufacturer_tier_label,
                        })

                    df_results = pd.DataFrame(rows)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)

                    # ── Savings bar chart
                    if len(result.alternatives) > 1:
                        fig = go.Figure()
                        names = [a.name[:25] for a in result.alternatives]
                        savings = [a.unit_price_savings_pct for a in result.alternatives]
                        colors_bar = ["#2ECC71" if s > 50 else "#F39C12" if s > 0 else "#E74C3C" for s in savings]
                        fig.add_trace(go.Bar(
                            x=names, y=savings,
                            marker_color=colors_bar,
                            text=[f"{s:.1f}%" for s in savings],
                            textposition="outside",
                        ))
                        fig.update_layout(
                            title="Unit Price Savings vs Query Medicine (%)",
                            yaxis_title="Savings (%)",
                            height=350,
                            margin=dict(t=40, b=80, l=20, r=20),
                            plot_bgcolor="white",
                            showlegend=False,
                        )
                        fig.update_xaxes(tickangle=-35)
                        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
                        st.plotly_chart(fig, use_container_width=True)

                    # ── Download
                    csv_out = df_results.to_csv(index=False)
                    st.download_button(
                        "Download Results CSV",
                        csv_out,
                        f"pharmiq_alternatives_{query_input[:20].replace(' ','_')}.csv",
                        "text/csv",
                    )
    else:
        st.markdown("""
        <div style="text-align:center; color:#aaa; padding:2.5rem 0;">
            <div style="font-size:3rem;">🔍</div>
            <div>Enter a medicine name or salt composition and click Find Alternatives</div>
            <div style="margin-top:0.5rem; font-size:0.9rem;">Example: <code>Azithral 500 Tablet</code> → finds 1,902 cheaper alternatives, best saving 95.9%</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Therapeutic Category Classifier
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🏷️ Therapeutic Category Classifier")
    st.markdown("Classify a medicine's therapeutic category from its salt composition — 13 categories, **AUC 0.999**, accuracy 96%.")

    @st.cache_resource
    def load_category_model():
        import joblib
        return joblib.load(PROJECT_ROOT / "models" / "category_classifier_v1.pkl")

    @st.cache_resource
    def get_label_engine():
        from src.classifier.label_engine import assign_label, PRIORITY_ORDER, CODE_TO_CATEGORY
        return assign_label, PRIORITY_ORDER, CODE_TO_CATEGORY

    try:
        cat_model = load_category_model()
        assign_label_fn, CATS, CODE_MAP = get_label_engine()
        cat_loaded = True
    except Exception as e:
        cat_model = None
        cat_loaded = False
        st.error(f"Category model not found: {e}")

    CAT_COLORS = {
        "Antibiotic": "#E74C3C", "Analgesic": "#E67E22", "Anti-diabetic": "#27AE60",
        "Cardiac": "#C0392B", "Respiratory": "#2980B9", "Gastrointestinal": "#8E44AD",
        "Neurological": "#2C3E50", "Vitamin/Supplement": "#F1C40F", "Hormonal": "#E91E63",
        "Dermatology": "#FF5722", "Musculoskeletal": "#795548", "Anti-parasitic": "#009688",
        "Ophthalmic": "#3F51B5", "Other": "#95A5A6",
    }

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### Input")
        cat_salt = st.text_area(
            "Salt Composition",
            value="Azithromycin (500mg)",
            height=100,
            help="e.g. Metformin (500mg) + Glimepiride (1mg)",
        )
        cat_quantity = st.text_input(
            "Quantity / Pack",
            value="strip of 10 tablets",
            help="Used to extract dosage form",
        )
        classify_btn = st.button("Classify", type="primary", use_container_width=True)

    with col_right:
        st.markdown("#### Result")

        if classify_btn:
            if not cat_loaded:
                st.error("Model not loaded.")
            else:
                from src.features.engineer import extract_dosage_form, extract_salt_count

                dosage_form = extract_dosage_form(cat_quantity)
                salt_count = extract_salt_count(cat_salt)
                features_df = pd.DataFrame([{
                    "Salt_Composition": cat_salt,
                    "dosage_form": dosage_form,
                    "salt_count": salt_count,
                }])

                pred_code = int(cat_model.predict(features_df)[0])
                proba = cat_model.predict_proba(features_df)[0]
                classes = cat_model.classes_
                pred_category = CODE_MAP.get(pred_code, "Other")
                rule_label = assign_label_fn(cat_salt)
                color = CAT_COLORS.get(pred_category, "#95A5A6")

                st.markdown(
                    f'<div style="display:inline-block;padding:0.4rem 1.2rem;'
                    f'border-radius:20px;background:{color};color:white;'
                    f'font-size:1.3rem;font-weight:700;">{pred_category}</div>',
                    unsafe_allow_html=True,
                )

                # Rule vs model agreement
                if rule_label == pred_category:
                    st.success(f"Rule engine agrees: **{rule_label}**")
                elif rule_label == "Other":
                    st.info(f"Rule engine: **Other** (unlabelled) → model predicts **{pred_category}**")
                else:
                    st.warning(f"Rule engine: **{rule_label}** | Model: **{pred_category}**")

                st.caption(f"Dosage form: `{dosage_form}` | Salts: `{salt_count}`")

                # Top probabilities
                proba_dict = {CODE_MAP.get(int(c), str(c)): float(p) for c, p in zip(classes, proba)}
                top5 = sorted(proba_dict.items(), key=lambda x: -x[1])[:5]

                fig = go.Figure(go.Bar(
                    x=[t[0] for t in top5],
                    y=[round(t[1]*100, 1) for t in top5],
                    marker_color=[CAT_COLORS.get(t[0], "#95A5A6") for t in top5],
                    text=[f"{round(t[1]*100,1)}%" for t in top5],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Top 5 Category Probabilities",
                    yaxis_title="Probability (%)",
                    yaxis_range=[0, 115],
                    height=300,
                    margin=dict(t=40, b=20, l=10, r=10),
                    plot_bgcolor="white",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center;color:#aaa;padding:3rem 0;">
                <div style="font-size:3rem;">🏷️</div>
                <div>Enter salt composition and click Classify</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Supported Categories")
    cols = st.columns(4)
    cats_list = list(CAT_COLORS.items())[:-1] if cat_loaded else []
    for i, (cat, color) in enumerate(cats_list):
        cols[i % 4].markdown(
            f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:10px;font-size:0.8rem;">{cat}</span>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Batch Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
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
                    df_upload["is_branded"] = 0
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
# TAB 5: Model Insights
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Model Card")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC OvR Macro", "0.8701")
    col2.metric("Test Accuracy", "64.5%")
    col3.metric("Training Samples", "218,924")
    col4.metric("Test Suite", "98/98 ✓")

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
