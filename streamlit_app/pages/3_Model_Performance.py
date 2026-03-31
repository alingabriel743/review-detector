"""Page 3 — Model performance comparison: XGBoost, Random Forest, LightGBM."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import DATA_DIR, MARKER_NAMES, MODELS_DIR, OUTPUTS_DIR

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("Model Performance Comparison")

MARKER_LABELS = {
    "standardized_structure": "Standardized Structure",
    "predictable_criticism": "Predictable Criticism",
    "excessive_balance": "Excessive Balance",
    "linguistic_homogeneity": "Linguistic Homogeneity",
    "generic_domain_language": "Generic Domain Language",
    "conceptual_feedback": "Conceptual Feedback",
    "absence_personal_signals": "No Personal Signals",
    "repetition_patterns": "Repetition Patterns",
}

MODEL_COLORS = {
    "XGBoost": "#3498db",
    "RandomForest": "#27ae60",
    "LightGBM": "#e74c3c",
    "LogisticRegression": "#9b59b6",
}


# ── Load results ─────────────────────────────────────────────────────────────
comparison_path = OUTPUTS_DIR / "classifier_comparison.json"
single_path = OUTPUTS_DIR / "classifier_results.json"

if comparison_path.exists():
    with open(comparison_path) as f:
        all_results = json.load(f)
elif single_path.exists():
    with open(single_path) as f:
        xgb = json.load(f)
    all_results = {"XGBoost": xgb}
else:
    st.error("No results found. Run `python pipeline.py` first.")
    st.stop()

# ── Comparison table ─────────────────────────────────────────────────────────
st.subheader("Model Comparison")

comp_rows = []
for name, res in all_results.items():
    cm = np.array(res["confusion_matrix"])
    comp_rows.append({
        "Model": name,
        "Accuracy": f"{res['accuracy']:.4f}",
        "AUC-ROC": f"{res['auc_roc']:.4f}",
        "False Positives": int(cm[0, 1]),
        "False Negatives": int(cm[1, 0]),
        "Total Errors": int(cm[0, 1] + cm[1, 0]),
    })

st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

# ── Bar chart comparison ─────────────────────────────────────────────────────
st.divider()
col_acc, col_auc = st.columns(2)

with col_acc:
    st.subheader("Accuracy")
    names = list(all_results.keys())
    accs = [all_results[n]["accuracy"] for n in names]
    colors = [MODEL_COLORS.get(n, "#95a5a6") for n in names]
    fig = go.Figure(go.Bar(x=names, y=accs, marker_color=colors, text=[f"{a:.4f}" for a in accs], textposition="auto"))
    fig.update_layout(yaxis=dict(range=[min(accs) - 0.02, 1.0], title="Accuracy"), height=300, margin=dict(t=10, b=40))
    st.plotly_chart(fig, width="stretch")

with col_auc:
    st.subheader("AUC-ROC")
    aucs = [all_results[n]["auc_roc"] for n in names]
    fig = go.Figure(go.Bar(x=names, y=aucs, marker_color=colors, text=[f"{a:.4f}" for a in aucs], textposition="auto"))
    fig.update_layout(yaxis=dict(range=[min(aucs) - 0.02, 1.0], title="AUC-ROC"), height=300, margin=dict(t=10, b=40))
    st.plotly_chart(fig, width="stretch")

# ── Per-model details ────────────────────────────────────────────────────────
st.divider()
st.subheader("Detailed Results per Model")

selected_model = st.selectbox("Select model:", list(all_results.keys()))
res = all_results[selected_model]

col_report, col_cm = st.columns([1, 1])

with col_report:
    st.markdown(f"**Classification Report — {selected_model}**")
    st.code(res["classification_report"])

with col_cm:
    st.markdown(f"**Confusion Matrix — {selected_model}**")
    cm = np.array(res["confusion_matrix"])
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred: Human", "Pred: AI"], y=["True: Human", "True: AI"],
        text=cm, texttemplate="%{text}", textfont=dict(size=20),
        colorscale=[[0, "#d5f5e3"], [1, "#fadbd8"]],
        showscale=False,
    ))
    fig.update_layout(height=300, margin=dict(t=10, b=40), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

# ── Feature Importance comparison ────────────────────────────────────────────
st.divider()
st.subheader("Feature Importance Comparison")

# Build comparison dataframe
imp_data = []
for name, res in all_results.items():
    feat_imp = res.get("feature_importance", {})
    for m in MARKER_NAMES:
        imp_data.append({
            "Model": name,
            "Marker": MARKER_LABELS[m],
            "Importance": feat_imp.get(m, 0.0),
        })

if imp_data:
    imp_df = pd.DataFrame(imp_data)
    fig = px.bar(
        imp_df, x="Importance", y="Marker", color="Model",
        orientation="h", barmode="group",
        color_discrete_map=MODEL_COLORS,
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=450, margin=dict(t=10, b=40, l=10, r=10),
    )
    st.plotly_chart(fig, width="stretch")

# ── SHAP Global Plots ───────────────────────────────────────────────────────
st.divider()
st.subheader("SHAP Global Analysis (XGBoost)")

imp_path = OUTPUTS_DIR / "global_importance.json"
if imp_path.exists():
    with open(imp_path) as f:
        shap_importance = json.load(f)

    sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
    names_s = [MARKER_LABELS.get(k, k) for k, _ in sorted_shap]
    values_s = [v for _, v in sorted_shap]

    fig = go.Figure(go.Bar(
        x=values_s, y=names_s, orientation="h",
        marker_color="#e74c3c",
        text=[f"{v:.4f}" for v in values_s], textposition="auto",
    ))
    fig.update_layout(
        xaxis_title="Mean |SHAP Value|", yaxis=dict(autorange="reversed"),
        height=350, margin=dict(t=10, b=40, l=10, r=10),
    )
    st.plotly_chart(fig, width="stretch")

col_s1, col_s2 = st.columns(2)
summary_path = OUTPUTS_DIR / "shap_summary.png"
bar_path = OUTPUTS_DIR / "shap_bar_importance.png"

with col_s1:
    if summary_path.exists():
        st.image(str(summary_path), caption="SHAP Summary (Beeswarm)", width="stretch")
with col_s2:
    if bar_path.exists():
        st.image(str(bar_path), caption="SHAP Bar Importance", width="stretch")
