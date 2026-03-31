"""Page 2 — Explore the dataset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DATA_DIR, MARKER_NAMES

st.set_page_config(page_title="Dataset Explorer", page_icon="📊", layout="wide")
st.title("Dataset Explorer")

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


@st.cache_data
def load_data():
    dataset_path = DATA_DIR / "dataset.csv"
    features_path = DATA_DIR / "features_cache.csv"
    df = pd.read_csv(dataset_path) if dataset_path.exists() else None
    feat = pd.read_csv(features_path) if features_path.exists() else None
    return df, feat

df, feat = load_data()

if df is None:
    st.error("No dataset found. Run `python pipeline.py` first.")
    st.stop()

# ── Overview ─────────────────────────────────────────────────────────────────
st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", len(df))
col2.metric("Human", (df["label"] == 0).sum())
col3.metric("AI-Generated", (df["label"] == 1).sum())
col4.metric("Sources", df["source"].nunique())

# Class distribution
st.divider()
col_dist, col_src = st.columns(2)

with col_dist:
    st.subheader("Class Distribution")
    counts = df["label"].value_counts().rename({0: "Human", 1: "AI-Generated"})
    fig = px.pie(values=counts.values, names=counts.index, color_discrete_sequence=["#27ae60", "#e74c3c"])
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

with col_src:
    st.subheader("Source Distribution")
    src_counts = df["source"].value_counts()
    fig = px.bar(x=src_counts.index, y=src_counts.values, labels={"x": "Source", "y": "Count"})
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

# Review length distribution
st.divider()
st.subheader("Review Length Distribution")
df["word_count"] = df["review_text"].str.split().str.len()
df["label_name"] = df["label"].map({0: "Human", 1: "AI-Generated"})

fig = px.histogram(
    df, x="word_count", color="label_name",
    color_discrete_map={"Human": "#27ae60", "AI-Generated": "#e74c3c"},
    barmode="overlay", nbins=50,
    labels={"word_count": "Word Count", "label_name": "Class"},
)
fig.update_layout(height=350, margin=dict(t=10, b=40))
st.plotly_chart(fig, width="stretch")

# ── Marker distributions ────────────────────────────────────────────────────
if feat is not None and all(m in feat.columns for m in MARKER_NAMES):
    st.divider()
    st.subheader("Marker Score Distributions (Human vs AI)")

    feat["label_name"] = feat["label"].map({0: "Human", 1: "AI-Generated"}) if "label" in feat.columns else "Unknown"

    selected_marker = st.selectbox("Select marker:", MARKER_NAMES, format_func=lambda m: MARKER_LABELS[m])

    fig = px.histogram(
        feat, x=selected_marker, color="label_name",
        color_discrete_map={"Human": "#27ae60", "AI-Generated": "#e74c3c"},
        barmode="overlay", nbins=30,
        labels={selected_marker: MARKER_LABELS[selected_marker], "label_name": "Class"},
    )
    fig.update_layout(height=350, margin=dict(t=10, b=40))
    st.plotly_chart(fig, width="stretch")

    # Comparison table
    st.subheader("Marker Statistics by Class")
    stats_rows = []
    for m in MARKER_NAMES:
        human = feat[feat["label_name"] == "Human"][m]
        ai = feat[feat["label_name"] == "AI-Generated"][m]
        stats_rows.append({
            "Marker": MARKER_LABELS[m],
            "Human Mean": f"{human.mean():.3f}",
            "Human Std": f"{human.std():.3f}",
            "AI Mean": f"{ai.mean():.3f}",
            "AI Std": f"{ai.std():.3f}",
            "Difference": f"{ai.mean() - human.mean():+.3f}",
        })
    st.dataframe(pd.DataFrame(stats_rows), width="stretch", hide_index=True)

# ── Browse reviews ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Browse Reviews")

filter_label = st.selectbox("Filter by class:", ["All", "Human", "AI-Generated"])
if filter_label == "Human":
    filtered = df[df["label"] == 0]
elif filter_label == "AI-Generated":
    filtered = df[df["label"] == 1]
else:
    filtered = df

st.markdown(f"Showing {len(filtered)} reviews")

for i, (_, row) in enumerate(filtered.head(20).iterrows()):
    label = "Human" if row["label"] == 0 else "AI-Generated"
    color = "#27ae60" if row["label"] == 0 else "#e74c3c"
    with st.expander(f"[{label}] {row['source']} — {row['review_text'][:100]}..."):
        st.markdown(f"**Label:** <span style='color:{color}'>{label}</span> | **Source:** {row['source']} | **Paper:** {row.get('paper_id', 'N/A')}", unsafe_allow_html=True)
        st.text(row["review_text"][:2000])
