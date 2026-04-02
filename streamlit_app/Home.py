"""
Home page — AI-Generated Peer Review Detection Framework
"""

import os
import sys
import warnings
from pathlib import Path

# Suppress noisy warnings from transformers/torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*__path__.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# Add src to path so we can import pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st

st.set_page_config(
    page_title="AI Review Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("Is This Peer Review AI-Generated?")
st.markdown(
    "**An Explainable AI & RAG-Based Detection Framework** based on a LLM-Feature Extractor"
)
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### How It Works

    This framework detects AI-generated peer reviews through a **6-stage pipeline**:

    1. **Data Collection** — Human reviews from PeerRead + AI reviews from Gen-Review (GPT-4o)
    2. **Feature Extraction** — 8 linguistic markers scored per review (rule-based or LLM)
    3. **Classification** — XGBoost classifier trained on marker features
    4. **SHAP Explainability** — Which markers drove the prediction and by how much
    5. **RAG Retrieval** — Find similar reviews from the knowledge base for evidence
    6. **Editor Report** — Combined report with probability, markers, explanations, and evidence

    ### Marker Taxonomy

    | Category | Markers |
    |---|---|
    | **Structural** | Standardized Structure |
    | **Argumentative** | Predictable Criticism, Excessive Balance |
    | **Linguistic** | Linguistic Homogeneity, Generic Domain Language |
    | **Behavioral** | Conceptual Feedback, No Personal Signals, Repetition Patterns |
    """)

with col2:
    st.markdown("### Quick Navigation")
    st.page_link("pages/1_Analyze_Review.py", label="Analyze a Review", icon="🔬")
    st.page_link("pages/2_Dataset_Explorer.py", label="Explore the Dataset", icon="📊")
    st.page_link("pages/3_Model_Performance.py", label="Model Performance", icon="📈")
    st.page_link("pages/4_RAG_Search.py", label="RAG Similar Reviews", icon="🔎")

    st.divider()
    st.markdown("### Dataset")
    try:
        from config import DATA_DIR
        import pandas as pd
        df = pd.read_csv(DATA_DIR / "dataset.csv")
        n_human = (df["label"] == 0).sum()
        n_ai = (df["label"] == 1).sum()
        st.metric("Total Reviews", len(df))
        c1, c2 = st.columns(2)
        c1.metric("Human", n_human)
        c2.metric("AI", n_ai)
    except Exception:
        st.warning("Dataset not loaded. Run the pipeline first.")

st.divider()
st.markdown("""
### Architecture

```
Peer Review (text)
       │
       ▼
┌─────────────────────────┐
│  LLM Feature Extractor  │  → 8 marker scores (0.0 – 1.0)
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  XGBoost Classifier     │  → Human vs AI probability
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  SHAP Explanation       │  → Which markers drove the decision
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  RAG Retrieval          │  → Similar reviews as evidence
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  Editor Report          │  → Actionable assessment
└─────────────────────────┘
```
""")

st.caption("Research POC — LLM Feature Extractor + XGBoost + SHAP + RAG")
