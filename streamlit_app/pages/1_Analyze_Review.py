"""Page 1 — Analyze a single peer review."""

import os
import sys
import warnings
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*__path__.*")
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DATA_DIR, MARKER_NAMES, MODELS_DIR

st.set_page_config(page_title="Analyze Review", page_icon="🔬", layout="wide")
st.title("Analyze a Peer Review")
st.markdown("Paste a review or load a sample, then click **Analyze**.")

# ── Constants ────────────────────────────────────────────────────────────────
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
MARKER_CATEGORIES = {
    "Standardized Structure": "Structural",
    "Predictable Criticism": "Argumentative",
    "Excessive Balance": "Argumentative",
    "Linguistic Homogeneity": "Linguistic",
    "Generic Domain Language": "Linguistic",
    "Conceptual Feedback": "Behavioral",
    "No Personal Signals": "Behavioral",
    "Repetition Patterns": "Behavioral",
}
CATEGORY_COLORS = {
    "Structural": "#e74c3c",
    "Argumentative": "#f39c12",
    "Linguistic": "#3498db",
    "Behavioral": "#2ecc71",
}

SAMPLE_HUMAN = """This paper proposes a method for jointly learning word embeddings and topic models. The approach is interesting but I have several concerns.

First, the evaluation is limited to intrinsic measures only. I would like to see extrinsic evaluation on downstream tasks like text classification or sentiment analysis. Second, the comparison with existing methods is incomplete - LDA2Vec and TopicVec should be included as baselines.

I think the writing could be improved in Section 3 where the mathematical notation becomes inconsistent. On page 5, equation (7) seems to contradict the claim made in the previous paragraph. Also, Figure 2 is too small to read properly.

After re-reading the paper twice, I'm still not fully convinced that the proposed regularization term in eq. (4) is necessary. Could the authors provide ablation results without it?

Minor: several typos on pages 3 and 7."""

SAMPLE_AI = """### Summary of the Paper

This paper presents a novel approach to jointly learning word embeddings and topic models through a unified neural framework. The authors propose an innovative architecture that bridges the gap between distributed word representations and probabilistic topic modeling.

### Strengths

1. **Novel Architecture**: The proposed model elegantly combines word embedding learning with topic modeling in a principled manner.
2. **Comprehensive Experiments**: The authors conduct extensive experiments on multiple benchmark datasets.
3. **Strong Results**: The model achieves state-of-the-art performance on several evaluation metrics.
4. **Clear Presentation**: The paper is well-written and the methodology is clearly explained.

### Weaknesses

1. **Limited Baselines**: The comparison could be strengthened by including more recent baseline methods.
2. **Scalability Concerns**: The paper would benefit from a more thorough analysis of computational complexity.
3. **Evaluation Scope**: Additional evaluation on downstream tasks would strengthen the claims.

### Questions for Authors

1. How does the model perform with varying vocabulary sizes?
2. Can you provide more details on the hyperparameter sensitivity analysis?

### Overall Recommendation

The paper addresses an important problem and proposes a promising solution. While there are some areas for improvement, the contributions are significant enough to warrant acceptance with minor revisions."""


# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    import joblib
    models = {}
    for name, filename in [("XGBoost", "classifier.joblib"), ("RandomForest", "classifier_rf.joblib"), ("LightGBM", "classifier_lgbm.joblib"), ("LogisticRegression", "classifier_lr.joblib")]:
        path = MODELS_DIR / filename
        if path.exists():
            models[name] = joblib.load(path)
    return models if models else None

@st.cache_resource
def load_rag():
    from rag_retrieval import ReviewRAG
    rag = ReviewRAG()
    try:
        rag.load_index()
        return rag
    except Exception:
        return None

@st.cache_data
def load_dataset():
    path = DATA_DIR / "dataset.csv"
    return pd.read_csv(path) if path.exists() else None


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    extraction_method = st.radio(
        "Feature Extraction",
        ["Rule-Based (instant)", "LLM via OpenAI", "LLM via Anthropic", "LLM via Gemini"],
        index=0,
    )

    openai_key = None
    openai_model = None
    anthropic_key = None
    anthropic_model = None
    gemini_key = None
    gemini_model = None

    if "OpenAI" in extraction_method:
        try:
            from config import OPENAI_API_KEY, OPENAI_MODEL
        except ImportError:
            OPENAI_API_KEY = ""
            OPENAI_MODEL = "gpt-4o"
        st.markdown("---")
        st.subheader("OpenAI Config")
        openai_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("openai_key", OPENAI_API_KEY),
            type="password",
            help="Get a key at https://platform.openai.com/api-keys",
            key="openai_key_input",
        )
        if openai_key:
            st.session_state["openai_key"] = openai_key
        OPENAI_MODELS = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]
        openai_model = st.selectbox(
            "Model",
            OPENAI_MODELS,
            index=0,
        )

    elif "Anthropic" in extraction_method:
        try:
            from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
        except ImportError:
            ANTHROPIC_API_KEY = ""
            ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        st.markdown("---")
        st.subheader("Anthropic Config")
        anthropic_key = st.text_input(
            "Anthropic API Key",
            value=st.session_state.get("anthropic_key", ANTHROPIC_API_KEY),
            type="password",
            help="Get a key at https://console.anthropic.com/settings/keys",
            key="anthropic_key_input",
        )
        if anthropic_key:
            st.session_state["anthropic_key"] = anthropic_key
        ANTHROPIC_MODELS = [
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250414",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
        ]
        anthropic_model = st.selectbox(
            "Model",
            ANTHROPIC_MODELS,
            index=0,
        )

    elif "Gemini" in extraction_method:
        try:
            from config import GOOGLE_API_KEY, GOOGLE_MODEL
        except ImportError:
            GOOGLE_API_KEY = ""
            GOOGLE_MODEL = "gemini-3.1-pro-preview"
        st.markdown("---")
        st.subheader("Gemini Config")
        gemini_key = st.text_input(
            "Google API Key",
            value=st.session_state.get("gemini_key", GOOGLE_API_KEY),
            type="password",
            help="Get a key at https://aistudio.google.com/app/apikey",
            key="gemini_key_input",
        )
        if gemini_key:
            st.session_state["gemini_key"] = gemini_key
        gemini_model = st.text_input(
            "Model",
            value=GOOGLE_MODEL,
            help="Default: gemini-3.1-pro-preview",
        )

    all_models = load_models()
    model_choice = "XGBoost"
    if all_models and len(all_models) > 1:
        model_choice = st.selectbox("Classifier", list(all_models.keys()))

    st.divider()
    st.header("Load Sample")
    if st.button("Human Review Example", width="stretch"):
        st.session_state["review_text"] = SAMPLE_HUMAN
    if st.button("AI Review Example", width="stretch"):
        st.session_state["review_text"] = SAMPLE_AI

    dataset = load_dataset()
    if dataset is not None:
        st.divider()
        st.header("Random from Dataset")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Random Human", width="stretch"):
                st.session_state["review_text"] = dataset[dataset["label"]==0].sample(1).iloc[0]["review_text"]
        with c2:
            if st.button("Random AI", width="stretch"):
                st.session_state["review_text"] = dataset[dataset["label"]==1].sample(1).iloc[0]["review_text"]


# ── Input ────────────────────────────────────────────────────────────────────
review_text = st.text_area(
    "Paste a peer review:",
    value=st.session_state.get("review_text", ""),
    height=250,
    placeholder="Paste a peer review here...",
)

analyze_btn = st.button("Analyze Review", type="primary", width="stretch")

if not analyze_btn or not review_text.strip():
    st.info("Paste a review and click **Analyze Review**.")
    st.stop()

# ── Run pipeline ─────────────────────────────────────────────────────────────
if all_models is None:
    st.error("No trained models. Run `python pipeline.py` first.")
    st.stop()
model = all_models[model_choice]

# Extract markers
with st.spinner("Extracting markers..."):
    if "OpenAI" in extraction_method:
        from feature_extractor import extract_markers_openai
        if not openai_key:
            st.error("Please enter an OpenAI API key in the sidebar.")
            st.stop()
        markers = extract_markers_openai(review_text, api_key=openai_key, model=openai_model)
        if all(v == 0.0 for v in markers.values()):
            st.warning("OpenAI returned empty scores. Falling back to rule-based.")
            from feature_extractor import extract_markers_rulebased
            markers = extract_markers_rulebased(review_text)
    elif "Anthropic" in extraction_method:
        from feature_extractor import extract_markers_anthropic
        if not anthropic_key:
            st.error("Please enter an Anthropic API key in the sidebar.")
            st.stop()
        markers = extract_markers_anthropic(review_text, api_key=anthropic_key, model=anthropic_model)
        if all(v == 0.0 for v in markers.values()):
            st.warning("Anthropic returned empty scores. Falling back to rule-based.")
            from feature_extractor import extract_markers_rulebased
            markers = extract_markers_rulebased(review_text)
    elif "Gemini" in extraction_method:
        from feature_extractor import extract_markers_gemini
        if not gemini_key:
            st.error("Please enter a Google API key in the sidebar.")
            st.stop()
        markers = extract_markers_gemini(review_text, api_key=gemini_key, model=gemini_model)
        if all(v == 0.0 for v in markers.values()):
            st.warning("Gemini returned empty scores. Falling back to rule-based.")
            from feature_extractor import extract_markers_rulebased
            markers = extract_markers_rulebased(review_text)
    else:
        from feature_extractor import extract_markers_rulebased
        markers = extract_markers_rulebased(review_text)

# Classify
X = np.array([[markers.get(m, 0.0) for m in MARKER_NAMES]])
proba = model.predict_proba(X)[0]
ai_prob = float(proba[1])
pred_label = "AI-Generated" if ai_prob > 0.5 else "Human"

# SHAP — use the right explainer for the model type
import shap
try:
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X)
    sv = shap_vals.values[0]
    if sv.ndim > 1:
        sv = sv[:, 1]
    shap_dict = dict(zip(MARKER_NAMES, sv.tolist()))
except Exception:
    try:
        # Fallback: LinearExplainer for logistic regression / linear models
        bg = np.zeros((1, len(MARKER_NAMES)))
        explainer = shap.LinearExplainer(model, bg)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0]
        shap_dict = dict(zip(MARKER_NAMES, sv.tolist()))
    except Exception:
        # If all else fails, use feature coefficients or zeros
        shap_dict = {m: 0.0 for m in MARKER_NAMES}

# ── Results ──────────────────────────────────────────────────────────────────
st.divider()

# Prediction + gauge
col_pred, col_gauge = st.columns([1, 1])

with col_pred:
    is_ai = ai_prob > 0.5
    color = "#e74c3c" if is_ai else "#27ae60"
    st.markdown(f"### Prediction: <span style='color:{color}; font-size:1.4em;'>{pred_label}</span>", unsafe_allow_html=True)

    if ai_prob > 0.8 or ai_prob < 0.2:
        conf, cc = "High", color
    elif ai_prob > 0.6 or ai_prob < 0.4:
        conf, cc = "Medium", "#f39c12"
    else:
        conf, cc = "Low", "#95a5a6"
    st.markdown(f"**Confidence:** <span style='color:{cc}'>{conf}</span>", unsafe_allow_html=True)

    high_markers = [MARKER_LABELS[m] for m in MARKER_NAMES if markers[m] > 0.7]
    if high_markers:
        st.warning(f"High markers: {', '.join(high_markers)}")

with col_gauge:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=ai_prob * 100,
        title={"text": "AI Probability"}, number={"suffix": "%", "font": {"size": 36}},
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color="#e74c3c" if is_ai else "#27ae60"),
            steps=[
                dict(range=[0, 40], color="#d5f5e3"),
                dict(range=[40, 60], color="#fdebd0"),
                dict(range=[60, 100], color="#fadbd8"),
            ],
            threshold=dict(line=dict(color="black", width=2), thickness=0.75, value=50),
        ),
    ))
    fig.update_layout(height=250, margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch")

st.divider()

# Markers: radar + bar
st.subheader("Detected Markers")
col_r, col_b = st.columns([1, 1])

with col_r:
    labels = [MARKER_LABELS[m] for m in MARKER_NAMES]
    vals = [markers[m] for m in MARKER_NAMES]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]],
        fill="toself", fillcolor="rgba(52,152,219,0.2)",
        line=dict(color="#3498db", width=2),
    ))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), showlegend=False, height=400, margin=dict(t=30, b=30, l=60, r=60))
    st.plotly_chart(fig, width="stretch")

with col_b:
    colors = [CATEGORY_COLORS[MARKER_CATEGORIES[l]] for l in labels]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors, text=[f"{v:.2f}" for v in vals], textposition="auto"))
    for i, (m, l) in enumerate(zip(MARKER_NAMES, labels)):
        s = shap_dict.get(m, 0)
        if abs(s) > 0.001:
            arrow = "AI" if s > 0 else "Human"
            fig.add_annotation(x=max(vals)+0.05, y=l, text=f"SHAP:{s:+.2f} {arrow}", showarrow=False,
                             font=dict(size=10, color="#e74c3c" if s > 0 else "#27ae60"), xanchor="left")
    fig.update_layout(xaxis=dict(range=[0, max(vals)*1.5+0.1], title="Score"), yaxis=dict(autorange="reversed"),
                     height=400, margin=dict(t=10, b=40, l=10, r=150))
    st.plotly_chart(fig, width="stretch")

# Marker details
with st.expander("Marker Details Table"):
    rows = []
    for m in MARKER_NAMES:
        l = MARKER_LABELS[m]
        score = markers[m]
        s = shap_dict.get(m, 0)
        rows.append({
            "Category": MARKER_CATEGORIES[l], "Marker": l,
            "Score": f"{score:.3f}",
            "Severity": "High" if score > 0.7 else "Medium" if score > 0.4 else "Low",
            "SHAP": f"{s:+.4f}", "Direction": "toward AI" if s > 0 else "toward Human",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.divider()

# SHAP
st.subheader("SHAP Explanation")
sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
for m, s in sorted_shap:
    if abs(s) < 0.001:
        continue
    l = MARKER_LABELS[m]
    d = "AI" if s > 0 else "Human"
    c = "#e74c3c" if s > 0 else "#27ae60"
    w = min(abs(s) / 5 * 100, 100)
    st.markdown(
        f"<div style='margin:4px 0;'><span style='display:inline-block;width:220px;'>{l}</span>"
        f"<span style='color:{c};font-weight:bold;'>{s:+.4f}</span> <span style='color:{c};'>→ {d}</span>"
        f"<div style='background:{c}22;border-radius:4px;height:12px;width:{w}%;'>"
        f"<div style='background:{c};border-radius:4px;height:100%;width:100%;'></div></div></div>",
        unsafe_allow_html=True,
    )

# RAG
rag = load_rag()
if rag:
    st.divider()
    st.subheader("Similar Reviews (RAG)")
    with st.spinner("Searching..."):
        rag_results = rag.retrieve_with_context(review_text)
    summary = rag_results["summary"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Retrieved", summary["total_retrieved"])
    c2.metric("Human Matches", summary["human_matches"])
    c3.metric("AI Matches", summary["ai_matches"])
    for r in rag_results["retrieved_reviews"][:5]:
        lc = "#e74c3c" if r["label"] == "AI-Generated" else "#27ae60"
        with st.expander(f"[{r['label']}] sim={r['similarity_score']:.3f} — {r['review_text'][:80]}..."):
            st.markdown(f"**Label:** <span style='color:{lc}'>{r['label']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Similarity:** {r['similarity_score']:.4f} | **Source:** {r['source']}")
            st.text(r.get("full_text", r["review_text"]))
