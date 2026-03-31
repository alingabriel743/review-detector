"""Page 4 — RAG similarity search across the review knowledge base."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from config import DATA_DIR

st.set_page_config(page_title="RAG Search", page_icon="🔎", layout="wide")
st.title("RAG Similar Review Search")
st.markdown("Search the review knowledge base for similar reviews using semantic embeddings.")


@st.cache_resource
def load_rag():
    from rag_retrieval import ReviewRAG
    rag = ReviewRAG()
    try:
        rag.load_index()
        return rag
    except Exception:
        return None


rag = load_rag()

if rag is None:
    st.error("RAG index not found. Run `python pipeline.py` first.")
    st.stop()

# ── Input ────────────────────────────────────────────────────────────────────
query = st.text_area(
    "Enter a review to search for similar ones:",
    height=200,
    placeholder="Paste a peer review here to find similar reviews in the knowledge base...",
)

col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.slider("Number of results", 1, 20, 5)

search_btn = st.button("Search", type="primary", width="stretch")

if not search_btn or not query.strip():
    st.info("Enter a review text and click **Search**.")
    st.stop()

# ── Search ───────────────────────────────────────────────────────────────────
with st.spinner("Searching knowledge base..."):
    results = rag.retrieve_with_context(query, top_k=top_k)

# ── Results ──────────────────────────────────────────────────────────────────
st.divider()

summary = results["summary"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Retrieved", summary["total_retrieved"])
col2.metric("Human Matches", summary["human_matches"])
col3.metric("AI Matches", summary["ai_matches"])
col4.metric("Avg Similarity", f"{summary['avg_similarity']:.3f}")

# Similarity chart
st.subheader("Similarity Scores")
chart_data = []
for i, r in enumerate(results["retrieved_reviews"]):
    chart_data.append({
        "Rank": i + 1,
        "Similarity": r["similarity_score"],
        "Label": r["label"],
        "Preview": r["review_text"][:60] + "...",
    })

cdf = pd.DataFrame(chart_data)
fig = px.bar(
    cdf, x="Rank", y="Similarity", color="Label",
    color_discrete_map={"Human": "#27ae60", "AI-Generated": "#e74c3c"},
    text="Similarity", hover_data=["Preview"],
)
fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig.update_layout(height=300, margin=dict(t=10, b=40), yaxis=dict(range=[0, 1]))
st.plotly_chart(fig, width="stretch")

# Individual results
st.subheader("Retrieved Reviews")
for i, r in enumerate(results["retrieved_reviews"]):
    lc = "#e74c3c" if r["label"] == "AI-Generated" else "#27ae60"
    with st.expander(f"#{i+1} [{r['label']}] Similarity: {r['similarity_score']:.4f} — {r['review_text'][:80]}..."):
        st.markdown(f"**Label:** <span style='color:{lc}'>{r['label']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Similarity:** {r['similarity_score']:.4f}")
        st.markdown(f"**Source:** {r['source']} | **Paper ID:** {r['paper_id']}")
        st.divider()
        st.text(r["review_text"])
