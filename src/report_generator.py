"""
Editor Report Generator — Combines all pipeline outputs into a final report.

For a given peer review, the report includes:
  1. AI probability score
  2. Detected markers with scores and justifications
  3. SHAP-based explanation (which features drove the prediction)
  4. Similar reviews from the RAG knowledge base
  5. Overall assessment and recommendation
"""

import json
from datetime import datetime

import numpy as np

from config import MARKER_NAMES, OUTPUTS_DIR
from classifier import load_classifier, predict_single
from explainer import explain_single, MARKER_LABELS
from feature_extractor import extract_markers_rulebased, extract_markers_llm
from config import AWS_REGION
from rag_retrieval import ReviewRAG


def generate_report(review_text: str, model=None, rag: ReviewRAG = None) -> dict:
    """Generate a full editor report for a single peer review.

    Args:
        review_text: The peer review text to analyze.
        model: Trained classifier (loaded from disk if None).
        rag: ReviewRAG instance (created fresh if None).

    Returns:
        Complete report dict.
    """
    if model is None:
        model = load_classifier()

    # Step 1: Extract markers via Bedrock (falls back to rule-based)
    try:
        import boto3
        client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
        markers = extract_markers_llm(review_text, client)
    except Exception:
        markers = extract_markers_rulebased(review_text)

    # Step 2: Classify
    prediction = predict_single(markers, model)

    # Step 3: SHAP explanation
    shap_explanation = explain_single(model, markers)

    # Step 4: RAG retrieval
    rag_results = None
    if rag is not None:
        rag_results = rag.retrieve_with_context(review_text)

    # Step 5: Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "review_text_preview": review_text[:300] + "..." if len(review_text) > 300 else review_text,

        # Classification result
        "classification": {
            "prediction": prediction["label"],
            "ai_probability": round(prediction["ai_probability"], 4),
            "human_probability": round(prediction["human_probability"], 4),
            "confidence": "High" if max(prediction["ai_probability"], prediction["human_probability"]) > 0.8 else "Medium" if max(prediction["ai_probability"], prediction["human_probability"]) > 0.6 else "Low",
        },

        # Marker scores
        "detected_markers": {
            MARKER_LABELS.get(m, m): {
                "score": round(markers[m], 3),
                "severity": "High" if markers[m] > 0.7 else "Medium" if markers[m] > 0.4 else "Low",
            }
            for m in MARKER_NAMES
        },

        # SHAP explanation
        "explanation": {
            "top_contributing_features": [
                {
                    "feature": MARKER_LABELS.get(m, m),
                    "shap_value": round(v["shap_value"], 4),
                    "direction": v["direction"],
                    "score": round(v["score"], 3),
                }
                for m, v in list(shap_explanation["feature_contributions"].items())[:5]
            ],
            "base_value": round(shap_explanation["base_value"], 4),
        },

        # RAG similar reviews
        "similar_reviews": None,
    }

    if rag_results:
        report["similar_reviews"] = {
            "summary": rag_results["summary"],
            "top_matches": [
                {
                    "label": r["label"],
                    "similarity": round(r["similarity_score"], 3),
                    "preview": r["review_text"][:200],
                    "source": r["source"],
                }
                for r in rag_results["retrieved_reviews"][:3]
            ],
        }

    # Overall assessment
    ai_prob = prediction["ai_probability"]
    high_markers = [m for m in MARKER_NAMES if markers[m] > 0.7]

    if ai_prob > 0.8 and len(high_markers) >= 3:
        assessment = "STRONG indicators of AI generation detected."
    elif ai_prob > 0.6 and len(high_markers) >= 2:
        assessment = "MODERATE indicators of AI generation detected."
    elif ai_prob > 0.4:
        assessment = "WEAK indicators — review may have AI assistance but is inconclusive."
    else:
        assessment = "Review appears to be human-authored."

    report["overall_assessment"] = assessment

    return report


def format_report_text(report: dict) -> str:
    """Format the report as readable text for editors."""
    lines = []
    lines.append("=" * 70)
    lines.append("  AI-GENERATED PEER REVIEW DETECTION REPORT")
    lines.append("=" * 70)
    lines.append(f"  Generated: {report['timestamp']}")
    lines.append("")

    # Classification
    c = report["classification"]
    lines.append(f"  PREDICTION: {c['prediction']}")
    lines.append(f"  AI Probability: {c['ai_probability']:.1%}  |  Confidence: {c['confidence']}")
    lines.append("")

    # Overall assessment
    lines.append(f"  ASSESSMENT: {report['overall_assessment']}")
    lines.append("-" * 70)

    # Markers
    lines.append("\n  DETECTED MARKERS:")
    for name, info in report["detected_markers"].items():
        bar = "█" * int(info["score"] * 10) + "░" * (10 - int(info["score"] * 10))
        lines.append(f"    {name:30s}  [{bar}] {info['score']:.2f}  ({info['severity']})")

    # SHAP explanation
    lines.append("\n  KEY EXPLANATORY FEATURES (SHAP):")
    for feat in report["explanation"]["top_contributing_features"]:
        arrow = "↑" if "AI" in feat["direction"] else "↓"
        lines.append(f"    {arrow} {feat['feature']:30s}  SHAP={feat['shap_value']:+.4f}  ({feat['direction']})")

    # Similar reviews
    if report["similar_reviews"]:
        s = report["similar_reviews"]["summary"]
        lines.append(f"\n  SIMILAR REVIEWS (RAG):")
        lines.append(f"    Found: {s['total_retrieved']} similar | "
                     f"{s['human_matches']} human, {s['ai_matches']} AI | "
                     f"Avg similarity: {s['avg_similarity']:.3f}")
        for match in report["similar_reviews"]["top_matches"]:
            lines.append(f"    [{match['label']}] sim={match['similarity']:.3f}: {match['preview'][:80]}...")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def save_report(report: dict, filename: str = None):
    """Save report as JSON and text."""
    if filename is None:
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    json_path = OUTPUTS_DIR / f"{filename}.json"
    txt_path = OUTPUTS_DIR / f"{filename}.txt"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    with open(txt_path, "w") as f:
        f.write(format_report_text(report))

    print(f"Report saved to:\n  {json_path}\n  {txt_path}")
    return json_path, txt_path
