"""
SHAP Explainability Module — Explains classifier predictions using SHAP values.

Provides:
  - Global feature importance (which markers matter most overall)
  - Local explanations (which markers drove a specific prediction)
  - Visualization exports (summary plots, force plots)
"""

import json

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import MARKER_NAMES, OUTPUTS_DIR, DATA_DIR
from classifier import load_classifier


# Readable labels for the marker features
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


def compute_shap_values(model, X: np.ndarray) -> shap.Explanation:
    """Compute SHAP values for the dataset using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values


def global_importance(shap_values: shap.Explanation) -> dict:
    """Compute mean absolute SHAP value per feature (global importance)."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    # Handle multi-output (binary classification) case
    if mean_abs.ndim > 1:
        mean_abs = mean_abs[:, 1]  # class 1 = AI-generated
    importance = dict(zip(MARKER_NAMES, mean_abs.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return importance


def plot_global_summary(shap_values: shap.Explanation, X: np.ndarray, save: bool = True):
    """Generate and save SHAP summary beeswarm plot."""
    vals = shap_values.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]  # class 1

    feature_names = [MARKER_LABELS.get(m, m) for m in MARKER_NAMES]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        vals,
        X,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    if save:
        path = OUTPUTS_DIR / "shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"SHAP summary plot saved to {path}")
    plt.close()


def plot_bar_importance(shap_values: shap.Explanation, save: bool = True):
    """Generate and save SHAP bar plot of mean feature importance."""
    vals = shap_values.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    mean_abs = np.abs(vals).mean(axis=0)
    feature_names = [MARKER_LABELS.get(m, m) for m in MARKER_NAMES]

    sorted_idx = np.argsort(mean_abs)
    plt.figure(figsize=(8, 5))
    plt.barh(
        [feature_names[i] for i in sorted_idx],
        mean_abs[sorted_idx],
        color="#1f77b4",
    )
    plt.xlabel("Mean |SHAP Value|")
    plt.title("Feature Importance for AI Review Detection")
    plt.tight_layout()
    if save:
        path = OUTPUTS_DIR / "shap_bar_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"SHAP bar plot saved to {path}")
    plt.close()


def explain_single(model, review_markers: dict) -> dict:
    """Generate SHAP explanation for a single review prediction.

    Returns:
        dict with feature-level SHAP contributions and the predicted class.
    """
    X = np.array([[review_markers.get(m, 0.0) for m in MARKER_NAMES]])
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X)

    vals = shap_vals.values[0]
    if vals.ndim > 1:
        vals = vals[:, 1]

    base_value = shap_vals.base_values[0]
    if isinstance(base_value, np.ndarray):
        base_value = base_value[1]

    contributions = {}
    for i, m in enumerate(MARKER_NAMES):
        contributions[m] = {
            "score": float(review_markers.get(m, 0.0)),
            "shap_value": float(vals[i]),
            "direction": "toward AI" if vals[i] > 0 else "toward Human",
        }

    # Sort by absolute contribution
    contributions = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]["shap_value"]), reverse=True)
    )

    proba = model.predict_proba(X)[0]
    return {
        "base_value": float(base_value),
        "ai_probability": float(proba[1]),
        "prediction": "AI-Generated" if proba[1] > 0.5 else "Human",
        "feature_contributions": contributions,
    }


def run_full_explanation(model=None):
    """Run SHAP analysis on the full test set and save all outputs."""
    if model is None:
        model = load_classifier()

    features_path = DATA_DIR / "features_cache.csv"
    df = pd.read_csv(features_path)
    X = df[MARKER_NAMES].values

    print("Computing SHAP values...")
    shap_values = compute_shap_values(model, X)

    importance = global_importance(shap_values)
    print("\nGlobal Feature Importance (mean |SHAP|):")
    for feat, val in importance.items():
        label = MARKER_LABELS.get(feat, feat)
        print(f"  {label:30s} {val:.4f}")

    # Save importance
    imp_path = OUTPUTS_DIR / "global_importance.json"
    with open(imp_path, "w") as f:
        json.dump(importance, f, indent=2)

    # Generate plots
    plot_global_summary(shap_values, X)
    plot_bar_importance(shap_values)

    print("SHAP explanation complete.")
    return shap_values


if __name__ == "__main__":
    run_full_explanation()
