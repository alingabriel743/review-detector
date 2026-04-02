"""Generate publication-quality figures for the research paper."""

import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, f1_score
)
from sklearn.model_selection import train_test_split

from config import DATA_DIR, MARKER_NAMES, MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

MARKER_LABELS = [
    "Standardized\nStructure",
    "Predictable\nCriticism",
    "Excessive\nBalance",
    "Linguistic\nHomogeneity",
    "Generic Domain\nLanguage",
    "Conceptual\nFeedback",
    "No Personal\nSignals",
    "Repetition\nPatterns",
]

MARKER_LABELS_SHORT = [
    "Struct.", "Crit.", "Bal.", "Homo.", "Gen.Lang.", "Concept.", "Pers.Sig.", "Repet."
]

MARKER_LABELS_FULL = [
    "Standardized Structure",
    "Predictable Criticism",
    "Excessive Balance",
    "Linguistic Homogeneity",
    "Generic Domain Language",
    "Conceptual Feedback",
    "Absence of Personal Signals",
    "Repetition Patterns",
]

CATEGORY_MAP = {
    0: "Structural", 1: "Argumentative", 2: "Argumentative",
    3: "Linguistic", 4: "Linguistic",
    5: "Behavioral", 6: "Behavioral", 7: "Behavioral",
}

CAT_COLORS = {
    "Structural": "#c0392b",
    "Argumentative": "#e67e22",
    "Linguistic": "#2980b9",
    "Behavioral": "#27ae60",
}

MODEL_NAMES = ["XGBoost", "RandomForest", "LightGBM", "LogisticRegression"]
MODEL_FILES = ["classifier.joblib", "classifier_rf.joblib", "classifier_lgbm.joblib", "classifier_lr.joblib"]
MODEL_COLORS = ["#3498db", "#27ae60", "#e74c3c", "#9b59b6"]

FIG_DIR = OUTPUTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_DIR / "features_cache.csv")
    X = df[MARKER_NAMES].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    return df, X_train, X_test, y_train, y_test


def load_models():
    models = {}
    for name, f in zip(MODEL_NAMES, MODEL_FILES):
        path = MODELS_DIR / f
        if path.exists():
            models[name] = joblib.load(path)
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Dataset composition
# ═══════════════════════════════════════════════════════════════════════════════
def fig_dataset_composition(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Class distribution
    counts = df["label"].value_counts().sort_index()
    bars = axes[0].bar(["Human", "AI-Generated"], [counts[0], counts[1]],
                       color=["#27ae60", "#e74c3c"], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, [counts[0], counts[1]]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     str(val), ha="center", fontweight="bold")
    axes[0].set_ylabel("Number of Reviews")
    axes[0].set_title("(a) Class Distribution")

    # (b) Source distribution
    src = df["source"].value_counts()
    src_labels = [s.replace("peerread_", "PR:").replace("adversarial_", "Adv:").replace("genreview_ai_neutral", "Gen-Review") for s in src.index]
    colors = ["#27ae60" if "peerread" in s else "#e74c3c" for s in src.index]
    bars = axes[1].barh(src_labels, src.values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, src.values):
        axes[1].text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                     str(val), va="center", fontsize=9)
    axes[1].set_xlabel("Count")
    axes[1].set_title("(b) Source Distribution")
    axes[1].invert_yaxis()

    # (c) Review length distribution
    df["wc"] = df["review_text"].str.split().str.len()
    human_wc = df[df["label"]==0]["wc"]
    ai_wc = df[df["label"]==1]["wc"]
    axes[2].hist(human_wc, bins=50, alpha=0.7, color="#27ae60", label="Human", edgecolor="black", linewidth=0.3)
    axes[2].hist(ai_wc, bins=50, alpha=0.7, color="#e74c3c", label="AI-Generated", edgecolor="black", linewidth=0.3)
    axes[2].set_xlabel("Word Count")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("(c) Review Length Distribution")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_dataset_composition.pdf")
    plt.savefig(FIG_DIR / "fig1_dataset_composition.png")
    plt.close()
    print("Fig 1: Dataset composition")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Marker distribution comparison (Human vs AI)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_marker_distributions(df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for i, m in enumerate(MARKER_NAMES):
        human = df[df["label"]==0][m]
        ai = df[df["label"]==1][m]
        axes[i].hist(human, bins=20, alpha=0.7, color="#27ae60", label="Human", density=True, edgecolor="black", linewidth=0.3)
        axes[i].hist(ai, bins=20, alpha=0.7, color="#e74c3c", label="AI", density=True, edgecolor="black", linewidth=0.3)
        axes[i].set_title(MARKER_LABELS_FULL[i], fontsize=10)
        axes[i].set_xlim(0, 1)
        if i == 0:
            axes[i].legend(fontsize=8)

    plt.suptitle("Marker Score Distributions: Human vs AI-Generated Reviews", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_marker_distributions.pdf")
    plt.savefig(FIG_DIR / "fig2_marker_distributions.png")
    plt.close()
    print("Fig 2: Marker distributions")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Marker comparison (box plot)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_marker_boxplot(df):
    fig, ax = plt.subplots(figsize=(12, 5))

    plot_data = []
    for i, m in enumerate(MARKER_NAMES):
        for _, row in df.iterrows():
            plot_data.append({
                "Marker": MARKER_LABELS_FULL[i],
                "Score": row[m],
                "Class": "Human" if row["label"] == 0 else "AI-Generated",
            })

    plot_df = pd.DataFrame(plot_data)
    sns.boxplot(data=plot_df, x="Marker", y="Score", hue="Class",
                palette={"Human": "#27ae60", "AI-Generated": "#e74c3c"},
                ax=ax, fliersize=2, linewidth=0.8)
    ax.set_xticklabels([l.replace(" ", "\n") for l in MARKER_LABELS_FULL], rotation=0, fontsize=9)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Marker Score Comparison: Human vs AI-Generated Reviews")
    ax.legend(title="Class")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_marker_boxplot.pdf")
    plt.savefig(FIG_DIR / "fig3_marker_boxplot.png")
    plt.close()
    print("Fig 3: Marker boxplot")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: ROC curves for all models
# ═══════════════════════════════════════════════════════════════════════════════
def fig_roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, color in zip(MODEL_NAMES, MODEL_COLORS):
        if name not in models:
            continue
        model = models[name]
        if name == "LightGBM":
            X_input = pd.DataFrame(X_test, columns=MARKER_NAMES)
        else:
            X_input = X_test
        y_proba = model.predict_proba(X_input)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — AI Review Detection")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_roc_curves.pdf")
    plt.savefig(FIG_DIR / "fig4_roc_curves.png")
    plt.close()
    print("Fig 4: ROC curves")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Precision-Recall curves
# ═══════════════════════════════════════════════════════════════════════════════
def fig_pr_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, color in zip(MODEL_NAMES, MODEL_COLORS):
        if name not in models:
            continue
        model = models[name]
        if name == "LightGBM":
            X_input = pd.DataFrame(X_test, columns=MARKER_NAMES)
        else:
            X_input = X_test
        y_proba = model.predict_proba(X_input)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        f1 = f1_score(y_test, model.predict(X_input))
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{name} (F1 = {f1:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — AI Review Detection")
    ax.legend(loc="lower left")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_pr_curves.pdf")
    plt.savefig(FIG_DIR / "fig5_pr_curves.png")
    plt.close()
    print("Fig 5: Precision-Recall curves")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Confusion matrices (2x2 grid)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_confusion_matrices(models, X_test, y_test):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for idx, (name, color) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
        if name not in models:
            continue
        model = models[name]
        if name == "LightGBM":
            X_input = pd.DataFrame(X_test, columns=MARKER_NAMES)
        else:
            X_input = X_test
        y_pred = model.predict(X_input)
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=["Human", "AI"], yticklabels=["Human", "AI"],
                    annot_kws={"size": 14}, linewidths=1, linecolor="black")
        axes[idx].set_title(f"{name}\nAcc: {acc:.2%}", fontsize=11)
        axes[idx].set_ylabel("True" if idx == 0 else "")
        axes[idx].set_xlabel("Predicted")

    plt.suptitle("Confusion Matrices — All Classifiers", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_confusion_matrices.pdf")
    plt.savefig(FIG_DIR / "fig6_confusion_matrices.png")
    plt.close()
    print("Fig 6: Confusion matrices")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Model comparison bar chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig_model_comparison(models, X_test, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    metrics = {"Accuracy": [], "AUC-ROC": [], "F1-Score": []}
    names_used = []

    for name in MODEL_NAMES:
        if name not in models:
            continue
        model = models[name]
        if name == "LightGBM":
            X_input = pd.DataFrame(X_test, columns=MARKER_NAMES)
        else:
            X_input = X_test
        y_pred = model.predict(X_input)
        y_proba = model.predict_proba(X_input)[:, 1]
        metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["AUC-ROC"].append(roc_auc_score(y_test, y_proba))
        metrics["F1-Score"].append(f1_score(y_test, y_pred))
        names_used.append(name)

    for idx, (metric_name, values) in enumerate(metrics.items()):
        bars = axes[idx].bar(names_used, values, color=MODEL_COLORS[:len(names_used)],
                             edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                          f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
        axes[idx].set_title(metric_name)
        axes[idx].set_ylim(min(values) - 0.05, 1.005)
        axes[idx].tick_params(axis="x", rotation=20)
        axes[idx].grid(axis="y", alpha=0.3)

    plt.suptitle("Classifier Performance Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_model_comparison.pdf")
    plt.savefig(FIG_DIR / "fig7_model_comparison.png")
    plt.close()
    print("Fig 7: Model comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: SHAP global importance (bar)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_shap_importance(models, X_test):
    from matplotlib.patches import Patch

    tree_models = ["XGBoost", "RandomForest", "LightGBM"]
    available = [m for m in tree_models if m in models]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for idx, name in enumerate(available):
        model = models[name]
        if name == "LightGBM":
            X_input = pd.DataFrame(X_test, columns=MARKER_NAMES)
        else:
            X_input = X_test
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_input)
        vals = shap_values.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]

        mean_abs = np.abs(vals).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)
        colors = [CAT_COLORS[CATEGORY_MAP[i]] for i in sorted_idx]

        bars = axes[idx].barh([MARKER_LABELS_FULL[i] for i in sorted_idx], mean_abs[sorted_idx],
                              color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, mean_abs[sorted_idx]):
            axes[idx].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                           f"{val:.3f}", va="center", fontsize=9)
        axes[idx].set_xlabel("Mean |SHAP Value|")
        axes[idx].set_title(f"{name}")

    # Category legend on last axis
    legend_elements = [Patch(facecolor=c, edgecolor="black", label=cat) for cat, c in CAT_COLORS.items()]
    axes[-1].legend(handles=legend_elements, loc="lower right", title="Category", fontsize=8)

    plt.suptitle("Global Feature Importance (SHAP) — Tree-Based Classifiers", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig8_shap_importance.pdf")
    plt.savefig(FIG_DIR / "fig8_shap_importance.png")
    plt.close()
    print(f"Fig 8: SHAP importance ({', '.join(available)})")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: SHAP beeswarm
# ═══════════════════════════════════════════════════════════════════════════════
def fig_shap_beeswarm(models, X_test):
    tree_models = ["XGBoost", "RandomForest", "LightGBM"]
    available = [m for m in tree_models if m in models]
    if not available:
        return

    # Compute SHAP values for each model
    all_vals = {}
    for name in available:
        model = models[name]
        if name == "LightGBM":
            X_input = pd.DataFrame(X_test, columns=MARKER_NAMES)
        else:
            X_input = X_test
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_input)
        vals = shap_values.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        all_vals[name] = vals

    # Side-by-side with shared y-axis, single colorbar
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 6),
                              sharey=True)
    if len(available) == 1:
        axes = [axes]

    for idx, name in enumerate(available):
        vals = all_vals[name]
        ax = axes[idx]

        # Manual beeswarm-style: dot strip plot colored by feature value
        feature_order = np.argsort(np.abs(vals).mean(axis=0))[::-1]

        for row_pos, feat_idx in enumerate(feature_order):
            shap_col = vals[:, feat_idx]
            feat_col = X_test[:, feat_idx]

            # Normalize feature values to 0-1 for coloring
            f_min, f_max = feat_col.min(), feat_col.max()
            if f_max > f_min:
                feat_norm = (feat_col - f_min) / (f_max - f_min)
            else:
                feat_norm = np.zeros_like(feat_col)

            # Add jitter for y
            jitter = np.random.RandomState(42).uniform(-0.3, 0.3, size=len(shap_col))

            colors = plt.cm.bwr(feat_norm)
            ax.scatter(shap_col, row_pos + jitter, c=colors, s=4, alpha=0.5,
                      rasterized=True, linewidths=0)

        ax.set_yticks(range(len(feature_order)))
        if idx == 0:
            ax.set_yticklabels([MARKER_LABELS_FULL[i] for i in feature_order], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("SHAP value", fontsize=10)
        ax.set_title(f"({chr(97+idx)}) {name}", fontsize=12)
        ax.invert_yaxis()

    # Single colorbar on the right
    sm = plt.cm.ScalarMappable(cmap="bwr", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.6, pad=0.02)
    cbar.set_label("Feature Value", fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])

    plt.suptitle("SHAP Beeswarm Plots — Tree-Based Classifiers", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig9_shap_beeswarm.pdf")
    plt.savefig(FIG_DIR / "fig9_shap_beeswarm.png")
    plt.close()

    # Also save individual clean plots
    for name in available:
        vals = all_vals[name]
        plt.figure(figsize=(10, 5))
        shap.summary_plot(vals, X_test, feature_names=MARKER_LABELS_FULL, show=False)
        plt.title(f"SHAP Beeswarm — {name}", fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"fig9_shap_beeswarm_{name.lower()}.pdf")
        plt.savefig(FIG_DIR / f"fig9_shap_beeswarm_{name.lower()}.png")
        plt.close()

    print(f"Fig 9: SHAP beeswarm ({', '.join(available)})")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Feature importance comparison across models
# ═══════════════════════════════════════════════════════════════════════════════
def fig_feature_importance_comparison(models):
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(MARKER_NAMES))
    width = 0.18
    n_models = 0

    for idx, (name, color) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
        if name not in models:
            continue
        model = models[name]
        if hasattr(model, "feature_importances_"):
            imp = np.array(model.feature_importances_, dtype=float)
        elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("lr", None), "coef_"):
            imp = np.abs(model.named_steps["lr"].coef_[0])
        else:
            continue
        # Normalize to 0-1 so all models are comparable
        imp_max = imp.max()
        if imp_max > 0:
            imp = imp / imp_max
        ax.bar(x + n_models * width, imp, width, label=name, color=color,
               edgecolor="black", linewidth=0.3, alpha=0.85)
        n_models += 1

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([l.replace(" ", "\n") for l in MARKER_LABELS_FULL], fontsize=9)
    ax.set_ylabel("Normalized Feature Importance")
    ax.set_title("Feature Importance Comparison Across Classifiers")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_feature_importance_comparison.pdf")
    plt.savefig(FIG_DIR / "fig10_feature_importance_comparison.png")
    plt.close()
    print("Fig 10: Feature importance comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: Correlation heatmap of features
# ═══════════════════════════════════════════════════════════════════════════════
def fig_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 7))

    corr = df[MARKER_NAMES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=MARKER_LABELS_SHORT, yticklabels=MARKER_LABELS_SHORT,
                linewidths=0.5, annot_kws={"size": 9})
    ax.set_title("Feature Correlation Matrix")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_correlation_heatmap.pdf")
    plt.savefig(FIG_DIR / "fig11_correlation_heatmap.png")
    plt.close()
    print("Fig 11: Correlation heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: Radar chart (mean markers per class)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_radar_chart(df):
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(MARKER_NAMES), endpoint=False).tolist()
    angles += angles[:1]

    human_means = df[df["label"]==0][MARKER_NAMES].mean().values.tolist()
    ai_means = df[df["label"]==1][MARKER_NAMES].mean().values.tolist()
    human_means += human_means[:1]
    ai_means += ai_means[:1]

    ax.plot(angles, human_means, "o-", color="#27ae60", linewidth=2, label="Human", markersize=5)
    ax.fill(angles, human_means, alpha=0.15, color="#27ae60")
    ax.plot(angles, ai_means, "s-", color="#e74c3c", linewidth=2, label="AI-Generated", markersize=5)
    ax.fill(angles, ai_means, alpha=0.15, color="#e74c3c")

    ax.set_thetagrids(np.degrees(angles[:-1]), MARKER_LABELS, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Mean Marker Scores: Human vs AI-Generated", y=1.08, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig12_radar_chart.pdf")
    plt.savefig(FIG_DIR / "fig12_radar_chart.png")
    plt.close()
    print("Fig 12: Radar chart")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading data and models...")
    df, X_train, X_test, y_train, y_test = load_data()
    models = load_models()
    print(f"Dataset: {len(df)} reviews | Models: {list(models.keys())}")

    print(f"\nGenerating figures to {FIG_DIR}/\n")

    fig_dataset_composition(df)
    fig_marker_distributions(df)
    fig_marker_boxplot(df)
    fig_roc_curves(models, X_test, y_test)
    fig_pr_curves(models, X_test, y_test)
    fig_confusion_matrices(models, X_test, y_test)
    fig_model_comparison(models, X_test, y_test)
    fig_shap_importance(models, X_test)
    fig_shap_beeswarm(models, X_test)
    fig_feature_importance_comparison(models)
    fig_correlation_heatmap(df)
    fig_radar_chart(df)

    print(f"\nAll figures saved to {FIG_DIR}/")
    print(f"Formats: PDF (vector) + PNG (300 DPI)")


if __name__ == "__main__":
    main()
