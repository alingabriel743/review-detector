"""
Classifier — Trains and evaluates multiple models on the 8 extracted marker features.

Models: XGBoost, Random Forest, LightGBM, Logistic Regression — all with hyperparameter tuning.
Outputs: trained models, classification reports, confusion matrices, comparison.
"""

import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from config import (
    CLASSIFIER_PATH,
    DATA_DIR,
    MARKER_NAMES,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    TEST_SPLIT,
)


def prepare_splits(df: pd.DataFrame, balance: bool = True):
    """Split dataset into train/test using the 8 marker features.

    If balance=True, undersample the majority class (human) to match
    the minority class (AI) count, producing balanced train/test sets.
    """
    if balance:
        human_df = df[df["label"] == 0]
        ai_df = df[df["label"] == 1]
        n_ai = len(ai_df)
        human_sampled = human_df.sample(n=n_ai, random_state=RANDOM_SEED)
        df_balanced = pd.concat([human_sampled, ai_df], ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"Balanced dataset: {len(df_balanced)} reviews "
              f"({(df_balanced['label']==0).sum()} human, {(df_balanced['label']==1).sum()} AI)")
        X = df_balanced[MARKER_NAMES].values
        y = df_balanced["label"].values
    else:
        X = df[MARKER_NAMES].values
        y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Train distribution: {np.bincount(y_train)} | Test distribution: {np.bincount(y_test)}")
    return X_train, X_test, y_train, y_test


def train_classifier(X_train, y_train) -> XGBClassifier:
    """Train an XGBoost classifier with grid search over key hyperparameters."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }

    # scale_pos_weight handles any residual imbalance after undersampling
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

    base_model = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        scale_pos_weight=scale_weight,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")

    # Save model
    joblib.dump(best_model, CLASSIFIER_PATH)
    print(f"Model saved to {CLASSIFIER_PATH}")
    return best_model


def evaluate_classifier(model, X_test, y_test):
    """Evaluate the classifier and save results."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["Human", "AI-Generated"])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    print(f"\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    # Save results
    results = {
        "accuracy": float(acc),
        "auc_roc": float(auc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "feature_importance": dict(zip(MARKER_NAMES, model.feature_importances_.tolist())),
    }
    results_path = OUTPUTS_DIR / "classifier_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Train a Random Forest classifier with grid search."""
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    weight = neg_count / pos_count if pos_count > 0 else 1.0

    base_model = RandomForestClassifier(
        random_state=RANDOM_SEED,
        class_weight={0: 1.0, 1: weight},
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"[RF] Best params: {grid_search.best_params_}")
    print(f"[RF] Best CV AUC: {grid_search.best_score_:.4f}")

    joblib.dump(best_model, MODELS_DIR / "classifier_rf.joblib")
    return best_model


def train_lightgbm(X_train, y_train) -> LGBMClassifier:
    """Train a LightGBM classifier with grid search."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "subsample": [0.8, 1.0],
    }

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    weight = neg_count / pos_count if pos_count > 0 else 1.0

    base_model = LGBMClassifier(
        random_state=RANDOM_SEED,
        scale_pos_weight=weight,
        verbose=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Use DataFrame with feature names to avoid LightGBM warnings
    X_df = pd.DataFrame(X_train, columns=MARKER_NAMES)

    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1,
    )
    grid_search.fit(X_df, y_train)

    best_model = grid_search.best_estimator_
    print(f"[LGBM] Best params: {grid_search.best_params_}")
    print(f"[LGBM] Best CV AUC: {grid_search.best_score_:.4f}")

    joblib.dump(best_model, MODELS_DIR / "classifier_lgbm.joblib")
    return best_model


def train_logistic_regression(X_train, y_train) -> Pipeline:
    """Train a Logistic Regression classifier with grid search (scaled features)."""
    param_grid = {
        "lr__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "lr__l1_ratio": [0.0, 0.5, 1.0],
    }

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    weight = neg_count / pos_count if pos_count > 0 else 1.0

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            random_state=RANDOM_SEED,
            class_weight={0: 1.0, 1: weight},
            max_iter=5000,
            solver="saga",
            penalty="elasticnet",
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"[LR] Best params: {grid_search.best_params_}")
    print(f"[LR] Best CV AUC: {grid_search.best_score_:.4f}")

    joblib.dump(best_model, MODELS_DIR / "classifier_lr.joblib")
    return best_model


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """Evaluate a single model and return results dict."""
    # Use DataFrame for LightGBM to avoid feature name warnings
    if isinstance(model, LGBMClassifier):
        X_eval = pd.DataFrame(X_test, columns=MARKER_NAMES)
    else:
        X_eval = X_test
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["Human", "AI-Generated"])
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance
    if hasattr(model, "feature_importances_"):
        feat_imp = dict(zip(MARKER_NAMES, model.feature_importances_.tolist()))
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("lr", None), "coef_"):
        # Logistic Regression inside a Pipeline — use absolute coefficients
        coefs = np.abs(model.named_steps["lr"].coef_[0])
        feat_imp = dict(zip(MARKER_NAMES, coefs.tolist()))
    else:
        feat_imp = {}

    print(f"\n[{name}] Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f}")
    print(report)
    print(f"Confusion Matrix:\n{cm}")

    return {
        "model_name": name,
        "accuracy": float(acc),
        "auc_roc": float(auc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "feature_importance": feat_imp,
    }


def train_all_classifiers(X_train, y_train, X_test, y_test):
    """Train XGBoost, Random Forest, LightGBM, and Logistic Regression. Save all results."""
    print("\n" + "=" * 60)
    print("Training XGBoost...")
    print("=" * 60)
    xgb_model = train_classifier(X_train, y_train)
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    print("\n" + "=" * 60)
    print("Training Random Forest...")
    print("=" * 60)
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "RandomForest")

    print("\n" + "=" * 60)
    print("Training LightGBM...")
    print("=" * 60)
    lgbm_model = train_lightgbm(X_train, y_train)
    lgbm_results = evaluate_model(lgbm_model, X_test, y_test, "LightGBM")

    print("\n" + "=" * 60)
    print("Training Logistic Regression...")
    print("=" * 60)
    lr_model = train_logistic_regression(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "LogisticRegression")

    # Save comparison
    all_results = {
        "XGBoost": xgb_results,
        "RandomForest": rf_results,
        "LightGBM": lgbm_results,
        "LogisticRegression": lr_results,
    }

    comparison_path = OUTPUTS_DIR / "classifier_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComparison saved to {comparison_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Model':<15s} {'Accuracy':>10s} {'AUC-ROC':>10s} {'FP':>5s} {'FN':>5s}")
    print("-" * 45)
    for name, res in all_results.items():
        cm = np.array(res["confusion_matrix"])
        print(f"{name:<15s} {res['accuracy']:>10.4f} {res['auc_roc']:>10.4f} {cm[0,1]:>5d} {cm[1,0]:>5d}")
    print("=" * 60)

    # Also save individual results for backwards compatibility
    with open(OUTPUTS_DIR / "classifier_results.json", "w") as f:
        json.dump(xgb_results, f, indent=2)

    return xgb_model, rf_model, lgbm_model, lr_model, all_results


def load_classifier():
    """Load a previously trained classifier."""
    if CLASSIFIER_PATH.exists():
        return joblib.load(CLASSIFIER_PATH)
    raise FileNotFoundError(f"No trained model at {CLASSIFIER_PATH}. Run training first.")


def predict_single(review_markers: dict, model=None) -> dict:
    """Predict on a single review given its marker scores.

    Returns:
        dict with 'label', 'probability', 'marker_scores'
    """
    if model is None:
        model = load_classifier()

    X = np.array([[review_markers.get(m, 0.0) for m in MARKER_NAMES]])
    proba = model.predict_proba(X)[0]
    label = int(model.predict(X)[0])

    return {
        "label": "AI-Generated" if label == 1 else "Human",
        "ai_probability": float(proba[1]),
        "human_probability": float(proba[0]),
        "marker_scores": review_markers,
    }


if __name__ == "__main__":
    features_path = DATA_DIR / "features_cache.csv"
    if features_path.exists():
        df = pd.read_csv(features_path)
        X_train, X_test, y_train, y_test = prepare_splits(df)
        train_all_classifiers(X_train, y_train, X_test, y_test)
    else:
        print("Run feature_extractor.py first.")
