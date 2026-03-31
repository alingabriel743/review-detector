"""Final training: 5772 human + 1000 adversarial AI (imbalanced), 70-30 split, all 4 classifiers."""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, MARKER_NAMES, MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED
from classifier import train_all_classifiers


def main():
    # Load adversarial dataset (already has features)
    df = pd.read_csv(DATA_DIR / "features_cache_adversarial.csv")

    human = df[df["label"] == 0]
    ai = df[df["label"] == 1]
    print(f"Human: {len(human)}, AI: {len(ai)}")
    print(f"Ratio: {len(human)/len(ai):.1f}:1")
    print(f"Sources: {df['source'].value_counts().to_dict()}")

    # 70-30 split (stratified, no undersampling)
    X = df[MARKER_NAMES].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    print(f"Train dist: {np.bincount(y_train)} (human={np.sum(y_train==0)}, ai={np.sum(y_train==1)})")
    print(f"Test dist:  {np.bincount(y_test)} (human={np.sum(y_test==0)}, ai={np.sum(y_test==1)})")

    # Train all 4 classifiers (scale_pos_weight handles imbalance)
    xgb, rf, lgbm, lr, results = train_all_classifiers(X_train, y_train, X_test, y_test)

    # Also run SHAP on XGBoost
    print("\n" + "=" * 60)
    print("SHAP Explainability (XGBoost)")
    print("=" * 60)
    from explainer import run_full_explanation

    # Save features for SHAP
    df.to_csv(DATA_DIR / "features_cache.csv", index=False)
    run_full_explanation(xgb)

    # Rebuild RAG index on full dataset
    print("\n" + "=" * 60)
    print("Rebuilding RAG index")
    print("=" * 60)
    from rag_retrieval import ReviewRAG
    df.to_csv(DATA_DIR / "dataset.csv", index=False)
    rag = ReviewRAG()
    rag.build_index(df)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
