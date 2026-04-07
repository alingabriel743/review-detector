"""Train on combined dataset: 5772 human + 1000 Gen-Review (GPT-4o) + 1000 adversarial AI."""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, MARKER_NAMES, MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED
from classifier import train_all_classifiers


def main():
    # Load human + adversarial (already scored with neutral LLM prompt)
    adversarial_df = pd.read_csv(DATA_DIR / "features_cache_adversarial.csv")
    human = adversarial_df[adversarial_df["label"] == 0]
    adversarial_ai = adversarial_df[adversarial_df["label"] == 1]

    # Load Gen-Review scored reviews
    genreview = pd.read_csv(DATA_DIR / "genreview_scored.csv")

    print(f"Human reviews: {len(human)}")
    print(f"Gen-Review AI (GPT-4o neutral): {len(genreview)}")
    print(f"Adversarial AI (Claude+GPT): {len(adversarial_ai)}")

    # Combine
    combined = pd.concat([human, genreview, adversarial_ai], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Verify no NaNs
    nans = combined[MARKER_NAMES].isna().any(axis=1).sum()
    if nans > 0:
        print(f"WARNING: Dropping {nans} rows with NaN scores")
        combined = combined.dropna(subset=MARKER_NAMES).reset_index(drop=True)

    print(f"\nCombined: {len(combined)} reviews")
    print(f"  Human: {(combined['label']==0).sum()}")
    print(f"  AI:    {(combined['label']==1).sum()}")
    print(f"  Sources: {combined['source'].value_counts().to_dict()}")

    combined.to_csv(DATA_DIR / "dataset_combined.csv", index=False)

    # 70-30 split
    X = combined[MARKER_NAMES].values
    y = combined["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    print(f"Train: human={np.sum(y_train==0)}, ai={np.sum(y_train==1)}")
    print(f"Test:  human={np.sum(y_test==0)}, ai={np.sum(y_test==1)}")

    # Save test set for reproducible figures
    np.save(DATA_DIR / "X_test.npy", X_test)
    np.save(DATA_DIR / "y_test.npy", y_test)
    print("Saved X_test.npy and y_test.npy")

    # Train all 4
    xgb, rf, lgbm, lr, results = train_all_classifiers(X_train, y_train, X_test, y_test)

    # Save comparison
    with open(OUTPUTS_DIR / "combined_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # SHAP
    print("\n" + "=" * 60)
    print("SHAP Explainability (XGBoost)")
    print("=" * 60)
    from explainer import run_full_explanation
    combined.to_csv(DATA_DIR / "features_cache.csv", index=False)
    run_full_explanation(xgb)

    # RAG
    print("\n" + "=" * 60)
    print("Rebuilding RAG index")
    print("=" * 60)
    from rag_retrieval import ReviewRAG
    combined.to_csv(DATA_DIR / "dataset.csv", index=False)
    rag = ReviewRAG()
    rag.build_index(combined)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
