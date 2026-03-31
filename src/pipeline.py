"""
Main Pipeline — Runs the full AI-generated peer review detection framework.

Usage:
    python pipeline.py                     # Run full pipeline (data → train → evaluate)
    python pipeline.py --analyze "text"    # Analyze a single review
    python pipeline.py --skip-data         # Skip data download, use existing dataset
"""

import argparse
import sys

import pandas as pd

from config import DATA_DIR, MARKER_NAMES


def run_full_pipeline(skip_data: bool = False, use_llm: bool = True):
    """Execute the complete pipeline end-to-end."""

    # ── Step 1: Data preparation ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)

    dataset_path = DATA_DIR / "dataset.csv"

    if skip_data and dataset_path.exists():
        print(f"Loading existing dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        from data_loader import build_dataset
        df = build_dataset()

    print(f"Dataset: {len(df)} reviews")
    print(f"  Human:        {(df['label'] == 0).sum()}")
    print(f"  AI-Generated: {(df['label'] == 1).sum()}")

    # ── Step 2: Feature extraction ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: LLM FEATURE EXTRACTION (8 Markers)")
    print("=" * 60)

    from feature_extractor import extract_features
    df = extract_features(df, use_llm=use_llm)

    print("\nMarker statistics:")
    print(df[MARKER_NAMES].describe().round(3))

    # ── Step 3: Classifier training ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: CLASSIFIER TRAINING (XGBoost + Random Forest + LightGBM)")
    print("=" * 60)

    from classifier import prepare_splits, train_all_classifiers
    X_train, X_test, y_train, y_test = prepare_splits(df)
    model, rf_model, lgbm_model, lr_model, all_results = train_all_classifiers(X_train, y_train, X_test, y_test)

    # ── Step 4: SHAP Explainability ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: SHAP EXPLAINABILITY")
    print("=" * 60)

    from explainer import run_full_explanation
    run_full_explanation(model)

    # ── Step 5: RAG Index ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: RAG RETRIEVAL INDEX")
    print("=" * 60)

    from rag_retrieval import ReviewRAG
    rag = ReviewRAG()
    rag.build_index(df)

    # Test retrieval
    sample_review = df.iloc[0]["review_text"]
    results = rag.retrieve_with_context(sample_review)
    print(f"\nSample retrieval: {results['summary']['total_retrieved']} results, "
          f"avg similarity: {results['summary']['avg_similarity']:.3f}")

    # ── Step 6: Generate sample report ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: SAMPLE EDITOR REPORT")
    print("=" * 60)

    from report_generator import generate_report, format_report_text, save_report
    report = generate_report(sample_review, model=model, rag=rag)
    print(format_report_text(report))
    save_report(report, "sample_report")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {DATA_DIR.parent / 'outputs'}")


def analyze_single_review(review_text: str):
    """Analyze a single review using the trained pipeline."""
    from classifier import load_classifier
    from rag_retrieval import ReviewRAG
    from report_generator import generate_report, format_report_text, save_report

    model = load_classifier()
    rag = ReviewRAG()
    rag.load_index()

    report = generate_report(review_text, model=model, rag=rag)
    print(format_report_text(report))
    save_report(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Generated Peer Review Detection Pipeline")
    parser.add_argument("--analyze", type=str, help="Analyze a single review text")
    parser.add_argument("--analyze-file", type=str, help="Analyze review from a text file")
    parser.add_argument("--skip-data", action="store_true", help="Skip data download")
    parser.add_argument("--no-llm", action="store_true", help="Use rule-based extraction only")

    args = parser.parse_args()

    if args.analyze:
        analyze_single_review(args.analyze)
    elif args.analyze_file:
        with open(args.analyze_file, "r") as f:
            text = f.read()
        analyze_single_review(text)
    else:
        run_full_pipeline(skip_data=args.skip_data, use_llm=not args.no_llm)
