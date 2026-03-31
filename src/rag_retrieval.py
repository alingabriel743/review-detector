"""
RAG Retrieval Module — Finds similar reviews from the knowledge base.

Uses sentence-transformers for embedding and FAISS for efficient search.
For each query review, retrieves the top-K most similar reviews (both human
and AI-generated) to provide contextual evidence for the editor report.
"""

import json

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    MODELS_DIR,
    RAG_TOP_K,
)


class ReviewRAG:
    """Retrieval-Augmented Generation module for peer review similarity search."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.reviews_df = None
        self.embeddings = None

    def build_index(self, df: pd.DataFrame):
        """Build FAISS index from the review dataset."""
        self.reviews_df = df.reset_index(drop=True)
        texts = df["review_text"].tolist()

        print(f"Encoding {len(texts)} reviews...")
        self.embeddings = self.model.encode(
            texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True
        )

        # Build FAISS index (cosine similarity via inner product on normalized vectors)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

        # Save index and metadata
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        embeddings_path = MODELS_DIR / "embeddings.npy"
        np.save(embeddings_path, self.embeddings)

        print(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")
        print(f"Index saved to {FAISS_INDEX_PATH}")

    def load_index(self):
        """Load a previously built FAISS index."""
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError("FAISS index not found. Run build_index first.")

        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.embeddings = np.load(MODELS_DIR / "embeddings.npy")

        dataset_path = DATA_DIR / "dataset.csv"
        self.reviews_df = pd.read_csv(dataset_path)
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")

    def retrieve(self, query_text: str, top_k: int = RAG_TOP_K) -> list[dict]:
        """Retrieve the top-K most similar reviews to a query review.

        Returns:
            List of dicts with: review_text, label, similarity_score, source, paper_id
        """
        if self.index is None:
            self.load_index()

        query_embedding = self.model.encode(
            [query_text], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k + 1)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.reviews_df):
                continue
            row = self.reviews_df.iloc[idx]
            # Skip if it's the exact same review
            if row["review_text"].strip() == query_text.strip():
                continue
            results.append({
                "review_text": row["review_text"][:500] + "..." if len(row["review_text"]) > 500 else row["review_text"],
                "full_text": row["review_text"],
                "label": "AI-Generated" if row["label"] == 1 else "Human",
                "similarity_score": float(score),
                "source": row.get("source", "unknown"),
                "paper_id": row.get("paper_id", "unknown"),
            })
            if len(results) >= top_k:
                break

        return results

    def retrieve_with_context(self, query_text: str, top_k: int = RAG_TOP_K) -> dict:
        """Retrieve similar reviews and provide a summary context.

        Returns:
            dict with retrieved reviews and aggregate statistics.
        """
        results = self.retrieve(query_text, top_k)

        human_count = sum(1 for r in results if r["label"] == "Human")
        ai_count = sum(1 for r in results if r["label"] == "AI-Generated")
        avg_similarity = np.mean([r["similarity_score"] for r in results]) if results else 0.0

        return {
            "retrieved_reviews": results,
            "summary": {
                "total_retrieved": len(results),
                "human_matches": human_count,
                "ai_matches": ai_count,
                "avg_similarity": float(avg_similarity),
                "most_similar_label": results[0]["label"] if results else "N/A",
                "most_similar_score": results[0]["similarity_score"] if results else 0.0,
            },
        }


if __name__ == "__main__":
    dataset_path = DATA_DIR / "dataset.csv"
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        rag = ReviewRAG()
        rag.build_index(df)

        # Test retrieval with the first review
        sample = df.iloc[0]["review_text"]
        results = rag.retrieve_with_context(sample)
        print(f"\nQuery: {sample[:100]}...")
        print(f"Retrieved {results['summary']['total_retrieved']} similar reviews")
        for r in results["retrieved_reviews"]:
            print(f"  [{r['label']}] sim={r['similarity_score']:.3f}: {r['review_text'][:80]}...")
    else:
        print("Run data_loader.py first.")
