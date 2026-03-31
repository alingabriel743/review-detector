"""
Data Loader — Downloads PeerRead from GitHub and prepares the dataset.

Steps:
  1. Clone the PeerRead repo (https://github.com/allenai/PeerRead)
  2. Parse human reviews from the JSON files
  3. Generate AI reviews using Claude via Amazon Bedrock
  4. Output a combined DataFrame with columns: review_text, label, source, paper_id
"""

import json
import subprocess
import time

import boto3
import pandas as pd
from tqdm import tqdm

from config import (
    ANTHROPIC_VERSION,
    AWS_REGION,
    BEDROCK_MODEL_ID,
    DATA_DIR,
    PEERREAD_DIR,
    PEERREAD_REVIEWS_DIRS,
    RANDOM_SEED,
)


def clone_peerread():
    """Clone PeerRead repo if not already present."""
    if PEERREAD_DIR.exists() and any(PEERREAD_DIR.iterdir()):
        print("PeerRead already cloned.")
        return
    print("Cloning PeerRead repository...")
    subprocess.run(
        ["git", "clone", "https://github.com/allenai/PeerRead.git", str(PEERREAD_DIR)],
        check=True,
    )
    print("PeerRead cloned successfully.")


def load_human_reviews() -> pd.DataFrame:
    """Parse human peer reviews from PeerRead JSON files."""
    records = []

    for venue_dir in PEERREAD_REVIEWS_DIRS:
        for split in ["train", "dev", "test"]:
            reviews_dir = venue_dir / split / "reviews"
            if not reviews_dir.exists():
                continue
            for json_file in sorted(reviews_dir.glob("*.json")):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                if isinstance(data, list):
                    reviews = data
                elif isinstance(data, dict) and "reviews" in data:
                    reviews = data["reviews"]
                elif isinstance(data, dict) and "comments" in data:
                    reviews = [data]
                else:
                    reviews = [data]

                for rev in reviews:
                    text = rev.get("comments", "") or rev.get("review", "") or rev.get("text", "")
                    if not text or len(text.strip()) < 50:
                        continue
                    paper_id = json_file.stem
                    venue = venue_dir.name
                    records.append({
                        "review_text": text.strip(),
                        "label": 0,  # 0 = human
                        "source": f"peerread_{venue}",
                        "paper_id": paper_id,
                    })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} human reviews from PeerRead.")
    return df


def load_paper_metadata() -> list[dict]:
    """Load paper titles and abstracts to use as prompts for AI review generation."""
    papers = []
    for venue_dir in PEERREAD_REVIEWS_DIRS:
        for split in ["train", "dev", "test"]:
            papers_dir = venue_dir / split / "parsed_pdfs"
            if not papers_dir.exists():
                continue
            for json_file in sorted(papers_dir.glob("*.json")):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                title = data.get("metadata", {}).get("title", "") or data.get("title", "")
                abstract = (
                    data.get("metadata", {}).get("abstractText", "")
                    or data.get("abstract", "")
                    or ""
                )
                if title and abstract and len(abstract) > 50:
                    papers.append({
                        "paper_id": json_file.stem,
                        "title": title,
                        "abstract": abstract[:2000],
                        "venue": venue_dir.name,
                    })
    print(f"Loaded metadata for {len(papers)} papers.")
    return papers


def _invoke_bedrock(client, prompt: str, max_tokens: int = 1500, temperature: float = 0.7) -> str:
    """Invoke Claude on Amazon Bedrock and return the response text."""
    request_body = {
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(request_body),
    )
    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


def generate_ai_reviews(papers: list[dict], n_reviews: int = 300) -> pd.DataFrame:
    """Generate AI peer reviews using Claude via Amazon Bedrock.

    Falls back to loading from a cached CSV if available.
    """
    cache_path = DATA_DIR / "ai_reviews_cache.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"Loaded {len(df)} cached AI reviews from {cache_path}")
        return df

    try:
        client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
    except Exception as e:
        print(f"WARNING: Could not create Bedrock client: {e}")
        print("Provide data/ai_reviews_cache.csv manually or configure AWS credentials.")
        return pd.DataFrame(columns=["review_text", "label", "source", "paper_id"])

    import random
    random.seed(RANDOM_SEED)
    sampled = random.sample(papers, min(n_reviews, len(papers)))

    records = []
    for paper in tqdm(sampled, desc="Generating AI reviews via Bedrock"):
        prompt = (
            f"You are a peer reviewer for a top AI/NLP conference. "
            f"Write a detailed peer review for the following paper.\n\n"
            f"Title: {paper['title']}\n\n"
            f"Abstract: {paper['abstract']}\n\n"
            f"Write a complete peer review including: Summary, Strengths, "
            f"Weaknesses, Questions for Authors, and Overall Recommendation. "
            f"Be specific and constructive."
        )
        try:
            review_text = _invoke_bedrock(client, prompt)
            records.append({
                "review_text": review_text.strip(),
                "label": 1,  # 1 = AI-generated
                "source": f"ai_{BEDROCK_MODEL_ID}",
                "paper_id": paper["paper_id"],
            })
        except Exception as e:
            print(f"Error generating review for {paper['paper_id']}: {e}")
        time.sleep(0.5)

    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    print(f"Generated and cached {len(df)} AI reviews.")
    return df


def build_dataset() -> pd.DataFrame:
    """Full pipeline: clone → load human → generate AI → combine."""
    clone_peerread()
    human_df = load_human_reviews()
    papers = load_paper_metadata()
    ai_df = generate_ai_reviews(papers)

    combined = pd.concat([human_df, ai_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    output_path = DATA_DIR / "dataset.csv"
    combined.to_csv(output_path, index=False)
    print(f"Combined dataset: {len(combined)} reviews ({(combined['label']==0).sum()} human, {(combined['label']==1).sum()} AI)")
    print(f"Saved to {output_path}")
    return combined


if __name__ == "__main__":
    build_dataset()
