"""Score the 5,772 human reviews using Claude via Bedrock (neutral prompt)."""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
from tqdm import tqdm

from config import ANTHROPIC_VERSION, AWS_REGION, BEDROCK_MODEL_ID, DATA_DIR, MARKER_NAMES
from feature_extractor import EXTRACTION_PROMPT

MAX_WORKERS = 10
CHECKPOINT_EVERY = 100


def score_one(args):
    idx, text, client = args
    prompt = EXTRACTION_PROMPT.format(review_text=text[:3000])
    try:
        body = {
            "anthropic_version": ANTHROPIC_VERSION,
            "max_tokens": 400,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = client.invoke_model(modelId=BEDROCK_MODEL_ID, body=json.dumps(body))
        content = json.loads(resp["body"].read())["content"][0]["text"]
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            data = json.loads(match.group())
            scores = {}
            for m in MARKER_NAMES:
                val = data.get(m, 0.0)
                scores[m] = float(val.get("score", 0.0)) if isinstance(val, dict) else float(val)
            return idx, scores
    except Exception as e:
        print(f"Error idx={idx}: {e}")
    return idx, None


def main():
    df = pd.read_csv(DATA_DIR / "features_cache.csv")
    human_indices = df[df["label"] == 0].index.tolist()
    print(f"Human reviews to score: {len(human_indices)}")

    # Check for checkpoint
    ckpt_path = DATA_DIR / "human_scores_checkpoint.json"
    completed = {}
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            completed = {int(k): v for k, v in json.load(f).items()}
        print(f"Resuming from checkpoint: {len(completed)} already done")

    remaining = [i for i in human_indices if i not in completed]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
    else:
        work = []
        for idx in remaining:
            client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
            work.append((idx, df.iloc[idx]["review_text"], client))

        pbar = tqdm(total=len(work), desc="Scoring human reviews via Bedrock")
        newly_done = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(score_one, item): item[0] for item in work}
            for future in as_completed(futures):
                idx, scores = future.result()
                if scores:
                    completed[idx] = scores
                newly_done += 1
                pbar.update(1)

                if newly_done % CHECKPOINT_EVERY == 0:
                    with open(ckpt_path, "w") as f:
                        json.dump(completed, f)
                    pbar.set_postfix({"saved": len(completed)})

        pbar.close()

        # Final checkpoint
        with open(ckpt_path, "w") as f:
            json.dump(completed, f)

    # Apply scores to features_cache
    applied = 0
    for idx_str, scores in completed.items():
        idx = int(idx_str)
        for m in MARKER_NAMES:
            df.at[idx, m] = scores[m]
        applied += 1

    df.to_csv(DATA_DIR / "features_cache.csv", index=False)
    print(f"\nApplied LLM scores to {applied} human reviews")
    print(f"Saved to features_cache.csv")

    # Show stats
    human = df[df["label"] == 0]
    ai = df[df["label"] == 1]
    print(f"\nHuman review marker means (neutral LLM-scored):")
    for m in MARKER_NAMES:
        print(f"  {m:30s}  {human[m].mean():.3f}")
    print(f"\nAI review marker means:")
    for m in MARKER_NAMES:
        print(f"  {m:30s}  {ai[m].mean():.3f}")


if __name__ == "__main__":
    main()
