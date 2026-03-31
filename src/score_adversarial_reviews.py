"""Merge 1000 adversarial reviews, score them via Bedrock, rebuild dataset and retrain."""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
from tqdm import tqdm

from config import ANTHROPIC_VERSION, AWS_REGION, BEDROCK_MODEL_ID, DATA_DIR, MARKER_NAMES
from feature_extractor import EXTRACTION_PROMPT

MAX_WORKERS = 10
CHECKPOINT_EVERY = 50


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
    # Step 1: Merge adversarial reviews
    bedrock_path = DATA_DIR / "adversarial_bedrock.csv"
    gpt_path = DATA_DIR / "adversarial_gpt.csv"

    dfs = []
    if bedrock_path.exists():
        dfs.append(pd.read_csv(bedrock_path))
        print(f"Loaded {len(dfs[-1])} Bedrock adversarial reviews")
    if gpt_path.exists():
        dfs.append(pd.read_csv(gpt_path))
        print(f"Loaded {len(dfs[-1])} GPT adversarial reviews")

    adversarial = pd.concat(dfs, ignore_index=True)
    print(f"Total adversarial reviews: {len(adversarial)}")
    print(f"Sources: {adversarial['source'].value_counts().to_dict()}")

    # Step 2: Score via Bedrock
    ckpt_path = DATA_DIR / "adversarial_scores_checkpoint.json"
    completed = {}
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            completed = {int(k): v for k, v in json.load(f).items()}
        print(f"Resuming from checkpoint: {len(completed)} already done")

    remaining = [i for i in range(len(adversarial)) if i not in completed]
    print(f"Remaining to score: {len(remaining)}")

    if remaining:
        work = []
        for idx in remaining:
            client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
            work.append((idx, adversarial.iloc[idx]["review_text"], client))

        pbar = tqdm(total=len(work), desc="Scoring adversarial reviews")
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
                        json.dump({str(k): v for k, v in completed.items()}, f)
                    pbar.set_postfix({"saved": len(completed)})

        pbar.close()
        with open(ckpt_path, "w") as f:
            json.dump({str(k): v for k, v in completed.items()}, f)

    # Step 3: Apply scores
    for idx, scores in completed.items():
        idx = int(idx)
        for m in MARKER_NAMES:
            adversarial.at[idx, m] = scores[m]

    # Step 4: Merge with existing human reviews
    existing = pd.read_csv(DATA_DIR / "features_cache.csv")
    human = existing[existing["label"] == 0]
    print(f"\nHuman reviews (existing): {len(human)}")
    print(f"Adversarial AI reviews: {len(adversarial)}")

    # Save adversarial features separately
    adversarial.to_csv(DATA_DIR / "adversarial_features.csv", index=False)

    # Build new combined dataset
    combined = pd.concat([human, adversarial], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    combined.to_csv(DATA_DIR / "dataset_adversarial.csv", index=False)
    combined.to_csv(DATA_DIR / "features_cache_adversarial.csv", index=False)

    print(f"\nCombined dataset: {len(combined)} reviews")
    print(f"  Human: {(combined['label']==0).sum()}")
    print(f"  AI: {(combined['label']==1).sum()}")
    print(f"  Sources: {combined['source'].value_counts().to_dict()}")

    # Show marker stats
    ai = combined[combined["label"] == 1]
    h = combined[combined["label"] == 0]
    print(f"\nMarker comparison (adversarial AI vs human):")
    print(f"{'Marker':30s} {'Human':>12s} {'AI':>12s} {'Overlap':>10s}")
    print("-" * 70)
    for m in MARKER_NAMES:
        hm, hs = h[m].mean(), h[m].std()
        am, as_ = ai[m].mean(), ai[m].std()
        overlap = (hm + hs) > (am - as_) and (am + as_) > (hm - hs)
        print(f"{m:30s} {hm:.2f}±{hs:.2f}    {am:.2f}±{as_:.2f}    {'YES' if overlap else 'NO'}")

    print(f"\nSaved to:")
    print(f"  {DATA_DIR / 'adversarial_features.csv'}")
    print(f"  {DATA_DIR / 'dataset_adversarial.csv'}")
    print(f"  {DATA_DIR / 'features_cache_adversarial.csv'}")


if __name__ == "__main__":
    main()
