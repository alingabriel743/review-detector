"""Generate 500 adversarial AI reviews via Bedrock (Claude) that mimic human writing style."""

import json
import random
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import pandas as pd
from tqdm import tqdm

from config import ANTHROPIC_VERSION, AWS_REGION, BEDROCK_MODEL_ID, DATA_DIR, RANDOM_SEED

MAX_WORKERS = 10
CHECKPOINT_EVERY = 50
N_REVIEWS = 500

ADVERSARIAL_PROMPT = """You are a real human peer reviewer at a top ML conference. You have been reviewing papers for 10+ years. Write a genuine, natural peer review for this paper.

IMPORTANT STYLE REQUIREMENTS — your review must feel authentically human:
- Do NOT use markdown headers like "### Summary" or "### Strengths". Write in flowing paragraphs or use simple formatting.
- Include personal voice: "I think", "I found this confusing", "after reading this twice", "in my experience"
- Reference specific parts: mention page numbers, figure numbers, equation numbers, line numbers, table numbers
- Be inconsistent in tone — mix formal and informal language naturally
- Show genuine uncertainty: "I might be wrong but...", "I'm not entirely sure about..."
- Vary your sentence length — mix short punchy sentences with longer ones
- Include minor imperfections: slight tangents, self-corrections, strong opinions
- Do NOT use balanced "Strengths/Weaknesses" lists. Real reviewers often focus more on one side
- Occasionally be blunt or even a bit harsh — real reviewers sometimes are
- Reference your own expertise or related work you've read

Title: {title}

Abstract: {abstract}

Write your review now. Make it 200-500 words, as a real busy reviewer would."""


def load_papers():
    db = sqlite3.connect(str(DATA_DIR / "gen_review_data" / "gen_review.db"))
    papers = pd.read_sql(
        "SELECT id as paper_id, title, abstract FROM SUBMISSION WHERE length(abstract) > 100 ORDER BY RANDOM() LIMIT 600",
        db,
    )
    db.close()
    return papers.to_dict("records")


def generate_one(args):
    idx, paper, client = args
    prompt = ADVERSARIAL_PROMPT.format(title=paper["title"], abstract=paper["abstract"][:2000])
    try:
        body = {
            "anthropic_version": ANTHROPIC_VERSION,
            "max_tokens": 1000,
            "temperature": 0.9,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = client.invoke_model(modelId=BEDROCK_MODEL_ID, body=json.dumps(body))
        content = json.loads(resp["body"].read())["content"][0]["text"].strip()
        return idx, {
            "review_text": content,
            "label": 1,
            "source": "adversarial_bedrock_claude",
            "paper_id": paper["paper_id"],
        }
    except Exception as e:
        print(f"Error idx={idx}: {e}")
        return idx, None


def main():
    ckpt_path = DATA_DIR / "adversarial_bedrock_checkpoint.json"
    completed = {}
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            completed = json.load(f)
        print(f"Resuming from checkpoint: {len(completed)} already done")

    remaining = N_REVIEWS - len(completed)
    if remaining <= 0:
        print("All 500 done!")
    else:
        print(f"Generating {remaining} adversarial reviews via Bedrock ({BEDROCK_MODEL_ID})...")
        papers = load_papers()
        random.seed(RANDOM_SEED)
        random.shuffle(papers)

        work = []
        start_idx = len(completed)
        for i in range(remaining):
            client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
            work.append((start_idx + i, papers[i % len(papers)], client))

        pbar = tqdm(total=len(work), desc="Adversarial (Bedrock)")
        newly_done = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(generate_one, item): item[0] for item in work}
            for future in as_completed(futures):
                idx, result = future.result()
                if result:
                    completed[str(idx)] = result
                newly_done += 1
                pbar.update(1)

                if newly_done % CHECKPOINT_EVERY == 0:
                    with open(ckpt_path, "w") as f:
                        json.dump(completed, f)
                    pbar.set_postfix({"saved": len(completed)})

        pbar.close()
        with open(ckpt_path, "w") as f:
            json.dump(completed, f)

    # Save as CSV
    records = list(completed.values())
    df = pd.DataFrame(records)
    out_path = DATA_DIR / "adversarial_bedrock.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} adversarial reviews to {out_path}")
    print(f"Sample: {df.iloc[0]['review_text'][:200]}...")


if __name__ == "__main__":
    main()
