"""
LLM Feature Extractor — Uses Claude via Amazon Bedrock to score 8 AI-generation markers.

Markers (from the paper's taxonomy):
  Structural:
    1. standardized_structure — rigid template (Summary/Strengths/Weaknesses)
  Argumentative:
    2. predictable_criticism — generic methodological critiques
    3. excessive_balance — overly diplomatic, balanced tone
  Linguistic:
    4. linguistic_homogeneity — uniform grammar, sentence length, tone
    5. generic_domain_language — broad academic phrases, no deep technical detail
  Behavioral:
    6. conceptual_feedback — high-level feedback, no line/page references
    7. absence_personal_signals — no personal reading signals ("I may be wrong…")
    8. repetition_patterns — templated / repeated phrasing patterns

Each marker is scored 0.0–1.0 by the LLM, plus a brief justification.
A rule-based fallback is provided when Bedrock is unavailable.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    ANTHROPIC_VERSION,
    AWS_REGION,
    BEDROCK_MODEL_ID,
    DATA_DIR,
    MARKER_NAMES,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)


# ── Prompt template for LLM-based extraction ────────────────────────────────

EXTRACTION_PROMPT = """You are a linguistic analyst evaluating the writing characteristics of an academic peer review.

Score each of the following 8 textual properties from 0.0 (not present at all) to 1.0 (very strongly present). Be precise and use the full range of scores.

PROPERTIES:
1. standardized_structure: How rigidly does the text follow a templated structure with clearly labeled sections (e.g., Summary, Strengths, Weaknesses)?
2. predictable_criticism: How much does the text rely on common, formulaic critique phrases (e.g., "needs ablation study", "stronger baselines") rather than paper-specific criticism?
3. excessive_balance: How diplomatically balanced is the tone? Does it systematically pair criticism with positive framing?
4. linguistic_homogeneity: How uniform are the grammar, sentence length, and tone throughout the text?
5. generic_domain_language: How much does the text use broad academic phrases (e.g., "novel approach", "significant contribution") rather than precise technical language?
6. conceptual_feedback: How much does the feedback stay at a high/conceptual level without referencing specific lines, pages, figures, or tables?
7. absence_personal_signals: How absent are personal voice markers (e.g., "I think", "I found", "in my experience", expressions of uncertainty)?
8. repetition_patterns: How much repetitive or templated phrasing appears across sections?

PEER REVIEW TEXT:
\"\"\"
{review_text}
\"\"\"

Respond ONLY with valid JSON containing the 8 scores (no justifications):
{{
  "standardized_structure": 0.0,
  "predictable_criticism": 0.0,
  "excessive_balance": 0.0,
  "linguistic_homogeneity": 0.0,
  "generic_domain_language": 0.0,
  "conceptual_feedback": 0.0,
  "absence_personal_signals": 0.0,
  "repetition_patterns": 0.0
}}
"""


def extract_markers_llm(review_text: str, client) -> dict:
    """Use Claude via Bedrock to extract marker scores from a single review."""
    prompt = EXTRACTION_PROMPT.format(review_text=review_text[:3000])
    try:
        request_body = {
            "anthropic_version": ANTHROPIC_VERSION,
            "max_tokens": 800,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(request_body),
        )
        response_body = json.loads(response["body"].read())
        content = response_body["content"][0]["text"].strip()

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            data = json.loads(json_match.group())
            scores = {}
            for marker in MARKER_NAMES:
                val = data.get(marker, 0.0)
                if isinstance(val, dict):
                    scores[marker] = float(val.get("score", 0.0))
                else:
                    scores[marker] = float(val)
            return scores
    except Exception as e:
        print(f"LLM extraction error: {e}")
    return {m: 0.0 for m in MARKER_NAMES}


def extract_markers_openrouter(review_text: str, api_key: str = None, model: str = None) -> dict:
    """Use an open-source LLM via OpenRouter to extract marker scores."""
    import requests

    api_key = api_key or OPENROUTER_API_KEY
    model = model or OPENROUTER_MODEL
    prompt = EXTRACTION_PROMPT.format(review_text=review_text[:3000])

    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://ai-review-detector.app",
                "X-Title": "AI Review Detector",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.0,
            },
            timeout=60,
        )
        if response.status_code != 200:
            error_detail = response.text[:200]
            print(f"OpenRouter API error {response.status_code}: {error_detail}")
            return {m: 0.0 for m in MARKER_NAMES}
        content = response.json()["choices"][0]["message"]["content"].strip()

        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            data = json.loads(json_match.group())
            scores = {}
            for marker in MARKER_NAMES:
                val = data.get(marker, 0.0)
                if isinstance(val, dict):
                    scores[marker] = float(val.get("score", 0.0))
                else:
                    scores[marker] = float(val)
            return scores
    except Exception as e:
        print(f"OpenRouter extraction error: {e}")
    return {m: 0.0 for m in MARKER_NAMES}


# ── Rule-based fallback extractor ────────────────────────────────────────────

SECTION_HEADERS = re.compile(
    r"\b(summary|strengths?|weaknesses?|suggestions?|recommendation|overall|"
    r"major\s+comments?|minor\s+comments?|questions?)\b",
    re.IGNORECASE,
)

GENERIC_CRITICISM_PHRASES = [
    "ablation stud", "stronger baseline", "additional dataset",
    "robustness analysis", "sensitivity analysis", "evaluation protocol",
    "the paper would benefit", "more comprehensive evaluation",
    "further validation", "scalability", "generalizability",
]

DIPLOMATIC_PHRASES = [
    "the paper addresses an important", "promising but",
    "interesting contribution", "well-written overall",
    "the authors are encouraged", "could be strengthened",
    "would benefit from", "minor concerns",
]

GENERIC_ACADEMIC_PHRASES = [
    "end-to-end pipeline", "integration of multiple components",
    "practical motivation", "important application domain",
    "state-of-the-art", "benchmark dataset", "comprehensive framework",
    "novel approach", "significant contribution",
]

PERSONAL_SIGNALS = [
    "i may be", "i might be", "i struggled", "after re-reading",
    "after multiple readings", "i'm not sure", "i am not sure",
    "i could be wrong", "my understanding", "in my experience",
    "i found it", "i think", "i believe", "i noticed",
]

LINE_REF_PATTERN = re.compile(
    r"(line\s+\d|page\s+\d|figure\s+\d|fig\.\s*\d|table\s+\d|equation\s+\d|eq\.\s*\d|section\s+\d)",
    re.IGNORECASE,
)


def _count_matches(text: str, phrases: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for p in phrases if p.lower() in text_lower)


def _sentence_lengths(text: str) -> list[int]:
    sentences = re.split(r"[.!?]+", text)
    return [len(s.split()) for s in sentences if len(s.split()) > 2]


def extract_markers_rulebased(review_text: str) -> dict:
    """Rule-based heuristic marker extraction (no API needed)."""
    text = review_text.strip()
    text_lower = text.lower()

    # 1. Standardized structure: count section headers
    header_count = len(SECTION_HEADERS.findall(text))
    standardized = min(header_count / 5.0, 1.0)

    # 2. Predictable criticism
    crit_count = _count_matches(text, GENERIC_CRITICISM_PHRASES)
    predictable = min(crit_count / 4.0, 1.0)

    # 3. Excessive balance / diplomatic tone
    dipl_count = _count_matches(text, DIPLOMATIC_PHRASES)
    balance = min(dipl_count / 3.0, 1.0)

    # 4. Linguistic homogeneity: low std deviation in sentence lengths
    sent_lens = _sentence_lengths(text)
    if len(sent_lens) > 3:
        cv = np.std(sent_lens) / (np.mean(sent_lens) + 1e-9)
        homogeneity = max(0.0, 1.0 - cv)  # lower variation → higher score
    else:
        homogeneity = 0.5

    # 5. Generic domain language
    gen_count = _count_matches(text, GENERIC_ACADEMIC_PHRASES)
    generic_lang = min(gen_count / 4.0, 1.0)

    # 6. Conceptual feedback: absence of line/page references
    ref_count = len(LINE_REF_PATTERN.findall(text))
    conceptual = 1.0 if ref_count == 0 else max(0.0, 1.0 - ref_count / 5.0)

    # 7. Absence of personal signals
    personal_count = _count_matches(text, PERSONAL_SIGNALS)
    absence_personal = 1.0 if personal_count == 0 else max(0.0, 1.0 - personal_count / 3.0)

    # 8. Repetition patterns: check for repeated n-grams
    words = text_lower.split()
    trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    if trigrams:
        unique_ratio = len(set(trigrams)) / len(trigrams)
        repetition = max(0.0, 1.0 - unique_ratio) * 3  # amplify
        repetition = min(repetition, 1.0)
    else:
        repetition = 0.0

    return {
        "standardized_structure": round(standardized, 3),
        "predictable_criticism": round(predictable, 3),
        "excessive_balance": round(balance, 3),
        "linguistic_homogeneity": round(homogeneity, 3),
        "generic_domain_language": round(generic_lang, 3),
        "conceptual_feedback": round(conceptual, 3),
        "absence_personal_signals": round(absence_personal, 3),
        "repetition_patterns": round(repetition, 3),
    }


# ── Main extraction pipeline ────────────────────────────────────────────────

MAX_WORKERS = 10  # parallel Bedrock calls
CHECKPOINT_EVERY = 50  # save progress every N reviews


def _extract_one(args):
    """Worker function for parallel extraction."""
    idx, text, client = args
    if client:
        scores = extract_markers_llm(text, client)
    else:
        scores = extract_markers_rulebased(text)
    return idx, scores


def extract_features(df: pd.DataFrame, use_llm: bool = True) -> pd.DataFrame:
    """Extract marker features for every review in the dataset.

    Uses parallel Bedrock calls (ThreadPoolExecutor) for speed.
    Checkpoints progress every 50 reviews so work isn't lost on interruption.

    Args:
        df: DataFrame with a 'review_text' column.
        use_llm: If True, use Claude via Bedrock for extraction.
                 Otherwise, fall back to rule-based extraction.

    Returns:
        DataFrame with 8 additional marker score columns.
    """
    cache_path = DATA_DIR / "features_cache.csv"
    checkpoint_path = DATA_DIR / "features_checkpoint.csv"

    # Check for completed cache
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if len(cached) == len(df) and all(m in cached.columns for m in MARKER_NAMES):
            print(f"Loaded cached features from {cache_path}")
            for m in MARKER_NAMES:
                df[m] = cached[m].values
            return df

    # Resume from checkpoint if available
    completed = {}
    if checkpoint_path.exists():
        ckpt = pd.read_csv(checkpoint_path)
        for _, row in ckpt.iterrows():
            completed[int(row["_idx"])] = {m: row[m] for m in MARKER_NAMES}
        print(f"Resuming from checkpoint: {len(completed)}/{len(df)} already done.")

    # Set up client
    use_bedrock = False
    if use_llm:
        try:
            # Test the client first
            test_client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
            use_bedrock = True
            print(f"Using LLM-based feature extraction via Bedrock ({BEDROCK_MODEL_ID}).")
            print(f"Parallel workers: {MAX_WORKERS}")
        except Exception as e:
            print(f"Could not create Bedrock client: {e}")

    if not use_bedrock:
        print("Using rule-based feature extraction.")

    # Build work items (skip already-completed)
    work_items = []
    for idx in range(len(df)):
        if idx in completed:
            continue
        text = df.iloc[idx]["review_text"]
        # Each thread gets its own boto3 client (thread-safe)
        client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION) if use_bedrock else None
        work_items.append((idx, text, client))

    print(f"Remaining reviews to process: {len(work_items)}")

    # Parallel extraction
    newly_done = 0
    pbar = tqdm(total=len(work_items), desc="Extracting markers")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS if use_bedrock else 4) as executor:
        futures = {executor.submit(_extract_one, item): item[0] for item in work_items}

        for future in as_completed(futures):
            idx, scores = future.result()
            completed[idx] = scores
            newly_done += 1
            pbar.update(1)

            # Checkpoint periodically
            if newly_done % CHECKPOINT_EVERY == 0:
                _save_checkpoint(completed, checkpoint_path)
                pbar.set_postfix({"saved": len(completed)})

    pbar.close()

    # Final save
    _save_checkpoint(completed, checkpoint_path)

    # Apply scores to dataframe
    for idx in range(len(df)):
        for m in MARKER_NAMES:
            df.at[idx, m] = completed[idx][m]

    # Save final cache and clean up checkpoint
    df.to_csv(cache_path, index=False)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Features extracted and cached to {cache_path}")
    return df


def _save_checkpoint(completed: dict, path):
    """Save completed scores to a checkpoint CSV."""
    rows = []
    for idx, scores in completed.items():
        row = {"_idx": idx}
        row.update(scores)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)



if __name__ == "__main__":
    dataset_path = DATA_DIR / "dataset.csv"
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        df = extract_features(df)
        print(df[MARKER_NAMES].describe())
    else:
        print("Run data_loader.py first to build the dataset.")
