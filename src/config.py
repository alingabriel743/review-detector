"""Configuration for the AI-generated peer review detection framework."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# PeerRead dataset
PEERREAD_DIR = DATA_DIR / "PeerRead"
PEERREAD_REVIEWS_DIRS = [
    PEERREAD_DIR / "data" / "acl_2017",
    PEERREAD_DIR / "data" / "conll_2016",
    PEERREAD_DIR / "data" / "iclr_2017",
]

# Amazon Bedrock (used for AI review generation and LLM feature extraction)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
ANTHROPIC_VERSION = "bedrock-2023-05-31"

# OpenRouter (alternative LLM provider for deployment without Bedrock)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.6-plus-preview:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Marker categories and names (from the paper's Table 1)
MARKER_NAMES = [
    "standardized_structure",       # Structural
    "predictable_criticism",        # Argumentative
    "excessive_balance",            # Argumentative
    "linguistic_homogeneity",       # Linguistic
    "generic_domain_language",      # Linguistic
    "conceptual_feedback",          # Behavioral
    "absence_personal_signals",     # Behavioral
    "repetition_patterns",          # Behavioral
]

MARKER_CATEGORIES = {
    "structural": ["standardized_structure"],
    "argumentative": ["predictable_criticism", "excessive_balance"],
    "linguistic": ["linguistic_homogeneity", "generic_domain_language"],
    "behavioral": ["conceptual_feedback", "absence_personal_signals", "repetition_patterns"],
}

# Classifier
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"
TEST_SPLIT = 0.2
RANDOM_SEED = 42

# RAG
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"
RAG_TOP_K = 5

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
