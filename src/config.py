"""
Configuration for the PAN 2026 VK-Detect AI detection system.

Data statistics (from analysis of pan26-generative-ai-detection-task1-train):
  Train: 23,707 examples (after dedup: 23,682) | Val: 3,589 examples
  Label split: 38.4% human (label=0) / 61.6% AI (label=1)
  Genres: fiction 62% (50/50 split), essays 20% (80% AI), news 18% (80% AI)
  Text length (words): min=33, mean=621, median=635, p90=816, max=4892
  Token budget (1 word ≈ 1.3 tokens):
    MAX_LENGTH=512  → truncates 89.2% of texts  (too aggressive)
    MAX_LENGTH=1024 → truncates 12.9%           ✓ sweet spot

Hardest models to detect (highest human-likeness on validation set):
  - llama-3.1-8b-instruct      borderline=15.8%, human-like=0.753
  - gemini-1.5-pro             borderline= 9.0%, human-like=0.737
  - mistral-7b-instruct-v0.2   borderline= 2.3%, human-like=0.752
  - ministral-8b-instruct-2410 human-like=0.741
  - qwen1.5-72b-chat-8bit      human-like=0.735
These models are oversampled 1.5× in prepare_data.py before training.
"""
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
DEBERTA_DIR = MODELS_DIR / "deberta"          # Stores ModernBERT checkpoint
TFIDF_PATH = MODELS_DIR / "tfidf_pipeline.pkl"
LGBM_PATH = MODELS_DIR / "lgbm_model.pkl"
PPL_CALIBRATION_PATH = MODELS_DIR / "perplexity_model.pkl"
ENSEMBLE_PATH = MODELS_DIR / "ensemble_config.pkl"

# Prepared (deduped + oversampled) training file — output of prepare_data.py.
PREPARED_TRAIN_PATH = ROOT_DIR / "data" / "train_prepared.jsonl"

# ─── Transformer hyperparameters ─────────────────────────────────────────────
# ModernBERT-base (Warner et al., 2024): post-2024 training data, 8192-token
# context window, Flash Attention 2.
TRANSFORMER_MODEL_NAME = "answerdotai/ModernBERT-base"

# 1024 tokens ≈ 787 words → covers 87.1% of training texts without truncation.
MAX_LENGTH = 1024

# Effective batch = TRAIN_BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS = 32
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 8             # Upper bound; EarlyStoppingCallback(patience=2) stops earlier
WARMUP_RATIO = 0.06        # ~6% of total steps
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05     # Prevents overconfidence; improves Brier Score
GRADIENT_ACCUMULATION_STEPS = 2

# BF16 is stable on A100/H100. Set BF16=False, FP16=True for V100/T4/3090.
FP16 = False
BF16 = True

# ─── Data preparation ─────────────────────────────────────────────────────────
HARD_MODELS = [
    "llama-3.1-8b-instruct",
    "gemini-1.5-pro",
    "mistral-7b-instruct-v0.2",
    "ministral-8b-instruct-2410",
    "qwen1.5-72b-chat-8bit",
]
HARD_MODEL_OVERSAMPLE_FACTOR = 1.5

# ─── Data augmentation ────────────────────────────────────────────────────────
AUG_FRACTION = 0.5        # Fraction of examples augmented by Tier 1 EDA
AUG_N_OPS = 2             # Number of EDA operations per augmented text

# ─── Classical model hyperparameters ─────────────────────────────────────────
# TF-IDF: char n-grams (2-6) + word n-grams (1-3), 200K features total.
TFIDF_CHAR_MAX_FEATURES = 100_000
TFIDF_WORD_MAX_FEATURES = 100_000
TFIDF_CHAR_NGRAM_RANGE = (2, 6)
TFIDF_WORD_NGRAM_RANGE = (1, 3)
LR_C = 1.0

# LightGBM for stylometric features (54 features including SBERT coherence).
LGBM_N_ESTIMATORS = 500
LGBM_LEARNING_RATE = 0.05
LGBM_NUM_LEAVES = 63
LGBM_MIN_CHILD_SAMPLES = 20
LGBM_COLSAMPLE_BYTREE = 0.8
LGBM_SUBSAMPLE = 0.8

# ─── Perplexity model (Binoculars) ────────────────────────────────────────────
# Observer: EleutherAI/pythia-1b — downloaded to models/pythia-1b/
# Performer: EleutherAI/pythia-70m — downloaded to models/pythia-70m/
# The /opt/pan26_detector paths below are the Docker runtime paths.
# For local training, point these to where you downloaded the Pythia weights.
PPL_MODEL_NAME = "/opt/pan26_detector/models/pythia-1b"
PPL_PERFORMER_NAME = "/opt/pan26_detector/models/pythia-70m"
PPL_MAX_LENGTH = 512
PPL_STRIDE = 256
PPL_CALIBRATION_FRACTION = 0.25

# ─── Ensemble ────────────────────────────────────────────────────────────────
DEFAULT_ENSEMBLE_WEIGHTS = [0.55, 0.18, 0.12, 0.15]  # [ModernBERT, TF-IDF, LGBM, Binoculars]

# Initial abstention thresholds — refined by grid search on validation set.
# Grid search maximizes mean of all 5 PAN metrics (not just C@1).
ABSTENTION_THRESHOLD = 0.10   # |score - 0.5| < threshold → output 0.5
VAR_THRESHOLD = 0.05          # prediction_variance > threshold → output 0.5

SEED = 42