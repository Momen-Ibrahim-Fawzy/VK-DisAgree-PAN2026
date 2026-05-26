"""
Classical ML models for AI text detection:
  1. TF-IDF (char + word n-grams) + Logistic Regression
  2. Stylometric features + LightGBM

Both models are trained independently and serve as complementary signals
in the ensemble, providing robustness against obfuscation.
"""
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    TFIDF_CHAR_MAX_FEATURES,
    TFIDF_WORD_MAX_FEATURES,
    TFIDF_CHAR_NGRAM_RANGE,
    TFIDF_WORD_NGRAM_RANGE,
    LR_C,
    LGBM_N_ESTIMATORS,
    LGBM_LEARNING_RATE,
    LGBM_NUM_LEAVES,
    LGBM_MIN_CHILD_SAMPLES,
    LGBM_COLSAMPLE_BYTREE,
    LGBM_SUBSAMPLE,
    TFIDF_PATH,
    LGBM_PATH,
    SEED,
)
from .features import batch_extract_features


# ─── TF-IDF + Logistic Regression ────────────────────────────────────────────

def build_tfidf_pipeline() -> Pipeline:
    """Build the TF-IDF feature extraction + LogisticRegression pipeline."""
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=TFIDF_CHAR_NGRAM_RANGE,
        max_features=TFIDF_CHAR_MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=TFIDF_WORD_NGRAM_RANGE,
        max_features=TFIDF_WORD_MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"\b[a-zA-Z][a-zA-Z']+\b",
    )
    union = FeatureUnion([
        ("char", char_tfidf),
        ("word", word_tfidf),
    ])
    clf = LogisticRegression(
        C=LR_C,
        max_iter=1000,
        solver="saga",
        n_jobs=-1,
        random_state=SEED,
    )
    return Pipeline([("features", union), ("clf", clf)])


def train_tfidf(
    train_texts: list[str],
    train_labels: list[int],
    save_path: Path = TFIDF_PATH,
) -> Pipeline:
    """Train the TF-IDF pipeline and save it."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("Training TF-IDF + LogisticRegression pipeline...")
    pipeline = build_tfidf_pipeline()
    pipeline.fit(train_texts, train_labels)

    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved TF-IDF pipeline to {save_path}")
    return pipeline


def load_tfidf(path: Path = TFIDF_PATH) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Stylometric Features + LightGBM ─────────────────────────────────────────

def build_lgbm_pipeline():
    """Build the LightGBM pipeline for stylometric features."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    clf = lgb.LGBMClassifier(
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LEARNING_RATE,
        num_leaves=LGBM_NUM_LEAVES,
        min_child_samples=LGBM_MIN_CHILD_SAMPLES,
        colsample_bytree=LGBM_COLSAMPLE_BYTREE,
        subsample=LGBM_SUBSAMPLE,
        n_jobs=-1,
        random_state=SEED,
        verbose=-1,
    )
    # Scaler + classifier (LightGBM doesn't strictly need scaling, but helps)
    return {"scaler": StandardScaler(), "clf": clf}


def train_lgbm(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: Optional[list[str]] = None,
    val_labels: Optional[list[int]] = None,
    save_path: Path = LGBM_PATH,
) -> dict:
    """Train LightGBM on stylometric features and save the model."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("Extracting stylometric features (train)...")
    X_train = batch_extract_features(train_texts, verbose=True)
    y_train = np.array(train_labels)

    components = build_lgbm_pipeline()
    scaler = components["scaler"]
    clf = components["clf"]

    X_train_scaled = scaler.fit_transform(X_train)
    np.nan_to_num(X_train_scaled, nan=0.0, posinf=1e6, neginf=-1e6, copy=False)

    if val_texts is not None and val_labels is not None:
        print("Extracting stylometric features (val)...")
        X_val = batch_extract_features(val_texts, verbose=True)
        X_val_scaled = scaler.transform(X_val)
        np.nan_to_num(X_val_scaled, nan=0.0, posinf=1e6, neginf=-1e6, copy=False)
        y_val = np.array(val_labels)

        print("Training LightGBM with early stopping...")
        clf.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_val_scaled, y_val)],
            callbacks=[__import__("lightgbm").early_stopping(50, verbose=False),
                       __import__("lightgbm").log_evaluation(100)],
        )
    else:
        print("Training LightGBM...")
        clf.fit(X_train_scaled, y_train)

    model = {"scaler": scaler, "clf": clf}
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved LightGBM model to {save_path}")
    return model


def load_lgbm(path: Path = LGBM_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Inference helpers ────────────────────────────────────────────────────────

class TFIDFDetector:
    def __init__(self, path: Path = TFIDF_PATH):
        self.pipeline = load_tfidf(path)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return probability of AI-generated class."""
        return self.pipeline.predict_proba(texts)[:, 1]


class LGBMDetector:
    def __init__(self, path: Path = LGBM_PATH):
        model = load_lgbm(path)
        self.scaler = model["scaler"]
        self.clf = model["clf"]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return probability of AI-generated class from stylometric features."""
        X = batch_extract_features(texts, verbose=False)
        X_scaled = self.scaler.transform(X)
        np.nan_to_num(X_scaled, nan=0.0, posinf=1e6, neginf=-1e6, copy=False)
        return self.clf.predict_proba(X_scaled)[:, 1]