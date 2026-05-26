"""
Ensemble combiner with:
  - Stacked meta-learner that includes model disagreement as a feature
  - Temperature scaling for Brier Score calibration
  - Disagreement-aware abstention: abstain when models disagree strongly
    OR when the final score is close to 0.5

This replaces the naive |score - 0.5| < θ abstention with a two-dimensional
uncertainty criterion that uses actual prediction variance across models.

Intuition: If ModernBERT says 0.8 (AI) but Binoculars says 0.3 (human),
the models fundamentally disagree — this is a case where outputting 0.5
is more honest than trusting either model.
"""
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
# LightGBM is imported lazily inside train_meta_learner to avoid import errors
# when the saved model is loaded (lightgbm must be installed)

from .config import (
    DEFAULT_ENSEMBLE_WEIGHTS,
    ABSTENTION_THRESHOLD,
    ENSEMBLE_PATH,
    SEED,
)


# ─── PAN evaluation metrics ───────────────────────────────────────────────────

def c_at_1(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    C@1: Accuracy where undecidable predictions (score == 0.5) are rewarded
    at the accuracy rate of the decided subset.
    """
    n = len(labels)
    preds = np.where(scores < 0.5 - 1e-6, 0, np.where(scores > 0.5 + 1e-6, 1, -1))
    decided = preds != -1
    n_decided = int(decided.sum())

    if n_decided == 0:
        return 0.5

    acc_decided = float((preds[decided] == labels[decided]).mean())
    return (1 / n) * (n_decided * acc_decided + (n - n_decided) * acc_decided)


def f05u_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """F0.5u: undecidable (0.5) predictions treated as false negatives."""
    preds = np.where(scores < 0.5 - 1e-6, 0, np.where(scores > 0.5 + 1e-6, 1, -1))
    effective_preds = np.where(preds == -1, 1 - labels, preds)
    tp = int(((effective_preds == 1) & (labels == 1)).sum())
    fp = int(((effective_preds == 1) & (labels == 0)).sum())
    fn = int(((effective_preds == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    beta = 0.5
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def evaluate_all_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Compute all five PAN evaluation metrics."""
    from sklearn.metrics import f1_score
    binary_preds = (scores > 0.5).astype(int)
    auc = roc_auc_score(labels, scores)
    brier = 1.0 - brier_score_loss(labels, scores)
    cat1 = c_at_1(labels, scores)
    f1 = f1_score(labels, binary_preds, zero_division=0)
    f05u = f05u_score(labels, scores)
    return {
        "roc_auc": auc,
        "brier": brier,
        "c@1": cat1,
        "f1": f1,
        "f0.5u": f05u,
        "mean": float(np.mean([auc, brier, cat1, f1, f05u])),
    }


# ─── Ensemble model ───────────────────────────────────────────────────────────

class EnsembleModel:
    """
    Combines predictions from: ModernBERT, TF-IDF, LightGBM, Binoculars.

    Pipeline:
      1. Stack base model probabilities + their variance as features
      2. Meta-learner (LightGBM) trained on validation set
      3. Temperature scaling for Brier Score calibration
      4. Two-dimensional abstention:
           abstain if  (|score - 0.5| < score_thresh)
                    OR (pred_variance > var_thresh)
    """

    def __init__(
        self,
        weights: Optional[list[float]] = None,
        score_threshold: float = ABSTENTION_THRESHOLD,
        var_threshold: float = 0.05,
    ):
        self.weights = weights or DEFAULT_ENSEMBLE_WEIGHTS
        self.score_threshold = score_threshold
        self.var_threshold = var_threshold
        self.meta_learner = None   # LightGBM classifier
        self.temperature: float = 1.0

    # ── Meta-learner (LightGBM) ───────────────────────────────────────────────

    def train_meta_learner(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        """
        Train a LightGBM meta-learner on stacked validation predictions.

        Why LightGBM instead of Logistic Regression:
          - Captures non-linear interactions between signals, e.g.:
            "if Binoculars says AI but ModernBERT says human → trust ModernBERT"
          - Decision-tree splits handle "regime changes" (clean vs obfuscated texts
            require different weighting of signals)
          - LR can only draw linear decision boundaries in the 6D feature space;
            LightGBM can draw arbitrary boundaries

        Input features: [p1, p2, p3, p4, variance, range]
        predictions: shape (n_samples, n_models)
        labels: shape (n_samples,)
        """
        import lightgbm as lgb
        print("Training LightGBM meta-learner (disagreement-aware)...")
        X = self._augment_with_disagreement(predictions)
        self.meta_learner = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,       # Shallow trees → avoids overfitting on small val set
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbose=-1,
        )
        self.meta_learner.fit(X, labels)
        val_probs = self.meta_learner.predict_proba(X)[:, 1]
        print(f"  Meta-learner val AUC: {roc_auc_score(labels, val_probs):.4f}")
        # Log feature importances
        importances = self.meta_learner.feature_importances_
        names = [f"model_{i}" for i in range(predictions.shape[1])] + ["variance", "range"]
        for n, imp in sorted(zip(names, importances), key=lambda x: -x[1]):
            print(f"    {n:12s}: {imp}")

    def _augment_with_disagreement(self, predictions: np.ndarray) -> np.ndarray:
        """Add prediction variance and range as features to the stacked predictions."""
        variance = predictions.var(axis=1, keepdims=True)          # Disagreement
        pred_range = (predictions.max(axis=1) - predictions.min(axis=1)).reshape(-1, 1)
        return np.hstack([predictions, variance, pred_range])

    # ── Temperature calibration ───────────────────────────────────────────────

    def calibrate_temperature(
        self, predictions: np.ndarray, labels: np.ndarray, n_steps: int = 100
    ) -> None:
        """
        Find temperature T that minimizes Brier Score via logit scaling.
        sigmoid(logit(p) / T): T > 1 softens, T < 1 sharpens.
        """
        raw_probs = self._get_meta_probs(predictions)
        best_T = 1.0
        best_brier = brier_score_loss(labels, raw_probs)

        for T in np.linspace(0.3, 4.0, n_steps):
            cal = self._apply_temperature(raw_probs, T)
            b = brier_score_loss(labels, cal)
            if b < best_brier:
                best_brier = b
                best_T = T

        self.temperature = best_T
        print(f"Temperature calibration: T={best_T:.3f} | Brier={best_brier:.4f}")

    @staticmethod
    def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
        logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7) / np.clip(1 - probs, 1e-7, 1 - 1e-7))
        return 1.0 / (1.0 + np.exp(-logits / T))

    # ── Disagreement-aware abstention ─────────────────────────────────────────

    def find_abstention_thresholds(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> None:
        """
        Grid search over (score_threshold, var_threshold) to maximize the
        MEAN of all five PAN metrics — not just C@1.

        Optimizing only C@1 causes over-abstention, which destroys F0.5u
        (undecidable outputs are treated as false negatives in F0.5u).
        Maximizing the mean naturally balances the trade-off:
          - Too much abstention → F0.5u drops
          - Too little abstention → C@1 drops
          - Optimum is where the mean of all five is highest

        Two conditions independently trigger abstention:
          1. |final_score - 0.5| < score_thresh  (borderline score)
          2. prediction_variance > var_thresh     (models strongly disagree)
        """
        calibrated = self._get_calibrated_probs(predictions)
        variance = predictions.var(axis=1)

        # Baseline: no abstention at all
        baseline_metrics = evaluate_all_metrics(labels, calibrated)
        best_mean = baseline_metrics["mean"]
        best_score_thresh = 0.0
        best_var_thresh = 999.0  # Effectively disabled

        score_grid = np.linspace(0.0, 0.20, 25)
        var_grid = np.concatenate([
            np.linspace(0.001, 0.05, 15),
            [999.0]  # Disabled
        ])

        for st in score_grid:
            for vt in var_grid:
                abstained = self._apply_abstention(calibrated, variance, st, vt)
                m = evaluate_all_metrics(labels, abstained)
                if m["mean"] > best_mean:
                    best_mean = m["mean"]
                    best_score_thresh = st
                    best_var_thresh = vt

        self.score_threshold = best_score_thresh
        self.var_threshold = best_var_thresh

        # Report the final balanced metrics
        final = evaluate_all_metrics(
            labels,
            self._apply_abstention(calibrated, variance, best_score_thresh, best_var_thresh)
        )
        print(
            f"Abstention thresholds (optimized for mean score):\n"
            f"  score_thresh={best_score_thresh:.3f}, "
            f"var_thresh={'∞' if best_var_thresh >= 999 else f'{best_var_thresh:.4f}'}\n"
            f"  mean={final['mean']:.4f} | C@1={final['c@1']:.4f} | "
            f"F0.5u={final['f0.5u']:.4f} | Brier={final['brier']:.4f}"
        )

    @staticmethod
    def _apply_abstention(
        scores: np.ndarray, variance: np.ndarray,
        score_thresh: float, var_thresh: float
    ) -> np.ndarray:
        """Return scores with 0.5 substituted where abstention criteria are met."""
        abstain = (np.abs(scores - 0.5) < score_thresh) | (variance > var_thresh)
        return np.where(abstain, 0.5, scores)

    # ── Inference ─────────────────────────────────────────────────────────────

    def _get_meta_probs(self, predictions: np.ndarray) -> np.ndarray:
        if self.meta_learner is not None:
            X = self._augment_with_disagreement(predictions)
            return self.meta_learner.predict_proba(X)[:, 1]
        # Fallback: weighted average (truncate weights to match n_models)
        w = np.array(self.weights[: predictions.shape[1]])
        w = w / w.sum()
        return predictions @ w

    def _get_calibrated_probs(self, predictions: np.ndarray) -> np.ndarray:
        raw = self._get_meta_probs(predictions)
        if self.temperature != 1.0:
            return self._apply_temperature(raw, self.temperature)
        return raw

    def predict(self, predictions: np.ndarray, apply_abstention: bool = True) -> np.ndarray:
        """
        Generate final predictions.

        predictions: shape (n_samples, n_models) — probabilities from each base model
        Returns: scores in [0, 1] where 0.5 = undecidable
        """
        probs = self._get_calibrated_probs(predictions)
        if apply_abstention:
            variance = predictions.var(axis=1)
            probs = self._apply_abstention(
                probs, variance, self.score_threshold, self.var_threshold
            )
        return probs

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, path: Path = ENSEMBLE_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "weights": self.weights,
                "score_threshold": self.score_threshold,
                "var_threshold": self.var_threshold,
                "meta_learner": self.meta_learner,
                "temperature": self.temperature,
            }, f)
        print(f"Saved ensemble config to {path}")

    @classmethod
    def load(cls, path: Path = ENSEMBLE_PATH) -> "EnsembleModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        model = cls(
            weights=state["weights"],
            score_threshold=state.get("score_threshold", state.get("abstention_threshold", 0.1)),
            var_threshold=state.get("var_threshold", 999.0),
        )
        model.meta_learner = state["meta_learner"]
        model.temperature = state["temperature"]
        return model