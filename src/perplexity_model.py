"""
Generation-based detection using the Binoculars cross-model ratio.

**Why cross-model ratio instead of single-model perplexity:**

Single-model perplexity (e.g., Pythia-1B alone) has a critical flaw:
a long, technical passage about astrophysics will have high perplexity under
*any* language model — not because it's human-written, but because the *topic*
is hard to predict. This is called domain difficulty, and it contaminates the signal.

The Binoculars insight (Hans et al., 2024): use TWO models and compute a RATIO.

  binoculars(x) = log_ppl_observer(x) / CE(observer → performer)(x)

Where:
  - log_ppl_observer = mean -log P_obs(token_i | context) under the observer model
  - CE(obs → perf) = mean cross-entropy between observer's distribution and the
    performer model's predicted distribution at each token position

**Why the ratio works:**
  - Topic difficulty increases BOTH log_ppl_obs AND CE equally → ratio stays ~1.0
  - AI text is "easy" for the observer (LM was trained on similar text) → low log_ppl_obs
    But the performer (smaller model) predicts these tokens as "typical" too → low CE
    The key: for AI text, log_ppl_obs drops faster than CE → ratio < 1.0
  - Human text: log_ppl_obs is higher; performer's predictions are less aligned
    with observer → ratio closer to 1.0

This ratio is robust to domain shift and topic difficulty in a way that
single-model perplexity fundamentally cannot be.

**Model choices:**
  - Observer: `EleutherAI/pythia-1b` (1B params, 2022, trained on The Pile)
  - Performer: `EleutherAI/pythia-70m` (70M params, same training distribution)

Using two models from the same training distribution (both Pythia) is deliberate:
they share the same understanding of what "expected" text looks like, so the ratio
cleanly measures AI-ness rather than model capability differences.

References:
  - Hans et al. (2024), "Spotting LLMs with Binoculars: Zero-Shot Detection of
    LLM-Generated Text", ICML 2024
  - Mitchell et al. (2023), "DetectGPT: Zero-Shot Machine-Generated Text Detection"
  - Biderman et al. (2023), "Pythia: A Suite for Analyzing Large Language Models"
"""
import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODELS_DIR, PPL_MODEL_NAME, PPL_PERFORMER_NAME, SEED


PPL_PATH = MODELS_DIR / "perplexity_model.pkl"

# Observer model: large, accurate reference model
# Docker: /opt/pan26_detector/models/pythia-1b  |  Local: auto-download from HuggingFace
OBSERVER_MODEL = PPL_MODEL_NAME        # EleutherAI/pythia-1b
PERFORMER_MODEL = PPL_PERFORMER_NAME   # EleutherAI/pythia-70m

_DEFAULT_THRESHOLD = 0.9      # binoculars scores below this → AI
_DEFAULT_SCALE = 10.0         # sigmoid sharpness


class BinocularsDetector:
    """
    Detects AI-generated text via the cross-model log-perplexity ratio.

    binoculars(x) = log_ppl_observer(x) / CE(observer, performer)(x)

    Lower ratio → text is "expected" by observer, but performer's predictions
    also align well → typical LM output → more likely AI-generated.

    Higher ratio → topic is genuinely hard OR observer and performer disagree on
    what tokens to expect → more likely human.

    The model weights are NEVER updated. Only a calibration threshold and scale
    are learned from the training data labels.
    """

    def __init__(
        self,
        observer_name: str = OBSERVER_MODEL,
        performer_name: str = PERFORMER_MODEL,
        device: Optional[str] = None,
        max_length: int = 512,
        stride: int = 256,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_length = max_length
        self.stride = stride

        # Calibration parameters
        self.threshold = _DEFAULT_THRESHOLD
        self.scale = _DEFAULT_SCALE

        print(f"Loading observer ({observer_name}) on {device}...")
        self._obs_tok = AutoTokenizer.from_pretrained(observer_name)
        if self._obs_tok.pad_token is None:
            self._obs_tok.pad_token = self._obs_tok.eos_token
        self._obs = AutoModelForCausalLM.from_pretrained(observer_name)
        self._obs.to(device)
        self._obs.eval()

        print(f"Loading performer ({performer_name}) on {device}...")
        self._perf_tok = AutoTokenizer.from_pretrained(performer_name)
        if self._perf_tok.pad_token is None:
            self._perf_tok.pad_token = self._perf_tok.eos_token
        self._perf = AutoModelForCausalLM.from_pretrained(performer_name)
        self._perf.to(device)
        self._perf.eval()

    def _tokenize(self, text: str, tokenizer) -> torch.Tensor:
        """Tokenize text and return input_ids tensor."""
        enc = tokenizer(text, return_tensors="pt")
        return enc.input_ids.to(self.device)

    def _sliding_window_logits(self, input_ids: torch.Tensor, model) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get per-token logits using a sliding window to handle long texts.

        Returns:
          token_ids: shape (n_tokens,) — the actual token IDs
          logits_list: list of per-position logit vectors, one per token (excluding first)
        """
        seq_len = input_ids.size(1)
        all_obs_logits = []
        all_token_ids = []
        prev_end = 0

        for begin in range(0, seq_len, self.stride):
            end = min(begin + self.max_length, seq_len)
            target_start = max(begin, prev_end)

            chunk = input_ids[:, begin:end]
            with torch.no_grad():
                out = model(chunk)
                logits = out.logits  # (1, chunk_len, vocab)

            # Collect predictions for tokens after the target_start
            local_start = target_start - begin
            if local_start < logits.size(1) - 1:
                all_obs_logits.append(logits[0, local_start:-1, :])  # (n, vocab)
                all_token_ids.append(chunk[0, local_start + 1:])     # actual tokens

            prev_end = end
            if end == seq_len:
                break

        if not all_obs_logits:
            return torch.tensor([]), torch.tensor([])

        return torch.cat(all_obs_logits, dim=0), torch.cat(all_token_ids, dim=0)

    def _compute_binoculars_score(self, text: str) -> float:
        """
        Compute the binoculars score for a single text.

        Returns a scalar where:
          Low value  → text is AI-generated (observer and performer agree on tokens)
          High value → text is human-written (harder to predict, models disagree)
        """
        # Get logits from both models
        obs_ids = self._tokenize(text, self._obs_tok)
        perf_ids = self._tokenize(text, self._perf_tok)

        obs_logits, obs_tokens = self._sliding_window_logits(obs_ids, self._obs)
        perf_logits, _ = self._sliding_window_logits(perf_ids, self._perf)

        if len(obs_logits) == 0 or len(perf_logits) == 0:
            return 0.9  # Neutral score for very short texts

        # Align lengths (tokenizers may differ slightly)
        n = min(len(obs_logits), len(perf_logits))
        obs_logits = obs_logits[:n]
        perf_logits = perf_logits[:n]
        obs_tokens = obs_tokens[:n]

        # ── log_ppl_observer ─────────────────────────────────────────────────
        # Mean -log P_obs(actual_token | context)
        log_probs_obs = torch.log_softmax(obs_logits, dim=-1)  # (n, vocab)
        actual_log_probs = log_probs_obs.gather(
            1, obs_tokens.unsqueeze(1)
        ).squeeze(1)  # (n,)
        log_ppl_obs = float(-actual_log_probs.mean())

        # ── CE(observer → performer) ─────────────────────────────────────────
        # Mean cross-entropy: -sum_v P_perf(v) * log P_obs(v)
        # = E_{v ~ P_perf}[-log P_obs(v)]
        log_probs_perf = torch.log_softmax(perf_logits, dim=-1)  # (n, vocab)
        probs_perf = torch.exp(log_probs_perf)                   # (n, vocab)

        # Cross-entropy at each position: -sum_v P_perf(v) * log P_obs(v)
        cross_entropy_per_token = -(probs_perf * log_probs_obs).sum(dim=-1)  # (n,)
        mean_ce = float(cross_entropy_per_token.mean())

        if mean_ce < 1e-6:
            return 0.5

        return log_ppl_obs / mean_ce

    def _binoculars_to_ai_prob(self, score: float) -> float:
        """
        Convert a binoculars score to P(AI-generated) in [0, 1].

        Lower binoculars score → higher AI probability.
        Sigmoid: P(AI) = 1 / (1 + exp(scale * (score - threshold)))
        """
        return 1.0 / (1.0 + math.exp(self.scale * (score - self.threshold)))

    def predict_proba_single(self, text: str) -> float:
        score = self._compute_binoculars_score(text)
        return self._binoculars_to_ai_prob(score)

    def predict_proba(self, texts: list[str], verbose: bool = True) -> np.ndarray:
        from tqdm import tqdm
        iterator = tqdm(texts, desc="Binoculars detection") if verbose else texts
        return np.array([self.predict_proba_single(t) for t in iterator])

    def calibrate(self, texts: list[str], labels: list[int]) -> None:
        """
        Learn threshold and scale from labeled data.
        The binoculars score distribution is fit by a 1D logistic regression.
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.linear_model import LogisticRegression

        print("Computing Binoculars scores for calibration...")
        from tqdm import tqdm
        raw_scores = np.array([
            self._compute_binoculars_score(t) for t in tqdm(texts)
        ])
        labels_arr = np.array(labels)

        # Fit logistic regression on the raw scores
        # P(AI) = sigmoid(-(scale * score + intercept))
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(-raw_scores.reshape(-1, 1), labels_arr)

        self.scale = float(abs(lr.coef_[0, 0]))
        self.threshold = float(-lr.intercept_[0] / lr.coef_[0, 0])

        final_probs = np.array([self._binoculars_to_ai_prob(s) for s in raw_scores])
        print(
            f"Binoculars calibration: threshold={self.threshold:.3f}, "
            f"scale={self.scale:.3f}, AUC={roc_auc_score(labels_arr, final_probs):.4f}"
        )

    def save_calibration(self, path: Path = PPL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"threshold": self.threshold, "scale": self.scale}, f)
        print(f"Saved Binoculars calibration to {path}")

    def load_calibration(self, path: Path = PPL_PATH) -> None:
        with open(path, "rb") as f:
            params = pickle.load(f)
        self.threshold = params["threshold"]
        self.scale = params["scale"]
        print(
            f"Loaded Binoculars calibration from {path} "
            f"(threshold={self.threshold:.3f}, scale={self.scale:.3f})"
        )