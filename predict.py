#!/usr/bin/env python3
"""
Inference CLI for the PAN 2026 Voight-Kampff AI Detection system.

Usage:
    predict.py INPUT_FILE OUTPUT_DIRECTORY

Runs four detectors in sequence and combines them with the trained ensemble:
  1. ModernBERT-base (fine-tuned, GPU if available)
  2. TF-IDF n-gram + LogisticRegression
  3. Stylometric features + LightGBM
  4. Binoculars cross-model ratio (Pythia-1B / Pythia-70M)

Output: OUTPUT_DIRECTORY/predictions.jsonl
  {"id": "...", "label": <float 0.0-1.0>}
  score > 0.5 → AI-generated
  score < 0.5 → human-written
  score = 0.5 → undecidable (uncertain)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    if len(sys.argv) < 3:
        print("Usage: predict.py INPUT_FILE OUTPUT_DIRECTORY", file=sys.stderr)
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_file}")
    print(f"Output: {output_dir}")

    from src.data import load_jsonl, save_jsonl
    from src.config import (
        DEBERTA_DIR, TFIDF_PATH, LGBM_PATH, PPL_CALIBRATION_PATH, ENSEMBLE_PATH
    )
    import numpy as np

    print("Loading input data...")
    records = load_jsonl(input_file)
    texts = [r["text"] for r in records]
    ids = [r["id"] for r in records]
    print(f"  {len(texts)} texts to process")

    # ── Component 1: ModernBERT ───────────────────────────────────────────
    print("\nLoading ModernBERT model...")
    from src.transformer_model import TransformerDetector
    deberta = TransformerDetector()
    deberta_probs = deberta.predict_proba(texts)
    _log_stats("ModernBERT", deberta_probs)
    del deberta  # Free GPU memory

    # ── Component 2: TF-IDF ───────────────────────────────────────────────
    print("\nLoading TF-IDF model...")
    from src.classical_models import TFIDFDetector
    tfidf_probs = TFIDFDetector().predict_proba(texts)
    _log_stats("TF-IDF", tfidf_probs)

    # ── Component 3: LightGBM ─────────────────────────────────────────────
    print("\nLoading LightGBM model...")
    from src.classical_models import LGBMDetector
    lgbm_probs = LGBMDetector().predict_proba(texts)
    _log_stats("LightGBM", lgbm_probs)

    # ── Component 4: Binoculars (cross-model ratio) ───────────────────────
    ppl_path = PPL_CALIBRATION_PATH
    if ppl_path.exists():
        print("\nLoading Binoculars detector (Pythia-1B observer / Pythia-70M performer)...")
        from src.perplexity_model import BinocularsDetector
        bino = BinocularsDetector()
        bino.load_calibration(ppl_path)
        ppl_probs = bino.predict_proba(texts, verbose=True)
        _log_stats("Binoculars", ppl_probs)
        del bino
        predictions = np.column_stack([deberta_probs, tfidf_probs, lgbm_probs, ppl_probs])
    else:
        print("\nBinoculars calibration not found — using 3-model ensemble")
        predictions = np.column_stack([deberta_probs, tfidf_probs, lgbm_probs])

    # ── Ensemble ──────────────────────────────────────────────────────────
    print("\nApplying ensemble with disagreement-aware abstention...")
    from src.ensemble import EnsembleModel
    ensemble = EnsembleModel.load(ENSEMBLE_PATH)
    final_scores = ensemble.predict(predictions, apply_abstention=True)

    # ── Write output ──────────────────────────────────────────────────────
    output_records = [
        {"id": id_, "label": float(score)}
        for id_, score in zip(ids, final_scores)
    ]
    output_file = output_dir / "predictions.jsonl"
    save_jsonl(output_records, output_file)

    n_human = int((final_scores < 0.5).sum())
    n_ai = int((final_scores > 0.5).sum())
    n_undecided = int((final_scores == 0.5).sum())
    print(f"\nPredictions written to {output_file}")
    print(f"  Human (< 0.5):     {n_human}")
    print(f"  AI    (> 0.5):     {n_ai}")
    print(f"  Undecided (= 0.5): {n_undecided}")


def _log_stats(name: str, probs: "np.ndarray") -> None:
    import numpy as np
    print(
        f"  {name:12s}: min={probs.min():.3f} "
        f"max={probs.max():.3f} mean={probs.mean():.3f}"
    )


if __name__ == "__main__":
    main()