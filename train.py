#!/usr/bin/env python3
"""
Training script for the PAN 2026 Voight-Kampff AI Detection system.

Models trained (in order):
  1. ModernBERT-base  — fine-tuned with two-tier adversarial augmentation
  2. TF-IDF (char+word n-grams) + LogisticRegression
  3. Stylometric features (54 features incl. SBERT coherence) + LightGBM
  4. Binoculars cross-model detector (Pythia-1B observer / Pythia-70M performer)

Then:
  5. Stack all 4 base model predictions on validation set
  6. Train LightGBM meta-learner with disagreement features (variance, range)
  7. Find optimal temperature for Brier Score calibration
  8. Grid-search over (score_threshold, var_threshold) for mean score maximization

Usage:
    # Download PAN 2026 data and place it under data/
    python train.py --train data/train.jsonl --val data/val.jsonl

    # With extra data (after running extra_data collection scripts):
    python prepare_data.py --train data/train.jsonl --extra extra_data/merged/extra_train.jsonl
    python train.py --val data/val.jsonl

    # Skip slow components (for quick testing):
    python train.py --skip-perplexity --skip-transformer
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DEBERTA_DIR, TFIDF_PATH, LGBM_PATH, PPL_CALIBRATION_PATH,
    ENSEMBLE_PATH, SEED, PPL_CALIBRATION_FRACTION, PREPARED_TRAIN_PATH,
)
from src.data import load_jsonl, get_texts_and_labels
from src.ensemble import EnsembleModel, evaluate_all_metrics
from src.training_logger import TrainingLogger


def parse_args():
    p = argparse.ArgumentParser(description="Train PAN 2026 AI detection models")
    p.add_argument(
        "--train", type=Path,
        default=PREPARED_TRAIN_PATH,
        help="Path to training JSONL. Defaults to data/train_prepared.jsonl "
             "(output of prepare_data.py). Falls back to data/train.jsonl if not found.",
    )
    p.add_argument(
        "--val", type=Path,
        default=Path("data/val.jsonl"),
        help="Path to validation JSONL (default: data/val.jsonl)",
    )
    p.add_argument("--skip-transformer", action="store_true")
    p.add_argument("--skip-tfidf", action="store_true")
    p.add_argument("--skip-lgbm", action="store_true")
    p.add_argument("--skip-perplexity", action="store_true",
                   help="Skip Binoculars calibration (saves time; lower quality)")
    p.add_argument("--no-augmentation", action="store_true",
                   help="Disable adversarial augmentation for ModernBERT training")
    p.add_argument("--no-t5-augmentation", action="store_true",
                   help="Disable T5 paraphrase augmentation (Tier 2)")
    p.add_argument("--no-gpu", action="store_true")
    return p.parse_args()


def _auc(labels, probs):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, probs)


def main():
    args = parse_args()
    np.random.seed(SEED)

    logger = TrainingLogger()

    # ── Load data ──────────────────────────────────────────────────────────────
    train_path = args.train
    if not train_path.exists():
        raw_fallback = Path("data/train.jsonl")
        if train_path == PREPARED_TRAIN_PATH and raw_fallback.exists():
            print(
                f"[WARNING] Prepared train file not found: {train_path}\n"
                f"  Run prepare_data.py first for best results.\n"
                f"  Falling back to raw: {raw_fallback}"
            )
            train_path = raw_fallback
        else:
            print(f"[ERROR] Training file not found: {train_path}", file=sys.stderr)
            print("  Place the PAN 2026 training data at data/train.jsonl", file=sys.stderr)
            sys.exit(1)

    print(f"Loading training data from {train_path}")
    train_records = load_jsonl(train_path)
    train_texts, train_labels = get_texts_and_labels(train_records)
    print(f"  Train: {len(train_texts)} examples")

    print(f"Loading validation data from {args.val}")
    val_records = load_jsonl(args.val)
    val_texts, val_labels = get_texts_and_labels(val_records)
    print(f"  Val:   {len(val_texts)} examples")
    val_labels_arr = np.array(val_labels)

    # ── STEP 1: Fine-tune ModernBERT ───────────────────────────────────────────
    aug_desc = ""
    if not args.no_augmentation:
        aug_desc = " [Tier1 EDA" + ("" if args.no_t5_augmentation else " + Tier2 T5") + "]"
    header(f"STEP 1: Fine-tuning ModernBERT-base{aug_desc}", skip=args.skip_transformer)
    if args.skip_transformer:
        logger.skip_step("ModernBERT")
    else:
        logger.start_step("ModernBERT")
        # Optional Tier 2: T5 paraphrase augmentation (run before transformer training)
        t5_texts, t5_labels = train_texts, train_labels
        if not args.no_augmentation and not args.no_t5_augmentation:
            header("  STEP 1a: Tier 2 T5 paraphrase augmentation")
            from src.augmentation import augment_dataset_with_paraphrase
            t5_texts, t5_labels = augment_dataset_with_paraphrase(
                train_texts, train_labels, paraphrase_fraction=0.25, seed=SEED
            )

        from src.transformer_model import train as train_transformer
        from src.training_logger import ModernBERTLogCallback
        train_transformer(
            train_texts=t5_texts,
            train_labels=t5_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            use_augmentation=not args.no_augmentation,
            log_callback=ModernBERTLogCallback(logger)(),
        )
        logger.finish_step("ModernBERT")

    # ── STEP 2: TF-IDF + LogisticRegression ────────────────────────────────────
    header("STEP 2: Training TF-IDF + LogisticRegression", skip=args.skip_tfidf)
    if args.skip_tfidf:
        logger.skip_step("TF-IDF")
    else:
        logger.start_step("TF-IDF")
        from src.classical_models import train_tfidf
        train_tfidf(train_texts, train_labels)
        logger.finish_step("TF-IDF")

    # ── STEP 3: LightGBM on stylometric features ───────────────────────────────
    header("STEP 3: Training LightGBM on stylometric features", skip=args.skip_lgbm)
    if args.skip_lgbm:
        logger.skip_step("LightGBM stylometric")
    else:
        logger.start_step("LightGBM stylometric")
        from src.classical_models import train_lgbm
        train_lgbm(train_texts, train_labels, val_texts, val_labels)
        logger.finish_step("LightGBM stylometric")

    # ── STEP 4: Binoculars cross-model calibration ─────────────────────────────
    header("STEP 4: Calibrating Binoculars (Pythia-1B / Pythia-70M)", skip=args.skip_perplexity)
    if args.skip_perplexity:
        logger.skip_step("Binoculars")
    else:
        logger.start_step("Binoculars")
        from src.perplexity_model import BinocularsDetector
        bino = BinocularsDetector()

        import random
        random.seed(SEED)
        n_cal = max(200, int(len(train_texts) * PPL_CALIBRATION_FRACTION))
        cal_indices = random.sample(range(len(train_texts)), n_cal)
        cal_texts = [train_texts[i] for i in cal_indices]
        cal_labels = [train_labels[i] for i in cal_indices]
        print(f"Calibrating on {n_cal} examples...")

        bino.calibrate(cal_texts, cal_labels)
        bino.save_calibration(PPL_CALIBRATION_PATH)
        logger.finish_step("Binoculars")

    # ── STEP 5: Validation predictions from all models ─────────────────────────
    header("STEP 5: Computing validation predictions for ensemble training")
    logger.start_step("Validation predictions")

    device = "cpu" if args.no_gpu else None

    print("ModernBERT predictions...")
    from src.transformer_model import TransformerDetector
    deberta = TransformerDetector(device=device)
    deberta_val = deberta.predict_proba(val_texts)
    deberta_auc = _auc(val_labels_arr, deberta_val)
    print(f"  AUC: {deberta_auc:.4f}")
    logger.log_component_auc("ModernBERT", deberta_auc)
    del deberta

    print("TF-IDF predictions...")
    from src.classical_models import TFIDFDetector
    tfidf_val = TFIDFDetector().predict_proba(val_texts)
    tfidf_auc = _auc(val_labels_arr, tfidf_val)
    print(f"  AUC: {tfidf_auc:.4f}")
    logger.log_component_auc("TF-IDF", tfidf_auc)

    print("LightGBM predictions...")
    from src.classical_models import LGBMDetector
    lgbm_val = LGBMDetector().predict_proba(val_texts)
    lgbm_auc = _auc(val_labels_arr, lgbm_val)
    print(f"  AUC: {lgbm_auc:.4f}")
    logger.log_component_auc("LightGBM", lgbm_auc)

    if not args.skip_perplexity:
        print("Binoculars predictions on validation set...")
        from src.perplexity_model import BinocularsDetector
        bino = BinocularsDetector()
        bino.load_calibration(PPL_CALIBRATION_PATH)
        ppl_val = bino.predict_proba(val_texts, verbose=True)
        ppl_auc = _auc(val_labels_arr, ppl_val)
        print(f"  AUC: {ppl_auc:.4f}")
        logger.log_component_auc("Binoculars", ppl_auc)
        del bino
        val_predictions = np.column_stack([deberta_val, tfidf_val, lgbm_val, ppl_val])
    else:
        print("Skipping perplexity (--skip-perplexity set) — using 3-model ensemble")
        ppl_val = None
        val_predictions = np.column_stack([deberta_val, tfidf_val, lgbm_val])

    logger.finish_step("Validation predictions")

    # ── STEP 6: Train meta-learner + calibration + abstention ──────────────────
    header("STEP 6: Training ensemble meta-learner + calibration + abstention")
    logger.start_step("Ensemble meta-learner")

    ensemble = EnsembleModel()
    ensemble.train_meta_learner(val_predictions, val_labels_arr)
    ensemble.calibrate_temperature(val_predictions, val_labels_arr)
    ensemble.find_abstention_thresholds(val_predictions, val_labels_arr)
    logger.finish_step("Ensemble meta-learner")

    # ── STEP 7: Final evaluation ────────────────────────────────────────────────
    header("STEP 7: Final validation evaluation")
    logger.start_step("Final evaluation")

    final_scores = ensemble.predict(val_predictions, apply_abstention=True)
    metrics = evaluate_all_metrics(val_labels_arr, final_scores)

    logger.log_ensemble_metrics(metrics)
    logger.log_val_scores(val_labels_arr, final_scores)

    # Log abstention thresholds if available
    if hasattr(ensemble, "score_threshold") and hasattr(ensemble, "var_threshold"):
        logger.log_abstention(ensemble.score_threshold, ensemble.var_threshold)

    logger.finish_step("Final evaluation")

    print("\nFinal Ensemble Metrics on Validation Set:")
    print("-" * 42)
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    print("\nIndividual Model Metrics (for reference):")
    model_preds = [
        ("ModernBERT", deberta_val),
        ("TF-IDF", tfidf_val),
        ("LightGBM", lgbm_val),
    ]
    if ppl_val is not None:
        model_preds.append(("Binoculars", ppl_val))

    for name, probs in model_preds:
        m = evaluate_all_metrics(val_labels_arr, probs)
        print(
            f"  {name:12s}: mean={m['mean']:.4f} | "
            f"AUC={m['roc_auc']:.4f} | C@1={m['c@1']:.4f} | Brier={m['brier']:.4f}"
        )

    ensemble.save(ENSEMBLE_PATH)
    print(f"\nAll models saved to {ENSEMBLE_PATH.parent}/")
    print("Training complete!")


def header(title: str, skip: bool = False) -> None:
    status = "[SKIPPED]" if skip else ""
    print(f"\n{'=' * 60}")
    print(f"{title} {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()