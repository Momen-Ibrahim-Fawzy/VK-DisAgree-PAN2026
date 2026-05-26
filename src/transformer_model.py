"""
Fine-tuned ModernBERT-base for binary AI/human text classification.

ModernBERT (Warner et al., 2024) replaces DeBERTa as the primary backbone:
  - Trained on more recent data (post-2024), giving better priors on modern AI text
  - Supports 8192 token context (vs 512 for DeBERTa) — we use 1024 for practical training
  - Flash Attention 2 based, faster and more memory efficient
  - Consistently outperforms DeBERTa-v3 on classification benchmarks

Training includes:
  - Label smoothing (prevents overconfidence, improves Brier Score)
  - Class-weighted loss (handles 38% human / 62% AI imbalance)
  - Adversarial augmentation (exposes model to obfuscated variants during training)
  - Early stopping on validation ROC-AUC
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import roc_auc_score

from .config import (
    TRANSFORMER_MODEL_NAME,
    MAX_LENGTH,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    GRADIENT_ACCUMULATION_STEPS,
    FP16,
    BF16,
    AUG_FRACTION,
    AUG_N_OPS,
    DEBERTA_DIR,
    SEED,
)


class TextDataset(Dataset):
    def __init__(self, encodings: dict, labels: Optional[list[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class LabelSmoothingTrainer(Trainer):
    """Trainer with label smoothing + optional class weighting."""

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None,
                 smoothing: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.smoothing = smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        num_classes = logits.size(-1)

        # Soft labels via label smoothing
        with torch.no_grad():
            smooth_labels = torch.full_like(logits, self.smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = nn.functional.log_softmax(logits, dim=-1)

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)[labels]
            loss = -(smooth_labels * log_probs).sum(dim=-1)
            loss = (loss * weights).mean()
        else:
            loss = -(smooth_labels * log_probs).sum(dim=-1).mean()

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Compute all five PAN evaluation metrics reported at each validation epoch."""
    from sklearn.metrics import brier_score_loss, f1_score
    from .ensemble import c_at_1, f05u_score

    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]

    auc   = roc_auc_score(labels, probs)
    brier = 1.0 - brier_score_loss(labels, probs)          # higher = better
    cat1  = c_at_1(labels, probs)
    f1    = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)
    f05u  = f05u_score(labels, probs)
    mean  = float(np.mean([auc, brier, cat1, f1, f05u]))

    # HuggingFace metric keys must be valid Python identifiers (no dots)
    return {
        "roc_auc": auc,
        "brier":   brier,
        "c_at_1":  cat1,
        "f1":      f1,
        "f0_5u":   f05u,
        "mean":    mean,
    }


def train(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    output_dir: Path = DEBERTA_DIR,
    model_name: str = TRANSFORMER_MODEL_NAME,
    use_augmentation: bool = True,
    log_callback=None,
):
    # ── GPU guard ──────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA-capable GPU found. Aborting transformer training.\n"
            "Fix options:\n"
            "  1. Update your NVIDIA driver: https://www.nvidia.com/Download/index.aspx\n"
            "  2. Reinstall PyTorch for your CUDA version: https://pytorch.org\n"
            "  3. Pass --skip-transformer to skip this step.\n"
            f"  PyTorch CUDA version: {torch.version.cuda}\n"
            f"  Run `nvidia-smi` to check your driver."
        )
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    # ──────────────────────────────────────────────────────────────────────────

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply adversarial augmentation to training data
    if use_augmentation:
        from .augmentation import augment_dataset
        print("Applying adversarial augmentation to training data...")
        aug_texts, aug_labels = augment_dataset(
            train_texts, train_labels,
            aug_fraction=AUG_FRACTION,
            n_augmentations=AUG_N_OPS,
            seed=SEED,
        )
    else:
        aug_texts, aug_labels = train_texts, train_labels

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Tokenizing {len(aug_texts)} training examples...")
    train_encodings = tokenizer(
        aug_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors=None,
    )
    print(f"Tokenizing {len(val_texts)} validation examples...")
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors=None,
    )

    train_dataset = TextDataset(train_encodings, aug_labels)
    val_dataset = TextDataset(val_encodings, val_labels)

    # Inverse-frequency class weights
    from collections import Counter
    label_counts = Counter(aug_labels)
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / (2 * label_counts[i]) for i in range(2)], dtype=torch.float32
    )
    print(f"Class weights: {class_weights.tolist()}")

    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )

    steps_per_epoch = len(train_dataset) // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mean",
        greater_is_better=True,
        fp16=FP16,
        bf16=BF16,
        seed=SEED,
        dataloader_num_workers=4,
        report_to="none",
        logging_steps=200,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    if log_callback is not None:
        callbacks.append(log_callback)

    trainer = LabelSmoothingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        smoothing=LABEL_SMOOTHING,
        callbacks=callbacks,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving best model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer


class TransformerDetector:
    """Inference wrapper for the fine-tuned ModernBERT model."""

    def __init__(self, model_dir: Path = DEBERTA_DIR, device: Optional[str] = None):
        model_dir = Path(model_dir)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading ModernBERT model from {model_dir} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(device)
        self.model.eval()

    def predict_proba(self, texts: list[str], batch_size: int = EVAL_BATCH_SIZE) -> np.ndarray:
        """Return AI-class probability scores."""
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encodings = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            with torch.no_grad():
                logits = self.model(**encodings).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
        return np.array(all_probs)