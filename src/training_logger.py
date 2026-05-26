"""
Training logger — writes structured metrics to training_log.json during training.

Usage in train.py:
    from src.training_logger import TrainingLogger
    logger = TrainingLogger()

    logger.start_step("ModernBERT")
    # ... training ...
    logger.finish_step("ModernBERT")

    logger.log_component_auc("ModernBERT", 0.995)
    logger.log_ensemble_metrics({"roc_auc": 0.997, ...})
    logger.log_val_scores(labels, scores)
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

LOG_PATH = Path(__file__).parent.parent / "training_log.json"

# Step names in order
STEPS = [
    "ModernBERT",
    "TF-IDF",
    "LightGBM stylometric",
    "Binoculars",
    "Validation predictions",
    "Ensemble meta-learner",
    "Final evaluation",
]


class TrainingLogger:
    """
    Writes training progress to training_log.json.
    Thread-safe enough for single-process training.
    """

    def __init__(self, path: Path = LOG_PATH):
        self.path = Path(path)
        self._state = {
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "steps": {s: {"status": "pending", "started_at": None,
                          "finished_at": None, "elapsed_sec": None}
                      for s in STEPS},
            "train_steps": [],         # [{epoch_frac, step, loss, lr}] — every logging_steps
            "modernbert_epochs": [],   # [{epoch, val_loss, roc_auc, brier, c_at_1, f1, f0_5u, mean}]
            "component_aucs": {},      # {"ModernBERT": 0.995, ...}
            "ensemble_metrics": {},    # {"roc_auc": ..., "brier": ..., ...}
            "abstention": {},          # {"score_threshold": ..., "var_threshold": ...}
            "val_scores": {},          # {"human": [...], "ai": [...]}
        }
        self._save()

    # ── Step tracking ─────────────────────────────────────────────────────────

    def start_step(self, name: str) -> None:
        if name in self._state["steps"]:
            self._state["steps"][name]["status"] = "running"
            self._state["steps"][name]["started_at"] = datetime.now().isoformat()
            self._save()

    def finish_step(self, name: str) -> None:
        if name in self._state["steps"]:
            step = self._state["steps"][name]
            step["status"] = "done"
            step["finished_at"] = datetime.now().isoformat()
            if step["started_at"]:
                started = datetime.fromisoformat(step["started_at"])
                step["elapsed_sec"] = round((datetime.now() - started).total_seconds(), 1)
            self._save()

    def skip_step(self, name: str) -> None:
        if name in self._state["steps"]:
            self._state["steps"][name]["status"] = "skipped"
            self._save()

    # ── ModernBERT per-step training loss ────────────────────────────────────

    def log_train_step(
        self,
        step: int,
        epoch_frac: float,
        loss: float,
        lr: float,
    ) -> None:
        """Called every logging_steps (default 200) during ModernBERT training."""
        self._state["train_steps"].append({
            "step":       step,
            "epoch_frac": round(epoch_frac, 4),
            "loss":       round(loss, 5),
            "lr":         round(lr, 8),
        })
        self._save()

    # ── ModernBERT per-epoch metrics ──────────────────────────────────────────

    def log_modernbert_epoch(
        self,
        epoch: float,
        metrics: dict,
    ) -> None:
        """
        Store per-epoch val metrics from the HuggingFace Trainer evaluation.

        metrics keys: val_loss, roc_auc, brier, c_at_1, f1, f0_5u, mean
        Step-level train loss is stored separately via log_train_step().
        """
        entry = {"epoch": round(epoch, 2)}
        for k, v in metrics.items():
            entry[k] = round(float(v), 5)
        self._state["modernbert_epochs"].append(entry)
        self._save()

    # ── Component AUCs ────────────────────────────────────────────────────────

    def log_component_auc(self, name: str, auc: float) -> None:
        self._state["component_aucs"][name] = round(auc, 5)
        self._save()

    # ── Ensemble metrics ──────────────────────────────────────────────────────

    def log_ensemble_metrics(self, metrics: dict) -> None:
        self._state["ensemble_metrics"] = {
            k: round(float(v), 5) for k, v in metrics.items()
        }
        self._save()

    def log_abstention(self, score_threshold: float, var_threshold: float) -> None:
        self._state["abstention"] = {
            "score_threshold": round(score_threshold, 4),
            "var_threshold": round(var_threshold, 4) if var_threshold < 100 else None,
        }
        self._save()

    # ── Validation score distribution ─────────────────────────────────────────

    def log_val_scores(self, labels, scores, max_samples: int = 2000) -> None:
        """
        Store a sample of val scores for the histogram chart.
        Keeps at most max_samples per class to keep the JSON small.
        """
        import random
        human_scores = [float(s) for s, l in zip(scores, labels) if l == 0]
        ai_scores    = [float(s) for s, l in zip(scores, labels) if l == 1]

        if len(human_scores) > max_samples:
            human_scores = random.sample(human_scores, max_samples)
        if len(ai_scores) > max_samples:
            ai_scores = random.sample(ai_scores, max_samples)

        self._state["val_scores"] = {
            "human": [round(s, 4) for s in human_scores],
            "ai":    [round(s, 4) for s in ai_scores],
        }
        self._save()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _save(self) -> None:
        self._state["updated_at"] = datetime.now().isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)
        tmp.replace(self.path)   # Atomic replace


# ── HuggingFace Trainer callback ───────────────────────────────────────────────

class ModernBERTLogCallback:
    """
    Plugs into HuggingFace Trainer to log per-epoch metrics to TrainingLogger.

    Usage:
        callback = ModernBERTLogCallback(logger)
        trainer = Trainer(..., callbacks=[callback, EarlyStoppingCallback(...)])
    """

    def __init__(self, logger: TrainingLogger):
        self.logger = logger
        # Import here so the file can be imported without transformers installed
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def __init__(self_, logger_):
                self_._logger = logger_

            def on_log(self_, args, state, control, logs=None, **kwargs):
                if logs is None or "loss" not in logs:
                    return   # skip eval-phase log entries
                self_._logger.log_train_step(
                    step=state.global_step,
                    epoch_frac=logs.get("epoch", state.epoch or 0),
                    loss=logs["loss"],
                    lr=logs.get("learning_rate", 0.0),
                )

            def on_evaluate(self_, args, state, control, metrics=None, **kwargs):
                if metrics is None:
                    return
                epoch = state.epoch or 0

                # Val loss (always present) + all PAN metrics
                epoch_metrics = {}
                if "eval_loss" in metrics:
                    epoch_metrics["val_loss"] = metrics["eval_loss"]
                for k in ("roc_auc", "brier", "c_at_1", "f1", "f0_5u", "mean"):
                    if f"eval_{k}" in metrics:
                        epoch_metrics[k] = metrics[f"eval_{k}"]

                self_._logger.log_modernbert_epoch(epoch, epoch_metrics)

        self._callback = _Callback(logger)

    def __call__(self):
        return self._callback