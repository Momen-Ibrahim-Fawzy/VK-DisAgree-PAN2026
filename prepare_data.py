#!/usr/bin/env python3
"""
Data preparation script — run this BEFORE train.py.

Performs three steps:
  1. Deduplication   — removes exact-duplicate texts found in analysis
  2. Oversampling    — duplicates the 5 hardest-to-detect AI models by 1.5×
  3. Extra data      — optionally merges extra_data/merged/extra_train.jsonl
                       (Track A new models + Track B obfuscations + Track C human)

Output: data/train_prepared.jsonl  ← train.py reads this by default

Usage:
    # Standard (no extra data):
    python prepare_data.py --train data/train.jsonl

    # With extra data (recommended for robustness):
    python prepare_data.py \\
        --train data/train.jsonl \\
        --extra extra_data/merged/extra_train.jsonl

    Then train as usual:
    python train.py --val data/val.jsonl
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.config import HARD_MODELS, HARD_MODEL_OVERSAMPLE_FACTOR, SEED


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """Remove exact duplicate texts (keep first occurrence)."""
    seen = set()
    clean = []
    n_removed = 0
    for rec in records:
        h = hash(rec["text"])
        if h in seen:
            n_removed += 1
        else:
            seen.add(h)
            clean.append(rec)
    return clean, n_removed


def oversample_hard_models(
    records: list[dict],
    hard_models: list[str],
    factor: float,
    seed: int,
) -> tuple[list[dict], int]:
    """
    Add extra copies of hard-model texts.

    factor=1.5 means each hard-model text gets 0.5 extra copies on average.
    We randomly sample floor(n_hard * (factor - 1.0)) additional records.
    """
    random.seed(seed)
    hard_recs = [r for r in records if r.get("model", "") in hard_models]
    n_extra = int(len(hard_recs) * (factor - 1.0))

    if n_extra == 0:
        return records, 0

    extra = random.choices(hard_recs, k=n_extra)
    return records + extra, n_extra


def main():
    p = argparse.ArgumentParser(description="Prepare training data: dedup + oversample")
    p.add_argument(
        "--train", type=Path,
        default=Path("data/train.jsonl"),
        help="Path to raw train.jsonl (default: data/train.jsonl)",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("data/train_prepared.jsonl"),
        help="Where to save the prepared file (default: data/train_prepared.jsonl)",
    )
    p.add_argument(
        "--no-oversample", action="store_true",
        help="Skip oversampling (apply dedup only)",
    )
    p.add_argument(
        "--no-dedup", action="store_true",
        help="Skip deduplication (apply oversampling only)",
    )
    p.add_argument(
        "--extra", type=Path,
        default=None,
        help="Path to extra_data/merged/extra_train.jsonl to append after preparation",
    )
    args = p.parse_args()

    print(f"Loading {args.train}...")
    records = load_jsonl(args.train)
    n_raw = len(records)
    print(f"  Raw examples: {n_raw:,}")

    # ── Step 1: Deduplication ─────────────────────────────────────────────────
    if not args.no_dedup:
        records, n_removed = deduplicate(records)
        print(f"\nDeduplication:")
        print(f"  Removed {n_removed} duplicate texts")
        print(f"  Remaining: {len(records):,}")
    else:
        print("\nDeduplication: skipped")

    # ── Step 2: Oversample hard models ────────────────────────────────────────
    if not args.no_oversample:
        # Report model counts before
        model_counts_before = collections.Counter(
            r.get("model", "?") for r in records if r.get("model") in HARD_MODELS
        )
        print(f"\nOversampling hard models (factor={HARD_MODEL_OVERSAMPLE_FACTOR}×):")
        print(f"  Hard models: {HARD_MODELS}")
        print(f"  Counts before:")
        for m in HARD_MODELS:
            print(f"    {m:<42} {model_counts_before.get(m, 0):>5}")

        records, n_extra = oversample_hard_models(
            records, HARD_MODELS, HARD_MODEL_OVERSAMPLE_FACTOR, SEED
        )

        model_counts_after = collections.Counter(
            r.get("model", "?") for r in records if r.get("model") in HARD_MODELS
        )
        print(f"  Added {n_extra:,} extra hard-model copies")
        print(f"  Counts after:")
        for m in HARD_MODELS:
            print(f"    {m:<42} {model_counts_after.get(m, 0):>5}")
    else:
        print("\nOversampling: skipped")

    # ── Step 3: Merge extra data ──────────────────────────────────────────────
    if args.extra is not None:
        if not args.extra.exists():
            print(f"\n[ERROR] --extra file not found: {args.extra}", file=sys.stderr)
            sys.exit(1)

        extra_records = load_jsonl(args.extra)
        print(f"\nMerging extra data from {args.extra}:")
        print(f"  Extra records: {len(extra_records):,}")

        # Dedup extra against already-seen texts (by full text hash)
        seen_hashes = {hash(r["text"]) for r in records}
        before = len(extra_records)
        extra_records = [r for r in extra_records if hash(r["text"]) not in seen_hashes]
        n_extra_dupes = before - len(extra_records)
        if n_extra_dupes:
            print(f"  Removed {n_extra_dupes} duplicates already in base set")
        print(f"  Adding {len(extra_records):,} extra records")

        records = records + extra_records

    # ── Summary ───────────────────────────────────────────────────────────────
    n_final = len(records)
    labels = [r["label"] for r in records]
    n_human = labels.count(0)
    n_ai    = labels.count(1)
    print(f"\nFinal dataset: {n_final:,} examples "
          f"(+{n_final - n_raw:,} net change from raw)")
    print(f"  Human: {n_human:,} ({n_human/n_final*100:.1f}%)")
    print(f"  AI:    {n_ai:,}    ({n_ai/n_final*100:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    # Shuffle before saving so records are well-mixed
    random.seed(SEED)
    random.shuffle(records)

    save_jsonl(records, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()