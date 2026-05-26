#!/usr/bin/env python3
"""
Merge all extra-data tracks into a single training supplement file.

Tracks:
  A – extra_data/ai_texts/*.jsonl       (new AI model texts, label=1)
  B – extra_data/obfuscations/*.jsonl   (obfuscated AI texts, label=1)
  C – extra_data/human_texts/*.jsonl    (pre-LLM human texts, label=0)

Filters applied:
  - 200–1200 words
  - ≥80% ASCII (English check)
  - Dedup against train_prepared.jsonl by text prefix hash
  - Dedup within extra data by text prefix hash

Output:
  extra_data/merged/extra_train.jsonl
  (drop-in supplement for train.py — same JSONL schema as train_prepared.jsonl)

Usage:
    python merge_extra_data.py
    python merge_extra_data.py --min-words 150 --max-words 1500
    python merge_extra_data.py --dry-run
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

EXTRA_DIR  = Path(__file__).parent
TRAIN_FILE = EXTRA_DIR.parent / "data" / "train_prepared.jsonl"
OUTPUT     = EXTRA_DIR / "merged" / "extra_train.jsonl"

SOURCE_DIRS = {
    # A and B: use CLEANED versions (artifacts removed)
    "A_ai":         EXTRA_DIR / "cleaned" / "ai_texts",
    "B_obfuscated": EXTRA_DIR / "cleaned" / "obfuscations",
    # C: no cleaning needed — human texts have no model artifacts
    "C_human":      EXTRA_DIR / "human_texts",
}

DEFAULT_MIN_WORDS = 200
DEFAULT_MAX_WORDS = 1200


# ── Helpers ───────────────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def is_english(text: str, threshold: float = 0.80) -> bool:
    if not text:
        return False
    return sum(1 for c in text if ord(c) < 128) / len(text) >= threshold


def text_fingerprints(text: str) -> tuple[str, str]:
    """Return (prefix_hash, full_hash) for two-level deduplication.

    prefix_hash: MD5 of first 200 chars — catches texts that start identically
                 (same Gutenberg excerpt, same AI prompt output)
    full_hash:   MD5 of entire normalised text — catches identical full texts
                 even when they have different beginnings (obfuscations, etc.)
    """
    normalised = " ".join(text.strip().lower().split())  # collapse whitespace
    prefix_hash = hashlib.md5(normalised[:200].encode("utf-8")).hexdigest()
    full_hash   = hashlib.md5(normalised.encode("utf-8")).hexdigest()
    return prefix_hash, full_hash


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def load_all_jsonl_in_dir(directory: Path) -> list[dict]:
    if not directory.exists():
        return []
    records = []
    for f in sorted(directory.glob("*.jsonl")):
        records.extend(load_jsonl(f))
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Merge extra training data tracks")
    p.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    p.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    p.add_argument("--dry-run", action="store_true",
                   help="Report stats without writing output")
    return p.parse_args()


def main():
    args = parse_args()

    # Load train fingerprints for dedup (both prefix and full hash)
    print(f"Loading training data fingerprints from {TRAIN_FILE}…")
    seen_prefix: set[str] = set()
    seen_full:   set[str] = set()
    if TRAIN_FILE.exists():
        for rec in load_jsonl(TRAIN_FILE):
            text = rec.get("text", "")
            if text:
                ph, fh = text_fingerprints(text)
                seen_prefix.add(ph)
                seen_full.add(fh)
    print(f"  {len(seen_full)} texts fingerprinted from training data")

    all_records: list[dict] = []
    track_stats: dict[str, Counter] = {}

    for track_name, directory in SOURCE_DIRS.items():
        recs = load_all_jsonl_in_dir(directory)
        print(f"\nTrack {track_name}: {len(recs)} raw records from {directory}")

        kept = []
        dropped = Counter()
        for rec in recs:
            text = rec.get("text", "")
            wc = word_count(text)

            if wc < args.min_words:
                dropped["too_short"] += 1
                continue
            if wc > args.max_words:
                dropped["too_long"] += 1
                continue
            if not is_english(text):
                dropped["not_english"] += 1
                continue

            ph, fh = text_fingerprints(text)
            if ph in seen_prefix or fh in seen_full:
                dropped["duplicate"] += 1
                continue

            seen_prefix.add(ph)
            seen_full.add(fh)
            kept.append(rec)

        label_dist = Counter(r.get("label") for r in kept)
        genre_dist = Counter(r.get("genre") for r in kept)
        print(f"  Kept: {len(kept)} | Dropped: {dict(dropped)}")
        print(f"  Labels: {dict(label_dist)} | Genres: {dict(genre_dist)}")
        track_stats[track_name] = Counter({"kept": len(kept), **dropped})
        all_records.extend(kept)

    # Summary
    total_label = Counter(r.get("label") for r in all_records)
    total_genre = Counter(r.get("genre") for r in all_records)
    print(f"\n{'='*50}")
    print(f"Total records to write: {len(all_records)}")
    print(f"By label: {dict(total_label)}")
    print(f"By genre: {dict(total_genre)}")

    if args.dry_run:
        print("\nDry run — not writing output.")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for rec in all_records:
            # Normalise to standard schema: id, text, label, model, genre
            out = {
                "id":    rec.get("id", ""),
                "text":  rec.get("text", "").strip(),
                "label": rec.get("label", 0),
                "model": rec.get("model", "unknown"),
                "genre": rec.get("genre", "unknown"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\nWritten to {OUTPUT}")

    # Verify
    written = load_jsonl(OUTPUT)
    print(f"Verified: {len(written)} records in output file")


if __name__ == "__main__":
    main()
