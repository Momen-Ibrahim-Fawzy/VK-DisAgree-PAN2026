#!/usr/bin/env python3
"""
Track B – Generate obfuscated/paraphrased variants of existing AI texts.

For each source AI text, we ask a Groq LLM to rewrite it in a way that
disguises its AI origin while preserving meaning (i.e. "humanisation" /
obfuscation).  This gives us label=1 texts that look more human-written,
making the detector more robust against adversarial obfuscation at test time.

Usage:
    pip install groq
    export GROQ_API_KEY=your_key_here
    python collect_obfuscations.py

    # Limit sources (useful for testing):
    python collect_obfuscations.py --max-sources 50

    # Dry run:
    python collect_obfuscations.py --dry-run --max-sources 5

Output:
    extra_data/obfuscations/obfuscated_texts.jsonl
    Each record: {id, text, label:1, model:"<orig_model>_para_via_<groq_model>", genre}
"""
import argparse
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_FILE   = Path(__file__).parent.parent / "data" / "train_prepared.jsonl"
OUTPUT_FILE  = Path(__file__).parent / "obfuscations" / "obfuscated_texts.jsonl"

# Groq models to use for paraphrasing (rotated to spread rate-limit load)
PARAPHRASE_MODELS = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct",
]

# How many source AI texts to sample from training data
DEFAULT_MAX_SOURCES = 1000

# Genre distribution weights for source sampling (match training distribution)
GENRE_WEIGHTS = {"fiction": 0.60, "essays": 0.20, "news": 0.20}

# Minimum word count for source texts
MIN_WORDS = 200

# ── Prompts ───────────────────────────────────────────────────────────────────

PARAPHRASE_PROMPT = """\
IMPORTANT: Respond ONLY in English. Do not use any other language.

Below is a piece of {genre} text. Rewrite it so that it reads as naturally \
human-written as possible — vary sentence length and rhythm, use occasional \
informality or minor imperfection where appropriate, and avoid the polished \
uniformity typical of AI-generated prose. Preserve all the facts, arguments, \
and narrative content of the original. Do not add or remove major ideas.

Output only the rewritten text — no preamble, no explanation, no heading.

ORIGINAL TEXT:
{text}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def word_count(text: str) -> int:
    return len(text.split())


def is_english(text: str) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) >= 0.80


def sample_sources(train_records: list[dict], n: int, seed: int) -> list[dict]:
    """Sample n AI texts from training data, respecting genre weights."""
    rng = random.Random(seed)
    ai_texts = [r for r in train_records
                if r.get("label") == 1
                and word_count(r.get("text", "")) >= MIN_WORDS
                and is_english(r.get("text", ""))]

    by_genre: dict[str, list] = {}
    for r in ai_texts:
        g = r.get("genre", "unknown")
        by_genre.setdefault(g, []).append(r)

    targets = {g: round(n * w) for g, w in GENRE_WEIGHTS.items()}
    diff = n - sum(targets.values())
    targets["fiction"] += diff

    sampled = []
    for genre, count in targets.items():
        pool = by_genre.get(genre, [])
        k = min(count, len(pool))
        sampled.extend(rng.sample(pool, k))

    rng.shuffle(sampled)
    return sampled


def paraphrase(client, text: str, genre: str, model: str) -> str | None:
    prompt = PARAPHRASE_PROMPT.format(genre=genre, text=text)
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"    [retry {attempt+1}/4] {e} — waiting {wait:.1f}s")
            time.sleep(wait)
    return None


def already_done(output_file: Path) -> set[str]:
    """Return set of source IDs already processed (to allow resuming)."""
    if not output_file.exists():
        return set()
    done = set()
    for rec in load_jsonl(output_file):
        src_id = rec.get("source_id")
        model = rec.get("model", "")
        if src_id:
            done.add(f"{src_id}::{model}")
    return done


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate obfuscated AI text variants (Track B)")
    p.add_argument("--max-sources", type=int, default=DEFAULT_MAX_SOURCES,
                   help=f"Number of source AI texts to paraphrase (default: {DEFAULT_MAX_SOURCES})")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Print prompts without calling the API")
    p.add_argument("--models", nargs="+", default=PARAPHRASE_MODELS,
                   help="Groq models to use for paraphrasing")
    return p.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key and not args.dry_run:
        print("[ERROR] Set GROQ_API_KEY environment variable first.", file=sys.stderr)
        print("  export GROQ_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        try:
            from groq import Groq
        except ImportError:
            print("[ERROR] groq package not installed. Run: pip install groq", file=sys.stderr)
            sys.exit(1)
        client = Groq(api_key=api_key)
    else:
        client = None

    if not TRAIN_FILE.exists():
        print(f"[ERROR] Training file not found: {TRAIN_FILE}", file=sys.stderr)
        print("  Run prepare_data.py first to generate data/train_prepared.jsonl", file=sys.stderr)
        sys.exit(1)

    print(f"Loading training data from {TRAIN_FILE}...")
    train = load_jsonl(TRAIN_FILE)
    print(f"  Loaded {len(train)} records.")

    models = args.models
    total_jobs = args.max_sources * len(models)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done_keys = already_done(OUTPUT_FILE)
    print(f"Already processed: {len(done_keys)} source×model pairs")
    print(f"Plan: {args.max_sources} sources × {len(models)} models = {total_jobs} jobs "
          f"(each model gets its own random sample)")

    n_generated = 0
    n_skipped = 0
    job = 0

    for m_idx, model in enumerate(models):
        # Each model gets an independent random sample — seed offset by model index
        model_seed = args.seed + m_idx * 1000
        sources = sample_sources(train, args.max_sources, model_seed)
        genre_dist = Counter(r.get("genre") for r in sources)
        print(f"\n--- Model {m_idx+1}/{len(models)}: {model.split('/')[-1]} "
              f"| seed={model_seed} | {dict(genre_dist)} ---")

        for source in sources:
            job += 1
            src_id = source.get("id", str(job))
            genre  = source.get("genre", "fiction")
            text   = source.get("text", "")
            orig_model = source.get("model", "unknown")
            key = f"{src_id}::{orig_model}_para_via_{model.split('/')[-1]}"

            if key in done_keys:
                print(f"  [{job}/{total_jobs}] SKIP (already done) — {src_id[:8]}…")
                continue

            if args.dry_run:
                print(f"[{job}/{total_jobs}] {genre} | {model.split('/')[-1]} | {word_count(text)}w source")
                n_generated += 1
                continue

            result = paraphrase(client, text, genre, model)

            if result is None or not is_english(result) or word_count(result) < MIN_WORDS:
                n_skipped += 1
                print(f"  [{job}/{total_jobs}] SKIPPED (failed/non-English/too short)")
                continue

            record = {
                "id":        str(uuid.uuid4()),
                "source_id": src_id,
                "text":      result,
                "label":     1,
                "model":     f"{orig_model}_para_via_{model.split('/')[-1]}",
                "genre":     genre,
            }
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            n_generated += 1
            done_keys.add(key)
            wc = word_count(result)
            print(f"  [{job}/{total_jobs}] {genre} | {wc}w | via {model.split('/')[-1][:20]}…")
            time.sleep(0.3)  # gentle rate limit

    print(f"\nDone. Generated: {n_generated}, Skipped: {n_skipped}")
    if OUTPUT_FILE.exists():
        final = load_jsonl(OUTPUT_FILE)
        gc = Counter(r.get("genre") for r in final)
        mc = Counter(r.get("model", "").split("_para_via_")[-1] for r in final)
        print(f"Total obfuscated texts: {len(final)}")
        print(f"By genre: {dict(gc)}")
        print(f"By paraphrase model: {dict(mc)}")


if __name__ == "__main__":
    main()