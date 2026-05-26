#!/usr/bin/env python3
"""
Generate Claude texts for Track A using the Anthropic API.

This script generates the remaining Claude texts needed to reach 300 total.
It resumes from where the existing claude-sonnet-4-6.jsonl left off.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=your_key_here
    python generate_claude_texts.py

    # Generate specific number:
    python generate_claude_texts.py --target 300

    # Dry run (prints prompts without calling API):
    python generate_claude_texts.py --dry-run --target 5
"""
import argparse
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path

OUTPUT_FILE = Path(__file__).parent / "ai_texts" / "claude-sonnet-4-6.jsonl"
MODEL = "claude-sonnet-4-6"

GENRE_WEIGHTS = {"fiction": 0.60, "essays": 0.20, "news": 0.20}

# ── Topic seeds ───────────────────────────────────────────────────────────────

FICTION_TOPICS = [
    "A postal worker discovers a bag of undelivered letters from the 1970s in a sorting facility",
    "Two neighbors who have feuded for years must cooperate during a prolonged power outage",
    "A high school principal must decide whether to report a colleague she's worked with for twenty years",
    "An archaeologist uncovers something that contradicts everything her mentor taught her",
    "A woman discovers her late mother kept a second household in another city",
    "A competitive baker must train the rival who beat her three years running",
    "An interpreter at a peace negotiation begins to question whether to translate everything",
    "A firefighter returns to the town where he grew up after a wildfire destroys it",
    "A twins' estrangement comes to a head when they must care for their ailing father together",
    "A hospice musician plays for a patient who turns out to be a significant figure from her past",
    "A newspaper editor must decide what to print in the paper's final edition before it closes",
    "A woman who has kept her terminal diagnosis from her family for six months finally tells them",
    "A teacher accidentally discovers one of her students is homeless and must decide how to help",
    "Two strangers realize they've been writing letters to the same prisoner for three years",
    "An engineer is sent to decommission a lighthouse that has been staffed for 120 years",
    "A competitive chess player begins coaching the child of the man who beat her in her last tournament",
    "A soldier's service dog is adopted by a family who must return him when the soldier returns",
    "A botanist cataloguing a remote area discovers that someone has been living there undocumented",
    "A novelist realizes her fictional town shares too many details with a real crime she witnessed",
    "A night security guard at a hospital develops an unexpected friendship with a long-term patient",
    "An elderly calligrapher takes on a student who is terminally ill and wants to leave something behind",
    "A woman returns to her childhood home for the first time in forty years and finds it changed less than she has",
    "A sommelier diagnosed with anosmia must decide whether to continue practicing",
    "A public defender realizes her client knows something about a case from her own family's past",
    "A long-distance swimmer attempts a crossing her mother failed thirty years before",
    "A small-town mayor must choose between a factory bringing jobs and protecting the watershed",
    "A retired schoolteacher begins corresponding with a student she failed and still thinks about",
    "A wilderness guide loses his bearings for the first time in a career spanning thirty years",
    "A museum conservator discovers a painting that may have been stolen during wartime",
    "A woman joins a grief support group and begins to realize it is changing her in ways she didn't expect",
]

ESSAY_TOPICS = [
    "The case for and against prison labor programs in the American penal system",
    "Whether the gig economy represents liberation or exploitation for low-income workers",
    "How the decline of local news has changed civic life in mid-sized American cities",
    "The limits and possibilities of food banks as a response to food insecurity",
    "Why suburban sprawl is primarily a policy problem rather than a preference problem",
    "The effects of sleep deprivation on workplace productivity and public health policy",
    "Whether standardized testing should continue to play a role in college admissions",
    "The ethics of charging higher insurance premiums based on genetic risk factors",
    "How pandemic-era school closures will affect educational outcomes for a decade",
    "The case for four-day workweeks as an economic and public health policy",
    "Whether cities should invest in light rail or bus rapid transit for urban mobility",
    "How the American foster care system fails teenagers aging out of the system",
]

NEWS_TOPICS = [
    "A long-struggling urban school district reports its first year of consistent attendance improvement",
    "A small city launches a guaranteed income pilot program for residents below the poverty line",
    "A national park service removes an invasive plant species after a decade-long restoration effort",
    "A county jail implements a first-in-state educational credential program for incarcerated residents",
    "A regional food bank launches a produce distribution network connecting farms and food-insecure households",
    "A rust belt manufacturing town attracts its first major employer in fifteen years",
    "A public university announces free tuition for students from families earning under sixty thousand dollars",
    "A coastal city begins relocating a neighborhood that has flooded three times in five years",
    "A youth mentorship program credits a ten-year-old partnership with local businesses for its retention rates",
    "A state legislature passes a bill requiring insurance coverage for fertility treatments",
]

ALL_TOPICS = {"fiction": FICTION_TOPICS, "essays": ESSAY_TOPICS, "news": NEWS_TOPICS}

PROMPTS = {
    "fiction": (
        "IMPORTANT: Respond ONLY in English. Do not use any other language.\n\n"
        "Write a self-contained literary fiction passage on this premise:\n\n{topic}\n\n"
        "Requirements:\n"
        "- English only\n"
        "- 500 to 650 words\n"
        "- Focus on character and specific concrete detail\n"
        "- Show through scene and dialogue; do not summarize\n"
        "- No title or headings — prose only\n"
    ),
    "essays": (
        "IMPORTANT: Respond ONLY in English. Do not use any other language.\n\n"
        "Write a well-argued analytical essay on:\n\n{topic}\n\n"
        "Requirements:\n"
        "- English only\n"
        "- 480 to 560 words\n"
        "- Clear position with specific evidence and reasoning\n"
        "- Continuous prose — no bullets or lists\n"
        "- No title or headings — essay body only\n"
    ),
    "news": (
        "IMPORTANT: Respond ONLY in English. Do not use any other language.\n\n"
        "Write a factual news article about:\n\n{topic}\n\n"
        "Requirements:\n"
        "- English only\n"
        "- 440 to 540 words\n"
        "- Inverted pyramid structure\n"
        "- At least two attributed quotes from named sources\n"
        "- No headline or byline — article body only\n"
    ),
}


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def is_english(text: str) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) >= 0.85


def word_count(text: str) -> int:
    return len(text.split())


def build_genre_list(n: int) -> list[str]:
    counts = {g: round(n * w) for g, w in GENRE_WEIGHTS.items()}
    diff = n - sum(counts.values())
    counts["fiction"] += diff
    genres = []
    for g, c in counts.items():
        genres.extend([g] * c)
    random.shuffle(genres)
    return genres


def generate_text(client, genre: str, topic: str) -> str | None:
    prompt = PROMPTS[genre].format(topic=topic)
    for attempt in range(4):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"  [retry {attempt+1}/4] {e} — waiting {wait:.1f}s")
            time.sleep(wait)
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Generate Claude texts for Track A")
    p.add_argument("--target", type=int, default=300,
                   help="Target total number of Claude texts (default: 300)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dry-run", action="store_true",
                   help="Print prompts without calling the API")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("[ERROR] Set ANTHROPIC_API_KEY environment variable first.", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        try:
            import anthropic
        except ImportError:
            print("[ERROR] anthropic package not installed. Run: pip install anthropic", file=sys.stderr)
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
    else:
        client = None

    existing = load_existing(OUTPUT_FILE)
    n_done = len(existing)
    n_needed = max(0, args.target - n_done)

    print(f"Existing: {n_done} | Target: {args.target} | To generate: {n_needed}")
    if n_needed == 0:
        print("Already at target. Done.")
        return

    genre_list = build_genre_list(n_needed)
    used_topics = {g: set() for g in ALL_TOPICS}

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    n_generated = 0
    n_skipped = 0

    for i, genre in enumerate(genre_list):
        pool = ALL_TOPICS[genre]
        available = [t for t in pool if t not in used_topics[genre]]
        if not available:
            used_topics[genre] = set()
            available = pool
        topic = random.choice(available)
        used_topics[genre].add(topic)

        if args.dry_run:
            print(f"[{i+1}/{n_needed}] {genre} | {topic[:60]}…")
            print(PROMPTS[genre].format(topic=topic)[:200])
            print("---")
            n_generated += 1
            continue

        text = generate_text(client, genre, topic)
        if text is None or not is_english(text):
            n_skipped += 1
            print(f"  [{i+1}/{n_needed}] SKIPPED (failed or non-English)")
            continue

        wc = word_count(text)
        record = {
            "id":    str(uuid.uuid4()),
            "text":  text,
            "label": 1,
            "model": "claude-sonnet-4-6",
            "genre": genre,
        }
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        n_generated += 1

        total_now = n_done + n_generated
        print(f"  [{i+1}/{n_needed}] {genre} | {wc}w | {topic[:50]}… (total: {total_now})")
        time.sleep(0.5)

    print(f"\nDone. Generated: {n_generated}, Skipped: {n_skipped}")
    final = load_existing(OUTPUT_FILE)
    print(f"Total in file: {len(final)}")


if __name__ == "__main__":
    main()
