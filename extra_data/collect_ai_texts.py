#!/usr/bin/env python3
"""
Track A — Generate AI texts using new models via Groq API.

Models targeted (none are in the PAN training data except llama-3.3-70b which is marginal):
  - meta-llama/llama-4-maverick-17b-128e-instruct
  - meta-llama/llama-4-scout-17b-16e-instruct
  - qwen/qwen3-32b
  - moonshotai/kimi-k2-instruct
  - allam-2-7b  (primarily Arabic — non-English outputs filtered automatically)

Usage:
    pip install groq
    export GROQ_API_KEY=your_key_here
    python collect_ai_texts.py

    # Generate fewer texts per model (for quick testing):
    python collect_ai_texts.py --n-per-genre 50

    # Run only specific models:
    python collect_ai_texts.py --models llama-4-maverick qwen3-32b
"""
import argparse
import json
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent / "ai_texts"

# (groq_model_id, short_name_for_filename)
MODELS = [
    ("meta-llama/llama-4-maverick-17b-128e-instruct", "llama-4-maverick"),
    ("meta-llama/llama-4-scout-17b-16e-instruct",     "llama-4-scout"),
    ("qwen/qwen3-32b",                                 "qwen3-32b"),
    ("moonshotai/kimi-k2-instruct",                    "kimi-k2"),
    ("allam-2-7b",                                     "allam-2-7b"),
]

# Genre distribution matching PAN training set: 60% fiction, 20% essays, 20% news
GENRE_WEIGHTS = {"fiction": 0.60, "essays": 0.20, "news": 0.20}

# ── Topic seeds (diverse, non-AI-related, covers PAN genres) ──────────────────

FICTION_TOPICS = [
    "A father attempts to reconnect with his estranged adult daughter at a family funeral",
    "A competitive swimmer discovers her rival has been hiding a serious illness",
    "A retired detective receives a letter reopening a case she closed twenty years ago",
    "Two strangers become stranded together during a severe snowstorm",
    "A small-town librarian discovers a collection of unpublished manuscripts hidden in the walls",
    "A teenage musician must choose between a scholarship and caring for a sick parent",
    "A marine biologist observes strange migratory behavior in a whale pod",
    "An immigrant family celebrates their first Thanksgiving in a new country",
    "A chef prepares the final service at a restaurant that is closing after thirty years",
    "A park ranger discovers an abandoned camp deep in a protected wilderness",
    "A surgeon must perform an operation on someone she once loved",
    "An elderly woman sorts through her late husband's belongings and finds unexpected letters",
    "A war correspondent returns home and struggles to readjust to ordinary life",
    "A young architect is tasked with demolishing the building where she grew up",
    "Two childhood friends reunite at their thirtieth high school reunion",
    "A translator working on a manuscript begins to suspect it contains encoded messages",
    "A community organizer fights to preserve a neighborhood threatened by development",
    "A grieving mother starts running competitively for the first time at age fifty-two",
    "A night-shift nurse befriends an elderly patient who claims to be someone famous",
    "A fisherman discovers an unusual object washed ashore during a storm",
    "A schoolteacher in a remote village must decide whether to accept a city position",
    "A woman inherits her grandmother's bakery and discovers an old recipe book with notes",
    "A journalist investigates rumors of illegal dumping near a small river town",
    "A soldier writes letters home that he never intends to send",
    "A young botanist catalogues an unexplored region and makes an unsettling discovery",
    "An astronomer at a rural observatory encounters an unexplained signal",
    "A foster child ages out of the system and must navigate life entirely alone",
    "A competitive chess player prepares to face the opponent who humiliated her as a child",
    "A coastal town debates whether to allow a large resort to be built on the last undeveloped beach",
    "A physical therapist helps a stroke patient recover and learns unexpected truths about perseverance",
]

ESSAY_TOPICS = [
    "The effect of social media on adolescent mental health and what schools can do",
    "Whether universities should require community service hours for graduation",
    "How remote work has permanently changed the geography of American cities",
    "The ethics of gene editing in agricultural crops",
    "Whether voting should be mandatory in democratic nations",
    "The impact of fast fashion on workers and ecosystems in developing countries",
    "The case for banning single-use plastics in retail packaging",
    "Why arts education deserves equal funding to STEM programs in public schools",
    "How climate change is threatening food security in coastal regions",
    "The role of public libraries in bridging the digital divide",
    "Whether social media platforms should be regulated as public utilities",
    "The hidden costs of convenience culture and its effect on local businesses",
]

NEWS_TOPICS = [
    "A city council votes to convert a decommissioned rail yard into a public park",
    "Researchers at a regional university develop a low-cost water filtration system for rural areas",
    "A historic theater announces its final season after ninety years of continuous operation",
    "A fishing community on the Gulf Coast adapts to new catch limits designed to protect declining stocks",
    "A high school robotics team from a rural district wins a national engineering competition",
    "A wildlife corridor project successfully reconnects two forest reserves separated by a highway",
    "A network of community gardens transforms vacant lots in a mid-sized Midwestern city",
    "A small-town newspaper celebrates its one-hundred-and-fiftieth anniversary amid industry decline",
    "A county health department launches a mobile clinic program to reach underserved rural populations",
    "Scientists document the return of a bird species not seen in the region for forty years",
]

ALL_TOPICS = {
    "fiction": FICTION_TOPICS,
    "essays":  ESSAY_TOPICS,
    "news":    NEWS_TOPICS,
}

# ── Prompt templates ──────────────────────────────────────────────────────────

PROMPTS = {
    "fiction": (
        "IMPORTANT: You must respond ONLY in English. Do not use any other language.\n\n"
        "Write a self-contained short story or literary fiction passage on the following premise:\n\n"
        "{topic}\n\n"
        "Requirements:\n"
        "- ENGLISH ONLY — any non-English output will be discarded\n"
        "- Approximately 550 to 650 words\n"
        "- Focus on character interiority and specific sensory detail\n"
        "- Do not summarise or explain the story; show it through scene and dialogue\n"
        "- No title or section headings — output only the prose\n"
    ),
    "essays": (
        "IMPORTANT: You must respond ONLY in English. Do not use any other language.\n\n"
        "Write a well-argued analytical essay on the following topic:\n\n"
        "{topic}\n\n"
        "Requirements:\n"
        "- ENGLISH ONLY — any non-English output will be discarded\n"
        "- Approximately 480 to 560 words\n"
        "- Present a clear position and support it with specific reasoning and examples\n"
        "- Avoid bullet points and numbered lists — write continuous prose\n"
        "- No title or headings — output only the essay body\n"
    ),
    "news": (
        "IMPORTANT: You must respond ONLY in English. Do not use any other language.\n\n"
        "Write a factual English-language news article on the following event or development:\n\n"
        "{topic}\n\n"
        "Requirements:\n"
        "- ENGLISH ONLY — any non-English output will be discarded\n"
        "- Approximately 450 to 550 words\n"
        "- Follow inverted-pyramid structure: most important facts first\n"
        "- Include at least two attributed quotes from named sources\n"
        "- No headline or byline — output only the article body\n"
    ),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_english(text: str, threshold: float = 0.85) -> bool:
    """
    Fast heuristic: check fraction of ASCII printable characters.
    Filters out Arabic/Chinese/etc. responses from allam or other multilingual models.
    """
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) >= threshold


def word_count(text: str) -> int:
    return len(text.split())


def build_genre_list(n_total: int) -> list[str]:
    """Return a shuffled list of genres matching the target distribution."""
    counts = {g: round(n_total * w) for g, w in GENRE_WEIGHTS.items()}
    # Adjust rounding errors
    diff = n_total - sum(counts.values())
    counts["fiction"] += diff
    genres = []
    for g, c in counts.items():
        genres.extend([g] * c)
    random.shuffle(genres)
    return genres


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Core generation ───────────────────────────────────────────────────────────

def generate_one(client, model_id: str, genre: str, topic: str, max_retries: int = 4) -> str | None:
    prompt = PROMPTS[genre].format(topic=topic)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=1200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"    [retry {attempt+1}/{max_retries}] {e} — waiting {wait:.1f}s")
            time.sleep(wait)
    return None


def generate_for_model(
    client,
    model_id: str,
    short_name: str,
    n_total: int,
    seed: int,
) -> None:
    random.seed(seed)
    output_path = OUTPUT_DIR / f"{short_name}.jsonl"
    existing = load_existing(output_path)
    n_done = len(existing)

    if n_done >= n_total:
        print(f"  {short_name}: already complete ({n_done}/{n_total}) — skipping")
        return

    print(f"  {short_name}: generating {n_total - n_done} more (have {n_done}/{n_total})")

    genre_list = build_genre_list(n_total)
    # Skip already-done items (deterministic order)
    genre_list = genre_list[n_done:]

    used_topics = {g: set() for g in ALL_TOPICS}

    n_generated = 0
    n_skipped = 0

    for genre in genre_list:
        topic_pool = ALL_TOPICS[genre]
        available = [t for t in topic_pool if t not in used_topics[genre]]
        if not available:
            used_topics[genre] = set()  # cycle topics
            available = topic_pool
        topic = random.choice(available)
        used_topics[genre].add(topic)

        text = generate_one(client, model_id, genre, topic)
        if text is None:
            n_skipped += 1
            print(f"    [{n_done + n_generated + 1}/{n_total}] FAILED — skipping")
            continue

        if not is_english(text):
            n_skipped += 1
            print(f"    [{n_done + n_generated + 1}/{n_total}] non-English — skipping")
            continue

        wc = word_count(text)
        if wc < 150 or wc > 1300:
            n_skipped += 1
            print(f"    [{n_done + n_generated + 1}/{n_total}] word count {wc} out of range — skipping")
            continue

        record = {
            "id":    str(uuid.uuid4()),
            "text":  text,
            "label": 1,
            "model": short_name,
            "genre": genre,
        }
        append_record(output_path, record)
        n_generated += 1

        current = n_done + n_generated
        print(f"    [{current}/{n_total}] {genre} | {wc} words | topic: {topic[:50]}…")

        # Polite rate-limiting: ~1 request/sec average
        time.sleep(0.8)

    print(f"  {short_name}: done. Generated={n_generated}, skipped={n_skipped}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate AI texts via Groq (Track A)")
    p.add_argument("--n-per-model", type=int, default=500,
                   help="Total texts per model (default 500)")
    p.add_argument("--models", nargs="*",
                   help="Short names of models to run (default: all). "
                        "Options: llama-4-maverick llama-4-scout qwen3-32b kimi-k2 allam-2-7b")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] Set GROQ_API_KEY environment variable first.", file=sys.stderr)
        print("  export GROQ_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)

    try:
        from groq import Groq
    except ImportError:
        print("[ERROR] groq package not installed. Run: pip install groq", file=sys.stderr)
        sys.exit(1)

    client = Groq(api_key=api_key)

    model_filter = set(args.models) if args.models else None
    selected = [(mid, sn) for mid, sn in MODELS
                if model_filter is None or sn in model_filter]

    if not selected:
        print(f"[ERROR] No models matched: {args.models}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Models: {[sn for _, sn in selected]}")
    print(f"Texts per model: {args.n_per_model}")
    print()

    for i, (model_id, short_name) in enumerate(selected):
        print(f"[{i+1}/{len(selected)}] Model: {short_name} ({model_id})")
        generate_for_model(
            client=client,
            model_id=model_id,
            short_name=short_name,
            n_total=args.n_per_model,
            seed=args.seed,
        )
        print()

    print("All done.")


if __name__ == "__main__":
    main()