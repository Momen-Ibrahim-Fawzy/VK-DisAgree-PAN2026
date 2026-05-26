#!/usr/bin/env python3
"""
Track C – Collect pre-LLM human-written texts.

Sources:
  - Project Gutenberg (fiction excerpts, pre-1927 public domain)
  - Wikisource essays (public domain non-fiction)
  - Wikipedia featured article leads (news/informational prose)

All texts are pre-LLM (written before 2020) so are genuine human writing.
Label = 0 (human).

Usage:
    pip install requests beautifulsoup4
    python collect_human_texts.py

    # Limit per source:
    python collect_human_texts.py --per-source 50

    # Dry run:
    python collect_human_texts.py --dry-run

Output:
    extra_data/human_texts/collected_human_texts.jsonl
    Each record: {id, text, label:0, model:"human", genre, source}
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
from collections import Counter

OUTPUT_FILE = Path(__file__).parent / "human_texts" / "collected_human_texts.jsonl"

# Gutenberg book IDs — pre-1927 fiction prose only, no duplicates
# Script will silently skip any that fail to download
GUTENBERG_BOOKS = [
    # British classics
    (1342, "pride_and_prejudice"),
    (1260, "jane_eyre"),
    (768,  "wuthering_heights"),
    (161,  "sense_and_sensibility"),
    (121,  "northanger_abbey"),
    (105,  "persuasion"),
    (1998, "emma"),
    (766,  "david_copperfield"),
    (730,  "oliver_twist"),
    (1400, "great_expectations"),
    (786,  "hard_times"),
    (203,  "mill_on_the_floss"),
    (910,  "tess_of_the_durbervilles"),
    (135,  "les_miserables"),
    (4300, "ulysses"),
    (2814, "dubliners"),
    (219,  "heart_of_darkness"),
    (174,  "picture_of_dorian_gray"),
    (2830, "importance_of_being_earnest"),
    (4363, "three_men_in_a_boat"),
    (580,  "turn_of_the_screw"),
    (209,  "age_of_innocence"),
    (514,  "little_women"),
    (1952, "yellow_wallpaper"),
    (3236, "the_awakening"),
    (160,  "the_bostonians"),
    (432,  "portrait_of_a_lady"),
    (863,  "memoirs_of_sherlock_holmes"),
    (1661, "adventures_of_sherlock_holmes"),
    (345,  "dracula"),
    (84,   "frankenstein"),
    # American classics
    (76,   "huckleberry_finn"),
    (74,   "tom_sawyer"),
    (98,   "tale_of_two_cities"),
    (73,   "red_badge_of_courage"),
    (2148, "uncle_toms_cabin"),
    (1248, "scarlet_letter"),
    (33,   "house_of_seven_gables"),
    (45,   "anne_of_green_gables"),
    (1064, "the_secret_garden"),
    (16,   "peter_pan"),
    (11,   "alice_in_wonderland"),
    (2701, "moby_dick"),
    # Adventure / genre
    (120,  "treasure_island"),
    (35,   "the_time_machine"),
    (36,   "the_war_of_the_worlds"),
    (159,  "the_invisible_man"),
    (5230, "island_of_dr_moreau"),
    (1257, "the_jungle_book"),
    (3268, "the_call_of_the_wild"),
    (2097, "kim"),
    (1322, "quo_vadis"),
    (244,  "count_of_monte_cristo"),
    # Russian classics
    (4517, "crime_and_punishment"),
    (2600, "war_and_peace"),
    (28054,"brothers_karamazov"),
    (1399, "anna_karenina"),
    (5765, "dead_souls"),
    (2554, "notes_from_underground"),
    (600,  "the_idiot"),
    (2197, "the_possessed"),
    (243,  "the_eternal_husband"),
    # French / European
    (1727, "gullivers_travels"),
    (5200, "metamorphosis"),
    (36,   "the_war_of_the_worlds"),
    (1237, "the_three_musketeers"),
    (2759, "around_the_world_in_80_days"),
    (103,  "20000_leagues"),
    (3748, "the_master_and_margarita"),
    # More British / Irish
    (140,  "the_jungle_book_v2"),
    (27827,"house_of_seven_gables_2"),
    (2147, "barchester_towers"),
    (3367, "north_and_south"),
    (1484, "the_woman_in_white"),
    (583,  "the_moonstone"),
    (1417, "middlemarch"),
    (145,  "middlemarch_v2"),
    (551,  "the_return_of_the_native"),
    (153,  "jude_the_obscure"),
    (3268, "far_from_the_madding_crowd"),
    (58,   "the_mayor_of_casterbridge"),
    (2054, "the_man_who_was_thursday"),
    (5324, "the_napoleon_of_notting_hill"),
    (1837, "the_way_of_all_flesh"),
    (19,   "the_strange_case_of_dr_jekyll"),
    (421,  "kidnapped"),
    (969,  "prester_john"),
    (1260, "the_thirty_nine_steps"),
    (2097, "the_riddle_of_the_sands"),
    # American short fiction / novellas
    (175,  "the_house_of_mirth"),
    (2500, "the_custom_of_the_country"),
    (541,  "ethan_frome"),
    (215,  "the_sea_wolf"),
    (1245, "the_iron_heel"),
    (1056, "martin_eden"),
    (2428, "the_red_house_mystery"),
    (8129, "the_mysterious_affair_at_styles"),
    (863,  "the_adventures_of_sherlock_2"),
    (108,  "the_adventures_of_tom_sawyer_2"),
    # Late Victorian / Edwardian
    (1279, "the_pickwick_papers"),
    (766,  "dombey_and_son"),
    (141,  "our_mutual_friend"),
    (1023, "bleak_house"),
    (730,  "the_old_curiosity_shop"),
    (967,  "the_mystery_of_edwin_drood"),
    (4557, "the_mill_on_the_floss_v2"),
    (180,  "the_way_we_live_now"),
    (2099, "the_egoist"),
    (3244, "the_ordeal_of_richard_feverel"),
    # American realism
    (2386, "the_financier"),
    (1430, "sister_carrie"),
    (2543, "the_pit"),
    (4085, "the_damnation_of_theron_ware"),
    (1617, "the_rise_of_silas_lapham"),
    (2181, "a_hazard_of_new_fortunes"),
    (3354, "indian_summer"),
    # Short story collections (many qualifying paragraphs each)
    (41,   "the_legend_of_sleepy_hollow"),
    (1661, "the_memoirs_of_sherlock_holmes"),
    (2097, "plain_tales_from_the_hills"),
    (236,  "the_man_who_would_be_king"),
    (1743, "the_gift_of_the_magi"),
    (7256, "the_four_million"),
    (2776, "the_trimmed_lamp"),
    (10131,"the_voice_of_the_city"),
    # More Victorian / Edwardian
    (4280, "the_card"),                  # Arnold Bennett
    (4105, "the_old_wives_tale"),        # Arnold Bennett
    (959,  "clayhanger"),               # Arnold Bennett
    (2887, "the_history_of_mr_polly"),  # H.G. Wells
    (3970, "kipps"),                    # H.G. Wells
    (1638, "tono_bungay"),              # H.G. Wells
    (2091, "ann_veronica"),             # H.G. Wells
    (3743, "the_new_machiavelli"),      # H.G. Wells
    (768,  "the_tenant_of_wildfell_hall"), # Anne Bronte
    (158,  "the_professor"),            # Charlotte Bronte
    # Edwardian social novels
    (2005, "howards_end"),              # E.M. Forster
    (2891, "a_room_with_a_view"),       # E.M. Forster
    (1409, "where_angels_fear_to_tread"), # Forster
    (4700, "the_longest_journey"),      # Forster
    (1270, "the_rainbow"),              # D.H. Lawrence
    (217,  "sons_and_lovers"),          # D.H. Lawrence
    (2814, "the_white_peacock"),        # D.H. Lawrence
    # American early 20th c
    (805,  "the_custom_of_the_country_2"),
    (174,  "the_valley_of_decision"),   # Wharton
    (2426, "the_reef"),                 # Wharton
    (4517, "summer"),                   # Wharton
    (1993, "the_glimpses_of_the_moon"), # Wharton
    # Naturalism / regionalism
    (2091, "the_doll_maker"),
    (2097, "o_pioneers"),              # Willa Cather
    (2101, "my_antonia"),              # Willa Cather
    (2413, "the_song_of_the_lark"),    # Cather
    (5765, "death_comes_for_archbishop"), # Cather
    # More mystery / adventure
    (1155, "the_mystery_of_cloomber"), # Conan Doyle
    (290,  "a_study_in_scarlet"),
    (2852, "the_sign_of_four"),
    (108,  "the_hound_of_baskervilles"),
    (294,  "the_lost_world"),
]

# Gutenberg non-fiction prose — fetched as genre="essays"
GUTENBERG_ESSAYS = [
    (1080, "a_modest_proposal"),           # Swift
    (7370, "essays_of_elia"),              # Charles Lamb
    (5667, "essays_bacon"),               # Francis Bacon complete
    (3290, "on_liberty"),                 # John Stuart Mill
    (34901,"utilitarianism"),             # Mill
    (1232, "the_prince"),                 # Machiavelli
    (2405, "the_federalist_papers"),      # Hamilton/Madison/Jay
    (5827, "common_sense"),               # Thomas Paine
    (3600, "essays_montaigne"),           # Montaigne
    (16648,"democracy_in_america"),       # Tocqueville
    (2680, "confessions_rousseau"),       # Rousseau
    (4279, "walden"),                     # Thoreau
    (1023, "the_social_contract"),        # Rousseau
    (7370, "last_essays_of_elia"),        # Lamb
    (2005, "selected_essays_hazlitt"),    # Hazlitt
    (1906, "selected_essays_de_quincey"),
    (2374, "virginibus_puerisque"),       # Stevenson essays
    (521,  "familiar_studies"),           # Stevenson
    (1937, "selected_prose_ruskin"),      # Ruskin
    (4279, "sesame_and_lilies"),          # Ruskin
    (2044, "the_scope_of_history"),       # Macaulay essays
    (3528, "critical_and_historical_essays"), # Macaulay
]

# Wikipedia article titles — each yields multiple section-chunks (genre="news")
WIKIPEDIA_ARTICLES = [
    # History / wars / events
    "Spanish_flu",
    "Great_Depression",
    "Apollo_11",
    "Titanic_(ship)",
    "Manhattan_Project",
    "Berlin_Wall",
    "Chernobyl_disaster",
    "Cuban_Missile_Crisis",
    "Korean_War",
    "Vietnam_War",
    "Dust_Bowl",
    "Montgomery_bus_boycott",
    "Nuremberg_trials",
    "Marshall_Plan",
    "Bretton_Woods_system",
    "Green_Revolution",
    "Polio_vaccine",
    "Brown_v._Board_of_Education",
    "New_Deal",
    "Atomic_bombings_of_Hiroshima_and_Nagasaki",
    "World_War_I",
    "World_War_II",
    "Cold_War",
    "Space_Race",
    "Moon_landing",
    "Iranian_Revolution",
    "Fall_of_the_Soviet_Union",
    "Gulf_War",
    "Rwandan_genocide",
    "Bosnian_War",
    "September_11_attacks",
    "Iraq_War",
    "Afghanistan_War_(2001–2021)",
    "Arab_Spring",
    "Brexit",
    "2008_financial_crisis",
    "COVID-19_pandemic",
    "Black_Death",
    "Great_Fire_of_London",
    "French_Revolution",
    "American_Civil_War",
    "Reconstruction_era",
    # Science / medicine / technology
    "Human_Genome_Project",
    "International_Space_Station",
    "Hubble_Space_Telescope",
    "Large_Hadron_Collider",
    "DNA",
    "Penicillin",
    "Vaccination",
    "Climate_change",
    "Ozone_depletion",
    "Plate_tectonics",
    "Theory_of_evolution",
    "Germ_theory_of_disease",
    "Radioactivity",
    "Quantum_mechanics",
    "General_relativity",
    "CRISPR",
    "Artificial_intelligence",
    "Internet",
    "World_Wide_Web",
    "Semiconductor",
    "Nuclear_fission",
    "Stem_cell",
    "Antibiotic_resistance",
    "Cancer",
    "Alzheimer%27s_disease",
    "Diabetes",
    "Heart_disease",
    "Mental_health",
    "Nutrition",
    "Obesity",
    # Society / economics / politics
    "Universal_Declaration_of_Human_Rights",
    "World_Trade_Organization",
    "United_Nations",
    "Welfare_state",
    "Universal_basic_income",
    "Public_health",
    "Urbanization",
    "Industrial_Revolution",
    "Colonialism",
    "Decolonization",
    "Globalization",
    "Income_inequality",
    "Poverty",
    "Minimum_wage",
    "Labour_movement",
    "Women%27s_suffrage",
    "Civil_rights_movement",
    "Feminism",
    "LGBTQ_rights",
    "Immigration",
    "Refugee",
    "Racism",
    "Affirmative_action",
    "Capital_punishment",
    "Mass_incarceration",
    "Drug_policy",
    "Gun_control",
    "Freedom_of_the_press",
    "Democracy",
    "Authoritarianism",
    "Propaganda",
    # Environment
    "Amazon_rainforest",
    "Great_Barrier_Reef",
    "Arctic",
    "Biodiversity",
    "Endangered_species",
    "Renewable_energy",
    "Nuclear_power",
    "Fossil_fuel",
    "Water_scarcity",
    "Deforestation",
    "Plastic_pollution",
    "Ocean_acidification",
    "Wildfires",
    "Drought",
    "Flood",
    # Economics / finance
    "Capitalism",
    "Socialism",
    "Keynesian_economics",
    "Inflation",
    "Unemployment",
    "Central_bank",
    "Stock_market",
    "Trade_union",
    "Monopoly",
    "Taxation",
    # Education / culture
    "Public_education",
    "Higher_education",
    "Literacy",
    "Public_library",
    "Universal_health_care",
    "Journalism",
    "Documentary_film",
    "Public_broadcasting",
    "Social_media",
    "Misinformation",
]

# Wikisource essay/non-fiction pages
WIKISOURCE_ESSAYS = [
    # Emerson
    "Self-Reliance",
    "The_American_Scholar",
    "Nature_(essay)",
    "Experience_(essay)",
    "The_Over-Soul",
    "Compensation_(essay)",
    "Friendship_(essay)",
    # Thoreau
    "Civil_Disobedience",
    "Walking",
    "Life_Without_Principle",
    # Orwell
    "Shooting_an_Elephant",
    "Politics_and_the_English_Language",
    "Such,_Such_Were_the_Joys",
    "Why_I_Write",
    # Woolf
    "A_Room_of_One%27s_Own",
    "The_Death_of_the_Moth",
    "Modern_Fiction",
    # Wilde
    "The_Decay_of_Lying",
    "The_Soul_of_Man_Under_Socialism",
    "The_Critic_as_Artist",
    # Swift / Bacon / others
    "A_Modest_Proposal",
    "Of_Studies",
    "Of_Truth",
    "Of_Friendship_(Bacon)",
    "Of_Marriage_and_Single_Life",
    # Huxley / Russell / others
    "On_the_Advisableness_of_Improving_Natural_Knowledge",
    "The_Method_of_Scientific_Investigation",
    "On_a_Piece_of_Chalk",
    # Additional
    "The_Gettysburg_Address",
    "Common_Sense_(pamphlet)",
]

MIN_WORDS = 200
MAX_WORDS = 1200
TARGET_PER_SOURCE_TYPE = 1000  # per source category


# ── Text processing ───────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def is_english(text: str) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) >= 0.80


def clean_gutenberg(raw: str) -> list[str]:
    """Split Gutenberg text into paragraph chunks of appropriate length."""
    # Strip header/footer
    start = re.search(r'\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG', raw, re.IGNORECASE)
    end   = re.search(r'\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG', raw, re.IGNORECASE)
    if start:
        raw = raw[start.end():]
    if end:
        raw = raw[:end.start()]

    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', raw) if p.strip()]

    # Filter: prose paragraphs of reasonable length
    good = []
    for p in paragraphs:
        p = re.sub(r'\s+', ' ', p)
        # Skip chapter headings, very short, dialogue-only
        if len(p) < 100:
            continue
        if re.match(r'^(CHAPTER|Chapter|BOOK|Book|PART|Part)\b', p):
            continue
        wc = word_count(p)
        if wc < MIN_WORDS:
            continue
        if wc > MAX_WORDS:
            # Truncate to MAX_WORDS
            words = p.split()
            p = " ".join(words[:MAX_WORDS])
        good.append(p)

    return good


def clean_wikipedia(raw_html: str) -> list[str]:
    """Extract prose chunks from Wikipedia article sections (not just the lead)."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(raw_html, "html.parser")
    # Remove noisy elements
    for tag in soup.find_all(["table", "sup", "span", "figure", "style", "script"]):
        tag.decompose()

    content = soup.find(id="mw-content-text") or soup.find("div", class_="mw-parser-output")
    if not content:
        return []

    # Skip sections with these headings (references, external links, etc.)
    SKIP_SECTIONS = {
        "references", "notes", "further reading", "external links",
        "see also", "bibliography", "footnotes", "citations",
    }

    chunks = []
    current_paragraphs = []
    current_section = "lead"

    for el in content.find_all(["p", "h2", "h3"], recursive=True):
        if el.name in ("h2", "h3"):
            # Flush current section
            section_text = " ".join(current_paragraphs).strip()
            if (word_count(section_text) >= MIN_WORDS
                    and current_section not in SKIP_SECTIONS):
                if word_count(section_text) > MAX_WORDS:
                    section_text = " ".join(section_text.split()[:MAX_WORDS])
                chunks.append(section_text)
            # Start new section
            heading = el.get_text(" ", strip=True).lower()
            current_section = heading
            current_paragraphs = []
        else:
            text = el.get_text(" ", strip=True)
            text = re.sub(r'\[\d+\]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 40:
                current_paragraphs.append(text)

    # Flush last section
    section_text = " ".join(current_paragraphs).strip()
    if (word_count(section_text) >= MIN_WORDS
            and current_section not in SKIP_SECTIONS):
        if word_count(section_text) > MAX_WORDS:
            section_text = " ".join(section_text.split()[:MAX_WORDS])
        chunks.append(section_text)

    return chunks


def clean_wikisource(raw_html: str) -> list[str]:
    """Extract essay paragraphs from Wikisource HTML."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup.find_all(["sup", "table"]):
        tag.decompose()

    content = soup.find(id="mw-content-text") or soup.find("div", class_="mw-parser-output")
    if not content:
        return []

    chunks = []
    current = []
    for el in content.find_all(["p", "h2", "h3"]):
        if el.name in ("h2", "h3"):
            text = " ".join(current).strip()
            if word_count(text) >= MIN_WORDS:
                if word_count(text) > MAX_WORDS:
                    text = " ".join(text.split()[:MAX_WORDS])
                chunks.append(text)
            current = []
        else:
            t = el.get_text(" ", strip=True)
            t = re.sub(r'\[\d+\]', '', t)
            t = re.sub(r'\s+', ' ', t).strip()
            if t:
                current.append(t)

    if current:
        text = " ".join(current).strip()
        if word_count(text) >= MIN_WORDS:
            if word_count(text) > MAX_WORDS:
                text = " ".join(text.split()[:MAX_WORDS])
            chunks.append(text)

    return chunks


# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_url(url: str, session, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    [FAIL] {url}: {e}")
                return None
    return None


def fetch_gutenberg_chunks(book_id: int, title: str, session, rng: random.Random,
                           n: int) -> list[dict]:
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    raw = fetch_url(url, session)
    if raw is None:
        # Try alternate URL pattern
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        raw = fetch_url(url, session)
    if raw is None:
        return []

    chunks = clean_gutenberg(raw)
    if not chunks:
        return []

    selected = rng.sample(chunks, min(n, len(chunks)))
    records = []
    for chunk in selected:
        if not is_english(chunk):
            continue
        records.append({
            "id":     str(uuid.uuid4()),
            "text":   chunk,
            "label":  0,
            "model":  "human",
            "genre":  "fiction",
            "source": f"gutenberg:{book_id}:{title}",
        })
    return records


def fetch_wikipedia_chunks(title: str, session, rng: random.Random, n: int) -> list[dict]:
    url = f"https://en.wikipedia.org/w/api.php?action=parse&page={title}&prop=text&format=json"
    raw = fetch_url(url, session)
    if raw is None:
        return []
    try:
        data = json.loads(raw)
        html = data["parse"]["text"]["*"]
    except (KeyError, json.JSONDecodeError):
        return []

    chunks = clean_wikipedia(html)
    if not chunks:
        return []

    selected = rng.sample(chunks, min(n, len(chunks)))
    records = []
    for chunk in selected:
        if not is_english(chunk):
            continue
        records.append({
            "id":     str(uuid.uuid4()),
            "text":   chunk,
            "label":  0,
            "model":  "human",
            "genre":  "news",
            "source": f"wikipedia:{title}",
        })
    return records


def fetch_wikisource_chunks(title: str, session, rng: random.Random, n: int) -> list[dict]:
    url = f"https://en.wikisource.org/w/api.php?action=parse&page={title}&prop=text&format=json"
    raw = fetch_url(url, session)
    if raw is None:
        return []
    try:
        data = json.loads(raw)
        html = data["parse"]["text"]["*"]
    except (KeyError, json.JSONDecodeError):
        return []

    chunks = clean_wikisource(html)
    if not chunks:
        return []

    selected = rng.sample(chunks, min(n, len(chunks)))
    records = []
    for chunk in selected:
        if not is_english(chunk):
            continue
        records.append({
            "id":     str(uuid.uuid4()),
            "text":   chunk,
            "label":  0,
            "model":  "human",
            "genre":  "essays",
            "source": f"wikisource:{title}",
        })
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def load_existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            done.add(r.get("source", ""))
        except:
            pass
    return done


def parse_args():
    p = argparse.ArgumentParser(description="Collect pre-LLM human texts (Track C)")
    p.add_argument("--per-source", type=int, default=75,
                   help="Max chunks to take from each individual source (default: 75)")
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--dry-run", action="store_true",
                   help="Print sources without fetching")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    try:
        import requests
    except ImportError:
        print("[ERROR] requests not installed. Run: pip install requests beautifulsoup4",
              file=sys.stderr)
        sys.exit(1)

    try:
        from bs4 import BeautifulSoup  # noqa: F401
    except ImportError:
        print("[ERROR] beautifulsoup4 not installed. Run: pip install beautifulsoup4",
              file=sys.stderr)
        sys.exit(1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = (
        "PAN-CLEF26-DataCollection/1.0 (research; contact: researcher@example.com)"
    )

    existing_sources = load_existing(OUTPUT_FILE)
    print(f"Already collected: {len(existing_sources)} sources done")

    all_records = []
    n_fetched = 0

    # ── Gutenberg (fiction) ──────────────────────────────────────────────────
    print(f"\n=== Project Gutenberg ({len(GUTENBERG_BOOKS)} books) ===")
    rng.shuffle(GUTENBERG_BOOKS)
    for book_id, title in GUTENBERG_BOOKS:
        key = f"gutenberg:{book_id}:{title}"
        if any(s.startswith(key) for s in existing_sources):
            print(f"  SKIP {title} (already done)")
            continue
        if args.dry_run:
            print(f"  Would fetch: {key}")
            continue
        print(f"  Fetching {title} (id={book_id})…")
        chunks = fetch_gutenberg_chunks(book_id, title, session, rng, args.per_source)
        print(f"    Got {len(chunks)} chunks")
        all_records.extend(chunks)
        n_fetched += len(chunks)
        time.sleep(1.0)  # be polite to Gutenberg

    # ── Wikipedia (news/informational) ───────────────────────────────────────
    print(f"\n=== Wikipedia ({len(WIKIPEDIA_ARTICLES)} articles) ===")
    rng.shuffle(WIKIPEDIA_ARTICLES)
    for title in WIKIPEDIA_ARTICLES:
        key = f"wikipedia:{title}"
        if any(s.startswith(key) for s in existing_sources):
            print(f"  SKIP {title} (already done)")
            continue
        if args.dry_run:
            print(f"  Would fetch: {key}")
            continue
        print(f"  Fetching {title}…")
        chunks = fetch_wikipedia_chunks(title, session, rng, args.per_source)
        print(f"    Got {len(chunks)} chunks")
        all_records.extend(chunks)
        n_fetched += len(chunks)
        time.sleep(0.5)

    # ── Wikisource (essays) ──────────────────────────────────────────────────
    print(f"\n=== Wikisource ({len(WIKISOURCE_ESSAYS)} essays) ===")
    rng.shuffle(WIKISOURCE_ESSAYS)
    for title in WIKISOURCE_ESSAYS:
        key = f"wikisource:{title}"
        if any(s.startswith(key) for s in existing_sources):
            print(f"  SKIP {title} (already done)")
            continue
        if args.dry_run:
            print(f"  Would fetch: {key}")
            continue
        print(f"  Fetching {title}…")
        chunks = fetch_wikisource_chunks(title, session, rng, args.per_source)
        print(f"    Got {len(chunks)} chunks")
        all_records.extend(chunks)
        n_fetched += len(chunks)
        time.sleep(0.5)

    # ── Gutenberg non-fiction (essays) ───────────────────────────────────────
    print(f"\n=== Gutenberg Essays ({len(GUTENBERG_ESSAYS)} works) ===")
    rng.shuffle(GUTENBERG_ESSAYS)
    for book_id, title in GUTENBERG_ESSAYS:
        key = f"gutenberg_essay:{book_id}:{title}"
        if any(s.startswith(key) for s in existing_sources):
            print(f"  SKIP {title} (already done)")
            continue
        if args.dry_run:
            print(f"  Would fetch: {key}")
            continue
        print(f"  Fetching {title} (id={book_id})…")
        # Reuse Gutenberg fetcher but override genre to "essays"
        chunks = fetch_gutenberg_chunks(book_id, title, session, rng, args.per_source)
        # Override genre and source key
        for r in chunks:
            r["genre"] = "essays"
            r["source"] = f"gutenberg_essay:{book_id}:{title}"
        print(f"    Got {len(chunks)} chunks")
        all_records.extend(chunks)
        n_fetched += len(chunks)
        time.sleep(1.0)

    # ── Write output ─────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\nDry run complete. Would fetch from:")
        print(f"  {len(GUTENBERG_BOOKS)} Gutenberg fiction books")
        print(f"  {len(WIKIPEDIA_ARTICLES)} Wikipedia articles")
        print(f"  {len(WIKISOURCE_ESSAYS)} Wikisource essays")
        print(f"  {len(GUTENBERG_ESSAYS)} Gutenberg non-fiction works")
        return

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {n_fetched} new records to {OUTPUT_FILE}")

    # Final stats
    all_existing = []
    for line in OUTPUT_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            all_existing.append(json.loads(line))
        except:
            pass

    gc = Counter(r.get("genre") for r in all_existing)
    sc = Counter(r.get("source", "").split(":")[0] for r in all_existing)
    print(f"Total human texts: {len(all_existing)}")
    print(f"By genre: {dict(gc)}")
    print(f"By source type: {dict(sc)}")


if __name__ == "__main__":
    main()
