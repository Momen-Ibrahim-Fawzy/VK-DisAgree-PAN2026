"""
Stylometric and statistical feature extraction for AI text detection.

Features are designed to be robust to paraphrasing and style obfuscation,
capturing deep structural and statistical properties of text.
"""
import math
import re
import string
from collections import Counter
from typing import Optional

import numpy as np

# Try to import NLTK sentence tokenizer; fall back to regex
try:
    import nltk
    nltk.data.find("tokenizers/punkt_tab")
    _USE_NLTK = True
except (ImportError, LookupError):
    try:
        import nltk
        nltk.download("punkt_tab", quiet=True)
        _USE_NLTK = True
    except Exception:
        _USE_NLTK = False


def _sent_tokenize(text: str) -> list[str]:
    if _USE_NLTK:
        from nltk import sent_tokenize
        return sent_tokenize(text)
    # Fallback: split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _word_tokenize(text: str) -> list[str]:
    """Simple word tokenizer: lowercase alpha tokens."""
    return re.findall(r"\b[a-zA-Z']+\b", text)


def _char_entropy(text: str) -> float:
    """Shannon entropy over character frequencies."""
    counts = Counter(text)
    total = len(text)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _word_entropy(words: list[str]) -> float:
    """Shannon entropy over word frequencies."""
    if not words:
        return 0.0
    counts = Counter(w.lower() for w in words)
    total = len(words)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


# Common transition words used heavily in AI-generated text
_TRANSITION_WORDS = {
    "furthermore", "moreover", "additionally", "however", "therefore",
    "consequently", "nevertheless", "nonetheless", "subsequently", "accordingly",
    "thereby", "hence", "thus", "whereas", "conversely", "notwithstanding",
    "specifically", "notably", "importantly", "significantly", "essentially",
    "ultimately", "generally", "typically", "particularly", "primarily",
    "approximately", "effectively", "clearly", "obviously", "certainly",
    "literally", "basically", "simply", "overall", "overall",
}

# English contractions (more common in human text)
_CONTRACTIONS = re.compile(
    r"\b(don't|can't|won't|isn't|aren't|wasn't|weren't|haven't|hasn't"
    r"|hadn't|doesn't|didn't|wouldn't|shouldn't|couldn't|i'm|i've|i'll"
    r"|i'd|you're|you've|you'll|you'd|he's|she's|it's|we're|we've|we'll"
    r"|we'd|they're|they've|they'll|they'd|that's|who's|what's|there's"
    r"|here's|let's|how's|'s|n't)\b",
    re.IGNORECASE,
)


def extract_features(text: str) -> np.ndarray:
    """
    Extract stylometric features from a single text.

    Returns a 1D numpy array of floats.
    """
    features = {}

    # ── Basic length stats ──────────────────────────────────────────────
    char_count = len(text)
    features["char_count"] = char_count

    words = _word_tokenize(text)
    word_count = len(words)
    features["word_count"] = word_count

    sentences = _sent_tokenize(text)
    sent_count = max(len(sentences), 1)
    features["sent_count"] = sent_count

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    para_count = max(len(paragraphs), 1)
    features["para_count"] = para_count

    # ── Word length stats ───────────────────────────────────────────────
    word_lengths = [len(w) for w in words] if words else [0]
    features["avg_word_len"] = np.mean(word_lengths)
    features["std_word_len"] = np.std(word_lengths)
    features["min_word_len"] = np.min(word_lengths)
    features["max_word_len"] = np.max(word_lengths)
    features["short_word_ratio"] = sum(1 for l in word_lengths if l < 4) / max(word_count, 1)
    features["long_word_ratio"] = sum(1 for l in word_lengths if l > 8) / max(word_count, 1)

    # ── Sentence length stats ───────────────────────────────────────────
    sent_lengths = [len(_word_tokenize(s)) for s in sentences]
    sent_lengths = sent_lengths if sent_lengths else [0]
    features["avg_sent_len"] = np.mean(sent_lengths)
    features["std_sent_len"] = np.std(sent_lengths)
    features["min_sent_len"] = np.min(sent_lengths)
    features["max_sent_len"] = np.max(sent_lengths)
    features["sent_len_cv"] = (  # Coefficient of variation: high in human text
        np.std(sent_lengths) / np.mean(sent_lengths) if np.mean(sent_lengths) > 0 else 0
    )

    # ── Vocabulary richness ─────────────────────────────────────────────
    lower_words = [w.lower() for w in words]
    unique_words = set(lower_words)
    vocab_size = len(unique_words)
    features["vocab_size"] = vocab_size
    features["type_token_ratio"] = vocab_size / max(word_count, 1)

    word_freq = Counter(lower_words)
    hapax = sum(1 for c in word_freq.values() if c == 1)
    features["hapax_ratio"] = hapax / max(word_count, 1)
    features["dis_legomena_ratio"] = (
        sum(1 for c in word_freq.values() if c == 2) / max(word_count, 1)
    )

    # ── Punctuation ─────────────────────────────────────────────────────
    features["comma_per_sent"] = text.count(",") / sent_count
    features["semicolon_per_sent"] = text.count(";") / sent_count
    features["colon_per_sent"] = text.count(":") / sent_count
    features["exclamation_density"] = text.count("!") / max(char_count, 1)
    features["question_density"] = text.count("?") / max(char_count, 1)
    features["ellipsis_density"] = text.count("...") / max(char_count, 1)
    features["dash_density"] = (text.count("—") + text.count("–") + text.count(" - ")) / max(char_count, 1)
    features["quote_density"] = (text.count('"') + text.count("'") + text.count('“') + text.count('”')) / max(char_count, 1)
    features["paren_density"] = (text.count("(") + text.count(")")) / max(char_count, 1)
    features["punct_total_density"] = sum(1 for c in text if c in string.punctuation) / max(char_count, 1)

    # ── Character ratios ────────────────────────────────────────────────
    features["uppercase_ratio"] = sum(1 for c in text if c.isupper()) / max(char_count, 1)
    features["digit_ratio"] = sum(1 for c in text if c.isdigit()) / max(char_count, 1)
    features["alpha_ratio"] = sum(1 for c in text if c.isalpha()) / max(char_count, 1)
    features["space_ratio"] = sum(1 for c in text if c.isspace()) / max(char_count, 1)

    # ── Entropy measures ────────────────────────────────────────────────
    features["char_entropy"] = _char_entropy(text)
    features["word_entropy"] = _word_entropy(words)

    # Sentence-level entropy (entropy over sentence length distribution)
    if len(sent_lengths) > 1:
        hist, _ = np.histogram(sent_lengths, bins=10)
        hist = hist[hist > 0].astype(float)
        hist /= hist.sum()
        features["sent_len_entropy"] = -np.sum(hist * np.log2(hist))
    else:
        features["sent_len_entropy"] = 0.0

    # ── Lexical features ────────────────────────────────────────────────
    features["transition_word_ratio"] = (
        sum(1 for w in lower_words if w in _TRANSITION_WORDS) / max(word_count, 1)
    )
    features["contraction_count"] = len(_CONTRACTIONS.findall(text)) / max(word_count, 1)

    # Numeric words ratio
    numeric_words = sum(1 for w in words if any(c.isdigit() for c in w))
    features["numeric_word_ratio"] = numeric_words / max(word_count, 1)

    # Average word frequency rank proxy: rare words ratio
    # (words > 8 chars tend to be rare/formal)
    features["rare_word_proxy"] = sum(1 for w in lower_words if len(w) > 8) / max(word_count, 1)

    # ── Repetition / redundancy ─────────────────────────────────────────
    # Bigram diversity: ratio of unique bigrams to total bigrams
    bigrams = [(lower_words[i], lower_words[i + 1]) for i in range(len(lower_words) - 1)]
    if bigrams:
        features["bigram_diversity"] = len(set(bigrams)) / len(bigrams)
    else:
        features["bigram_diversity"] = 0.0

    # Avg repetitions per unique word
    if word_freq:
        features["avg_word_repetition"] = np.mean(list(word_freq.values()))
    else:
        features["avg_word_repetition"] = 0.0

    # ── Text structure ──────────────────────────────────────────────────
    features["avg_para_len"] = word_count / para_count
    features["words_per_sent"] = word_count / sent_count
    features["chars_per_sent"] = char_count / sent_count
    features["chars_per_word"] = char_count / max(word_count, 1)

    # Sentence variety: std of sentence lengths normalized by mean
    if len(sent_lengths) > 1:
        features["sent_variety"] = np.std(sent_lengths) / max(np.mean(sent_lengths), 1)
    else:
        features["sent_variety"] = 0.0

    # ── Readability proxies ─────────────────────────────────────────────
    # Simplified Flesch-Kincaid grade level
    syllable_count = sum(_approx_syllables(w) for w in words)
    features["syllables_per_word"] = syllable_count / max(word_count, 1)
    features["syllables_per_sent"] = syllable_count / sent_count

    # Gunning Fog index proxy: (avg_sent_len + % complex words) * 0.4
    complex_words = sum(1 for w in words if _approx_syllables(w) >= 3)
    features["complex_word_ratio"] = complex_words / max(word_count, 1)
    features["gunning_fog_proxy"] = 0.4 * (
        features["avg_sent_len"] + 100 * features["complex_word_ratio"]
    )

    # ── First-person pronoun usage ──────────────────────────────────────
    first_person = re.findall(r"\b(i|me|my|myself|mine|we|us|our|ours|ourselves)\b", text.lower())
    features["first_person_ratio"] = len(first_person) / max(word_count, 1)

    third_person = re.findall(r"\b(he|she|they|him|her|them|his|hers|their|theirs)\b", text.lower())
    features["third_person_ratio"] = len(third_person) / max(word_count, 1)

    # ── Semantic sentence coherence ─────────────────────────────────────
    # LLM-generated texts exhibit lower variance in local semantic similarity
    # between adjacent sentences — each sentence stays on-topic with smooth
    # transitions. Human text naturally drifts: digressions, personal tangents,
    # and topic shifts create higher variance in consecutive-sentence similarity.
    coherence_feats = _sentence_coherence(sentences)
    features["coherence_mean"] = coherence_feats[0]   # High = AI
    features["coherence_std"] = coherence_feats[1]    # High = human
    features["coherence_min"] = coherence_feats[2]    # Low = human (big topic shift)
    features["coherence_cv"] = coherence_feats[3]     # Coefficient of variation

    # Return as numpy array (sorted by key for consistency)
    return np.array([features[k] for k in sorted(features.keys())], dtype=np.float32)


# ─── Module-level SBERT model cache ──────────────────────────────────────────
# Loaded once per process to avoid reloading for every text
_sbert_model = None
# Docker runtime path: /opt/pan26_detector/models/sbert (all-MiniLM-L6-v2)
# For local use, set to the HuggingFace model name and it will auto-download.
_SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_sbert():
    """Lazily load SBERT model; falls back to None if not available."""
    global _sbert_model
    if _sbert_model is not None:
        return _sbert_model
    try:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(_SBERT_MODEL_NAME)
        return _sbert_model
    except ImportError:
        return None


def _sentence_coherence(sentences: list[str]) -> tuple[float, float, float, float]:
    """
    Compute semantic coherence between consecutive sentences.

    Tries two approaches in order:
    1. SBERT embeddings (preferred) — captures true semantic similarity.
    2. TF-IDF cosine similarity (fallback) — vocabulary overlap as a proxy.

    Returns: (mean_sim, std_sim, min_sim, coeff_of_variation)
      - High mean_sim → consistent/focused → more AI-like
      - High std_sim  → variable coherence → more human-like
      - Low min_sim   → at least one abrupt topic shift → more human-like
    """
    if len(sentences) < 3:
        return 0.5, 0.0, 0.5, 0.0

    # ── Try SBERT first ───────────────────────────────────────────────────────
    sbert = _get_sbert()
    if sbert is not None:
        try:
            embeddings = sbert.encode(sentences, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False)
            # Consecutive cosine similarities (embeddings are L2-normalized)
            sims = [float(embeddings[i] @ embeddings[i + 1])
                    for i in range(len(embeddings) - 1)]
            sims_arr = np.array(sims)
            mean_s = float(np.mean(sims_arr))
            std_s = float(np.std(sims_arr))
            min_s = float(np.min(sims_arr))
            cv = std_s / max(mean_s, 1e-6)
            return mean_s, std_s, min_s, cv
        except Exception:
            pass  # Fall through to TF-IDF

    # ── TF-IDF fallback ───────────────────────────────────────────────────────
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
        X = vect.fit_transform(sentences)
        sims = [float(cos_sim(X[i], X[i + 1])[0, 0]) for i in range(len(sentences) - 1)]
        sims_arr = np.array(sims)
        mean_s, std_s = float(np.mean(sims_arr)), float(np.std(sims_arr))
        min_s = float(np.min(sims_arr))
        cv = std_s / max(mean_s, 1e-6)
        return mean_s, std_s, min_s, cv
    except Exception:
        return 0.5, 0.0, 0.5, 0.0


def _approx_syllables(word: str) -> int:
    """Approximate syllable count using vowel-group heuristic."""
    word = word.lower().strip(".,!?;:")
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Subtract silent 'e' at end
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def get_feature_names() -> list[str]:
    """Return sorted list of feature names (for debugging)."""
    return sorted([
        "alpha_ratio", "avg_para_len", "avg_sent_len", "avg_word_len",
        "avg_word_repetition", "bigram_diversity", "char_count", "char_entropy",
        "chars_per_sent", "chars_per_word", "colon_per_sent", "comma_per_sent",
        "complex_word_ratio", "contraction_count", "dash_density", "digit_ratio",
        "dis_legomena_ratio", "ellipsis_density", "exclamation_density",
        "first_person_ratio", "gunning_fog_proxy", "hapax_ratio", "long_word_ratio",
        "max_sent_len", "max_word_len", "min_sent_len", "min_word_len",
        "numeric_word_ratio", "para_count", "paren_density", "punct_total_density",
        "question_density", "quote_density", "rare_word_proxy", "sent_count",
        "sent_len_cv", "sent_len_entropy", "sent_variety", "semicolon_per_sent",
        "short_word_ratio", "space_ratio", "std_sent_len", "std_word_len",
        "syllables_per_sent", "syllables_per_word", "third_person_ratio",
        "transition_word_ratio", "type_token_ratio", "uppercase_ratio",
        "vocab_size", "word_count", "word_entropy", "words_per_sent",
        "coherence_mean", "coherence_std", "coherence_min", "coherence_cv",
    ])


def batch_extract_features(texts: list[str], verbose: bool = False) -> np.ndarray:
    """Extract features from a list of texts, returning a 2D array."""
    from tqdm import tqdm
    iterator = tqdm(texts, desc="Extracting features") if verbose else texts
    feature_list = [extract_features(t) for t in iterator]
    return np.vstack(feature_list)