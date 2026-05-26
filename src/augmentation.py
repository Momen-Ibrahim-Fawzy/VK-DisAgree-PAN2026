"""
Adversarial data augmentation for robust training.

The PAN test set contains texts obfuscated with:
  - LLM-based paraphrasing (DIPPER): semantics preserved, all words changed
  - Style transfer ("write like a 7-year-old"): grammar simplified
  - Noise injection: random tokens inserted mid-generation
  - Word order: subject-object-verb reordering
  - Temperature scaling: stochastic, less predictable output

We use TWO tiers of augmentation:

TIER 1 — Lightweight (EDA-style, always active):
  Five word/character-level perturbations requiring no external models.
  Fast, can augment the full training set in minutes.

TIER 2 — Semantic (T5-based paraphrasing, optional):
  Uses t5-small to paraphrase sentences, preserving meaning while changing
  words and sentence structure. This directly simulates DIPPER-style attacks,
  which EDA-level augmentation cannot replicate.

  A text paraphrased by T5 still has the same label (human/AI) but looks
  different on the surface. Training on these examples teaches the model to
  distinguish based on *meaning* and *structure*, not vocabulary.

  Requires `transformers` (already installed). Falls back gracefully to
  Tier 1 only if T5 download fails.

References:
  - Wei & Zou (2019), "EDA: Easy Data Augmentation"
  - Krishnan et al. (2021), "DIPPER: Discourse Paraphrasing"
  - Morris et al. (2020), "TextAttack"
"""
import random
import re
from typing import Callable


# ─── Individual augmentation functions ───────────────────────────────────────

def random_deletion(text: str, p: float = 0.10) -> str:
    """Delete each word with probability p. Keeps at least 10 words."""
    words = text.split()
    if len(words) <= 10:
        return text
    new_words = [w for w in words if random.random() > p]
    if len(new_words) < 10:
        new_words = words  # Revert if too many deleted
    return " ".join(new_words)


def random_swap(text: str, n_swaps: int = None) -> str:
    """Randomly swap adjacent word pairs n times."""
    words = text.split()
    if len(words) < 4:
        return text
    if n_swaps is None:
        n_swaps = max(1, len(words) // 20)  # ~5% of words
    for _ in range(n_swaps):
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return " ".join(words)


def sentence_shuffle(text: str, p_shuffle: float = 0.5) -> str:
    """
    Shuffle sentences within each paragraph with probability p_shuffle.
    This disrupts the sequential coherence that ModernBERT might rely on.
    """
    if random.random() > p_shuffle:
        return text
    paragraphs = text.split("\n\n")
    shuffled = []
    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", para.strip())
        if len(sentences) > 2:
            random.shuffle(sentences)
        shuffled.append(" ".join(sentences))
    return "\n\n".join(shuffled)


def character_noise(text: str, p: float = 0.02) -> str:
    """
    Inject random character-level noise (deletion, insertion, swap).
    Probability p applies per character. Simulates keyboard errors.
    """
    CHARS = "abcdefghijklmnopqrstuvwxyz "
    result = []
    for ch in text:
        if random.random() < p:
            noise_type = random.randint(0, 2)
            if noise_type == 0:
                pass  # Delete char
            elif noise_type == 1:
                result.append(random.choice(CHARS))  # Replace char
                result.append(ch)
            else:
                result.append(ch)
                result.append(random.choice(CHARS))  # Insert extra char
        else:
            result.append(ch)
    return "".join(result)


def punctuation_variation(text: str) -> str:
    """
    Randomly vary punctuation style: remove/add Oxford commas,
    change double-spaces to single, etc.
    These are common differences between human and AI text styles.
    """
    # Randomly collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    # With 30% chance, remove serial (Oxford) comma
    if random.random() < 0.3:
        text = re.sub(r",(\s+and\s)", r"\1", text)
    # With 30% chance, normalize em-dash to hyphen
    if random.random() < 0.3:
        text = text.replace("—", " - ").replace("–", " - ")
    return text


# ─── Augmentation pipeline ────────────────────────────────────────────────────

# Each augmentation is (function, probability_of_applying)
_AUGMENTATIONS: list[tuple[Callable, float]] = [
    (random_deletion, 0.5),
    (random_swap, 0.4),
    (sentence_shuffle, 0.3),
    (character_noise, 0.3),
    (punctuation_variation, 0.4),
]


def augment_text(text: str, n_augmentations: int = 2) -> str:
    """
    Apply n_augmentations randomly chosen transforms to a text.

    Each augmentation is applied independently with its own probability.
    Returns the (possibly modified) text.
    """
    result = text
    # Shuffle augmentation order for diversity
    augs = _AUGMENTATIONS.copy()
    random.shuffle(augs)
    applied = 0
    for func, prob in augs:
        if applied >= n_augmentations:
            break
        if random.random() < prob:
            result = func(result)
            applied += 1
    return result


def augment_dataset(
    texts: list[str],
    labels: list[int],
    aug_fraction: float = 0.5,
    n_augmentations: int = 2,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """
    Augment a dataset by creating additional examples from a random subset.

    For each selected example, we create one augmented copy and append it.
    The original examples are always kept intact.

    Args:
        texts: Original texts
        labels: Original labels
        aug_fraction: Fraction of examples to augment (0.5 = augment 50%)
        n_augmentations: Number of augmentation operations per text
        seed: Random seed for reproducibility

    Returns:
        (augmented_texts, augmented_labels) — originals + new examples
    """
    random.seed(seed)
    new_texts = list(texts)
    new_labels = list(labels)

    n_to_augment = int(len(texts) * aug_fraction)
    indices = random.sample(range(len(texts)), n_to_augment)

    print(f"Augmenting {n_to_augment} examples ({aug_fraction*100:.0f}% of dataset) [Tier 1 EDA]...")
    for idx in indices:
        aug = augment_text(texts[idx], n_augmentations=n_augmentations)
        new_texts.append(aug)
        new_labels.append(labels[idx])

    print(f"  Dataset size: {len(texts)} → {len(new_texts)}")
    return new_texts, new_labels


# ─── Tier 2: T5 semantic paraphrase augmentation ─────────────────────────────

# Module-level T5 cache
_t5_model = None
_t5_tokenizer = None
_T5_MODEL_NAME = "t5-small"  # 242MB; good quality paraphrasing at low cost


def _load_t5():
    """Lazily load T5-small for paraphrasing. Returns (model, tokenizer) or (None, None)."""
    global _t5_model, _t5_tokenizer
    if _t5_model is not None:
        return _t5_model, _t5_tokenizer
    try:
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        print(f"Loading T5-small for paraphrase augmentation...")
        _t5_tokenizer = AutoTokenizer.from_pretrained(_T5_MODEL_NAME)
        _t5_model = T5ForConditionalGeneration.from_pretrained(_T5_MODEL_NAME)
        _t5_model.eval()
        return _t5_model, _t5_tokenizer
    except Exception as e:
        print(f"  T5 not available ({e}), falling back to EDA-only augmentation")
        return None, None


def t5_paraphrase(text: str, n_beams: int = 4, max_input_len: int = 256) -> str:
    """
    Paraphrase a text using T5-small with diverse beam search.

    The prompt format "paraphrase: <text>" causes T5 to generate a semantically
    equivalent but structurally different version. This simulates DIPPER-style
    obfuscation where every surface token changes but meaning is preserved.

    Falls back to the original text if T5 is unavailable or fails.
    """
    model, tokenizer = _load_t5()
    if model is None:
        return text

    try:
        import torch
        # Truncate to avoid OOM on very long texts
        words = text.split()
        if len(words) > 200:
            # Paraphrase first 200 words, keep the rest unchanged
            truncated = " ".join(words[:200])
            remainder = " ".join(words[200:])
        else:
            truncated = text
            remainder = ""

        prompt = f"paraphrase: {truncated}"
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=max_input_len,
            truncation=True, padding=False
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=len(inputs.input_ids[0]) + 20,
                num_beams=n_beams,
                num_beam_groups=n_beams,
                diversity_penalty=1.0,   # Encourages diverse paraphrases
                num_return_sequences=1,
                early_stopping=True,
            )
        paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return (paraphrase + " " + remainder).strip() if remainder else paraphrase
    except Exception:
        return text


def augment_dataset_with_paraphrase(
    texts: list[str],
    labels: list[int],
    paraphrase_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """
    Tier 2 augmentation: paraphrase a subset of texts using T5-small.

    This targets a different failure mode than EDA: EDA teaches robustness
    to word deletion and noise; T5 paraphrasing teaches robustness to
    semantic-preserving rewriting (the hardest type of obfuscation).

    Only applies to a smaller fraction (25%) since T5 is slower than EDA.
    Combined with augment_dataset (Tier 1), the total augmentation creates:
      - EDA variants (noise/deletion/shuffle) — 50% of original
      - T5 paraphrases (semantic rewriting)   — 25% of original
      → Final training set = ~1.75× original size
    """
    from tqdm import tqdm

    random.seed(seed)
    model, _ = _load_t5()
    if model is None:
        print("T5 unavailable — skipping Tier 2 paraphrase augmentation")
        return texts, labels

    n_to_paraphrase = int(len(texts) * paraphrase_fraction)
    indices = random.sample(range(len(texts)), n_to_paraphrase)

    print(f"T5 paraphrase augmentation: {n_to_paraphrase} examples ({paraphrase_fraction*100:.0f}%)...")
    new_texts = list(texts)
    new_labels = list(labels)

    for idx in tqdm(indices, desc="T5 paraphrasing"):
        para = t5_paraphrase(texts[idx])
        new_texts.append(para)
        new_labels.append(labels[idx])

    print(f"  Dataset size: {len(texts)} → {len(new_texts)}")
    return new_texts, new_labels