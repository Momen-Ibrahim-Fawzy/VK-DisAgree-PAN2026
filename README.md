# VK-DisAgree: Disagreement-Aware Ensemble for Robust AI Text Detection

Official implementation of the VK-DisAgree system submitted to [PAN 2026 Voight-Kampff Generative AI Detection](https://pan.webis.de/clef26/pan26-web/generated-content-analysis.html) task at CLEF 2026.

Paper: *"When Detectors Disagree --- Disagreement-Aware Ensemble for Robust AI Text Detection"*

## System Overview

The system combines four complementary detectors whose **disagreement** is used as an explicit detection signal:

| Component | Model | Role |
|-----------|-------|------|
| ModernBERT | `answerdotai/ModernBERT-base` | Contextual transformer (primary) |
| TF-IDF | char 2–6-grams + word 1–3-grams, LogisticRegression | Lexical surface prior |
| Stylometric | 54 hand-crafted features + LightGBM | Surface-independent structure |
| Binoculars | Pythia-1B / Pythia-70M cross-perplexity ratio | Zero-shot generation signal |

These are combined by a **disagreement-aware meta-learner** that uses:

```
f = [p1, p2, p3, p4, var(p), range(p)]
```

where `var(p)` and `range(p)` are the inter-component variance and range. High disagreement under adversarial obfuscation is the central detection signal.

The output is a soft score in [0, 1]:
- `> 0.5` → AI-generated
- `< 0.5` → human-written  
- `= 0.5` → undecidable (abstained)

## Repository Structure

```
.
├── src/
│   ├── config.py              # All hyperparameters and paths
│   ├── data.py                # JSONL data loading utilities
│   ├── augmentation.py        # Two-tier adversarial augmentation (EDA + T5)
│   ├── features.py            # 54 stylometric features + SBERT coherence
│   ├── classical_models.py    # TF-IDF and LightGBM detectors
│   ├── perplexity_model.py    # Binoculars cross-model detector
│   ├── ensemble.py            # Disagreement-aware ensemble + PAN metrics
│   ├── transformer_model.py   # ModernBERT fine-tuning
│   └── training_logger.py     # Structured training progress logging
├── extra_data/                # Scripts to collect supplementary training data
│   ├── collect_ai_texts.py    # Track A: new AI models via Groq
│   ├── collect_obfuscations.py # Track B: LLM-paraphrased AI texts
│   ├── collect_human_texts.py  # Track C: pre-LLM human texts
│   ├── generate_claude_texts.py # Claude texts via Anthropic API
│   ├── merge_extra_data.py    # Combine all tracks
│   └── README.md              # Extra data collection guide
├── data/                      # PAN data goes here (not included — see below)
├── models/                    # Trained models go here (not included)
├── train.py                   # Full training pipeline
├── predict.py                 # Inference CLI (used by TIRA Docker)
├── prepare_data.py            # Data preparation (dedup + oversample)
├── Dockerfile                 # TIRA-compatible Docker image
├── requirements.txt           # Python dependencies
└── pyproject.toml             # Package metadata
```

## Setup

### Requirements

- Python 3.10+
- PyTorch with CUDA (for ModernBERT training)
- See `requirements.txt` for all dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
```

### Data

This repository does **not** include PAN competition data (not publicly redistributable). Download the official data from the [PAN 2026 task page](https://pan.webis.de/clef26/pan26-web/generative-authorship-verification.html) and place it as:

```
data/train.jsonl
data/val.jsonl
```

Each record must have the schema:
```json
{"id": "...", "text": "...", "label": 0_or_1}
```

### Models

Trained model weights are not included (too large for git). Download or train:

- **ModernBERT-base**: auto-downloaded from HuggingFace during training (`answerdotai/ModernBERT-base`)
- **Pythia-1B / Pythia-70M**: auto-downloaded from HuggingFace (`EleutherAI/pythia-1b`, `EleutherAI/pythia-70m`)
- **SBERT**: auto-downloaded from HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)

## Training

### Step 0: Prepare data

```bash
python prepare_data.py --train data/train.jsonl
```

This deduplicates and oversamples hard-to-detect models. Output: `data/train_prepared.jsonl`.

### Step 1: Train all components

```bash
python train.py --val data/val.jsonl
```

Full options:

```
--skip-transformer     Skip ModernBERT fine-tuning (loads existing checkpoint)
--skip-tfidf           Skip TF-IDF training
--skip-lgbm            Skip LightGBM training
--skip-perplexity      Skip Binoculars calibration
--no-augmentation      Disable EDA augmentation
--no-t5-augmentation   Disable T5 paraphrase augmentation (Tier 2)
```

Training takes approximately:
- ModernBERT fine-tuning: ~4–8 hours (A100 GPU)
- TF-IDF + LightGBM: ~30 minutes (CPU)
- Binoculars calibration: ~2 hours (GPU, 25% of training set)

### Step 2: Inference

```bash
python predict.py data/test.jsonl output/
# Output: output/predictions.jsonl
```

## Supplementary Training Data

Optional extra data collection scripts are in `extra_data/`. See [`extra_data/README.md`](extra_data/README.md).

## Docker (TIRA)

The `Dockerfile` targets the [TIRA](https://www.tira.io/) evaluation platform. Build locally with:

```bash
docker build -t pan26-detector .
docker run --gpus all -v /path/to/test:/input -v /path/to/output:/output \
    pan26-detector /input/dataset.jsonl /output
```

Note: The TIRA build uses `--from=torch_pkg` to inject a pre-built PyTorch wheel. For local builds, install PyTorch manually (comment out that `COPY` line and `pip install torch`).

## Key Hyperparameters

All hyperparameters are in [`src/config.py`](src/config.py):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `MAX_LENGTH` | 1024 | Covers 87% of training texts without truncation |
| `TRAIN_BATCH_SIZE` | 16 | Effective batch = 16 × 2 = 32 |
| `LEARNING_RATE` | 2e-5 | With 6% linear warmup |
| `LABEL_SMOOTHING` | 0.05 | Improves Brier Score |
| `AUG_FRACTION` | 0.5 | 50% of training examples augmented (Tier 1 EDA) |
| `HARD_MODEL_OVERSAMPLE_FACTOR` | 1.5 | Hard models oversampled 1.5× |

## Citation

```bibtex
@inproceedings{ibrahim:2026,
  author    = {Momen Ibrahim},
  title     = {When Detectors Disagree ---
               Disagreement-Aware Ensemble for Robust AI Text Detection},
  booktitle = {CLEF 2026 Working Notes},
  year      = {2026},
  publisher = {CEUR-WS.org},
}
```

## License

Code: MIT License. PAN competition data is subject to the [PAN data terms](https://pan.webis.de/).
