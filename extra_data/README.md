# Extra Data Collection

These scripts collect supplementary training data beyond the official PAN 2026 dataset.
The generated JSONL files are **not included** in this repository (see `.gitignore`).

## Three Data Tracks

| Track | Script | What it collects | Label |
|-------|--------|------------------|-------|
| A | `collect_ai_texts.py` | AI texts from new Groq-hosted models (Llama-4, Qwen3, Kimi-K2) | 1 |
| B | `collect_obfuscations.py` | LLM-paraphrased versions of existing AI texts | 1 |
| C | `collect_human_texts.py` | Pre-LLM human texts (Gutenberg, Wikipedia, Wikisource) | 0 |
| — | `generate_claude_texts.py` | AI texts via Anthropic Claude API | 1 |

After collecting, run `merge_extra_data.py` to combine all tracks into one file.

## Usage

### Prerequisites

```bash
pip install groq anthropic requests beautifulsoup4
```

### Track A — New AI models via Groq

```bash
export GROQ_API_KEY=your_key_here
python collect_ai_texts.py --n-per-model 500
```

Generated files: `extra_data/ai_texts/<model_name>.jsonl`

### Track B — Obfuscated AI texts

Requires `data/train_prepared.jsonl` (run `prepare_data.py` first).

```bash
export GROQ_API_KEY=your_key_here
python collect_obfuscations.py --max-sources 1000
```

Generated file: `extra_data/obfuscations/obfuscated_texts.jsonl`

### Track C — Human texts (no API key needed)

```bash
python collect_human_texts.py --per-source 75
```

Generated file: `extra_data/human_texts/collected_human_texts.jsonl`

### Claude texts via Anthropic API (optional)

```bash
export ANTHROPIC_API_KEY=your_key_here
python generate_claude_texts.py --target 300
```

Generated file: `extra_data/ai_texts/claude-sonnet-4-6.jsonl`

### Merge all tracks

After collecting from all tracks:

```bash
python merge_extra_data.py
```

Output: `extra_data/merged/extra_train.jsonl`

Then use with `prepare_data.py`:

```bash
python prepare_data.py --train data/train.jsonl --extra extra_data/merged/extra_train.jsonl
```

## Output Schema

All output JSONL files share the same schema as the PAN training data:

```json
{"id": "uuid", "text": "...", "label": 0_or_1, "model": "model_name", "genre": "fiction|essays|news"}
```

- `label=0` → human-written
- `label=1` → AI-generated (including obfuscated variants)