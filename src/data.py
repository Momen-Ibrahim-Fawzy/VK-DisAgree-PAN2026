"""
Data loading and saving utilities for JSONL format.
"""
import json
from pathlib import Path
from typing import Iterator


def load_jsonl(path: str | Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Iterate over records in a JSONL file without loading all into memory."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(records: list[dict], path: str | Path) -> None:
    """Save records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_texts_and_labels(records: list[dict]) -> tuple[list[str], list[int]]:
    """Extract texts and labels from records."""
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    return texts, labels


def get_texts(records: list[dict]) -> list[str]:
    """Extract texts from records (for inference)."""
    return [r["text"] for r in records]