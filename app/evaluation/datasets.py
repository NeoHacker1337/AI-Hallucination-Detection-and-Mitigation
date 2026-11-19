"""Benchmark loaders. Each loader returns a list of `DatasetItem`.

Adding a dataset = writing one function `(sample_size, seed, **kwargs) -> list[DatasetItem]` and
registering it in `LOADERS` (see README, "Adding a dataset").
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.config import get_settings


class DatasetUnavailableError(RuntimeError):
    """Raised when a dataset cannot be loaded; the message explains how to fix it."""


@dataclass
class DatasetItem:
    """One evaluation question with its reference answers and (optional) gold context."""

    id: str
    question: str
    correct_answers: list[str]
    incorrect_answers: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


_NO_COMMENT = {"i have no comment", "i have no comment."}


def _sample_indices(n: int, sample_size: int | None, seed: int) -> list[int]:
    """Seeded sample of indices (all indices if sample_size is None or >= n)."""
    if sample_size is None or sample_size >= n:
        return list(range(n))
    return sorted(random.Random(seed).sample(range(n), sample_size))


def _hf_load(dataset_id: str, config: str | None, split: str) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetUnavailableError("The 'datasets' package is not installed (pip install -r requirements.txt).") from exc
    try:
        return load_dataset(dataset_id, config, split=split)
    except Exception as exc:  # noqa: BLE001
        raise DatasetUnavailableError(
            f"Could not load Hugging Face dataset '{dataset_id}' (config={config}, split={split}): {exc}. "
            "Check your internet connection, the dataset id (environment variables HG_TRUTHFULQA_ID / "
            "HG_HALUEVAL_ID), or pre-download it with scripts/load_dataset.py."
        ) from exc


def load_truthfulqa(sample_size: int | None, seed: int, **_: Any) -> list[DatasetItem]:
    """TruthfulQA (generation split). No gold passages: retrieval uses a KB collection instead."""
    ds = _hf_load(get_settings().truthfulqa_hf_id, "generation", "validation")
    items: list[DatasetItem] = []
    for i in _sample_indices(len(ds), sample_size, seed):
        row = ds[i]
        correct = [a for a in row["correct_answers"] if a.strip().lower() not in _NO_COMMENT]
        if row.get("best_answer") and row["best_answer"] not in correct:
            correct.insert(0, row["best_answer"])
        items.append(
            DatasetItem(
                id=f"truthfulqa-{i}",
                question=row["question"],
                correct_answers=correct,
                incorrect_answers=list(row["incorrect_answers"]),
                contexts=[],
                metadata={"category": row.get("category"), "type": row.get("type"), "source": row.get("source")},
            )
        )
    return items


def load_halueval_qa(sample_size: int | None, seed: int, **_: Any) -> list[DatasetItem]:
    """HaluEval QA subset: each item has a knowledge passage, a right and a hallucinated answer."""
    ds = _hf_load(get_settings().halueval_hf_id, "qa", "data")
    items: list[DatasetItem] = []
    for i in _sample_indices(len(ds), sample_size, seed):
        row = ds[i]
        items.append(
            DatasetItem(
                id=f"halueval-qa-{i}",
                question=row["question"],
                correct_answers=[row["right_answer"]],
                incorrect_answers=[row["hallucinated_answer"]] if row.get("hallucinated_answer") else [],
                contexts=[row["knowledge"]] if row.get("knowledge") else [],
            )
        )
    return items


def load_custom_csv(sample_size: int | None, seed: int, csv_path: str | None = None, **_: Any) -> list[DatasetItem]:
    """Custom CSV with columns: question, correct_answers (required; '|' separated),
    incorrect_answers (optional), context (optional), id (optional)."""
    if not csv_path:
        raise DatasetUnavailableError("Dataset 'custom_csv' needs 'csv_path'.")
    path = Path(csv_path)
    if not path.exists():
        raise DatasetUnavailableError(f"CSV file not found: {path.resolve()}")
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise DatasetUnavailableError(f"CSV file {path} has no rows.")
    missing = {"question", "correct_answers"} - set(rows[0].keys())
    if missing:
        raise DatasetUnavailableError(f"CSV file {path} is missing required columns: {sorted(missing)}")

    def split(value: str | None) -> list[str]:
        return [v.strip() for v in (value or "").split("|") if v.strip()]

    items: list[DatasetItem] = []
    for i in _sample_indices(len(rows), sample_size, seed):
        row = rows[i]
        correct = split(row.get("correct_answers"))
        if not row.get("question") or not correct:
            raise DatasetUnavailableError(f"CSV row {i + 2} needs a question and at least one correct answer.")
        items.append(
            DatasetItem(
                id=row.get("id") or f"custom-{i}",
                question=row["question"].strip(),
                correct_answers=correct,
                incorrect_answers=split(row.get("incorrect_answers")),
                contexts=[row["context"].strip()] if (row.get("context") or "").strip() else [],
            )
        )
    return items


LOADERS: dict[str, Callable[..., list[DatasetItem]]] = {
    "truthfulqa": load_truthfulqa,
    "halueval_qa": load_halueval_qa,
    "custom_csv": load_custom_csv,
}


def load_benchmark(name: str, sample_size: int | None, seed: int, **kwargs: Any) -> list[DatasetItem]:
    """Load a registered dataset by name."""
    if name not in LOADERS:
        raise DatasetUnavailableError(f"Unknown dataset '{name}'. Available: {sorted(LOADERS)}")
    items = LOADERS[name](sample_size, seed, **kwargs)
    if not items:
        raise DatasetUnavailableError(f"Dataset '{name}' returned no items.")
    return items
