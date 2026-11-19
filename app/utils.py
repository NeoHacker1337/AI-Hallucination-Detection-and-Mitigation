"""Small shared helpers: seeding, timing, JSON parsing of LLM output."""

from __future__ import annotations

import json
import os
import random
import re
import time
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class Timer:
    """Wall-clock timer measuring milliseconds."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since creation."""
        return (time.perf_counter() - self.start) * 1000.0


@contextmanager
def timed() -> Iterator[Timer]:
    """Context manager yielding a Timer."""
    timer = Timer()
    yield timer


_JSON_BLOCK = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def extract_json(text: str) -> Any | None:
    """Parse the first JSON object or array embedded in an LLM response.

    Returns None when nothing parseable is found, so callers can fall back explicitly.
    """
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(cleaned)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and other objects into JSON-serialisable values."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float) and value != value:  # NaN
        return None
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    return value
