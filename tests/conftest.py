"""Shared test fixtures. Tests never download models: NLI and LLM are replaced by deterministic fakes."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    """Point settings at a temporary data dir so tests never touch real data."""
    monkeypatch.setenv("HG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HG_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class KeywordNLI:
    """Fake NLI: entailment when the normalised hypothesis words are a subset of the premise words,
    contradiction when the texts contain the marker 'NOT' asymmetrically, neutral otherwise."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        out = []
        for premise, hyp in pairs:
            p = set(premise.lower().replace(".", "").replace("?", "").split())
            h = set(hyp.lower().replace(".", "").replace("?", "").split())
            if ("not" in p) != ("not" in h) and len(p & h) >= 2:
                out.append([0.05, 0.05, 0.90])
            elif h <= p:
                out.append([0.90, 0.05, 0.05])
            else:
                out.append([0.05, 0.90, 0.05])
        return np.array(out, dtype=float)


@pytest.fixture
def keyword_nli() -> KeywordNLI:
    return KeywordNLI()


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
