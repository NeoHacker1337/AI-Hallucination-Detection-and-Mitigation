"""Natural-language-inference model wrapper (DeBERTa MNLI by default).

`predict` returns probabilities in a fixed column order: ENTAIL, NEUTRAL, CONTRADICT, independent
of the label order used by the underlying checkpoint.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Protocol, Sequence

import numpy as np

from app.config import get_settings

ENTAIL, NEUTRAL, CONTRADICT = 0, 1, 2


class NLIScorer(Protocol):
    """Anything that maps (premise, hypothesis) pairs to [entail, neutral, contradict] probabilities."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:  # pragma: no cover - protocol
        ...


def resolve_device(preference: str | None = None) -> str:
    """Pick a torch device: explicit setting, else CUDA, MPS, CPU."""
    pref = (preference or get_settings().device).lower()
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class HFNLIModel:
    """Hugging Face sequence-classification NLI model with an LRU cache of scored pairs."""

    def __init__(self, model_name: str | None = None, device: str | None = None, batch_size: int | None = None,
                 cache_size: int = 50_000) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.nli_model
        self.device = resolve_device(device)
        self.batch_size = batch_size or settings.nli_batch_size
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for the NLI model") from exc
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load NLI model '{self.model_name}': {exc}. Check HG_NLI_MODEL and your "
                "internet connection / Hugging Face cache."
            ) from exc
        self.model.to(self.device).eval()
        self._torch = torch
        self._order = self._label_order(self.model.config.id2label)
        self._cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    @staticmethod
    def _label_order(id2label: dict) -> list[int]:
        """Indices of the entail/neutral/contradict logits in the checkpoint's output."""
        lookup: dict[str, int] = {}
        for idx, label in id2label.items():
            name = str(label).lower()
            if name.startswith("entail"):
                lookup["e"] = int(idx)
            elif name.startswith("neutral"):
                lookup["n"] = int(idx)
            elif name.startswith("contradict"):
                lookup["c"] = int(idx)
        if set(lookup) != {"e", "n", "c"}:
            raise RuntimeError(f"NLI model labels {id2label} do not contain entailment/neutral/contradiction")
        return [lookup["e"], lookup["n"], lookup["c"]]

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Return an (n, 3) array of probabilities [entail, neutral, contradict]."""
        out = np.zeros((len(pairs), 3), dtype=np.float64)
        todo: list[int] = []
        with self._lock:
            for i, pair in enumerate(pairs):
                cached = self._cache.get(pair)
                if cached is not None:
                    out[i] = cached
                    self._cache.move_to_end(pair)
                else:
                    todo.append(i)
            torch = self._torch
            for start in range(0, len(todo), self.batch_size):
                batch_idx = todo[start : start + self.batch_size]
                premises = [pairs[i][0] for i in batch_idx]
                hypotheses = [pairs[i][1] for i in batch_idx]
                enc = self.tokenizer(
                    premises, hypotheses, truncation="only_first", max_length=512, padding=True,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[:, self._order]
                for row, i in enumerate(batch_idx):
                    out[i] = probs[row]
                    self._cache[pairs[i]] = probs[row]
                    if len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
        return out


_NLI: HFNLIModel | None = None
_NLI_LOCK = threading.Lock()


def get_nli() -> HFNLIModel:
    """Process-wide NLI model singleton (loaded on first use)."""
    global _NLI
    with _NLI_LOCK:
        if _NLI is None:
            _NLI = HFNLIModel()
        return _NLI
