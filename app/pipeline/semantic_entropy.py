"""Stage 5a: semantic entropy (Kuhn et al., 2023; Farquhar et al., 2024).

N answers are sampled at temperature > 0 and clustered greedily: an answer joins the first
cluster whose representative is semantically equivalent to it, otherwise it starts a new cluster.
Equivalence is bidirectional entailment judged by the NLI model on "question + answer" texts.
The discrete semantic entropy is H = -sum_c p(c) log p(c) with p(c) = |c| / N. The normalised value
H / log(N) lies in [0, 1] (0 = all samples agree, 1 = every sample means something different).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from app.pipeline.generation import strip_citations
from app.pipeline.nli import CONTRADICT, ENTAIL, NLIScorer


@dataclass
class SemanticEntropyResult:
    """Clusters and entropy values for one question."""

    clusters: list[list[int]]
    entropy: float
    normalized_entropy: float
    n_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


def cluster_by_equivalence(n: int, equivalent: Callable[[int, int], bool]) -> list[list[int]]:
    """Greedy clustering of items 0..n-1 using an equivalence oracle against cluster representatives."""
    clusters: list[list[int]] = []
    for i in range(n):
        for cluster in clusters:
            if equivalent(cluster[0], i):
                cluster.append(i)
                break
        else:
            clusters.append([i])
    return clusters


def entropy_from_clusters(clusters: list[list[int]], n: int | None = None) -> tuple[float, float]:
    """Return (entropy in nats, entropy normalised by log N)."""
    total = n if n is not None else sum(len(c) for c in clusters)
    if total <= 0:
        raise ValueError("Cannot compute entropy of zero samples")
    probs = np.array([len(c) / total for c in clusters if len(c) > 0], dtype=float)
    entropy = float(-(probs * np.log(probs)).sum())
    entropy = max(entropy, 0.0)
    normalized = entropy / math.log(total) if total > 1 else 0.0
    return entropy, min(max(normalized, 0.0), 1.0)


def nli_equivalence(
    question: str, answers: list[str], nli: NLIScorer, mode: str = "strict"
) -> Callable[[int, int], bool]:
    """Build an equivalence oracle backed by the NLI model.

    strict: both directions are classified as entailment (argmax).
    loose:  neither direction is a contradiction and at least one is entailment.
    """
    texts = [f"{question} {strip_citations(a)}".strip() for a in answers]

    def equivalent(i: int, j: int) -> bool:
        if not answers[i].strip() or not answers[j].strip():
            return answers[i].strip() == answers[j].strip()
        probs = nli.predict([(texts[i], texts[j]), (texts[j], texts[i])])
        labels = probs.argmax(axis=1)
        if mode == "strict":
            return bool(labels[0] == ENTAIL and labels[1] == ENTAIL)
        return bool(CONTRADICT not in labels and ENTAIL in labels)

    return equivalent


def semantic_entropy(question: str, samples: list[str], nli: NLIScorer, mode: str = "strict") -> SemanticEntropyResult:
    """Compute semantic entropy of sampled answers."""
    if not samples:
        raise ValueError("semantic_entropy needs at least one sample")
    clusters = cluster_by_equivalence(len(samples), nli_equivalence(question, samples, nli, mode))
    entropy, normalized = entropy_from_clusters(clusters, len(samples))
    return SemanticEntropyResult(clusters=clusters, entropy=entropy, normalized_entropy=normalized, n_samples=len(samples))
