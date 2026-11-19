"""Stage 5b: SelfCheckGPT-style self-consistency with NLI (Manakul et al., 2023).

For each claim and each sampled answer, the NLI model scores (premise = sample, hypothesis =
claim). Following SelfCheckGPT-NLI, the per-sample score is P(contradict) / (P(entail) +
P(contradict)), ignoring the neutral class; the claim score is the mean over samples. Higher
scores mean the samples disagree with the claim, i.e. a likely hallucination. Agreement = 1 - score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from app.pipeline.generation import strip_citations
from app.pipeline.nli import CONTRADICT, ENTAIL, NLIScorer

_EPS = 1e-8


@dataclass
class SelfCheckResult:
    """Per-claim and answer-level SelfCheck scores."""

    claim_scores: list[float]
    answer_score: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def selfcheck_from_probs(probs: np.ndarray) -> np.ndarray:
    """probs: array (n_claims, n_samples, 3) -> per-claim hallucination scores in [0, 1]."""
    if probs.ndim != 3 or probs.shape[-1] != 3:
        raise ValueError("probs must have shape (n_claims, n_samples, 3)")
    entail = probs[..., ENTAIL]
    contra = probs[..., CONTRADICT]
    per_sample = contra / (entail + contra + _EPS)
    return per_sample.mean(axis=1)


def self_consistency(claims: list[str], samples: list[str], nli: NLIScorer) -> SelfCheckResult:
    """Score each claim against the sampled answers."""
    if not claims or not samples:
        return SelfCheckResult(claim_scores=[], answer_score=None)
    clean_samples = [strip_citations(s) for s in samples]
    pairs = [(sample, claim) for claim in claims for sample in clean_samples]
    probs = nli.predict(pairs).reshape(len(claims), len(clean_samples), 3)
    scores = selfcheck_from_probs(probs)
    return SelfCheckResult(claim_scores=[float(s) for s in scores], answer_score=float(scores.mean()))
