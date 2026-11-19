"""Stage 5c: evidence check. Each claim is tested against every retrieved passage with NLI
(premise = passage, hypothesis = claim) and labelled supported / contradicted / not enough evidence
from its best-scoring passage."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from app.pipeline.nli import CONTRADICT, ENTAIL, NLIScorer
from app.pipeline.retrieval import Passage

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
NOT_ENOUGH = "not_enough_evidence"


@dataclass
class ClaimEvidence:
    """Evidence verdict for one claim."""

    claim: str
    label: str
    entail: float
    contradict: float
    best_source: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceResult:
    """Evidence verdicts plus summary fractions."""

    claims: list[ClaimEvidence]
    supported_fraction: float | None
    contradicted_fraction: float | None

    def to_dict(self) -> dict:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "supported_fraction": self.supported_fraction,
            "contradicted_fraction": self.contradicted_fraction,
        }


def label_claim(entail: float, contradict: float, support_threshold: float, contradict_threshold: float) -> str:
    """Label a claim from its best entailment and contradiction probabilities.

    `entail` and `contradict` are maxima over *all* retrieved passages, so they usually come from
    different passages. A claim entailed by one passage is SUPPORTED even if another passage scores
    high on contradiction: NLI models often label off-topic passages as "contradiction", and one
    passage that states the fact outweighs another that is about something else. A claim is
    CONTRADICTED only when no passage supports it.
    """
    if entail >= support_threshold:
        return SUPPORTED
    if contradict >= contradict_threshold:
        return CONTRADICTED
    return NOT_ENOUGH


def summarize(labels: list[str]) -> tuple[float | None, float | None]:
    """(supported fraction, contradicted fraction); None when there are no claims."""
    if not labels:
        return None, None
    n = len(labels)
    return labels.count(SUPPORTED) / n, labels.count(CONTRADICTED) / n


def evidence_check(
    claims: list[str],
    passages: list[Passage],
    nli: NLIScorer,
    support_threshold: float = 0.5,
    contradict_threshold: float = 0.5,
) -> EvidenceResult:
    """Run the NLI evidence check for all claims."""
    if not claims:
        return EvidenceResult(claims=[], supported_fraction=None, contradicted_fraction=None)
    if not passages:
        results = [ClaimEvidence(c, NOT_ENOUGH, 0.0, 0.0, None) for c in claims]
        sup, con = summarize([r.label for r in results])
        return EvidenceResult(results, sup, con)
    pairs = [(p.text, claim) for claim in claims for p in passages]
    probs = nli.predict(pairs).reshape(len(claims), len(passages), 3)
    results: list[ClaimEvidence] = []
    for ci, claim in enumerate(claims):
        entail = probs[ci, :, ENTAIL]
        contra = probs[ci, :, CONTRADICT]
        best_e = int(np.argmax(entail))
        best_c = int(np.argmax(contra))
        label = label_claim(float(entail[best_e]), float(contra[best_c]), support_threshold, contradict_threshold)
        source_idx = best_c if label == CONTRADICTED else best_e
        results.append(
            ClaimEvidence(
                claim=claim,
                label=label,
                entail=float(entail[best_e]),
                contradict=float(contra[best_c]),
                best_source=passages[source_idx].source_id,
            )
        )
    sup, con = summarize([r.label for r in results])
    return EvidenceResult(results, sup, con)
