"""Step 8 of the workflow: an honest verification status for the answer that is shown.

The decision (VERIFIED / UNCERTAIN / ...) says whether the answer is released. The verification
status says *what the release rests on*. An answer is only called "verified_by_evidence" when the
evidence check ran on the released answer, every claim was entailed by a retrieved passage, no
claim was contradicted, and no type-specific check found a problem. Everything else is labelled
for what it is (partially supported, consistent but not evidence-checked, unchecked, abstained).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from app.pipeline.decision import Decision, Signals
from app.pipeline.evidence_check import SUPPORTED, EvidenceResult
from app.pipeline.retrieval import Passage
from app.pipeline.type_checks import FACTUAL, Issue, TypeCheckResult

VERIFIED_BY_EVIDENCE = "verified_by_evidence"
PARTIALLY_SUPPORTED = "partially_supported"
CONSISTENT_NOT_EVIDENCE_CHECKED = "consistent_not_evidence_checked"
UNCHECKED = "unchecked"
ABSTAINED = "abstained_insufficient_evidence"
CLARIFICATION = "clarification_requested"
HUMAN_REVIEW = "needs_human_review"


@dataclass
class Assessment:
    """Verification status, typed issues, remaining uncertainty and the sources used."""

    verification_status: str
    issues: list[dict] = field(default_factory=list)
    remaining_uncertainty: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _sources(passages: list[Passage], cited_labels: set[str]) -> list[dict]:
    out = []
    for i, p in enumerate(passages, start=1):
        out.append({
            "label": f"S{i}", "cited": f"S{i}" in cited_labels, "title": p.title, "source_id": p.source_id,
            "source": p.source, "origin": p.origin, "score": round(p.score, 4), "last_modified": p.last_modified,
        })
    return out


def assess(
    decision: Decision,
    abstained: bool,
    clarification: bool,
    signals: Signals | None,
    evidence: EvidenceResult | None,
    type_checks: TypeCheckResult | None,
    passages: list[Passage],
    citations: list[dict],
    evidence_notes: list[str],
    evidence_check_enabled: bool,
) -> Assessment:
    """Build the final assessment for the released answer."""
    issues: list[Issue] = []
    unsupported: list[str] = []
    if evidence is not None:
        for c in evidence.claims:
            if c.label != SUPPORTED:
                unsupported.append(c.claim)
                issues.append(Issue(FACTUAL, f"claim {c.label.replace('_', ' ')}: '{c.claim}'",
                                    hard=c.label == "contradicted"))
    if type_checks is not None:
        issues.extend(type_checks.issues)
    remaining = list(evidence_notes)

    if decision == Decision.NEEDS_REVIEW:
        status = HUMAN_REVIEW
        remaining.append("the answer still failed verification after the allowed regenerations")
    elif clarification:
        status = CLARIFICATION
    elif abstained:
        status = ABSTAINED
    elif decision == Decision.UNCHECKED:
        status = UNCHECKED
        remaining.append("no hallucination checks ran in this configuration")
    elif not evidence_check_enabled or evidence is None:
        status = CONSISTENT_NOT_EVIDENCE_CHECKED
        remaining.append("released on consistency/uncertainty signals only; claims were not checked against evidence")
    elif not evidence.claims:
        status = PARTIALLY_SUPPORTED
        remaining.append("no checkable claims were extracted from the answer")
    elif unsupported or (type_checks is not None and type_checks.issues):
        status = PARTIALLY_SUPPORTED
        if unsupported:
            remaining.append(f"{len(unsupported)} of {len(evidence.claims)} claim(s) are not supported by the evidence")
        if type_checks is not None and type_checks.issues:
            remaining.append(f"{len(type_checks.issues)} non-blocking check issue(s) were found")
    else:
        status = VERIFIED_BY_EVIDENCE
    if signals is not None and signals.semantic_entropy_norm is not None and not abstained and signals.semantic_entropy_norm > 0:
        remaining.append(f"sampled answers were not all equivalent (normalised semantic entropy {signals.semantic_entropy_norm:.2f})")
    cited = {c["label"] for c in citations if c.get("valid")}
    return Assessment(
        verification_status=status,
        issues=[i.to_dict() for i in issues],
        remaining_uncertainty=remaining,
        sources=_sources(passages, cited),
        unsupported_claims=unsupported,
    )
