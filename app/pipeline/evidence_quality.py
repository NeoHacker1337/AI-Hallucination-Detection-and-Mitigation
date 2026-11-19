"""Validate retrieved evidence before the model sees it (C6): relevance, prompt injection and
freshness. RAG is only as good as its evidence, so every passage is checked and every decision
(kept, dropped, quarantined, stale) is logged in the trace."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from app.pipeline.retrieval import KnowledgeBase, Passage
from app.pipeline.safety import filter_passages
from app.pipeline.settings import PipelineSettings


@dataclass
class EvidenceReport:
    """What happened to the retrieved passages."""

    kept: list[Passage] = field(default_factory=list)
    dropped_irrelevant: list[dict] = field(default_factory=list)
    quarantined: list[dict] = field(default_factory=list)
    stale: list[dict] = field(default_factory=list)
    undated: int = 0

    def to_dict(self) -> dict:
        return {
            "kept": [p.to_dict() for p in self.kept],
            "dropped_irrelevant": self.dropped_irrelevant,
            "quarantined_injection": self.quarantined,
            "stale": self.stale,
            "undated_passages": self.undated,
        }


def rescore_with_embeddings(kb: KnowledgeBase | None, query: str, passages: list[Passage]) -> None:
    """Give web passages the same cosine-similarity score as knowledge-base passages, so one
    relevance threshold applies to both. No-op without an embedder."""
    if kb is None or not passages:
        return
    vectors = kb.embed([query] + [p.text for p in passages])
    q, docs = vectors[0], vectors[1:]
    norms = np.linalg.norm(docs, axis=1) * (np.linalg.norm(q) or 1.0)
    sims = docs @ q / np.where(norms == 0, 1.0, norms)
    for p, sim in zip(passages, sims):
        p.score = float(sim)


def _age_days(iso: str | None, now: datetime) -> float | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt).total_seconds() / 86400.0


def validate_evidence(
    passages: list[Passage], settings: PipelineSettings, time_sensitive: bool, now: datetime | None = None
) -> EvidenceReport:
    """Apply the relevance filter, the injection filter and the freshness check."""
    report = EvidenceReport()
    current = passages
    if settings.enable_relevance_filter:
        threshold = settings.thresholds.min_passage_score
        kept = [p for p in current if p.score >= threshold]
        report.dropped_irrelevant = [
            {"source_id": p.source_id, "title": p.title, "score": round(p.score, 4), "threshold": threshold}
            for p in current if p.score < threshold
        ]
        current = kept
    if settings.enable_injection_filter:
        current, findings = filter_passages(current)
        report.quarantined = [f.to_dict() for f in findings]
    now = now or datetime.now(timezone.utc)
    for p in current:
        age = _age_days(p.last_modified, now)
        if age is None:
            report.undated += 1
        elif time_sensitive and settings.max_evidence_age_days is not None and age > settings.max_evidence_age_days:
            report.stale.append({"source_id": p.source_id, "title": p.title, "last_modified": p.last_modified,
                                 "age_days": round(age, 1)})
    report.kept = current
    return report


class FilteredKnowledgeBase:
    """Wraps a knowledge base so that searches made by later stages (Chain-of-Verification) go
    through the same relevance and injection filters as the main retrieval."""

    def __init__(self, kb: KnowledgeBase, settings: PipelineSettings) -> None:
        self.kb = kb
        self.settings = settings
        self.reports: list[dict] = []

    def search(self, collection: str, query: str, k: int) -> list[Passage]:
        report = validate_evidence(self.kb.search(collection, query, k), self.settings, time_sensitive=False)
        if report.dropped_irrelevant or report.quarantined:
            self.reports.append({"query": query, "dropped_irrelevant": report.dropped_irrelevant,
                                 "quarantined_injection": report.quarantined})
        return report.kept

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.kb.embed(texts)
