"""Helpers to persist pipeline results and query them back."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.db.models import ReviewItem, Run, StageLog
from app.utils import to_jsonable

if TYPE_CHECKING:
    from app.pipeline.orchestrator import PipelineResult


def save_result(
    session: Session,
    result: "PipelineResult",
    question_id: str,
    experiment_id: int | None = None,
    queue_escalations: bool = True,
    source: str = "ask",
) -> Run:
    """Store a PipelineResult (run row + one row per stage) and queue it for review if escalated."""
    fs = result.final_signals
    run = Run(
        experiment_id=experiment_id,
        question_id=question_id,
        question=result.question,
        model=result.model,
        config_name=result.config_name,
        settings=to_jsonable(result.settings),
        initial_answer=result.initial_answer,
        candidate_answer=result.candidate_answer,
        final_answer=result.final_answer,
        first_decision=result.first_decision.value if result.first_decision else None,
        decision=result.decision.value,
        decision_reason="; ".join(result.decision_reasons),
        abstained=result.abstained,
        escalated=result.escalated,
        n_regenerations=result.n_regenerations,
        confidence=result.confidence,
        first_confidence=result.first_confidence,
        semantic_entropy=None,
        semantic_entropy_norm=result.first_signals.semantic_entropy_norm if result.first_signals else None,
        selfcheck_score=result.first_signals.selfcheck_score if result.first_signals else None,
        supported_fraction=fs.supported_fraction if fs else None,
        contradicted_fraction=fs.contradicted_fraction if fs else None,
        citations=to_jsonable(result.citations),
        passages=[to_jsonable(p.to_dict()) for p in result.passages],
        latency_ms=result.latency_ms,
        llm_calls=result.llm_calls,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
    for st in result.stages:
        if st.stage == "semantic_entropy":
            run.semantic_entropy = st.output.get("entropy")
        run.stages.append(
            StageLog(
                order=st.order,
                stage=st.stage,
                output=to_jsonable(st.output),
                latency_ms=st.latency_ms,
                llm_calls=st.llm_calls,
                prompt_tokens=st.prompt_tokens,
                completion_tokens=st.completion_tokens,
            )
        )
    session.add(run)
    session.flush()
    if result.escalated and queue_escalations:
        session.add(
            ReviewItem(
                run_id=run.id,
                question=result.question,
                proposed_answer=result.candidate_answer,
                reason="; ".join(result.decision_reasons),
                source=source,
            )
        )
    return run


def resolve_review(
    session: Session, item: ReviewItem, action: str, corrected_answer: str | None, note: str | None
) -> ReviewItem:
    """Apply a reviewer's decision."""
    status = {"approve": "approved", "reject": "rejected", "edit": "edited"}[action]
    if action == "edit" and not corrected_answer:
        raise ValueError("An 'edit' review needs corrected_answer.")
    item.status = status
    item.reviewer_answer = corrected_answer if action == "edit" else (item.proposed_answer if action == "approve" else None)
    item.reviewer_note = note
    item.reviewed_at = datetime.now(timezone.utc)
    session.add(item)
    return item
