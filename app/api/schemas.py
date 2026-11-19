"""Request and response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Run one question through one configuration."""

    question: str = Field(..., min_length=1, max_length=4000)
    model: str | None = Field(None, description="Model spec, e.g. 'ollama:llama3'. Defaults to HG_DEFAULT_MODEL.")
    configuration: str = Field("C6", description="C0-C6 (C6 = full workflow with type checks and evidence validation)")
    collection: str = Field("global", description="Knowledge-base collection used for retrieval")
    overrides: dict[str, Any] = Field(default_factory=dict, description="Pipeline setting overrides")


class StageOut(BaseModel):
    stage: str
    order: int
    latency_ms: float
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    output: dict[str, Any]


class AskResponse(BaseModel):
    run_id: int
    question: str
    model: str
    configuration: str
    final_answer: str
    candidate_answer: str
    decision: str
    decision_reasons: list[str]
    confidence: float | None
    abstained: bool
    escalated: bool
    review_item_id: int | None
    n_regenerations: int
    citations: list[dict[str, Any]]
    verification_status: str | None = Field(
        None, description="verified_by_evidence | partially_supported | consistent_not_evidence_checked | "
        "unchecked | abstained_insufficient_evidence | clarification_requested | needs_human_review")
    issues: list[dict[str, Any]] = Field(default_factory=list, description="Typed hallucination findings")
    remaining_uncertainty: list[str] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list, description="Evidence passages shown to the model")
    latency_ms: float
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    stages: list[StageOut]


class DocumentText(BaseModel):
    title: str
    text: str = Field(..., min_length=1)
    source: str | None = None


class DocumentsJSON(BaseModel):
    collection: str = "global"
    documents: list[DocumentText] = Field(..., min_length=1)


class DocumentsResponse(BaseModel):
    collection: str
    document_ids: list[int]
    total_chunks_in_collection: int


class ExperimentStatus(BaseModel):
    id: int
    name: str
    status: str
    progress_done: int
    progress_total: int
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    config: dict[str, Any]
    summary: dict[str, Any] | None


class ReviewOut(BaseModel):
    id: int
    run_id: int | None
    question: str
    proposed_answer: str | None
    reason: str | None
    source: str
    status: str
    reviewer_answer: str | None
    reviewer_note: str | None
    created_at: datetime
    reviewed_at: datetime | None


class ReviewAction(BaseModel):
    action: Literal["approve", "reject", "edit"]
    corrected_answer: str | None = None
    note: str | None = None


class JudgementOverride(BaseModel):
    """Manual correction of an automatic judge decision (used in metrics instead of the automatic label)."""

    target: Literal["initial", "candidate"] = "candidate"
    label: Literal["correct", "hallucinated", "abstained"]
