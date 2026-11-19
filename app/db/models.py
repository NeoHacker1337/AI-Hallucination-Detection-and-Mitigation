"""ORM models. Every pipeline stage output, score, latency and token count is persisted."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    """A document uploaded to (or built into) a knowledge-base collection."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    collection: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(String(512))
    source: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    content_date: Mapped[str | None] = mapped_column(String(32), nullable=True)  # ISO date of the content
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """A retrievable passage. Its integer id is the FAISS vector id."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    collection: Mapped[str] = mapped_column(String(128), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="chunks")


class Experiment(Base):
    """A batch experiment over a dataset, models and configurations."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|running|completed|failed
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    progress_done: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    runs: Mapped[list["Run"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")


class Run(Base):
    """One question answered by one model under one configuration."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), index=True, nullable=True
    )
    question_id: Mapped[str] = mapped_column(String(128), index=True)
    question: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(128), index=True)
    config_name: Mapped[str] = mapped_column(String(32), index=True)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON)

    initial_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    candidate_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    first_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False)
    escalated: Mapped[bool] = mapped_column(Boolean, default=False)
    n_regenerations: Mapped[int] = mapped_column(Integer, default=0)

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    first_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    semantic_entropy: Mapped[float | None] = mapped_column(Float, nullable=True)
    semantic_entropy_norm: Mapped[float | None] = mapped_column(Float, nullable=True)
    selfcheck_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    supported_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    contradicted_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)

    citations: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    passages: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)

    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    experiment: Mapped[Experiment | None] = relationship(back_populates="runs")
    stages: Mapped[list["StageLog"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="StageLog.order"
    )
    judgements: Mapped[list["Judgement"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class StageLog(Base):
    """Output, latency and token usage of a single pipeline stage within a run."""

    __tablename__ = "stage_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    order: Mapped[int] = mapped_column(Integer)
    stage: Mapped[str] = mapped_column(String(64))
    output: Mapped[dict[str, Any]] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)

    run: Mapped[Run] = relationship(back_populates="stages")


class Judgement(Base):
    """A correctness decision about one answer, saved so it can be audited by hand."""

    __tablename__ = "judgements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    target: Mapped[str] = mapped_column(String(16))  # "initial" | "candidate"
    answer: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(16))  # "nli" | "llm" | "both"
    label: Mapped[str] = mapped_column(String(16))  # "correct" | "hallucinated" | "abstained"
    nli_label: Mapped[str | None] = mapped_column(String(16), nullable=True)
    nli_correct_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    nli_incorrect_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    nli_matched_reference: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_label: Mapped[str | None] = mapped_column(String(16), nullable=True)
    llm_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    judges_disagree: Mapped[bool] = mapped_column(Boolean, default=False)
    uncertain: Mapped[bool] = mapped_column(Boolean, default=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    references: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    manual_label: Mapped[str | None] = mapped_column(String(16), nullable=True)

    run: Mapped[Run] = relationship(back_populates="judgements")


class ReviewItem(Base):
    """An escalated answer waiting for (or resolved by) a human reviewer."""

    __tablename__ = "review_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    proposed_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="ask")
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|approved|rejected|edited
    reviewer_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
