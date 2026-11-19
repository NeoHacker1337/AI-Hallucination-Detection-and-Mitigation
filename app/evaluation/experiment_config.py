"""Validated experiment specification (saved verbatim with every experiment)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.pipeline.settings import CONFIG_DESCRIPTIONS


class JudgeConfig(BaseModel):
    """How answers are judged correct (see app/evaluation/judge.py)."""

    mode: Literal["nli", "llm", "both"] = "nli"
    llm_model: str | None = Field(None, description="Judge model spec, required for 'llm' and 'both'")
    entail_threshold: float = Field(0.5, ge=0, le=1)


class ExperimentConfig(BaseModel):
    """Batch experiment: every model x configuration x question."""

    name: str = "experiment"
    dataset: str = Field("truthfulqa", description="truthfulqa | halueval_qa | custom_csv")
    csv_path: str | None = None
    sample_size: int = Field(50, ge=1, le=100_000)
    seed: int = 42
    models: list[str] = Field(..., min_length=1)
    configurations: list[str] = Field(default_factory=lambda: list(CONFIG_DESCRIPTIONS))
    kb_collection: str = Field(
        "global", description="Collection used for retrieval when the dataset has no gold passages"
    )
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    pipeline_overrides: dict[str, Any] = Field(
        default_factory=dict, description="Applied identically to every configuration"
    )
    queue_escalations: bool = False

    @field_validator("configurations")
    @classmethod
    def _valid_configs(cls, value: list[str]) -> list[str]:
        upper = [v.upper() for v in value]
        unknown = [v for v in upper if v not in CONFIG_DESCRIPTIONS]
        if unknown:
            raise ValueError(f"Unknown configurations {unknown}; use {list(CONFIG_DESCRIPTIONS)}")
        if not upper:
            raise ValueError("At least one configuration is required")
        return list(dict.fromkeys(upper))

    @model_validator(mode="after")
    def _judge_model(self) -> "ExperimentConfig":
        if self.judge.mode in ("llm", "both") and not self.judge.llm_model:
            raise ValueError("judge.llm_model is required when judge.mode is 'llm' or 'both'")
        if self.dataset == "custom_csv" and not self.csv_path:
            raise ValueError("csv_path is required for dataset 'custom_csv'")
        return self
