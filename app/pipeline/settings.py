"""Pipeline settings and the C0-C6 experimental configurations.

C0-C5 are the configurations of the original experiments and must not change (experiments are
resumable and compared across runs). C6 extends C5 with the workflow additions: type-specific
checks (citation, logical, mathematical, code), relevance and freshness checks on retrieved
evidence, prompt-injection filtering, claim-guided re-retrieval and optional web retrieval.

Every stage can be switched on or off. Defaults for thresholds are *a-priori* choices, not tuned
on any evaluation data; tune them on a held-out split before reporting final numbers.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Thresholds(BaseModel):
    """Thresholds used by the evidence check and the decision stage."""

    # Evidence check (NLI probabilities, claim vs. best passage)
    support_entail: float = Field(0.5, ge=0, le=1, description="P(entail) needed to mark a claim supported")
    contradict: float = Field(0.5, ge=0, le=1, description="P(contradict) needed to mark a claim contradicted")
    # Decision
    entropy_uncertain: float = Field(
        0.6, ge=0, le=1, description="Normalised semantic entropy at/above which the system abstains"
    )
    selfcheck_fail: float = Field(
        0.5, ge=0, le=1, description="Mean SelfCheck contradiction score at/above which the answer FAILS"
    )
    contradicted_fail: float = Field(
        0.25, ge=0, le=1, description="Fraction of contradicted claims at/above which the answer FAILS"
    )
    min_supported: float = Field(
        0.5, ge=0, le=1, description="Fraction of evidence-supported claims below which the system abstains"
    )
    verified_confidence: float = Field(0.6, ge=0, le=1, description="Confidence needed for VERIFIED")
    high_risk_margin: float = Field(
        0.15, ge=0, le=1, description="Added to verified_confidence for high-risk (medical/legal/...) queries"
    )
    weak_claim_selfcheck: float = Field(
        0.5, ge=0, le=1, description="SelfCheck score at/above which a claim is sent to verification"
    )
    logic_contradiction: float = Field(
        0.9, ge=0, le=1, description="P(contradict) between two claims of the same answer that marks a "
        "logical inconsistency (both directions must reach it)"
    )
    min_passage_score: float = Field(
        0.2, ge=-1, le=1, description="Cosine similarity below which a retrieved passage is treated as "
        "irrelevant and dropped (only with enable_relevance_filter)"
    )


class PipelineSettings(BaseModel):
    """Full description of one pipeline configuration. Saved with every run for reproducibility."""

    name: str = "custom"

    # Stage switches
    enable_query_analysis: bool = False
    enable_retrieval: bool = False
    enable_semantic_entropy: bool = False
    enable_self_consistency: bool = False
    enable_evidence_check: bool = False
    enable_verification: bool = False
    enable_decision: bool = False
    # C6 additions (all off in C0-C5)
    enable_type_checks: bool = False  # citation/reference, logical, mathematical and code checks
    enable_relevance_filter: bool = False  # drop passages below thresholds.min_passage_score
    enable_injection_filter: bool = False  # quarantine passages that contain instructions
    enable_claim_reretrieval: bool = False  # on regeneration, also retrieve with the failed claims
    web_search: Literal["none", "wikipedia"] = "none"  # external retrieval (sends the question out)
    web_top_k: int = Field(3, ge=1, le=10)
    max_evidence_age_days: int | None = Field(
        None, ge=1, description="For time-sensitive questions, evidence older than this is reported as stale"
    )

    # Query analysis
    query_analysis_mode: Literal["rule", "llm"] = "rule"
    ask_clarification_on_ambiguous: bool = True

    # Retrieval
    top_k: int = Field(4, ge=1, le=50)
    regen_topk_increment: int = Field(2, ge=0, le=20)

    # Decoding (identical across configurations)
    temperature: float = Field(0.0, ge=0, le=2)
    top_p: float = Field(1.0, gt=0, le=1)
    max_tokens: int = Field(256, ge=16, le=4096)
    seed: int = 42

    # Sampling for semantic entropy / self-consistency
    n_samples: int = Field(5, ge=2, le=30)
    sample_temperature: float = Field(1.0, gt=0, le=2)
    se_equivalence: Literal["strict", "loose"] = "strict"

    # Claims
    claim_extraction: Literal["rule", "llm"] = "rule"
    max_claims: int = Field(12, ge=1, le=50)

    # Chain-of-Verification
    max_verification_questions: int = Field(5, ge=1, le=20)

    # Decision / regeneration
    max_regenerations: int = Field(2, ge=0, le=10)
    thresholds: Thresholds = Field(default_factory=Thresholds)

    @property
    def needs_claims(self) -> bool:
        """Claim extraction runs when any claim-level stage is on."""
        return self.enable_self_consistency or self.enable_evidence_check or self.enable_verification

    @property
    def needs_samples(self) -> bool:
        """Stochastic samples are needed for semantic entropy and self-consistency."""
        return self.enable_semantic_entropy or self.enable_self_consistency


CONFIG_DESCRIPTIONS: dict[str, str] = {
    "C0": "Baseline: LLM only",
    "C1": "RAG only",
    "C2": "RAG + semantic entropy with abstention",
    "C3": "RAG + self-consistency (SelfCheck-NLI) with abstention/escalation",
    "C4": "RAG + Chain-of-Verification",
    "C5": "Full layered pipeline",
    "C6": "Full pipeline + type checks, evidence filtering and claim-guided re-retrieval",
}


def preset(name: str) -> PipelineSettings:
    """Return the settings for one of the C0-C6 configurations."""
    name = name.upper()
    base: dict[str, Any] = {"name": name}
    if name == "C0":
        pass
    elif name == "C1":
        base.update(enable_retrieval=True)
    elif name == "C2":
        base.update(enable_retrieval=True, enable_semantic_entropy=True, enable_decision=True, max_regenerations=0)
    elif name == "C3":
        base.update(enable_retrieval=True, enable_self_consistency=True, enable_decision=True, max_regenerations=0)
    elif name == "C4":
        base.update(enable_retrieval=True, enable_verification=True)
    elif name == "C5":
        base.update(
            enable_query_analysis=True,
            enable_retrieval=True,
            enable_semantic_entropy=True,
            enable_self_consistency=True,
            enable_evidence_check=True,
            enable_verification=True,
            enable_decision=True,
        )
    elif name == "C6":
        base.update(
            enable_query_analysis=True,
            enable_retrieval=True,
            enable_semantic_entropy=True,
            enable_self_consistency=True,
            enable_evidence_check=True,
            enable_verification=True,
            enable_decision=True,
            enable_type_checks=True,
            enable_relevance_filter=True,
            enable_injection_filter=True,
            enable_claim_reretrieval=True,
            max_evidence_age_days=365,
            # answers that agree but differ in added detail ("Spain." vs "Spain won, beating England
            # 2-1") are one meaning; strict bidirectional entailment split them (behaviour test v1)
            se_equivalence="loose",
        )
    else:
        raise ValueError(f"Unknown configuration '{name}'. Use one of {list(CONFIG_DESCRIPTIONS)}.")
    return PipelineSettings(**base)


def build_settings(name: str, overrides: dict[str, Any] | None = None) -> PipelineSettings:
    """Preset plus user overrides (overrides may include a nested 'thresholds' dict).

    Overrides are applied identically to every configuration in an experiment, so decoding and
    sampling settings stay the same across configurations.
    """
    settings = preset(name)
    if not overrides:
        return settings
    data = settings.model_dump()
    for key, value in overrides.items():
        if key == "thresholds" and isinstance(value, dict):
            data["thresholds"].update(value)
        elif key == "name":
            continue
        elif key in data:
            data[key] = value
        else:
            raise ValueError(f"Unknown pipeline setting '{key}'.")
    return PipelineSettings(**data)
