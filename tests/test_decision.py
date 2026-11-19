"""Decision logic, configurations and the orchestrator control flow (with fake LLM + NLI)."""

from __future__ import annotations

import pytest

from app.pipeline.decision import Decision, Signals, compute_confidence, decide, is_flagged
from app.pipeline.evidence_check import CONTRADICTED, NOT_ENOUGH, SUPPORTED, label_claim
from app.pipeline.settings import Thresholds, build_settings, preset

T = Thresholds()


def test_confidence_is_mean_of_available_components() -> None:
    assert compute_confidence(Signals()) is None
    assert compute_confidence(Signals(semantic_entropy_norm=0.2)) == pytest.approx(0.8)
    assert compute_confidence(Signals(semantic_entropy_norm=0.2, selfcheck_score=0.4, supported_fraction=1.0)) == \
        pytest.approx((0.8 + 0.6 + 1.0) / 3)


def test_abstention_and_ambiguity_lead_to_uncertain() -> None:
    assert decide(Signals(answer_is_abstention=True), T).decision == Decision.UNCERTAIN
    res = decide(Signals(ambiguous=True, ambiguity_reason="pronoun", semantic_entropy_norm=0.0), T)
    assert res.decision == Decision.UNCERTAIN and "ambiguous" in res.reasons[0]
    # clarification switched off: ambiguity is ignored
    assert decide(Signals(ambiguous=True, semantic_entropy_norm=0.0), T, ask_clarification=False).decision == Decision.VERIFIED


def test_contradiction_fails_before_entropy_is_considered() -> None:
    s = Signals(contradicted_fraction=0.5, semantic_entropy_norm=0.9, supported_fraction=0.5)
    assert decide(s, T).decision == Decision.FAILED


def test_selfcheck_failure() -> None:
    assert decide(Signals(selfcheck_score=0.7), T).decision == Decision.FAILED
    assert decide(Signals(selfcheck_score=0.1), T).decision == Decision.VERIFIED


def test_high_entropy_and_low_support_abstain() -> None:
    assert decide(Signals(semantic_entropy_norm=0.8), T).decision == Decision.UNCERTAIN
    assert decide(Signals(supported_fraction=0.2, contradicted_fraction=0.0), T).decision == Decision.UNCERTAIN


def test_threshold_boundaries_are_inclusive() -> None:
    assert decide(Signals(semantic_entropy_norm=T.entropy_uncertain), T).decision == Decision.UNCERTAIN
    assert decide(Signals(selfcheck_score=T.selfcheck_fail), T).decision == Decision.FAILED


def test_high_risk_requires_more_confidence() -> None:
    s = Signals(semantic_entropy_norm=0.35)  # confidence 0.65
    assert decide(s, T).decision == Decision.VERIFIED
    s.high_risk = True  # requires 0.6 + 0.15 = 0.75
    assert decide(s, T).decision == Decision.UNCERTAIN


def test_no_signals_passes_by_default() -> None:
    res = decide(Signals(), T)
    assert res.decision == Decision.VERIFIED and res.confidence is None


def test_configurable_thresholds() -> None:
    strict = Thresholds(entropy_uncertain=0.1)
    assert decide(Signals(semantic_entropy_norm=0.2), strict).decision == Decision.UNCERTAIN


def test_flagged() -> None:
    assert is_flagged(Decision.FAILED) and is_flagged(Decision.UNCERTAIN) and is_flagged(Decision.NEEDS_REVIEW)
    assert not is_flagged(Decision.VERIFIED) and not is_flagged(Decision.UNCHECKED)


def test_evidence_labels() -> None:
    assert label_claim(0.9, 0.1, 0.5, 0.5) == SUPPORTED
    assert label_claim(0.1, 0.8, 0.5, 0.5) == CONTRADICTED
    assert label_claim(0.3, 0.3, 0.5, 0.5) == NOT_ENOUGH
    # one passage entails the claim, another (off-topic) one "contradicts" it: supported wins
    assert label_claim(0.97, 0.974, 0.5, 0.5) == SUPPORTED
    assert label_claim(0.4, 0.7, 0.5, 0.5) == CONTRADICTED


def test_presets_switch_the_right_stages() -> None:
    c0, c5 = preset("C0"), preset("C5")
    assert not any([c0.enable_retrieval, c0.enable_decision, c0.needs_samples, c0.needs_claims])
    assert all([c5.enable_query_analysis, c5.enable_retrieval, c5.enable_semantic_entropy,
                c5.enable_self_consistency, c5.enable_evidence_check, c5.enable_verification, c5.enable_decision])
    assert preset("C2").needs_samples and not preset("C2").needs_claims
    assert preset("C4").needs_claims and not preset("C4").enable_decision
    with pytest.raises(ValueError):
        preset("C9")


def test_overrides_apply_and_validate() -> None:
    s = build_settings("C5", {"n_samples": 3, "thresholds": {"entropy_uncertain": 0.4}})
    assert s.n_samples == 3 and s.thresholds.entropy_uncertain == 0.4 and s.enable_decision
    with pytest.raises(ValueError):
        build_settings("C1", {"no_such_setting": 1})


# ---------------------------------------------------------------- orchestrator with fakes
class FakeProvider:
    """Deterministic fake LLM: returns scripted answers; samples are copies of the greedy answer."""

    prefix = "fake"

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls = 0
        self.model = "fake"

    @property
    def spec(self) -> str:
        return "fake:fake"

    def check_available(self) -> None:
        return None

    def chat(self, messages, params):  # type: ignore[no-untyped-def]
        from app.providers import Completion

        self.calls += 1
        text = self.answers[min(self.calls - 1, len(self.answers) - 1)]
        return Completion(text=text, model=self.spec, prompt_tokens=10, completion_tokens=5, latency_ms=1.0)

    def complete(self, prompt, params, system=None):  # type: ignore[no-untyped-def]
        return self.chat([], params)


class FakeKB:
    def __init__(self, text: str) -> None:
        from app.pipeline.retrieval import Passage

        self.passage = Passage(1, "doc1#c0", "Doc", text, 0.9)

    def search(self, collection, query, k):  # type: ignore[no-untyped-def]
        return [self.passage]


def test_c0_runs_one_call_and_is_unchecked(keyword_nli) -> None:  # type: ignore[no-untyped-def]
    from app.pipeline.orchestrator import Pipeline

    provider = FakeProvider(["Paris is the capital of France."])
    res = Pipeline(provider, preset("C0"), kb=None, nli=keyword_nli).run("What is the capital of France?")  # type: ignore[arg-type]
    assert res.decision == Decision.UNCHECKED
    assert res.llm_calls == 1 and res.prompt_tokens == 10
    assert res.final_answer == "Paris is the capital of France."
    assert [s.stage for s in res.stages] == ["generation", "final_assessment"]
    assert res.assessment is not None and res.assessment.verification_status == "unchecked"


def test_c2_consistent_samples_verify(keyword_nli) -> None:  # type: ignore[no-untyped-def]
    from app.pipeline.orchestrator import Pipeline

    provider = FakeProvider(["Paris is the capital of France [S1]."])
    kb = FakeKB("Paris is the capital of France.")
    res = Pipeline(provider, preset("C2"), kb=kb, nli=keyword_nli).run("What is the capital of France?")  # type: ignore[arg-type]
    assert res.decision == Decision.VERIFIED
    assert res.first_signals is not None and res.first_signals.semantic_entropy_norm == pytest.approx(0.0)
    assert res.llm_calls == 1 + preset("C2").n_samples
    assert res.citations and res.citations[0]["source_id"] == "doc1#c0"


def test_c5_contradicted_answer_is_regenerated_then_escalated(keyword_nli) -> None:  # type: ignore[no-untyped-def]
    from app.pipeline.orchestrator import Pipeline

    settings = build_settings("C5", {"max_regenerations": 1, "enable_verification": False, "n_samples": 2})
    provider = FakeProvider(["Paris is not the capital of France."])
    kb = FakeKB("Paris is the capital of France.")
    res = Pipeline(provider, settings, kb=kb, nli=keyword_nli).run("What is the capital of France?")  # type: ignore[arg-type]
    assert res.first_decision == Decision.FAILED
    assert res.n_regenerations == 1
    assert res.decision == Decision.NEEDS_REVIEW and res.escalated and res.abstained
    assert res.candidate_answer == "Paris is not the capital of France."
