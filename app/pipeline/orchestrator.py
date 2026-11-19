"""Runs one question through a configuration and records every stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.pipeline import prompts
from app.pipeline.abstention import is_abstention
from app.pipeline.assessment import Assessment, assess
from app.pipeline.claims import extract_claims
from app.pipeline.decision import Decision, DecisionResult, Signals, decide
from app.pipeline.evidence_check import SUPPORTED, EvidenceResult, evidence_check
from app.pipeline.evidence_quality import FilteredKnowledgeBase, rescore_with_embeddings, validate_evidence
from app.pipeline.generation import extract_citations, generate_answer, sample_answers
from app.pipeline.nli import NLIScorer, get_nli
from app.pipeline.query_analysis import QueryAnalysis, analyze_query
from app.pipeline.retrieval import KnowledgeBase, Passage, get_knowledge_base
from app.pipeline.self_consistency import SelfCheckResult, self_consistency
from app.pipeline.semantic_entropy import SemanticEntropyResult, semantic_entropy
from app.pipeline.settings import PipelineSettings
from app.pipeline.trace import StageRecord, Trace
from app.pipeline.type_checks import TypeCheckResult, run_type_checks
from app.pipeline.verification import chain_of_verification
from app.pipeline.web_search import wikipedia_search
from app.providers import LLMProvider

UNCERTAIN_MESSAGE = prompts.INSUFFICIENT_EVIDENCE_MESSAGE
REVIEW_MESSAGE = "I'm not confident in this answer, so it has been sent for human review."


@dataclass
class Detection:
    """Outputs of the detection stages for one answer."""

    claims: list[str] = field(default_factory=list)
    samples: list[str] | None = None
    se: SemanticEntropyResult | None = None
    selfcheck: SelfCheckResult | None = None
    evidence: EvidenceResult | None = None
    type_checks: TypeCheckResult | None = None

    def signals(self, analysis: QueryAnalysis | None, answer: str) -> Signals:
        """Collapse detector outputs into decision signals."""
        return Signals(
            semantic_entropy_norm=self.se.normalized_entropy if self.se else None,
            selfcheck_score=self.selfcheck.answer_score if self.selfcheck else None,
            supported_fraction=self.evidence.supported_fraction if self.evidence else None,
            contradicted_fraction=self.evidence.contradicted_fraction if self.evidence else None,
            answer_is_abstention=is_abstention(answer),
            ambiguous=bool(analysis and analysis.ambiguous),
            ambiguity_reason=analysis.ambiguity_reason if analysis else None,
            high_risk=bool(analysis and analysis.high_risk),
            hard_issues=[i.detail for i in self.type_checks.hard_issues] if self.type_checks else None,
        )


@dataclass
class PipelineResult:
    """Everything produced by one run; persisted by `app.db.crud.save_result`."""

    question: str
    normalized_question: str
    model: str
    config_name: str
    settings: dict[str, Any]
    passages: list[Passage]
    initial_answer: str
    candidate_answer: str
    final_answer: str
    decision: Decision
    decision_reasons: list[str]
    first_decision: Decision | None
    first_confidence: float | None
    confidence: float | None
    first_signals: Signals | None
    final_signals: Signals | None
    abstained: bool
    escalated: bool
    n_regenerations: int
    citations: list[dict]
    stages: list[StageRecord]
    latency_ms: float
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    assessment: Assessment | None = None


class Pipeline:
    """Layered hallucination detection and mitigation workflow."""

    def __init__(
        self,
        provider: LLMProvider,
        settings: PipelineSettings,
        kb: KnowledgeBase | None = None,
        nli: NLIScorer | None = None,
        collection: str = "global",
    ) -> None:
        self.provider = provider
        self.s = settings
        self.collection = collection
        self.kb = kb if kb is not None else (get_knowledge_base() if settings.enable_retrieval else None)
        needs_nli = (settings.enable_semantic_entropy or settings.enable_self_consistency
                     or settings.enable_evidence_check or settings.enable_type_checks)
        self.nli = nli if nli is not None else (get_nli() if needs_nli else None)
        self._notes: list[str] = []
        self._time_sensitive = False

    # ------------------------------------------------------------------------------------
    @property
    def _validates_evidence(self) -> bool:
        s = self.s
        return (s.enable_relevance_filter or s.enable_injection_filter or s.web_search != "none"
                or s.max_evidence_age_days is not None)

    def _retrieve(
        self, trace: Trace, question: str, k: int, label: str = "retrieval", extra_queries: list[str] | None = None
    ) -> list[Passage] | None:
        if not self.s.enable_retrieval:
            return None
        assert self.kb is not None
        with trace.stage(label) as st:
            if not self._validates_evidence and not extra_queries:  # C1-C5: unchanged behaviour
                passages = self.kb.search(self.collection, question, k)
                st.output = {"collection": self.collection, "top_k": k, "passages": [p.to_dict() for p in passages]}
                return passages
            out: dict[str, Any] = {"collection": self.collection, "top_k": k}
            passages = []
            try:
                passages = self.kb.search(self.collection, question, k)
            except LookupError as exc:
                if self.s.web_search == "none":
                    raise
                out["knowledge_base_error"] = str(exc)
                self._notes.append(f"local knowledge base unavailable: {exc}")
            if extra_queries:  # claim-guided re-retrieval (Algorithm 1: Re-retrieve(q, r))
                out["extra_queries"] = extra_queries
                for eq in extra_queries:
                    try:
                        passages.extend(self.kb.search(self.collection, eq, 2))
                    except LookupError:
                        break
            if self.s.web_search == "wikipedia":
                web = wikipedia_search(question, self.s.web_top_k)
                out["web_search"] = {"provider": "wikipedia", "query": web.query, "error": web.error,
                                     "n_results": len(web.passages)}
                if web.error:
                    self._notes.append(f"web search failed ({web.error}); answered from local evidence only")
                rescore_with_embeddings(self.kb, question, web.passages)
                passages.extend(web.passages)
            unique: dict[str, Passage] = {}
            for p in passages:
                if p.source_id not in unique or p.score > unique[p.source_id].score:
                    unique[p.source_id] = p
            passages = sorted(unique.values(), key=lambda p: p.score, reverse=True)
            report = validate_evidence(passages, self.s, self._time_sensitive)
            out.update(report.to_dict())
            out["passages"] = out.pop("kept")
            if report.dropped_irrelevant:
                self._notes.append(f"{len(report.dropped_irrelevant)} retrieved passage(s) dropped as irrelevant")
            if report.quarantined:
                self._notes.append(f"{len(report.quarantined)} passage(s) quarantined: they contained instructions "
                                   "aimed at the model")
            if report.stale:
                self._notes.append(f"{len(report.stale)} passage(s) are older than {self.s.max_evidence_age_days} "
                                   "days for a time-sensitive question; the answer may be outdated")
            if self._time_sensitive and report.kept and report.undated == len(report.kept):
                self._notes.append("time-sensitive question, but no retrieved passage has a known date")
            if not report.kept:
                self._notes.append("no relevant evidence passed validation")
            st.output = out
        return report.kept

    def _detect(
        self, trace: Trace, question: str, answer: str, passages: list[Passage] | None,
        samples: list[str] | None, se: SemanticEntropyResult | None, suffix: str = "",
    ) -> Detection:
        """Run claim extraction, sampling, semantic entropy, self-consistency and evidence check."""
        s = self.s
        det = Detection(samples=samples, se=se)
        if s.needs_claims:
            with trace.stage(f"claim_extraction{suffix}") as st:
                det.claims, mode_used = extract_claims(trace, question, answer, s.claim_extraction, s.max_claims)
                st.output = {"mode": mode_used, "claims": det.claims}
        if s.needs_samples and det.samples is None:
            with trace.stage(f"sampling{suffix}") as st:
                det.samples = sample_answers(trace, question, passages, s)
                st.output = {"n": len(det.samples), "temperature": s.sample_temperature, "samples": det.samples}
        if s.enable_semantic_entropy and det.se is None and det.samples:
            assert self.nli is not None
            with trace.stage(f"semantic_entropy{suffix}") as st:
                det.se = semantic_entropy(question, det.samples, self.nli, s.se_equivalence)
                st.output = det.se.to_dict()
        if s.enable_self_consistency and det.samples:
            assert self.nli is not None
            with trace.stage(f"self_consistency{suffix}") as st:
                det.selfcheck = self_consistency(det.claims, det.samples, self.nli)
                st.output = {
                    "claim_scores": dict(zip(det.claims, det.selfcheck.claim_scores)),
                    "answer_score": det.selfcheck.answer_score,
                }
        if s.enable_evidence_check:
            assert self.nli is not None
            with trace.stage(f"evidence_check{suffix}") as st:
                det.evidence = evidence_check(
                    det.claims, passages or [], self.nli, s.thresholds.support_entail, s.thresholds.contradict
                )
                st.output = det.evidence.to_dict()
        if s.enable_type_checks:
            with trace.stage(f"type_checks{suffix}") as st:
                det.type_checks = run_type_checks(
                    answer, det.claims, passages, self.nli, s.thresholds.logic_contradiction
                )
                st.output = det.type_checks.to_dict()
        return det

    def _weak_claims(self, det: Detection) -> list[str]:
        """Claims to verify: not supported by evidence, or inconsistent with samples. Without any
        claim-level detector every claim is treated as weak."""
        if det.evidence is None and det.selfcheck is None:
            return list(det.claims)
        weak: list[str] = []
        for i, claim in enumerate(det.claims):
            bad = False
            if det.evidence is not None and det.evidence.claims[i].label != SUPPORTED:
                bad = True
            if det.selfcheck is not None and det.selfcheck.claim_scores[i] >= self.s.thresholds.weak_claim_selfcheck:
                bad = True
            if bad:
                weak.append(claim)
        return weak

    @staticmethod
    def _bad_claims(det: Detection) -> list[str]:
        if det.evidence is None:
            return list(det.claims)
        return [c.claim for c in det.evidence.claims if c.label != SUPPORTED]

    # ------------------------------------------------------------------------------------
    def run(self, question: str) -> PipelineResult:
        """Run the configured pipeline on one question."""
        s = self.s
        trace = Trace(self.provider)
        self._notes = []
        self._time_sensitive = False

        analysis: QueryAnalysis | None = None
        q = question.strip()
        if s.enable_query_analysis:
            with trace.stage("query_analysis") as st:
                analysis = analyze_query(question, s.query_analysis_mode, trace)
                q = analysis.normalized
                self._time_sensitive = analysis.time_sensitive
                st.output = analysis.to_dict()

        k = s.top_k
        passages = self._retrieve(trace, q, k)

        with trace.stage("generation") as st:
            answer = generate_answer(trace, q, passages, s)
            st.output = {"answer": answer, "citations": extract_citations(answer, passages or [])}
        initial_answer = answer

        first_signals: Signals | None = None
        final_signals: Signals | None = None
        first_decision: Decision | None = None
        first_confidence: float | None = None
        result = DecisionResult(Decision.UNCHECKED, None, ["decision stage disabled"])
        n_regen = 0
        escalated = False

        any_check = s.needs_claims or s.needs_samples or s.enable_decision
        last_det: Detection | None = None
        for attempt in range(s.max_regenerations + 1):
            suffix = "" if attempt == 0 else f"_regen{attempt}"
            if not any_check:
                break
            if is_abstention(answer):
                final_signals = Signals(answer_is_abstention=True)
                if attempt == 0:
                    first_signals = final_signals
                    if s.enable_decision:
                        first_decision, first_confidence = Decision.UNCERTAIN, None
                if s.enable_decision:
                    with trace.stage(f"decision{suffix}") as st:
                        result = DecisionResult(Decision.UNCERTAIN, None, ["model abstained"])
                        st.output = result.to_dict()
                break

            det = self._detect(trace, q, answer, passages, None, None, suffix)
            signals = det.signals(analysis, answer)
            if attempt == 0:
                first_signals = signals
                if s.enable_decision:
                    first = decide(signals, s.thresholds, s.ask_clarification_on_ambiguous)
                    first_decision, first_confidence = first.decision, first.confidence

            if s.enable_verification and det.claims:
                weak = self._weak_claims(det)
                if weak:
                    with trace.stage(f"verification{suffix}") as st:
                        cove_kb = FilteredKnowledgeBase(self.kb, s) if (self.kb and self._validates_evidence) else self.kb
                        cove = chain_of_verification(
                            trace, q, answer, weak, passages, s, cove_kb, self.collection  # type: ignore[arg-type]
                        )
                        st.output = cove.to_dict()
                        if isinstance(cove_kb, FilteredKnowledgeBase):
                            st.output["evidence_filtering"] = cove_kb.reports
                    answer = cove.revised_answer
                    if not is_abstention(answer) and (s.enable_decision or s.needs_samples):
                        det = self._detect(trace, q, answer, passages, det.samples, det.se, suffix + "_post_cove")
                    signals = det.signals(analysis, answer)
            final_signals = signals
            last_det = det

            if not s.enable_decision:
                break
            with trace.stage(f"decision{suffix}") as st:
                result = decide(signals, s.thresholds, s.ask_clarification_on_ambiguous)
                st.output = {**result.to_dict(), "signals": signals.to_dict()}
            if result.decision != Decision.FAILED:
                break
            if attempt < s.max_regenerations:
                n_regen += 1
                k += s.regen_topk_increment
                bad = self._bad_claims(det)
                extra = bad[: s.max_verification_questions] if s.enable_claim_reretrieval else None
                passages = (self._retrieve(trace, q, k, label=f"retrieval_regen{n_regen}", extra_queries=extra)
                            if s.enable_retrieval else None)
                feedback = (
                    prompts.REGENERATION_FEEDBACK.format(bad_claims="\n".join(f"- {c}" for c in bad))
                    if bad and passages is not None else None
                )
                with trace.stage(f"regeneration{n_regen}") as st:
                    answer = generate_answer(trace, q, passages, s, feedback=feedback)
                    st.output = {"answer": answer, "feedback": feedback, "top_k": k}
            else:
                result = DecisionResult(
                    Decision.NEEDS_REVIEW, result.confidence,
                    result.reasons + [f"still FAILED after {n_regen} regeneration(s)"],
                )
                escalated = True

        candidate = answer
        clarification = False
        if result.decision in (Decision.VERIFIED, Decision.UNCHECKED):
            final = candidate
            abstained = is_abstention(candidate)
        elif result.decision == Decision.UNCERTAIN:
            if analysis and analysis.ambiguous and s.ask_clarification_on_ambiguous and not is_abstention(candidate):
                final = f"Your question seems ambiguous ({analysis.ambiguity_reason}). Could you clarify what you mean?"
                clarification = True
            else:
                final = UNCERTAIN_MESSAGE
            abstained = True
        else:  # NEEDS_REVIEW
            final = REVIEW_MESSAGE
            abstained = True

        final_citations = extract_citations(final, passages or []) if not abstained else []
        with trace.stage("final_assessment") as st:
            assessment = assess(
                result.decision, abstained, clarification, final_signals,
                last_det.evidence if last_det else None, last_det.type_checks if last_det else None,
                passages or [], final_citations, self._notes, s.enable_evidence_check,
            )
            st.output = assessment.to_dict()

        return PipelineResult(
            question=question,
            normalized_question=q,
            model=self.provider.spec,
            config_name=s.name,
            settings=s.model_dump(),
            passages=passages or [],
            initial_answer=initial_answer,
            candidate_answer=candidate,
            final_answer=final,
            decision=result.decision,
            decision_reasons=result.reasons,
            first_decision=first_decision,
            first_confidence=first_confidence,
            confidence=result.confidence,
            first_signals=first_signals,
            final_signals=final_signals,
            abstained=abstained,
            escalated=escalated,
            n_regenerations=n_regen,
            citations=final_citations,
            stages=trace.stages,
            latency_ms=trace.total_latency_ms,
            llm_calls=trace.llm_calls,
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            assessment=assessment,
        )
