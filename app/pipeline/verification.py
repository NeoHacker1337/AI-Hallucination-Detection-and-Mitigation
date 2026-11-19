"""Stage 6: Chain-of-Verification (Dhuliawala et al., 2023), "factored" variant.

1. Plan: one verification question per weak claim (single LLM call, JSON list).
2. Execute: answer each verification question independently - the model never sees the draft
   answer, which is what stops it from copying its own mistake. If retrieval is enabled, each
   question gets its own retrieval.
3. Revise: rewrite the draft so it agrees with the verification answers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from app.pipeline import prompts
from app.pipeline.generation import format_evidence
from app.pipeline.retrieval import KnowledgeBase, Passage
from app.pipeline.settings import PipelineSettings
from app.pipeline.trace import Trace
from app.providers import DecodingParams, Message
from app.utils import extract_json


@dataclass
class VerificationResult:
    """Outputs of the three CoVe steps."""

    weak_claims: list[str]
    questions: list[str]
    answers: list[str]
    revised_answer: str
    plan_parsed: bool
    extra_passages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def plan_questions(trace: Trace, question: str, claims: list[str]) -> tuple[list[str], bool]:
    """Ask the model for one verification question per claim. Falls back to templated questions."""
    claims_text = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
    completion = trace.complete(
        prompts.COVE_PLAN.format(question=question, claims=claims_text),
        DecodingParams(temperature=0.0, max_tokens=60 * len(claims) + 40),
    )
    parsed = extract_json(completion.text)
    if isinstance(parsed, list) and len(parsed) >= 1 and all(isinstance(q, str) for q in parsed):
        questions = [q.strip() for q in parsed][: len(claims)]
        while len(questions) < len(claims):
            questions.append(f"Is it true that {claims[len(questions)].rstrip('.')}?")
        return questions, True
    return [f"Is it true that {c.rstrip('.')}?" for c in claims], False


def chain_of_verification(
    trace: Trace,
    question: str,
    draft: str,
    weak_claims: list[str],
    passages: list[Passage] | None,
    settings: PipelineSettings,
    kb: KnowledgeBase | None = None,
    collection: str | None = None,
) -> VerificationResult:
    """Run plan -> independent answers -> revision. Must be called inside an open trace stage."""
    claims = weak_claims[: settings.max_verification_questions]
    questions, parsed = plan_questions(trace, question, claims)
    answers: list[str] = []
    extra: list[dict] = []
    params = DecodingParams(temperature=0.0, top_p=settings.top_p, max_tokens=120, seed=settings.seed)
    for vq in questions:
        if passages is not None and kb is not None and collection:
            vq_passages = kb.search(collection, vq, settings.top_k)
            extra.extend(p.to_dict() for p in vq_passages)
            messages = [
                Message("system", prompts.COVE_ANSWER_SYSTEM),
                Message("user", f"Evidence passages:\n{format_evidence(vq_passages)}\n\nQuestion: {vq}\nAnswer:"),
            ]
        else:
            messages = [Message("system", prompts.COVE_ANSWER_NO_EVIDENCE_SYSTEM), Message("user", vq)]
        answers.append(trace.chat(messages, params).text.strip())
    qa = "\n".join(f"Q{i}: {q}\nA{i}: {a}" for i, (q, a) in enumerate(zip(questions, answers), start=1))
    evidence_block = f"Evidence passages:\n{format_evidence(passages)}\n\n" if passages is not None else ""
    revised = trace.complete(
        prompts.COVE_REVISE.format(question=question, draft=draft, qa=qa, evidence_block=evidence_block),
        DecodingParams(temperature=0.0, top_p=settings.top_p, max_tokens=settings.max_tokens, seed=settings.seed),
    ).text.strip()
    return VerificationResult(
        weak_claims=claims,
        questions=questions,
        answers=answers,
        revised_answer=revised or draft,
        plan_parsed=parsed,
        extra_passages=extra,
    )
