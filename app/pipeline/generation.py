"""Stage 3: answer generation (plain or grounded in retrieved evidence, with citations)."""

from __future__ import annotations

import re

from app.pipeline import prompts
from app.pipeline.retrieval import Passage
from app.pipeline.settings import PipelineSettings
from app.pipeline.trace import Trace
from app.providers import DecodingParams, Message

_CITATION_RE = re.compile(r"\[S(\d+)\]")


def format_evidence(passages: list[Passage]) -> str:
    """Render passages as labelled evidence blocks [S1], [S2], ..."""
    if not passages:
        return "(no passages were retrieved)"
    return "\n\n".join(f"[S{i}] ({p.title}) {p.text}" for i, p in enumerate(passages, start=1))


def build_messages(
    question: str, passages: list[Passage] | None, feedback: str | None = None, guarded: bool = False
) -> list[Message]:
    """Build the chat messages for generation. `passages is None` means retrieval is disabled.
    `guarded` adds the rule that passages are untrusted data (injection filter on)."""
    if passages is None:
        return [Message("system", prompts.BASELINE_SYSTEM), Message("user", question)]
    user = prompts.RAG_USER.format(evidence=format_evidence(passages), question=question)
    if feedback:
        user += feedback
    system = prompts.RAG_SYSTEM_GUARDED if guarded else prompts.RAG_SYSTEM
    return [Message("system", system), Message("user", user)]


def generate_answer(
    trace: Trace,
    question: str,
    passages: list[Passage] | None,
    settings: PipelineSettings,
    temperature: float | None = None,
    seed: int | None = None,
    feedback: str | None = None,
) -> str:
    """Generate one answer. Uses the configuration's decoding settings unless overridden."""
    params = DecodingParams(
        temperature=settings.temperature if temperature is None else temperature,
        top_p=settings.top_p,
        max_tokens=settings.max_tokens,
        seed=settings.seed if seed is None else seed,
    )
    completion = trace.chat(build_messages(question, passages, feedback, settings.enable_injection_filter), params)
    return completion.text.strip()


def sample_answers(
    trace: Trace, question: str, passages: list[Passage] | None, settings: PipelineSettings
) -> list[str]:
    """Draw N stochastic answers (temperature > 0, distinct seeds) from the same prompt."""
    return [
        generate_answer(
            trace, question, passages, settings, temperature=settings.sample_temperature, seed=settings.seed + 1 + i
        )
        for i in range(settings.n_samples)
    ]


def extract_citations(answer: str, passages: list[Passage]) -> list[dict]:
    """Map [S#] markers in the answer to passage source ids (unknown labels are reported as invalid)."""
    cited: list[dict] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(answer or ""):
        n = int(match.group(1))
        if n in seen:
            continue
        seen.add(n)
        if 1 <= n <= len(passages):
            p = passages[n - 1]
            cited.append({"label": f"S{n}", "source_id": p.source_id, "title": p.title, "valid": True})
        else:
            cited.append({"label": f"S{n}", "source_id": None, "title": None, "valid": False})
    return cited


def strip_citations(text: str) -> str:
    """Remove [S#] markers and attribution phrases such as "According to [S1]," (used before NLI
    so the markers do not confuse the model)."""
    marker = r"\[S\d+\](?:\s*(?:,|and)?\s*\[S\d+\])*"
    text = re.sub(
        rf"\b(?:according to|as (?:stated|mentioned|described) in|based on|per|from)\s+(?:passages?\s+)?{marker}\s*,?\s*",
        "", text or "", flags=re.IGNORECASE,
    )
    text = re.sub(rf"\s*\(?{marker}\)?", "", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text).strip()
    return text[:1].upper() + text[1:] if text else text
