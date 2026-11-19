"""Retrieved content is untrusted data. This module keeps it from acting as instructions.

Two layers (both only with `enable_injection_filter`):
1. Passages that contain instruction-like text aimed at the model ("ignore previous instructions",
   "you are now ...", fake system/assistant turns, ...) are quarantined: they are not shown to the
   model and are listed in the trace so the decision can be audited.
2. The grounded-generation system prompt tells the model that passages are data and that any
   instruction inside them must be ignored (see prompts.RAG_SYSTEM_GUARDED).

The pipeline has no tools that act on the world (it only generates text and queries a read-only
knowledge base), so retrieved text cannot trigger actions; this filter protects the *answer*.
Pattern matching cannot catch every injection; it is a documented, conservative first layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.pipeline.retrieval import Passage

_INJECTION_PATTERNS = [
    r"\bignore (?:all |any |the )?(?:previous|prior|above|earlier|preceding) (?:instructions?|prompts?|messages?|rules?)",
    r"\bdisregard (?:all |any |the )?(?:previous|prior|above|earlier|system) (?:instructions?|prompts?|rules?)",
    r"\bforget (?:all |everything |your )(?:previous |prior )?(?:instructions?|rules?|you were told)",
    r"\byou are now (?:a|an|in|the)\b",
    r"\b(?:new|updated|override) (?:system )?instructions?\s*:",
    r"\bsystem prompt\b",
    r"\b(?:act|behave|respond) as (?:if you were |though you are )?(?:a|an|the) (?:different|new|unrestricted)\b",
    r"\bdo not (?:cite|mention|reveal) (?:this|these|the) (?:passage|source|instruction)s?\b",
    r"\b(?:answer|reply|respond) (?:only )?with [\"'“]",
    r"^\s*(?:system|assistant|developer)\s*:",
    r"<\s*/?\s*(?:system|instructions?|im_start|im_end)\s*>",
    r"\[\s*/?\s*(?:INST|SYS)\s*\]",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)


@dataclass
class InjectionFinding:
    """A quarantined passage and the text that triggered it."""

    source_id: str
    title: str
    matched: str

    def to_dict(self) -> dict:
        return {"source_id": self.source_id, "title": self.title, "matched": self.matched}


def find_injection(text: str) -> str | None:
    """Return the first instruction-like fragment in the text, or None."""
    match = _INJECTION_RE.search(text or "")
    return match.group(0) if match else None


def filter_passages(passages: list[Passage]) -> tuple[list[Passage], list[InjectionFinding]]:
    """Split passages into (kept, quarantined findings)."""
    kept: list[Passage] = []
    findings: list[InjectionFinding] = []
    for p in passages:
        hit = find_injection(p.text) or find_injection(p.title)
        if hit:
            findings.append(InjectionFinding(p.source_id, p.title, hit))
        else:
            kept.append(p)
    return kept, findings
