"""Stage 4: split an answer into short atomic claims.

Two modes:
* "rule" (default): deterministic sentence splitting plus splitting of coordinated clauses.
  Reproducible and free, but coarser than LLM extraction.
* "llm": the model lists atomic facts as JSON; falls back to "rule" if the output is unparseable
  (the fallback is recorded in the stage log).
"""

from __future__ import annotations

import re

from app.pipeline import prompts
from app.pipeline.abstention import is_abstention
from app.pipeline.generation import strip_citations
from app.pipeline.trace import Trace
from app.providers import DecodingParams
from app.utils import extract_json

_ABBREVIATIONS = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "vs.", "etc.", "e.g.", "i.e.", "U.S.", "U.K.",
    "Inc.", "Ltd.", "Co.", "No.", "approx.", "Fig.", "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sept.", "Oct.",
    "Nov.", "Dec.", "a.m.", "p.m.",
]
_PLACEHOLDER = "<<DOT>>"
_FILLER = re.compile(
    r"^(?:i hope this helps|let me know|feel free|in summary|in conclusion|overall|note that|"
    r"please note|as an ai|sure|certainly|of course)\b",
    re.IGNORECASE,
)
_CLAUSE_SPLIT = re.compile(r";\s+|,\s+(?:and|but|while|whereas)\s+|\s+but\s+|\s+whereas\s+", re.IGNORECASE)
_MIN_WORDS = 3


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, protecting common abbreviations and decimal numbers."""
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _PLACEHOLDER))
    protected = re.sub(r"(\d)\.(\d)", rf"\1{_PLACEHOLDER}\2", protected)
    protected = re.sub(r"\b([A-Z])\.", rf"\1{_PLACEHOLDER}", protected)  # initials such as "J. K."
    parts = re.split(r"(?<=[.!?])\s+|\n+", protected)
    sentences = [p.replace(_PLACEHOLDER, ".").strip(" -*\t") for p in parts]
    return [s for s in sentences if s]


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def rule_based_claims(answer: str, max_claims: int = 12) -> list[str]:
    """Deterministic claim extraction.

    1. remove citation markers and list bullets;
    2. split into sentences;
    3. split each sentence at ';' and at coordinating conjunctions (', and', ', but', 'whereas'),
       but only if every resulting part has at least three words (so "salt and pepper" stays whole);
    4. drop filler sentences and fragments shorter than three words; de-duplicate.
    """
    if not answer or is_abstention(answer):
        return []
    text = strip_citations(answer)
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text, flags=re.MULTILINE)
    claims: list[str] = []
    seen: set[str] = set()
    for sentence in split_sentences(text):
        if _FILLER.match(sentence):
            continue
        pieces = [p.strip(" ,;") for p in _CLAUSE_SPLIT.split(sentence)]
        if len(pieces) > 1 and all(_word_count(p) >= _MIN_WORDS for p in pieces):
            candidates = pieces
        else:
            candidates = [sentence.strip()]
        for cand in candidates:
            cand = cand.strip()
            if _word_count(cand) < _MIN_WORDS:
                continue
            if not cand.endswith((".", "!", "?")):
                cand += "."
            cand = cand[0].upper() + cand[1:]
            key = cand.lower()
            if key not in seen:
                seen.add(key)
                claims.append(cand)
    return claims[:max_claims]


def llm_claims(trace: Trace, question: str, answer: str, max_claims: int = 12) -> tuple[list[str], bool]:
    """LLM-based extraction. Returns (claims, used_fallback)."""
    if not answer or is_abstention(answer):
        return [], False
    completion = trace.complete(
        prompts.CLAIM_EXTRACTION.format(question=question, answer=strip_citations(answer)),
        DecodingParams(temperature=0.0, max_tokens=400),
    )
    parsed = extract_json(completion.text)
    if isinstance(parsed, list) and all(isinstance(c, str) for c in parsed) and parsed:
        cleaned = [c.strip() for c in parsed if _word_count(c) >= 2]
        return cleaned[:max_claims], False
    return rule_based_claims(answer, max_claims), True


def extract_claims(
    trace: Trace | None, question: str, answer: str, mode: str = "rule", max_claims: int = 12
) -> tuple[list[str], str]:
    """Extract claims with the chosen mode. Returns (claims, mode_actually_used)."""
    if mode == "llm" and trace is not None:
        claims, fallback = llm_claims(trace, question, answer, max_claims)
        return claims, "rule (llm fallback)" if fallback else "llm"
    return rule_based_claims(answer, max_claims), "rule"
