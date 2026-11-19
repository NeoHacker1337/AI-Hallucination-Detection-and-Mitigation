"""Stage 1: normalise the question and flag ambiguous or high-risk queries."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

from app.pipeline import prompts
from app.pipeline.trace import Trace
from app.providers import DecodingParams
from app.utils import extract_json

HIGH_RISK_TERMS: dict[str, list[str]] = {
    "medical": [
        "dose", "dosage", "overdose", "medication", "medicine", "drug", "symptom", "diagnos", "cancer",
        "treatment", "pregnan", "vaccine", "antibiotic", "disease", "surgery", "suicide", "self-harm",
    ],
    "legal": ["legal", "illegal", "lawsuit", "sue ", "court", "contract", "lawyer", "attorney", "custody"],
    "financial": ["invest", "stock", "crypto", "loan", "mortgage", "tax", "retirement", "bitcoin"],
    "safety": ["weapon", "explosive", "poison", "toxic", "lethal"],
}

_TIME_SENSITIVE = re.compile(
    r"\b(?:latest|current(?:ly)?|today|now|nowadays|recent(?:ly)?|this (?:year|month|week)|as of|"
    r"so far|newest|up to date|still|incumbent|this season)\b",
    re.IGNORECASE,
)

_PRONOUN_START = re.compile(
    r"^(?:what|who|where|when|why|how|is|was|did|does|do|can|will)\b[^?]*\b"
    r"(?:he|she|him|her|his|they|them|their|it|this|that|these|those)\b",
    re.IGNORECASE,
)


@dataclass
class QueryAnalysis:
    """Result of the query-analysis stage."""

    original: str
    normalized: str
    ambiguous: bool = False
    ambiguity_reason: str | None = None
    high_risk: bool = False
    risk_domains: list[str] = field(default_factory=list)
    mode: str = "rule"
    time_sensitive: bool = False  # asks about recent/current facts: evidence freshness matters

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_question(question: str) -> str:
    """Unicode-normalise and collapse whitespace. Does not change the wording."""
    text = unicodedata.normalize("NFKC", question)
    return re.sub(r"\s+", " ", text).strip()


def detect_high_risk(question: str) -> list[str]:
    """Return the high-risk domains whose keywords appear in the question."""
    lowered = f" {question.lower()} "
    return [domain for domain, terms in HIGH_RISK_TERMS.items() if any(t in lowered for t in terms)]


def rule_ambiguity(question: str) -> tuple[bool, str | None]:
    """Conservative heuristic ambiguity check.

    Flags (a) questions with fewer than three words, and (b) short questions that rely on a
    pronoun but contain no capitalised name, number or quoted entity that could resolve it.
    """
    words = question.split()
    if len(words) < 3:
        return True, "question has fewer than three words"
    tail_words = words[1:]
    has_entity = any(w[:1].isupper() for w in tail_words) or bool(re.search(r"\d|\"", question))
    if len(words) <= 8 and _PRONOUN_START.search(question) and not has_entity:
        return True, "question refers to an unidentified pronoun"
    return False, None


def analyze_query(question: str, mode: str, trace: Trace | None = None) -> QueryAnalysis:
    """Run query analysis in 'rule' mode or 'llm' mode (LLM ambiguity check, rule-based risk)."""
    normalized = normalize_question(question)
    domains = detect_high_risk(normalized)
    result = QueryAnalysis(
        original=question, normalized=normalized, high_risk=bool(domains), risk_domains=domains, mode=mode,
        time_sensitive=bool(_TIME_SENSITIVE.search(normalized)),
    )
    if mode == "llm" and trace is not None:
        completion = trace.complete(
            prompts.QUERY_AMBIGUITY.format(question=normalized), DecodingParams(temperature=0.0, max_tokens=80)
        )
        parsed = extract_json(completion.text)
        if isinstance(parsed, dict) and "ambiguous" in parsed:
            result.ambiguous = bool(parsed.get("ambiguous"))
            result.ambiguity_reason = str(parsed.get("reason") or "") or None
            return result
        result.mode = "rule (llm output unparseable)"
    result.ambiguous, result.ambiguity_reason = rule_ambiguity(normalized)
    return result
