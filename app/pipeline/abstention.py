"""Detect whether a text is an abstention ("I don't know", "cannot answer", ...)."""

from __future__ import annotations

import re

_ABSTAIN_PATTERNS = [
    r"\bi (?:do not|don't|dont) know\b",
    r"\bi(?:'m| am) (?:not sure|unsure|uncertain)\b",
    r"\bi (?:cannot|can't|can not|am unable to|'m unable to) (?:answer|determine|say|tell|find|provide)\b",
    r"\b(?:the )?(?:provided |given )?(?:passages|evidence|context|documents?) (?:do|does) not "
    r"(?:contain|provide|mention|include|say|specify|state)\b",
    r"\bthere is no (?:information|evidence|mention)\b",
    r"\bno (?:information|evidence) (?:is )?(?:available|provided|given)\b",
    r"\bnot enough (?:information|evidence)\b",
    r"\b(?:do not|don't) have enough (?:verified |reliable )?(?:information|evidence)\b",
    r"\bsent for human review\b",
    r"^\s*unknown\.?\s*$",
]
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS), re.IGNORECASE)


def is_abstention(text: str | None) -> bool:
    """True if the text declines to answer.

    Only the first 200 characters are inspected, so an answer that gives a fact and later says
    "the passages do not mention X" is *not* treated as an abstention.
    """
    if text is None:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    return bool(_ABSTAIN_RE.search(stripped[:200]))
