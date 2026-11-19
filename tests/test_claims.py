"""Claim extraction (rule-based) and related text helpers."""

from __future__ import annotations

from app.pipeline.abstention import is_abstention
from app.pipeline.claims import rule_based_claims, split_sentences
from app.pipeline.generation import extract_citations, strip_citations
from app.pipeline.retrieval import Passage


def test_sentences_are_split_and_abbreviations_kept() -> None:
    text = "Dr. Smith lives in the U.S. and works at Acme Inc. today. He earns 3.5 million dollars a year!"
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].startswith("Dr. Smith")
    assert "3.5 million" in sentences[1]


def test_citations_removed_from_claims() -> None:
    claims = rule_based_claims("Jane Austen wrote Pride and Prejudice [S1]. It was published in 1813 [S2][S3].")
    assert claims == ["Jane Austen wrote Pride and Prejudice.", "It was published in 1813."]


def test_coordinated_clauses_are_split_when_both_sides_are_clauses() -> None:
    claims = rule_based_claims("Mercury is the closest planet to the Sun, and it has no moons.")
    assert claims == ["Mercury is the closest planet to the Sun.", "It has no moons."]


def test_short_coordination_is_not_split() -> None:
    claims = rule_based_claims("The dish contains salt and pepper and some olive oil.")
    assert claims == ["The dish contains salt and pepper and some olive oil."]


def test_filler_short_fragments_and_duplicates_dropped() -> None:
    answer = "Water boils at 100 degrees Celsius. Water boils at 100 degrees Celsius. Yes. I hope this helps!"
    assert rule_based_claims(answer) == ["Water boils at 100 degrees Celsius."]


def test_semicolons_split_and_bullets_stripped() -> None:
    answer = "- The wall fell in 1989; Germany reunified in 1990.\n- Berlin became the capital again."
    assert rule_based_claims(answer) == [
        "The wall fell in 1989.", "Germany reunified in 1990.", "Berlin became the capital again.",
    ]


def test_abstention_has_no_claims() -> None:
    assert rule_based_claims("I don't know.") == []
    assert rule_based_claims("") == []


def test_max_claims_respected() -> None:
    answer = " ".join(f"Fact number {i} is true." for i in range(20))
    assert len(rule_based_claims(answer, max_claims=5)) == 5


def test_abstention_detection() -> None:
    assert is_abstention("I don't know.")
    assert is_abstention("The provided passages do not contain this information.")
    assert is_abstention("   ")
    assert not is_abstention("Jane Austen wrote it.")
    assert not is_abstention("Paris is the capital. " * 20 + "The passages do not mention the population.")


def test_citation_helpers() -> None:
    passages = [Passage(1, "doc1#c0", "A", "text a", 0.9), Passage(2, "doc2#c0", "B", "text b", 0.8)]
    cites = extract_citations("Claim [S1]. Other [S2]. Bad [S7]. Again [S1].", passages)
    assert [c["label"] for c in cites] == ["S1", "S2", "S7"]
    assert cites[0]["source_id"] == "doc1#c0" and cites[2]["valid"] is False
    assert strip_citations("A fact [S1], [S2].") == "A fact."
    assert strip_citations('According to [S1], the song is "Shatter Me".') == 'The song is "Shatter Me".'
    assert strip_citations("Based on [S1] and [S2], it was 1882 (S3).") == "It was 1882 (S3)."
    assert strip_citations("It was 1882 ([S3]).") == "It was 1882."
