"""How answers are judged correct. Every decision is stored in the `judgements` table and in
results.csv so it can be checked by hand.

Labels: "correct", "hallucinated", "abstained".

1. Abstention: if the answer declines to answer (see app/pipeline/abstention.py) -> "abstained".
2. NLI judge (mode "nli"):
   a. Lexical match: a reference of at most 6 words whose normalised text appears inside the
      normalised answer counts as a match (handles short entity answers such as HaluEval's).
   b. NLI match: premise = "question + answer", hypothesis = "question + reference"; the best
      P(entail) over the correct references (s_c) and over the incorrect references (s_i).
   c. correct  if it lexically matches a correct reference and no incorrect one,
               or s_c >= threshold and s_c > s_i;
      hallucinated otherwise. When neither s_c nor s_i reaches the threshold and there is no lexical
      match, the answer is still counted as hallucinated (it does not state a reference answer)
      but `uncertain=True` is recorded so these cases can be audited.
3. LLM judge (mode "llm"): a judge model receives question, references and answer and returns
   {"verdict": correct|incorrect|abstained}. Unparseable output is marked uncertain.
4. mode "both": the LLM verdict is the label; the NLI verdict is stored next to it and
   `judges_disagree` is set when they differ.
"""

from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass

from app.pipeline import prompts
from app.pipeline.abstention import is_abstention
from app.pipeline.generation import strip_citations
from app.pipeline.nli import ENTAIL, NLIScorer
from app.providers import DecodingParams, LLMProvider
from app.utils import extract_json

CORRECT, HALLUCINATED, ABSTAINED = "correct", "hallucinated", "abstained"
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_text(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation and articles, collapse spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def lexical_match(answer: str, references: list[str], max_words: int = 6) -> str | None:
    """Return the first short reference contained in the answer (after normalisation)."""
    norm_answer = f" {normalize_text(answer)} "
    for ref in references:
        norm_ref = normalize_text(ref)
        if norm_ref and len(norm_ref.split()) <= max_words and f" {norm_ref} " in norm_answer:
            return ref
    return None


@dataclass
class JudgeDecision:
    """A correctness verdict with all evidence used to reach it."""

    label: str
    mode: str
    nli_label: str | None = None
    nli_correct_score: float | None = None
    nli_incorrect_score: float | None = None
    matched_reference: str | None = None
    llm_label: str | None = None
    llm_raw: str | None = None
    judges_disagree: bool = False
    uncertain: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def nli_verdict(
    correct_lex: str | None, incorrect_lex: str | None, s_c: float, s_i: float, threshold: float
) -> tuple[str, bool, str]:
    """Pure decision rule for the NLI judge. Returns (label, uncertain, reason)."""
    if correct_lex and not incorrect_lex:
        return CORRECT, False, f"lexical match with correct reference '{correct_lex}'"
    if incorrect_lex and not correct_lex:
        return HALLUCINATED, False, f"lexical match with incorrect reference '{incorrect_lex}'"
    if s_c >= threshold and s_c > s_i:
        return CORRECT, False, f"NLI entails a correct reference (p={s_c:.2f} > incorrect p={s_i:.2f})"
    if s_i >= threshold and s_i >= s_c:
        return HALLUCINATED, False, f"NLI entails an incorrect reference (p={s_i:.2f} >= correct p={s_c:.2f})"
    return HALLUCINATED, True, (
        f"no reference matched (best correct p={s_c:.2f}, incorrect p={s_i:.2f} < {threshold}); "
        "counted as hallucinated, flagged uncertain"
    )


class AnswerJudge:
    """Judges answers against dataset references with NLI, an LLM judge, or both."""

    def __init__(
        self,
        mode: str = "nli",
        nli: NLIScorer | None = None,
        llm: LLMProvider | None = None,
        entail_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        if mode not in ("nli", "llm", "both"):
            raise ValueError("judge mode must be 'nli', 'llm' or 'both'")
        if mode in ("nli", "both") and nli is None:
            raise ValueError("NLI judge requires an NLI model")
        if mode in ("llm", "both") and llm is None:
            raise ValueError("LLM judge requires a judge model")
        self.mode = mode
        self.nli = nli
        self.llm = llm
        self.threshold = entail_threshold
        self.seed = seed

    def _nli(self, question: str, answer: str, correct: list[str], incorrect: list[str]) -> JudgeDecision:
        assert self.nli is not None
        c_lex = lexical_match(answer, correct)
        i_lex = lexical_match(answer, incorrect)
        premise = f"{question} {answer}"
        refs = correct + incorrect
        probs = self.nli.predict([(premise, f"{question} {r}") for r in refs]) if refs else None
        s_c = float(probs[: len(correct), ENTAIL].max()) if correct and probs is not None else 0.0
        s_i = float(probs[len(correct) :, ENTAIL].max()) if incorrect and probs is not None else 0.0
        label, uncertain, reason = nli_verdict(c_lex, i_lex, s_c, s_i, self.threshold)
        matched = c_lex or i_lex
        if matched is None and probs is not None and len(refs) > 0:
            matched = refs[int(probs[:, ENTAIL].argmax())]
        return JudgeDecision(
            label=label, mode="nli", nli_label=label, nli_correct_score=s_c, nli_incorrect_score=s_i,
            matched_reference=matched, uncertain=uncertain, reason=reason,
        )

    def _llm(self, question: str, answer: str, correct: list[str], incorrect: list[str]) -> JudgeDecision:
        assert self.llm is not None
        incorrect_block = f"Known incorrect answers: {incorrect}\n" if incorrect else ""
        prompt = prompts.LLM_JUDGE.format(
            question=question, correct=correct, incorrect_block=incorrect_block, answer=answer
        )
        raw = self.llm.complete(prompt, DecodingParams(temperature=0.0, max_tokens=150, seed=self.seed)).text
        parsed = extract_json(raw)
        verdict = str(parsed.get("verdict", "")).lower() if isinstance(parsed, dict) else ""
        uncertain = False
        if verdict not in ("correct", "incorrect", "abstained"):
            uncertain = True
            low = raw.lower()
            verdict = "incorrect" if "incorrect" in low else ("correct" if "correct" in low else "incorrect")
        label = {"correct": CORRECT, "incorrect": HALLUCINATED, "abstained": ABSTAINED}[verdict]
        explanation = parsed.get("explanation", "") if isinstance(parsed, dict) else "unparseable judge output"
        return JudgeDecision(
            label=label, mode="llm", llm_label=label, llm_raw=raw, uncertain=uncertain,
            reason=f"LLM judge ({self.llm.spec}): {explanation}",
        )

    def judge(self, question: str, answer: str | None, correct: list[str], incorrect: list[str]) -> JudgeDecision:
        """Judge one answer."""
        if answer is None or is_abstention(answer):
            return JudgeDecision(label=ABSTAINED, mode=self.mode, reason="answer is an abstention")
        clean = strip_citations(answer)
        if self.mode == "nli":
            return self._nli(question, clean, correct, incorrect)
        if self.mode == "llm":
            return self._llm(question, clean, correct, incorrect)
        nli_dec = self._nli(question, clean, correct, incorrect)
        llm_dec = self._llm(question, clean, correct, incorrect)
        return JudgeDecision(
            label=llm_dec.label,
            mode="both",
            nli_label=nli_dec.label,
            nli_correct_score=nli_dec.nli_correct_score,
            nli_incorrect_score=nli_dec.nli_incorrect_score,
            matched_reference=nli_dec.matched_reference,
            llm_label=llm_dec.llm_label,
            llm_raw=llm_dec.llm_raw,
            judges_disagree=nli_dec.label != llm_dec.label,
            uncertain=llm_dec.uncertain or nli_dec.label != llm_dec.label,
            reason=f"{llm_dec.reason} | NLI: {nli_dec.reason}",
        )
