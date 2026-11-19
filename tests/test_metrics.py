"""Metrics, statistics, judge decision rule and the summary table."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.evaluation.judge import CORRECT, HALLUCINATED, AnswerJudge, lexical_match, nli_verdict, normalize_text
from app.evaluation.metrics import (
    auroc,
    bootstrap_ci,
    expected_calibration_error,
    mcnemar_test,
    paired_bootstrap_diff_ci,
    precision_recall_f1,
    risk_coverage_curve,
)
from app.evaluation.report import summarize
from app.pipeline.self_consistency import selfcheck_from_probs


def test_ece_perfect_and_worst_calibration() -> None:
    assert expected_calibration_error([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0]) == pytest.approx(0.0)
    assert expected_calibration_error([1.0, 1.0], [0, 0]) == pytest.approx(1.0)


def test_ece_hand_computed() -> None:
    # bin (0.7,0.8]: conf 0.8 x2, acc 0.5 -> |0.5-0.8| ; bin (0.2,0.3]: conf 0.3 x2, acc 0.5 -> |0.5-0.3|
    ece = expected_calibration_error([0.8, 0.8, 0.3, 0.3], [1, 0, 1, 0])
    assert ece == pytest.approx(0.5 * 0.3 + 0.5 * 0.2)


def test_ece_ignores_nan_and_rejects_out_of_range() -> None:
    assert math.isnan(expected_calibration_error([], []))
    assert expected_calibration_error([np.nan, 1.0], [0, 1]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        expected_calibration_error([1.5], [1])


def test_auroc_values() -> None:
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)
    assert math.isnan(auroc([0.1, 0.2], [1, 1]))


def test_precision_recall_f1() -> None:
    r = precision_recall_f1([1, 1, 0, 0], [1, 0, 1, 0])
    assert r["precision"] == pytest.approx(0.5) and r["recall"] == pytest.approx(0.5) and r["f1"] == pytest.approx(0.5)
    assert math.isnan(precision_recall_f1([0, 0], [1, 0])["precision"])


def test_mcnemar_exact_and_chi2() -> None:
    a = np.array([True] * 10 + [False] * 10)
    b = np.array([False] * 10 + [False] * 10)
    res = mcnemar_test(a, b)
    assert res["n10"] == 10 and res["n01"] == 0 and res["method"] == "exact binomial"
    assert res["p_value"] == pytest.approx(2 * 0.5**10)
    big_a = np.array([True] * 40 + [False] * 10)
    big_b = np.array([False] * 40 + [True] * 10)
    big = mcnemar_test(big_a, big_b)
    assert big["method"].startswith("chi2")
    assert big["statistic"] == pytest.approx((abs(10 - 40) - 1) ** 2 / 50)
    assert mcnemar_test([True, False], [True, False])["p_value"] == 1.0


def test_bootstrap_ci_contains_mean_and_is_reproducible() -> None:
    values = np.array([0, 1] * 50, dtype=float)
    lo, hi = bootstrap_ci(values, seed=1)
    assert lo < 0.5 < hi
    assert (lo, hi) == bootstrap_ci(values, seed=1)
    d, dlo, dhi = paired_bootstrap_diff_ci([1, 1, 1, 1], [0, 0, 0, 0])
    assert d == pytest.approx(-1.0) and dlo == pytest.approx(-1.0) and dhi == pytest.approx(-1.0)


def test_risk_coverage_curve() -> None:
    cov, acc = risk_coverage_curve([0.9, 0.1, 0.5], [1, 0, 1])
    assert list(cov) == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert list(acc) == pytest.approx([1.0, 1.0, 2 / 3])


def test_selfcheck_from_probs() -> None:
    probs = np.array([[[0.9, 0.05, 0.05], [0.1, 0.1, 0.8]]])  # one claim, two samples
    score = selfcheck_from_probs(probs)[0]
    assert score == pytest.approx((0.05 / 0.95 + 0.8 / 0.9) / 2)


def test_judge_text_helpers_and_rule() -> None:
    assert normalize_text("The  Arthur's Magazine!") == "arthurs magazine"
    assert lexical_match("It was Arthur's Magazine, first.", ["Arthur's Magazine"]) == "Arthur's Magazine"
    assert lexical_match("Paris", ["a very long reference answer that has many words"]) is None
    assert nli_verdict("x", None, 0.0, 0.0, 0.5)[0] == CORRECT
    assert nli_verdict(None, "y", 0.0, 0.0, 0.5)[0] == HALLUCINATED
    assert nli_verdict(None, None, 0.8, 0.2, 0.5)[0] == CORRECT
    label, uncertain, _ = nli_verdict(None, None, 0.2, 0.1, 0.5)
    assert label == HALLUCINATED and uncertain


def test_nli_judge_end_to_end(keyword_nli) -> None:  # type: ignore[no-untyped-def]
    judge = AnswerJudge(mode="nli", nli=keyword_nli)
    q = "Which planet is closest to the Sun?"
    assert judge.judge(q, "Mercury is closest [S1].", ["Mercury"], ["Venus"]).label == CORRECT
    assert judge.judge(q, "Venus is closest.", ["Mercury"], ["Venus"]).label == HALLUCINATED
    assert judge.judge(q, "I don't know.", ["Mercury"], ["Venus"]).label == "abstained"


def _row(model: str, config: str, qid: str, label: str, abstained: bool = False, **kw) -> dict:  # type: ignore[no-untyped-def]
    base = dict(model=model, config=config, question_id=qid, final_label="abstained" if abstained else label,
                initial_label=label, candidate_label=label, abstained=abstained, escalated=False, error=None,
                confidence=np.nan, first_confidence=np.nan, semantic_entropy_norm=np.nan, selfcheck_score=np.nan,
                first_decision=None, faithfulness=np.nan, latency_ms=1000.0, llm_calls=1, total_tokens=10,
                judge_uncertain=False, judges_disagree=False)
    base.update(kw)
    return base


def test_summary_and_significance_from_rows() -> None:
    rows = []
    for i in range(10):
        rows.append(_row("m", "C0", f"q{i}", HALLUCINATED if i < 6 else CORRECT))
        if i < 4:
            rows.append(_row("m", "C5", f"q{i}", HALLUCINATED, abstained=True, latency_ms=4000.0, llm_calls=8,
                             first_decision="UNCERTAIN", semantic_entropy_norm=0.9, first_confidence=0.1, confidence=0.1))
        else:
            rows.append(_row("m", "C5", f"q{i}", HALLUCINATED if i < 6 else CORRECT, latency_ms=4000.0, llm_calls=8,
                             first_decision="VERIFIED", semantic_entropy_norm=0.1 if i >= 6 else 0.5,
                             first_confidence=0.9, confidence=0.9))
    summary, sig = summarize(pd.DataFrame(rows), seed=0)
    c0 = summary[summary.config == "C0"].iloc[0]
    c5 = summary[summary.config == "C5"].iloc[0]
    assert c0.hallucination_rate == pytest.approx(0.6) and c0.accuracy == pytest.approx(0.4)
    assert c5.hallucination_rate == pytest.approx(0.2) and c5.abstention_rate == pytest.approx(0.4)
    assert c5.selective_accuracy == pytest.approx(4 / 6)
    assert c5.auroc_semantic_entropy == pytest.approx(1.0)
    assert c5.detector_precision == pytest.approx(1.0) and c5.detector_recall == pytest.approx(4 / 6)
    s = sig.iloc[0]
    assert s.config == "C5" and s.n_paired == 10
    assert s.hallucination_diff == pytest.approx(-0.4)
    assert s.mcnemar_hallucination_n10 == 4 and s.mcnemar_hallucination_n01 == 0
    assert s.latency_ratio_vs_c0 == pytest.approx(4.0)
