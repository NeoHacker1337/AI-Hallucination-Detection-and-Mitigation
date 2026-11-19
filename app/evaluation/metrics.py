"""Metric functions (pure, numpy-based, unit-tested)."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def _arr(x) -> np.ndarray:  # type: ignore[no-untyped-def]
    return np.asarray(x, dtype=float)


def mean_or_nan(values) -> float:  # type: ignore[no-untyped-def]
    """Mean ignoring NaN; NaN for empty input."""
    a = _arr(values)
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")


def bootstrap_ci(
    values, statistic: Callable[[np.ndarray], float] = np.mean, n_boot: int = 2000,  # type: ignore[no-untyped-def]
    alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval. NaN bounds for empty input."""
    a = _arr(values)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = np.array([statistic(a[row]) for row in idx])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap_diff_ci(
    a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0  # type: ignore[no-untyped-def]
) -> tuple[float, float, float]:
    """Mean of (b - a) over paired items with a percentile bootstrap CI: (diff, lo, hi)."""
    x, y = _arr(a), _arr(b)
    if x.shape != y.shape:
        raise ValueError("paired arrays must have the same shape")
    d = y - x
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = bootstrap_ci(d, np.mean, n_boot, alpha, seed)
    return float(d.mean()), lo, hi


def mcnemar_test(a, b) -> dict:  # type: ignore[no-untyped-def]
    """McNemar's test for paired binary outcomes a (baseline) and b (treatment).

    n01 = a False & b True, n10 = a True & b False. Uses the exact binomial test when there are
    fewer than 25 discordant pairs, otherwise the chi-square statistic with continuity correction.
    """
    x = np.asarray(a, dtype=bool)
    y = np.asarray(b, dtype=bool)
    if x.shape != y.shape:
        raise ValueError("paired arrays must have the same shape")
    n01 = int(np.sum(~x & y))
    n10 = int(np.sum(x & ~y))
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0, "method": "no discordant pairs"}
    if n < 25:
        p = stats.binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue
        return {"n01": n01, "n10": n10, "statistic": float(min(n01, n10)), "p_value": float(min(p, 1.0)),
                "method": "exact binomial"}
    stat = (abs(n01 - n10) - 1) ** 2 / n
    p = stats.chi2.sf(stat, df=1)
    return {"n01": n01, "n10": n10, "statistic": float(stat), "p_value": float(p), "method": "chi2 (continuity)"}


def expected_calibration_error(confidence, correct, n_bins: int = 10) -> float:  # type: ignore[no-untyped-def]
    """ECE with equal-width bins: sum_b |B|/N * |acc(B) - conf(B)|. NaN for empty input."""
    conf = _arr(confidence)
    corr = _arr(correct)
    mask = ~np.isnan(conf) & ~np.isnan(corr)
    conf, corr = conf[mask], corr[mask]
    if conf.size == 0:
        return float("nan")
    if np.any((conf < 0) | (conf > 1)):
        raise ValueError("confidence values must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = bins == b
        if np.any(sel):
            ece += sel.mean() * abs(corr[sel].mean() - conf[sel].mean())
    return float(ece)


def auroc(scores, labels) -> float:  # type: ignore[no-untyped-def]
    """AUROC of `scores` for predicting `labels` (1 = hallucination). NaN if only one class."""
    s, y = _arr(scores), _arr(labels)
    mask = ~np.isnan(s) & ~np.isnan(y)
    s, y = s[mask], y[mask]
    if s.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def roc_points(scores, labels) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[no-untyped-def]
    """False-positive and true-positive rates for an ROC plot (empty arrays if undefined)."""
    s, y = _arr(scores), _arr(labels)
    mask = ~np.isnan(s) & ~np.isnan(y)
    s, y = s[mask], y[mask]
    if s.size == 0 or len(np.unique(y)) < 2:
        return np.array([]), np.array([])
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr


def precision_recall_f1(predicted, actual) -> dict[str, float]:  # type: ignore[no-untyped-def]
    """Detector precision/recall/F1 where positive = hallucination. NaN when undefined."""
    p = np.asarray(predicted, dtype=bool)
    a = np.asarray(actual, dtype=bool)
    tp = int(np.sum(p & a))
    fp = int(np.sum(p & ~a))
    fn = int(np.sum(~p & a))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    if math.isnan(precision) or math.isnan(recall) or precision + recall == 0:
        f1 = float("nan") if (math.isnan(precision) or math.isnan(recall)) else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def risk_coverage_curve(confidence, correct) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[no-untyped-def]
    """Accuracy of the k most confident answers versus coverage k/N, for k = 1..N."""
    conf, corr = _arr(confidence), _arr(correct)
    mask = ~np.isnan(conf) & ~np.isnan(corr)
    conf, corr = conf[mask], corr[mask]
    if conf.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(-conf, kind="stable")
    cum = np.cumsum(corr[order])
    k = np.arange(1, conf.size + 1)
    return k / conf.size, cum / k
