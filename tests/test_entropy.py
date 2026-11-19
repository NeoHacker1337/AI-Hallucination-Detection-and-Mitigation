"""Semantic entropy: clustering and entropy maths."""

from __future__ import annotations

import math

import pytest

from app.pipeline.semantic_entropy import cluster_by_equivalence, entropy_from_clusters, semantic_entropy


def test_all_samples_equivalent_gives_zero_entropy() -> None:
    clusters = cluster_by_equivalence(5, lambda i, j: True)
    assert clusters == [[0, 1, 2, 3, 4]]
    h, hn = entropy_from_clusters(clusters, 5)
    assert h == pytest.approx(0.0)
    assert hn == pytest.approx(0.0)


def test_all_samples_distinct_gives_max_entropy() -> None:
    clusters = cluster_by_equivalence(4, lambda i, j: False)
    assert len(clusters) == 4
    h, hn = entropy_from_clusters(clusters, 4)
    assert h == pytest.approx(math.log(4))
    assert hn == pytest.approx(1.0)


def test_mixed_clusters_match_hand_computation() -> None:
    labels = ["a", "a", "a", "b", "c"]
    clusters = cluster_by_equivalence(5, lambda i, j: labels[i] == labels[j])
    assert clusters == [[0, 1, 2], [3], [4]]
    h, hn = entropy_from_clusters(clusters, 5)
    expected = -(0.6 * math.log(0.6) + 2 * 0.2 * math.log(0.2))
    assert h == pytest.approx(expected)
    assert hn == pytest.approx(expected / math.log(5))


def test_greedy_clustering_compares_with_cluster_representative() -> None:
    # 0~1 and 1~2 but not 0~2: item 2 is compared with representative 0, so it opens a new cluster.
    eq = {(0, 1), (1, 2)}
    clusters = cluster_by_equivalence(3, lambda i, j: (i, j) in eq or (j, i) in eq)
    assert clusters == [[0, 1], [2]]


def test_single_sample_has_zero_normalised_entropy() -> None:
    assert entropy_from_clusters([[0]], 1) == (0.0, 0.0)


def test_zero_samples_rejected() -> None:
    with pytest.raises(ValueError):
        entropy_from_clusters([], 0)


def test_semantic_entropy_with_nli(keyword_nli) -> None:  # type: ignore[no-untyped-def]
    samples = ["Paris is the capital", "Paris is the capital", "Lyon is the capital"]
    res = semantic_entropy("capital?", samples, keyword_nli)
    assert res.clusters == [[0, 1], [2]]
    assert res.n_samples == 3
    assert 0 < res.normalized_entropy < 1
