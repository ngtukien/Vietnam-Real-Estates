"""vre.eval.metrics: nDCG, khớp theo dup_cluster_id (data-contract §2.2), thang điểm 0–2."""

import math

import pytest

from vre.eval import metrics


def test_ndcg_perfect_and_partial():
    assert metrics.ndcg_at_k([1, 2, 3], {1, 2}, 10) == pytest.approx(1.0)
    assert metrics.ndcg_at_k([9, 1], {1}, 10) == pytest.approx(1 / math.log2(3))
    assert metrics.ndcg_at_k([9, 8], {1}, 10) == 0.0
    assert math.isnan(metrics.ndcg_at_k([1], set(), 10))


def test_reposts_count_once_by_cluster():
    cluster_of = {11: 10, 12: 10, 20: 20}
    assert metrics.to_clusters([11, 12, 20, 30], cluster_of) == [10, 20, 30]
    scores = metrics.retrieval_scores([11, 12, 99], [10, 20], cluster_of)
    assert scores["recall_at_5"] == 0.5  # repost 12 cùng cụm 10 không được tính thêm
    assert scores["rr"] == 1.0
    full = metrics.retrieval_scores([12, 20], [10, 20], cluster_of)
    assert full["recall_at_5"] == 1.0 and full["ndcg_at_10"] == pytest.approx(1.0)


def test_points_scales():
    assert [metrics.retrieval_points(r) for r in (1.0, 0.4, 0.0)] == [2, 1, 0]
    assert metrics.value_points(5.3, 5.12, 0.15) == 2
    assert metrics.value_points(6.5, 5.12, 0.15) == 1  # lệch 27% ≤ 2×15%
    assert metrics.value_points(9.0, 5.12, 0.15) == 0
    assert metrics.value_points(None, 5.12, 0.15) == 0


def test_percentile_ignores_missing():
    assert metrics.percentile([100, None, 300], 50) == 200
    assert metrics.percentile([None], 50) is None
