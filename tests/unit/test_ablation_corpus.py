"""vre.eval.ablation: subcorpus chứa 100% tin GT, tất định theo seed; chỉ số theo cụm; cỡ chunk hiệu dụng."""

import math

import numpy as np
import pytest

from vre.eval import ablation


@pytest.fixture
def all_ids():
    return np.arange(10_000, 60_000, 3, dtype=np.int64)  # bảng sạch giả, id rời rạc


def test_subcorpus_contains_all_gt(all_ids):
    gt = [int(all_ids[5]), int(all_ids[-1]), int(all_ids[12_345])]
    ids = ablation.subcorpus_ids(all_ids, gt, n_random=100, seed=42)
    assert set(gt) <= set(ids.tolist())
    assert len(ids) >= 100 and len(ids) <= 103
    assert np.all(np.diff(ids) > 0)  # tăng dần, duy nhất
    assert set(ids.tolist()) <= set(all_ids.tolist())


def test_subcorpus_deterministic_with_seed(all_ids):
    gt = [int(all_ids[7])]
    a = ablation.subcorpus_ids(all_ids, gt, 500, seed=42)
    b = ablation.subcorpus_ids(all_ids, gt, 500, seed=42)
    c = ablation.subcorpus_ids(all_ids, gt, 500, seed=43)
    assert np.array_equal(a, b)
    assert ablation.ids_digest(a) == ablation.ids_digest(b)
    assert not np.array_equal(a, c)


def test_subcorpus_caps_at_corpus_and_rejects_unknown_gt(all_ids):
    assert len(ablation.subcorpus_ids(all_ids, [], 10**9)) == len(all_ids)
    with pytest.raises(ValueError, match="không có trong bảng sạch"):
        ablation.subcorpus_ids(all_ids, [1], 10)


def test_score_run_cluster_metrics():
    retrieved = [[7, 1, 2], [3, 4, 5, 6, 8, 9]]
    relevant = [{1}, set(range(100, 120))]
    s = ablation.score_run(retrieved, relevant, [10.0, 30.0])
    assert s["recall_at_5"] == pytest.approx(0.5)
    assert s["mrr"] == pytest.approx(0.25)
    assert s["ndcg_at_10"] == pytest.approx((1 / math.log2(3)) / 2)
    assert s["n_questions"] == 2 and s["latency_ms_p50"] == 20.0


def test_ndcg_local_perfect_and_capped():
    assert ablation._ndcg_local([1, 2, 3], {1, 2, 3}, 10) == pytest.approx(1.0)
    assert ablation._ndcg_local(list(range(10)), set(range(50)), 10) == pytest.approx(1.0)
    assert ablation._ndcg_local([9], {1}, 10) == 0.0


def test_dedup_clusters_keeps_order():
    assert ablation.dedup_clusters([5, 3, 5, 1, 3]) == [5, 3, 1]


def test_effective_chunk_tokens_fits_e5_max_length():
    assert ablation.effective_chunk_tokens(256, 512, 3) == 256
    assert ablation.effective_chunk_tokens(400, 512, 3) == 400
    assert ablation.effective_chunk_tokens(512, 512, 3) == 507  # [bos] + 3 tiền tố + 507 + [eos] = 512


def test_run_names_unique():
    names = {ablation.run_name(None, r, p, k) for r in ablation.RETRIEVALS for p in (False, True) for k in (False, True)}
    names |= {ablation.run_name(s, "hybrid", True, False) for s in ablation.CHUNK_SIZES}
    assert len(names) == 15
    assert ablation.run_name(512, "hybrid", True, False) == "c512_hybrid_prefilter"
