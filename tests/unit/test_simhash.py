"""SimHash + gom cụm repost (data-contract §2.2 tầng 3)."""

import numpy as np
import pandas as pd

from vre.data import simhash

BASE = "nhà phố mặt tiền đường lớn gần chợ trường học khu dân cư đông đúc an ninh tốt sổ hồng riêng chính chủ bán gấp"


def test_similar_texts_have_close_hashes():
    texts = pd.Series([BASE, BASE + " liên hệ ngay", "căn hộ cao cấp view sông hồ bơi gym tầng cao thoáng mát nội thất đầy đủ"])
    h, valid = simhash.simhashes(texts)
    assert valid.all()
    near = int(np.bitwise_count(h[0] ^ h[1]))
    far = int(np.bitwise_count(h[0] ^ h[2]))
    assert near < far


def test_empty_text_is_invalid_and_hash_is_deterministic():
    h1, v1 = simhash.simhashes(pd.Series(["", "hai từ", BASE]))
    h2, _ = simhash.simhashes(pd.Series(["", "hai từ", BASE]))
    assert v1.tolist() == [False, True, True]
    assert (h1 == h2).all()


def test_cluster_merges_identical_hashes_within_group_only():
    ids = np.array([50, 10, 30, 40], dtype=np.int64)
    group = np.array([0, 0, 0, 1], dtype=np.int64)
    hashes = np.array([7, 7, 2**63, 7], dtype=np.uint64)
    valid = np.array([True, True, True, True])
    cluster_ids, stats = simhash.cluster(ids, group, hashes, valid)
    assert cluster_ids.tolist() == [10, 10, 30, 40]
    assert stats["simhash_clusters_multi"] == 1 and stats["simhash_listings_in_multi"] == 2


def test_hamming_threshold_and_invalid_rows():
    ids = np.array([1, 2, 3], dtype=np.int64)
    group = np.zeros(3, dtype=np.int64)
    hashes = np.array([0, 0b111, 0b1111], dtype=np.uint64)  # khoảng cách 3 (nhận) và 4 (không)
    cluster_ids, _ = simhash.cluster(ids, group, hashes, np.array([True, True, True]))
    assert cluster_ids.tolist() == [1, 1, 1]  # 3 nối với 2 qua khoảng cách 1
    cluster_ids, _ = simhash.cluster(ids, group, np.array([0, 0b1111, 0], dtype=np.uint64), np.array([True, True, False]))
    assert cluster_ids.tolist() == [1, 2, 3]


def test_oversized_bucket_is_skipped(monkeypatch):
    monkeypatch.setattr(simhash, "MAX_BUCKET", 3)
    n = 5
    _, stats = simhash.cluster(np.arange(n, dtype=np.int64), np.zeros(n, np.int64), np.zeros(n, np.uint64), np.ones(n, bool))
    assert stats["simhash_skipped_buckets"] == simhash.BANDS
