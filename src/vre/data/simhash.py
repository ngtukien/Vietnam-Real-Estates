"""SimHash 64-bit trên 3-shingle từ của norm(description) và gom cụm repost (data-contract §2.2 tầng 3).

Tính hash theo shard (numpy vector hoá); gom cụm toàn cục: 4 band × 16 bit trong cùng (province, property_type),
Hamming ≤ 3, bucket > 500 phần tử bỏ qua (đếm). Cụm = thành phần liên thông; `dup_cluster_id` = listing_id nhỏ nhất.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

BITS = 64
BANDS = 4
BAND_BITS = 16
MAX_HAMMING = 3
MAX_BUCKET = 500
SHINGLE = 3
_WORD = re.compile(r"\w+")
_P1, _P2 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xC2B2AE3D27D4EB4F)


def _word_hash(word: str) -> int:
    return int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "little")


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer: trộn bit sau khi ghép 3 hash từ."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def simhashes(texts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(simhash uint64, hợp lệ bool) cho mỗi văn bản đã norm(); văn bản không có từ nào ⇒ không hợp lệ."""
    cache: dict[str, int] = {}

    def word_hash(word: str) -> int:
        value = cache.get(word)
        if value is None:
            value = cache[word] = _word_hash(word)
        return value

    words = [_WORD.findall(t) for t in texts.fillna("").tolist()]
    lengths = np.fromiter((len(w) for w in words), dtype=np.int64, count=len(words))
    flat = np.fromiter((word_hash(w) for ws in words for w in ws), dtype=np.uint64, count=int(lengths.sum()))
    doc = np.repeat(np.arange(len(words)), lengths)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    pos = np.arange(len(flat), dtype=np.int64) - starts[doc] if len(flat) else np.zeros(0, np.int64)
    # shingle i của tài liệu bắt đầu ở từ i, cần đủ 3 từ; tài liệu < 3 từ dùng một shingle gồm các từ có.
    n_sh = np.maximum(lengths - (SHINGLE - 1), (lengths > 0).astype(np.int64))
    keep = pos < n_sh[doc]
    with np.errstate(over="ignore"):
        h = flat.copy()
        for k, prime in ((1, _P1), (2, _P2)):
            nxt = np.zeros_like(flat)
            ok = pos + k < lengths[doc]
            if len(flat) > k:
                nxt[:-k] = flat[k:]
            h = h * prime ^ np.where(ok, nxt, np.uint64(0))
        shingles = _mix(h[keep])
    sdoc = doc[keep]
    out = np.zeros(len(words), dtype=np.uint64)
    for bit in range(BITS):
        vote = ((shingles >> np.uint64(bit)) & np.uint64(1)).astype(np.int8) * 2 - 1
        score = np.bincount(sdoc, weights=vote, minlength=len(words))
        out |= (score > 0).astype(np.uint64) << np.uint64(bit)
    return out, lengths > 0


def _pairs_in_bucket(idx: np.ndarray, hashes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = hashes[idx]
    dist = np.bitwise_count(x[:, None] ^ x[None, :])
    i, j = np.nonzero(np.triu(dist <= MAX_HAMMING, 1))
    return idx[i], idx[j]


def cluster(listing_id: np.ndarray, group: np.ndarray, hashes: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """`dup_cluster_id` cho mỗi tin + thống kê cho ingest_report. `group` = mã số (province, property_type)."""
    n = len(listing_id)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    skipped = 0
    cand = np.nonzero(valid)[0]
    for band in range(BANDS):
        value = ((hashes[cand] >> np.uint64(band * BAND_BITS)) & np.uint64(0xFFFF)).astype(np.int64)
        key = group[cand].astype(np.int64) * 65536 + value
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        bounds = np.flatnonzero(np.diff(sorted_key)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(sorted_key)]])
        sizes = ends - starts
        skipped += int((sizes > MAX_BUCKET).sum())
        for s, e in zip(starts[(sizes > 1) & (sizes <= MAX_BUCKET)], ends[(sizes > 1) & (sizes <= MAX_BUCKET)], strict=True):
            a, b = _pairs_in_bucket(cand[order[s:e]], hashes)
            if len(a):
                rows.append(a)
                cols.append(b)
    r = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    c = np.concatenate(cols) if cols else np.zeros(0, np.int64)
    graph = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    min_id = pd.Series(listing_id).groupby(labels).transform("min").to_numpy(dtype=np.int64)
    sizes = np.bincount(labels)
    stats = {"simhash_clusters_multi": int((sizes > 1).sum()),
             "simhash_listings_in_multi": int((sizes[labels] > 1).sum()),
             "simhash_skipped_buckets": skipped}
    return min_id, stats
