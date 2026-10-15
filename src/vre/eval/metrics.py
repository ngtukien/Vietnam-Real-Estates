"""Chỉ số đánh giá thuần (Phụ lục A.6): Recall@k, RR/MRR, nDCG, khớp theo cụm dup, thang điểm 0–2, Faithfulness."""

from __future__ import annotations

import numpy as np


def recall_at_k(retrieved, relevant, k, capped: bool = False):
    """Tỷ lệ tin đúng nằm trong top-k. capped=True chia cho min(|relevant|, k): khi có hơn k tin đúng,
    lấy đủ k tin đúng vẫn được 1,0 (dùng cho câu ràng buộc có hàng chục tin thoả điều kiện)."""
    if not relevant:
        return np.nan
    denominator = min(len(relevant), k) if capped else len(relevant)
    return len(set(retrieved[:k]) & set(relevant)) / denominator


def reciprocal_rank(retrieved, relevant):
    for pos, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1 / pos
    return 0.0


def mrr(runs):
    return sum(reciprocal_rank(r, rel) for r, rel in runs) / len(runs)


def faithfulness(claim_supported):
    return sum(claim_supported) / len(claim_supported) if claim_supported else np.nan


def _align(verdicts: list[bool], n: int) -> list[bool]:
    """Đúng n phán quyết: LLM trả thiếu thì phần thiếu tính là False (không được hỗ trợ), thừa thì cắt."""
    return (list(verdicts) + [False] * n)[:n]


def ndcg_at_k(retrieved, relevant, k) -> float:
    """nDCG@k nhị phân: DCG = Σ 1/log2(i+1) trên vị trí trúng; IDCG = cùng tổng với min(|relevant|, k) tin đúng đứng đầu.
    `relevant` rỗng ⇒ NaN (câu không có tập tin đúng không được tính)."""
    relevant = set(relevant)
    if not relevant:
        return float("nan")
    dcg = sum(1.0 / np.log2(pos + 1) for pos, doc in enumerate(retrieved[:k], start=1) if doc in relevant)
    ideal = sum(1.0 / np.log2(pos + 1) for pos in range(1, min(len(relevant), k) + 1))
    return float(dcg / ideal)


def to_clusters(sources, cluster_of: dict[int, int]) -> list[int]:
    """Danh sách nguồn (listing_id, theo thứ tự hạng) → danh sách `dup_cluster_id` không trùng, giữ thứ tự
    (data-contract §2.2: tin truy xuất đúng khi cùng cụm với tin đáp án; repost cùng cụm chỉ tính một lần).
    Tin không có trong bảng ánh xạ giữ chính id của nó (singleton)."""
    return list(dict.fromkeys(cluster_of.get(int(s), int(s)) for s in sources))


def retrieval_scores(sources, gt_clusters, cluster_of: dict[int, int], ks: tuple[int, ...] = (5, 10)) -> dict[str, float]:
    """Recall@k (capped), RR và nDCG@10 theo cụm dup cho câu `gt_kind=ids`."""
    ranked = to_clusters(sources, cluster_of)
    relevant = {int(c) for c in gt_clusters}
    out = {f"recall_at_{k}": float(recall_at_k(ranked, relevant, k, capped=True)) for k in ks}
    out["rr"] = float(reciprocal_rank(ranked, relevant))
    out["ndcg_at_10"] = ndcg_at_k(ranked, relevant, 10)
    return out


def retrieval_points(recall: float) -> int:
    """Thang 0–2 cũ cho câu tra cứu/ràng buộc: 2 = Recall@k (capped) đủ 1,0; 1 = có ít nhất một cụm đúng; 0 = không."""
    return 2 if recall >= 1 else int(recall > 0)


def value_points(predicted: float | None, truth: float, tolerance: float) -> int:
    """Thang 0–2 cho câu số (định giá): 2 = sai số tương đối ≤ tolerance; 1 = ≤ 2×tolerance; 0 = lệch hơn hoặc không có số."""
    if predicted is None or not np.isfinite(predicted) or truth == 0:
        return 0
    error = abs(predicted - truth) / abs(truth)
    return 2 if error <= tolerance else int(error <= 2 * tolerance)


def percentile(values, q: float) -> float | None:
    """Phân vị q (0–100) bỏ None/NaN; None khi rỗng (không nhầm "chưa đo" thành 0)."""
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.percentile(clean, q)) if clean else None
