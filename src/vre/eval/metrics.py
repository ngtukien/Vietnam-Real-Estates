"""Chỉ số đánh giá thuần (Phụ lục A.6): Recall@k, RR/MRR, Faithfulness (chuyển nguyên từ `src/eval.py`)."""

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
