"""Tìm kiếm: dense, BM25, hybrid (RRF), rerank và `retrieve` trên alias `vre_chunks` (data-contract §5).

Truy vấn Qdrant ở `vre.rag.qdrant_search`; bộ lọc biên dịch từ `vre.rag.filters` (một nguồn, B4).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from vre.config import get_settings
from vre.rag import qdrant_search as qs
from vre.rag.filters import FILTER_DROPPED, QueryFilters, merge, parse_filters, relaxations, relaxed_note, to_qdrant, unresolved


def tokenize_vi(text: str) -> list[str]:
    """Tách từ tiếng Việt bằng pyvi như phía chỉ mục: 'trường học' -> 'trường_học'; bỏ dấu câu."""
    return qs.tokenize(text)


def cosine(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _by_listing(hits: pd.DataFrame, k: int) -> pd.DataFrame:
    """Mỗi tin chỉ giữ chunk điểm cao nhất, để một tin dài không chiếm hết top-k."""
    hits = hits.sort_values("score", ascending=False).drop_duplicates("listing_id")
    return hits.head(k).reset_index(drop=True)


def search_dense(question: str, k: int | None = None, filters: QueryFilters | None = None) -> pd.DataFrame:
    """Dense search e5-large trên Qdrant, có pre-filter theo payload (không ngưỡng cosine — B5)."""
    k = get_settings().top_k if k is None else k
    (hits,) = qs.search_many([(qs.embed_query(question), "dense")], k * 4, to_qdrant(filters))
    return _by_listing(hits, k)


def search_bm25(question: str, k: int | None = None, filters: QueryFilters | None = None) -> pd.DataFrame:
    """BM25 trên Qdrant (sparse vector, IDF tính phía server) với từ đã tách bằng pyvi;
    khớp đúng từ như 'sổ hồng', '2PN'."""
    k = get_settings().top_k if k is None else k
    tokens = tokenize_vi(question)
    if not tokens:
        return qs.hits_frame([])
    (hits,) = qs.search_many([(qs.bm25_query(tokens), "bm25")], k * 4, to_qdrant(filters))
    return _by_listing(hits, k)


def rrf(rankings: list[list], k: int | None = None) -> list[tuple]:
    """rankings: danh sách các bảng xếp hạng, mỗi bảng là list ID theo thứ tự tốt dần (Phụ lục A.3).
    `k=None`: dùng `Settings.rrf_k`."""
    k = get_settings().rrf_k if k is None else k
    scores: dict = {}
    for ranking in rankings:
        for pos, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0) + 1 / (k + pos)
    return sorted(scores.items(), key=lambda x: -x[1])


def search_hybrid(question: str, k: int | None = None, filters: QueryFilters | None = None,
                  candidates: int = 20) -> pd.DataFrame:
    """Gộp dense và BM25 bằng RRF theo listing_id; giữ chunk tốt nhất của mỗi tin.
    Hai truy vấn đi chung một lượt gọi Qdrant (search_many), kết quả giống gọi search_dense + search_bm25."""
    k = get_settings().top_k if k is None else k
    candidates = max(candidates, k)
    queries: list[tuple[Any, str]] = [(qs.embed_query(question), "dense")]
    tokens = tokenize_vi(question)
    if tokens:  # câu hỏi toàn dấu câu: không có từ nào cho BM25
        queries.append((qs.bm25_query(tokens), "bm25"))
    rankings = [_by_listing(h, candidates) for h in qs.search_many(queries, candidates * 4, to_qdrant(filters))]
    fused = rrf([r["listing_id"].tolist() for r in rankings])[:k]
    pool = pd.concat(rankings).sort_values("score", ascending=False).drop_duplicates("listing_id")
    if not fused:
        return pool.assign(rrf=pd.Series(dtype=float)).reset_index(drop=True)  # rỗng nhưng đủ cột
    pool = pool.set_index("listing_id")
    return pd.DataFrame([{**pool.loc[lid].to_dict(), "listing_id": lid, "rrf": score}
                         for lid, score in fused])


def fmt_vnd(value) -> str:
    """7450000000 -> '7,45 tỷ'; 850000000 -> '850 triệu'."""
    if value is None or pd.isna(value):
        return "?"
    if value >= 1e9:
        return f"{value / 1e9:,.2f} tỷ".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{value / 1e6:,.0f} triệu".replace(",", ".")


def hits_table(hits: pd.DataFrame, score: str = "score", width: int = 70) -> pd.DataFrame:
    """Bảng kết quả gọn: ID, điểm, giá, diện tích, địa chỉ, đầu đoạn văn."""
    return pd.DataFrame({
        "listing_id": hits["listing_id"].astype(int),
        "điểm": hits[score].round(4),
        "giá": hits["price_vnd"].map(fmt_vnd),
        "m²": hits["area_m2"],
        "địa chỉ": hits[["ward", "district", "province"]].apply(
            lambda r: ", ".join(x for x in r if isinstance(x, str)), axis=1),
        "đoạn": hits["text"].str.replace("\n", " · ").str.slice(0, width),
    })


Search = Callable[[str, int, QueryFilters | None], pd.DataFrame]


def search_relaxed(search: Search, question: str, k: int, filters: QueryFilters | None) -> tuple[pd.DataFrame, str | None]:
    """`search` với bộ lọc; 0 kết quả ⇒ nới dần (`filters.relaxations`: tên dự án/đường → đẳng thức số) trước khi
    bỏ hết. `note`: None (khớp đủ) | "filter_relaxed:<trường,…>" (đã nới / tên không khớp tên thật nào) |
    FILTER_DROPPED (bỏ hết lọc)."""
    if filters is None:
        return search(question, k, None), None
    hits = search(question, k, filters)
    labels = unresolved(filters)
    if hits.empty:
        for loose, step_labels in relaxations(filters):
            hits = search(question, k, loose)
            if not hits.empty:
                return hits, relaxed_note([*labels, *step_labels])
        return search(question, k, None), FILTER_DROPPED
    return hits, relaxed_note(labels)


_reranker = None
_reranker_lock = threading.Lock()


def rerank(question: str, hits: pd.DataFrame, top: int | None = None) -> pd.DataFrame:
    """Cross-encoder chấm lại từng cặp (câu hỏi, chunk): chậm hơn nhưng chính xác hơn bi-encoder."""
    global _reranker
    s = get_settings()
    top = s.top_k if top is None else top
    with _reranker_lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(s.rerank_model)
        scores = _reranker.predict([(question, text) for text in hits["text"]])
    return hits.assign(rerank=scores).nlargest(top, "rerank").reset_index(drop=True)


def retrieve(question: str, k: int | None = None, use_filters: bool = True, use_rerank: bool | None = None,
             manual: dict | QueryFilters | None = None) -> dict:
    """Bộ lọc LLM (gộp với bộ lọc chỉnh tay `manual`, trường tay ghi đè) -> hybrid search có pre-filter
    (-> rerank nếu bật). Bộ lọc quá chặt (0 kết quả) ⇒ nới dần (`search_relaxed`), bỏ hết ⇒ `note = FILTER_DROPPED`.
    Trả `hits`, `filters` (QueryFilters | None), `source` (llm|manual|merged), `note`, `calls` (LLMResult)."""
    s = get_settings()
    k = s.top_k if k is None else k
    use_rerank = s.rerank if use_rerank is None else use_rerank
    extracted, parsed = parse_filters(question) if use_filters else (None, None)
    filters, source = merge(extracted, manual)
    active = None if filters.is_empty() else filters
    pool = max(s.rerank_candidates, k) if use_rerank else k
    hits, note = search_relaxed(search_hybrid, question, pool, active)
    if use_rerank and not hits.empty:
        hits = rerank(question, hits, top=k)
    return dict(hits=hits, filters=filters if (use_filters or active is not None) else None, source=source,
                note=note, calls=[parsed] if parsed else [])
