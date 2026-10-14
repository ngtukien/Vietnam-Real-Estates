"""Tìm kiếm: dense, BM25, hybrid (RRF), rerank và `retrieve` (chuyển nguyên từ `src/rag.py`).

Chỉ mục (`vre.index.qdrant_store`, `vre.index.embed`) do phiên khác sở hữu: import trễ trong thân hàm,
Phase 3 đối chiếu lại tên hàm.
"""

from __future__ import annotations

import re
import threading

import numpy as np
import pandas as pd

from vre.config import get_settings
from vre.rag.filters import FILTER_DROPPED, QueryFilters, parse_filters

_tokenizer_lock = threading.Lock()  # mô hình CRF của pyvi dùng chung giữa các luồng của chatbot


def tokenize_vi(text: str) -> list[str]:
    """Tách từ tiếng Việt bằng pyvi: 'trường học' -> 'trường_học'; bỏ dấu câu."""
    from pyvi import ViTokenizer

    with _tokenizer_lock:
        tokens = ViTokenizer.tokenize(text.lower()).split()
    return [t for t in tokens if re.search(r"\w", t)]


def cosine(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _by_listing(hits: pd.DataFrame, k: int) -> pd.DataFrame:
    """Mỗi tin chỉ giữ chunk điểm cao nhất, để một tin dài không chiếm hết top-k."""
    hits = hits.sort_values("score", ascending=False).drop_duplicates("listing_id")
    return hits.head(k).reset_index(drop=True)


def search_dense(question: str, k: int | None = None, filters: QueryFilters | None = None,
                 threshold: float | None = None) -> pd.DataFrame:
    """Dense search trên Qdrant, có pre-filter theo payload và ngưỡng cosine tuỳ chọn."""
    from vre.index import embed, qdrant_store

    k = get_settings().top_k if k is None else k
    hits = qdrant_store.search(embed.embed_query(question).tolist(), "dense", k * 4, filters)
    if threshold is not None:
        hits = hits[hits["score"] >= threshold]
    return _by_listing(hits, k)


def search_bm25(question: str, k: int | None = None, filters: QueryFilters | None = None) -> pd.DataFrame:
    """BM25 trên Qdrant (sparse vector, IDF tính phía server) với từ đã tách bằng pyvi;
    khớp đúng từ như 'sổ hồng', '2PN'."""
    from vre.index import qdrant_store

    k = get_settings().top_k if k is None else k
    hits = qdrant_store.search(qdrant_store.bm25_query(tokenize_vi(question)), "bm25", k * 4, filters)
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
    from vre.index import embed, qdrant_store

    k = get_settings().top_k if k is None else k
    candidates = max(candidates, k)
    queries = [(embed.embed_query(question).tolist(), "dense")]
    tokens = tokenize_vi(question)
    if tokens:  # câu hỏi toàn dấu câu: không có từ nào cho BM25
        queries.append((qdrant_store.bm25_query(tokens), "bm25"))
    rankings = [_by_listing(h, candidates) for h in qdrant_store.search_many(queries, candidates * 4, filters)]
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
        "giá": hits["price"].map(fmt_vnd),
        "m²": hits["area"],
        "địa chỉ": hits[["ward", "district", "province"]].apply(
            lambda r: ", ".join(x for x in r if isinstance(x, str)), axis=1),
        "đoạn": hits["text"].str.replace("\n", " · ").str.slice(0, width),
    })


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


def retrieve(question: str, k: int | None = None, use_filters: bool = True, use_rerank: bool | None = None) -> dict:
    """Bộ lọc từ LLM -> hybrid search có pre-filter (-> rerank nếu bật). Nếu bộ lọc quá chặt
    (0 kết quả), bỏ lọc và ghi lại trong 'note'."""
    s = get_settings()
    k = s.top_k if k is None else k
    use_rerank = s.rerank if use_rerank is None else use_rerank
    filters, parsed = parse_filters(question) if use_filters else (None, None)
    pool = max(s.rerank_candidates, k) if use_rerank else k
    hits, note = search_hybrid(question, pool, filters), ""
    if hits.empty and filters is not None:
        hits, note = search_hybrid(question, pool), FILTER_DROPPED
    if use_rerank and not hits.empty:
        hits = rerank(question, hits, top=k)
    return dict(hits=hits, filters=filters, note=note, calls=[parsed] if parsed else [])
