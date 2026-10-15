"""Truy vấn Qdrant cho RAG: alias `vre_chunks` (data-contract §5), dense e5-large + sparse BM25, lấy thẻ tin theo id.

Chỉ ĐỌC. Tạo/nạp collection thuộc `vre.index` (Phase 2); module này dùng lại encoder (`vre.index.embed`) và mã hoá
token BM25 (`vre.index.bm25`) để truy vấn khớp đúng cách chỉ mục đã được xây.
"""

from __future__ import annotations

import threading
from typing import Any

import pandas as pd

from vre.config import get_settings

CHUNKS_PER_LISTING = 64  # point_id = listing_id * 64 + chunk_index (data-contract §2.3)
PAYLOAD_COLUMNS = ("listing_id", "dup_cluster_id", "chunk_index", "text", "title", "description_excerpt",
                   "property_type", "deal_type", "province", "district", "ward", "street", "project",
                   "price_vnd", "area_m2", "price_m2_mil", "bedrooms", "bathrooms", "published_month", "amenities")
HIT_COLUMNS = (*PAYLOAD_COLUMNS, "score")

_client = None
_client_lock = threading.Lock()
_tokenize_lock = threading.Lock()  # mô hình CRF của pyvi không an toàn đa luồng


def client() -> Any:
    global _client
    with _client_lock:
        if _client is None:
            from vre.index import qdrant_store

            _client = qdrant_store.client(get_settings(), timeout=30)
        return _client


def collection() -> str:
    return get_settings().collection_alias


def embed_query(question: str) -> list[float]:
    """Vector truy vấn e5-large: tiền tố `query: `, mean pooling, L2 (data-contract §9.1) — CPU fp16, xem
    `vre.rag.query_encoder` (encoder GPU tốn ~6 GB commit trên Windows)."""
    from vre.rag import query_encoder

    return query_encoder.embed(question)


def tokenize(text: str) -> list[str]:
    from vre.index.bm25 import tokenize_vi

    with _tokenize_lock:
        return tokenize_vi(text)


def bm25_query(tokens: list[str]) -> Any:
    """Sparse vector truy vấn: mỗi token (crc32) trọng số 1; IDF do Qdrant tính."""
    from qdrant_client import models

    from vre.index.bm25 import token_id

    ids = sorted({token_id(t) for t in tokens})
    return models.SparseVector(indices=ids, values=[1.0] * len(ids))


def hits_frame(points: list[Any]) -> pd.DataFrame:
    """Point Qdrant → DataFrame cột cố định (khoá payload vắng ⇒ None, data-contract §5 bỏ khoá null)."""
    rows = [{**{c: (p.payload or {}).get(c) for c in PAYLOAD_COLUMNS}, "score": getattr(p, "score", None)}
            for p in points]
    return pd.DataFrame(rows, columns=list(HIT_COLUMNS))


def search_many(queries: list[tuple[Any, str]], limit: int, query_filter: Any = None) -> list[pd.DataFrame]:
    """Nhiều truy vấn (vector, tên vector `dense`|`bm25`) trong MỘT lượt `query_batch_points`."""
    from qdrant_client import models

    s = get_settings()
    dense_params = models.SearchParams(quantization=models.QuantizationSearchParams(
        rescore=True, oversampling=s.qdrant_oversampling))
    requests = [models.QueryRequest(query=vector, using=using, limit=limit, filter=query_filter, with_payload=True,
                                    params=dense_params if using == "dense" else None)
                for vector, using in queries]
    responses = client().query_batch_points(collection(), requests=requests)
    return [hits_frame(r.points) for r in responses]


def cards_by_ids(listing_ids: list[int]) -> pd.DataFrame:
    """Payload chunk 0 của từng tin (thẻ tin cho kết quả Cypher), giữ thứ tự `listing_ids`; tin không có bỏ qua."""
    ids = [int(i) for i in dict.fromkeys(listing_ids)]
    if not ids:
        return hits_frame([])
    points = client().retrieve(collection(), ids=[i * CHUNKS_PER_LISTING for i in ids], with_payload=True)
    frame = hits_frame(points).assign(score=None)
    order = {lid: pos for pos, lid in enumerate(ids)}
    return frame.sort_values("listing_id", key=lambda col: col.map(order)).reset_index(drop=True)
