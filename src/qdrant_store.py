"""Chỉ mục toàn bộ dataset trên Qdrant: dense (E5) + sparse BM25 trong cùng một collection.

BM25 dạng sparse vector: phía tài liệu lưu trọng số tf đã chuẩn hoá theo độ dài,
tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl)); IDF do Qdrant tính lúc truy vấn (modifier IDF).
Câu hỏi chỉ cần gửi các token với trọng số 1. Không giữ chỉ mục BM25 nào trong RAM của Python.
Notebook dùng collection mẫu (listings_sample), chatbot dùng toàn bộ (listings_full) qua use().
"""

import json
import math
import zlib
from collections import Counter

import numpy as np
import pandas as pd

from src.config import INDEX_DIR, QDRANT_COLLECTION, QDRANT_URL

K1, B = 1.2, 0.75
CARD_FIELDS = ("title", "property_type", "province", "district", "ward", "street", "project", "price",
               "area", "bedrooms", "bathrooms", "published_at", "price_m2_mil")
DESCRIPTION_CHARS = 400
_client = None
_collection = QDRANT_COLLECTION


def use(collection: str) -> None:
    """Chọn collection cho mọi hàm bên dưới (rag.search_* dùng collection này)."""
    global _collection
    _collection = collection


def manifest_path(collection: str | None = None):
    return INDEX_DIR / f"{collection or _collection}.json"


def read_manifest(collection: str | None = None) -> dict | None:
    """Số tin, số chunk, tổng giá đã nạp: chatbot dùng để kiểm tra đồ thị khớp chỉ mục."""
    path = manifest_path(collection)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_manifest(info: dict, collection: str | None = None) -> None:
    path = manifest_path(collection)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient

        _client = QdrantClient(url=QDRANT_URL, timeout=120)
    return _client


# ---------------------------------------------------------------- collection


def ensure_collection(reset: bool = False) -> None:
    """Tạo collection nếu chưa có. Vector gốc nằm trên đĩa, bản nén int8 ở RAM để tìm nhanh;
    HNSW tắt trong lúc nạp hàng loạt (indexing_threshold=0), bật lại bằng finish_indexing()."""
    from qdrant_client import models

    c = client()
    if reset and c.collection_exists(_collection):
        c.delete_collection(_collection)
    if c.collection_exists(_collection):
        return
    c.create_collection(
        _collection,
        vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE, on_disk=True)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        on_disk_payload=True,
    )
    for field in ("province", "district", "property_type"):
        c.create_payload_index(_collection, field, models.PayloadSchemaType.KEYWORD)
    for field in ("price", "area", "bedrooms"):
        c.create_payload_index(_collection, field, models.PayloadSchemaType.FLOAT)
    c.create_payload_index(_collection, "listing_id", models.PayloadSchemaType.INTEGER)


def finish_indexing() -> None:
    """Bật lại xây HNSW sau khi nạp xong (ngưỡng mặc định của Qdrant)."""
    from qdrant_client import models

    client().update_collection(_collection, optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000))


def count() -> int:
    c = client()
    return c.count(_collection, exact=True).count if c.collection_exists(_collection) else 0


def load(chunks: pd.DataFrame, listings: pd.DataFrame, dense: np.ndarray, tokens: list[list[str]],
         reset: bool = True, batch: int = 1024) -> int:
    """Nạp trọn một bảng chunk (notebook 01): tạo lại collection, upsert theo lô, bật HNSW."""
    ensure_collection(reset=reset)
    avgdl = float(np.mean([len(t) for t in tokens]))
    for start in range(0, len(chunks), batch):
        end = start + batch
        upsert(chunks.iloc[start:end], listings, dense[start:end], tokens[start:end], avgdl)
    finish_indexing()
    return count()


# ---------------------------------------------------------------- BM25 sparse


def token_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


def bm25_document(tokens: list[str], avgdl: float):
    from qdrant_client import models

    tf = Counter(token_id(t) for t in tokens)
    norm = K1 * (1 - B + B * len(tokens) / avgdl)
    ids = sorted(tf)
    return models.SparseVector(indices=ids, values=[tf[i] * (K1 + 1) / (tf[i] + norm) for i in ids])


def bm25_query(tokens: list[str]):
    from qdrant_client import models

    ids = sorted({token_id(t) for t in tokens})
    return models.SparseVector(indices=ids, values=[1.0] * len(ids))


# ---------------------------------------------------------------- nạp


def _clean(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def upsert(chunks: pd.DataFrame, listings: pd.DataFrame, dense: np.ndarray, tokens: list[list[str]],
           avgdl: float) -> None:
    """Mỗi chunk là một point: id = listing_id·1000 + chunk_index (nạp lại không nhân bản).
    Payload chứa văn bản chunk, metadata để lọc và thông tin thẻ tin cho chatbot."""
    from qdrant_client import models

    cards = listings.set_index("listing_id")
    points = []
    for row, vector, toks in zip(chunks.itertuples(index=False), dense, tokens):
        listing = cards.loc[row.listing_id]
        payload = {"listing_id": int(row.listing_id), "chunk_index": int(row.chunk_index), "text": row.text}
        payload.update({f: _clean(listing[f]) for f in CARD_FIELDS})
        description = listing["description"] if isinstance(listing["description"], str) else ""
        payload["description"] = description[:DESCRIPTION_CHARS]
        points.append(models.PointStruct(
            id=int(row.listing_id) * 1000 + int(row.chunk_index),
            vector={"dense": vector.tolist(), "bm25": bm25_document(toks, avgdl)},
            payload={k: v for k, v in payload.items() if v is not None}))
    client().upsert(_collection, points=points, wait=True)


# ---------------------------------------------------------------- tìm


def to_filter(filters):
    """QueryFilters -> Filter của Qdrant (pre-filter trên payload index)."""
    from qdrant_client import models

    if filters is None:
        return None
    f = filters
    must = [models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in (("province", f.province), ("district", f.district), ("property_type", f.property_type)) if v]
    if f.bedrooms is not None:
        must.append(models.FieldCondition(key="bedrooms", range=models.Range(gte=f.bedrooms, lte=f.bedrooms)))
    if f.min_price_mil is not None or f.max_price_mil is not None:
        must.append(models.FieldCondition(key="price", range=models.Range(
            gte=None if f.min_price_mil is None else f.min_price_mil * 1e6,
            lte=None if f.max_price_mil is None else f.max_price_mil * 1e6)))
    if f.min_area is not None or f.max_area is not None:
        must.append(models.FieldCondition(key="area", range=models.Range(gte=f.min_area, lte=f.max_area)))
    return models.Filter(must=must) if must else None


def _hits(points) -> pd.DataFrame:
    rows = [{**p.payload, "chunk_id": f"{p.payload['listing_id']}:{p.payload['chunk_index']}", "score": p.score}
            for p in points]
    columns = ["listing_id", "chunk_id", "text", "score", "price", "area", "bedrooms", "property_type",
               "ward", "district", "province"]
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df:
            df[col] = None
    return df


def search(query, using: str, limit: int, filters=None) -> pd.DataFrame:
    result = client().query_points(_collection, query=query, using=using, limit=limit,
                                   query_filter=to_filter(filters), with_payload=True)
    return _hits(result.points)


def listings(ids: list[int]) -> dict[int, dict]:
    """Thông tin thẻ tin theo listing_id (lấy từ payload chunk đầu tiên của mỗi tin)."""
    if not ids:
        return {}
    points = client().retrieve(_collection, ids=[int(i) * 1000 for i in ids], with_payload=True)
    return {p.payload["listing_id"]: p.payload for p in points}
