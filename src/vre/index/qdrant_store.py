"""Collection Qdrant của chỉ mục chunk (data-contract §5): tạo, alias, chờ HNSW, đếm theo part, smoke search.

Dense `dense` 1024 chiều (cosine, vector gốc trên đĩa, bản int8 trong RAM, truy vấn rescore) + sparse `bm25` (modifier IDF).
HNSW xây SONG SONG khi nạp (`indexing_threshold`, `max_indexing_threads` lấy từ `vre.config`).
"""

import time

from vre.config import EMBED_MODEL_SLUG, EMBED_MODELS

KEYWORD_FIELDS = ("property_type", "deal_type", "province", "district", "ward", "street", "project",
                  "published_month", "amenities")
INTEGER_FIELDS = ("listing_id", "dup_cluster_id")
FLOAT_FIELDS = ("price_vnd", "area_m2", "price_m2_mil", "bedrooms")
HNSW_M, HNSW_EF_CONSTRUCT = 16, 100
GREEN_POLL_S = 15
INDEXED_RATIO = 0.99


def client(settings, timeout: int = 120):
    from qdrant_client import QdrantClient

    return QdrantClient(url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, prefer_grpc=True,
                        timeout=timeout)


def create_collection(settings, qc, name: str) -> bool:
    """Tạo collection nếu chưa có (True nếu vừa tạo). Có sẵn mà khác số chiều ⇒ lỗi, không xoá ngầm."""
    from qdrant_client import models

    dim = EMBED_MODELS[EMBED_MODEL_SLUG][1]
    if qc.collection_exists(name):
        size = qc.get_collection(name).config.params.vectors["dense"].size
        if size != dim:
            raise ValueError(f"collection {name} có vector {size} chiều, model {EMBED_MODEL_SLUG} cho {dim} chiều")
        return False
    qc.create_collection(
        name,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE, on_disk=True)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        hnsw_config=models.HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT, on_disk=False,
                                          max_indexing_threads=settings.qdrant_max_indexing_threads),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=settings.qdrant_indexing_threshold),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)),
        on_disk_payload=True,
    )
    schema = models.PayloadSchemaType
    for fields, kind in ((KEYWORD_FIELDS, schema.KEYWORD), (INTEGER_FIELDS, schema.INTEGER),
                         (FLOAT_FIELDS, schema.FLOAT)):
        for field in fields:
            qc.create_payload_index(name, field, kind, wait=True)
    return True


def set_alias(qc, alias: str, collection: str) -> None:
    """Trỏ alias về collection trong một thao tác nguyên tử (xoá alias cũ nếu có)."""
    from qdrant_client import models

    ops = []
    if any(a.alias_name == alias for a in qc.get_aliases().aliases):
        ops.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    ops.append(models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=collection,
                                                                         alias_name=alias)))
    qc.update_collection_aliases(change_aliases_operations=ops)


def alias_target(qc, alias: str) -> str | None:
    return next((a.collection_name for a in qc.get_aliases().aliases if a.alias_name == alias), None)


def info(qc, name: str) -> dict:
    c = qc.get_collection(name)
    return {"status": str(getattr(c.status, "value", c.status)), "points": int(c.points_count or 0),
            "indexed_vectors": int(c.indexed_vectors_count or 0)}


def wait_green(qc, name: str, timeout_s: float, log=print) -> dict:
    """Chờ `status=green` và `indexed_vectors ≥ 0,99·points`; quá `timeout_s` ⇒ TimeoutError kèm trạng thái cuối."""
    deadline = time.monotonic() + timeout_s
    while True:
        state = info(qc, name)
        if state["status"] == "green" and state["indexed_vectors"] >= INDEXED_RATIO * state["points"]:
            return state
        if time.monotonic() > deadline:
            raise TimeoutError(f"HNSW chưa xong sau {timeout_s:.0f}s: {state}")
        log(f"chờ HNSW: {state}")
        time.sleep(GREEN_POLL_S)


def count_part(qc, name: str, lo: int, hi: int) -> int:
    """Số point có `listing_id ∈ [lo, hi]` (đếm chính xác; listing_id của một part là một khoảng liền)."""
    from qdrant_client import models

    flt = models.Filter(must=[models.FieldCondition(key="listing_id", range=models.Range(gte=lo, lte=hi))])
    return qc.count(name, count_filter=flt, exact=True).count


def smoke_search(settings, query: str, collection: str | None = None, limit: int = 10) -> list[dict]:
    """Dense + BM25 hợp nhất RRF, gộp theo tin (không LLM). Trả tối đa `limit` tin khác nhau."""
    from qdrant_client import models

    from vre.index.bm25 import query_vector
    from vre.index.embed import get_encoder

    name = collection or settings.collection_alias
    dense = get_encoder(settings).encode_texts([query], kind="query")[0].astype("float32").tolist()
    params = models.SearchParams(quantization=models.QuantizationSearchParams(
        rescore=True, oversampling=settings.qdrant_oversampling))
    prefetch = [models.Prefetch(query=dense, using="dense", limit=limit * 4, params=params)]
    sparse = query_vector(query)
    if sparse.indices:
        prefetch.append(models.Prefetch(query=sparse, using="bm25", limit=limit * 4))
    points = client(settings).query_points(name, prefetch=prefetch, query=models.FusionQuery(fusion=models.Fusion.RRF),
                                           limit=limit * 4, with_payload=True).points
    hits, seen = [], set()
    for p in points:
        lid = p.payload["listing_id"]
        if lid in seen:
            continue
        seen.add(lid)
        hits.append({"listing_id": lid, "chunk_index": p.payload.get("chunk_index"), "score": p.score,
                     **{k: p.payload.get(k) for k in ("title", "property_type", "province", "district", "price_vnd")}})
        if len(hits) == limit:
            break
    return hits
