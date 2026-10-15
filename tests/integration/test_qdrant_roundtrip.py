"""Round-trip Qdrant thật (collection tạm, 100 point): tạo đúng cấu hình §5, nạp, đếm theo part, dense + BM25 + filter."""

import os

import numpy as np
import pyarrow as pa
import pytest

pytestmark = pytest.mark.integration

N_LISTINGS, CHUNKS_PER_LISTING, DIM = 50, 2, 1024
PROVINCES = ("Hà Nội", "Hồ Chí Minh")


@pytest.fixture(scope="module")
def qdrant():
    from vre.config import load_settings
    from vre.index import qdrant_store

    settings = load_settings()
    qc = qdrant_store.client(settings, timeout=30)
    try:
        qc.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant không chạy: {exc}")
    name = f"test_roundtrip_{os.getpid()}"
    yield settings, qc, name
    if qc.collection_exists(name):
        qc.delete_collection(name)


def _part(rng):
    from vre.index import bm25, qdrant_upload
    from vre.index.chunk import point_id

    listings, rows = {}, []
    for lid in range(1000, 1000 + N_LISTINGS):
        province = PROVINCES[lid % 2]
        listings[lid] = qdrant_upload.listing_payload({
            "listing_id": lid, "dup_cluster_id": lid, "title": f"Tin {lid}", "description": "mô tả",
            "property_type": "Nhà", "deal_type": "sale", "province": province, "price_vnd": 1e9 * (lid % 7 + 1),
            "area_m2": 50.0, "is_price_outlier": False, "is_area_outlier": False, "amenities": []})
        for index in range(CHUNKS_PER_LISTING):
            text = "nhà mặt tiền thang máy" if lid == 1007 else f"căn hộ số {lid} chunk {index}"
            rows.append((point_id(lid, index), lid, index, text, bm25.doc_terms(bm25.tokenize_vi(text))))
    chunks = pa.table({"point_id": [r[0] for r in rows], "listing_id": [r[1] for r in rows],
                       "chunk_index": [r[2] for r in rows], "text": [r[3] for r in rows]})
    sparse = pa.table({"point_id": [r[0] for r in rows], "token_ids": [r[4][0] for r in rows],
                       "tfs": [r[4][1] for r in rows], "doc_len": [r[4][2] for r in rows]})
    emb = rng.normal(size=(len(rows), DIM)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return chunks, sparse, emb.astype(np.float16), listings


def test_roundtrip_dense_bm25_filter(qdrant):
    from qdrant_client import models

    from vre.index import bm25, qdrant_store, qdrant_upload

    settings, qc, name = qdrant
    assert qdrant_store.create_collection(settings, qc, name) is True
    assert qdrant_store.create_collection(settings, qc, name) is False  # idempotent
    config = qc.get_collection(name).config
    assert config.params.vectors["dense"].size == DIM and config.params.vectors["dense"].on_disk
    assert config.hnsw_config.m == qdrant_store.HNSW_M
    assert config.optimizer_config.indexing_threshold == settings.qdrant_indexing_threshold

    chunks, sparse, emb, listings = _part(np.random.default_rng(0))
    avgdl = float(np.mean(sparse.column("doc_len").to_numpy()))
    qc.upload_points(name, points=qdrant_upload.build_points(chunks, emb, sparse, listings, avgdl),
                     batch_size=32, wait=True)
    expected = N_LISTINGS * CHUNKS_PER_LISTING
    assert qdrant_store.count_part(qc, name, 1000, 1000 + N_LISTINGS - 1) == expected
    assert qdrant_store.count_part(qc, name, 1000, 1009) == 10 * CHUNKS_PER_LISTING

    params = models.SearchParams(quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0))
    hit = qc.query_points(name, query=emb[5].astype(np.float32).tolist(), using="dense", limit=1,
                          search_params=params).points[0]
    assert hit.id == chunks.column("point_id")[5].as_py()

    found = qc.query_points(name, query=bm25.query_vector("thang máy"), using="bm25", limit=3, with_payload=True).points
    assert found and found[0].payload["listing_id"] == 1007

    flt = models.Filter(must=[models.FieldCondition(key="province", match=models.MatchValue(value="Hà Nội"))])
    filtered = qc.query_points(name, query=emb[0].astype(np.float32).tolist(), using="dense", limit=20,
                               query_filter=flt, with_payload=True).points
    assert filtered and all(p.payload["province"] == "Hà Nội" for p in filtered)

    qdrant_store.set_alias(qc, f"{name}_alias", name)
    assert qdrant_store.alias_target(qc, f"{name}_alias") == name
    qdrant_store.set_alias(qc, f"{name}_alias", name)  # đặt lại không lỗi
    qc.delete_collection(name)
    assert qdrant_store.alias_target(qc, f"{name}_alias") is None
