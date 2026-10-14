"""Payload + point Qdrant (data-contract §5): khoá đúng, bỏ null, price_m2_mil như `v_listings`, thứ tự hàng nhất quán."""

import datetime as dt

import numpy as np
import pyarrow as pa
import pytest

from vre.index import qdrant_upload

ALLOWED = {"listing_id", "dup_cluster_id", "chunk_index", "text", "title", "description_excerpt", "property_type",
           "deal_type", "province", "district", "ward", "street", "project", "price_vnd", "area_m2", "price_m2_mil",
           "bedrooms", "bathrooms", "published_month", "amenities"}


def _row(**extra) -> dict:
    row = {"listing_id": 11, "dup_cluster_id": 7, "title": "Bán nhà", "description": "x" * 1000, "property_type": "Nhà",
           "deal_type": "sale", "province": "Hà Nội", "district": "Cầu Giấy", "ward": None, "street": None,
           "project": None, "price_vnd": 5e9, "area_m2": 50.0, "bedrooms": 3, "bathrooms": None,
           "published_at": dt.datetime(2025, 12, 31, 23, 0), "is_price_outlier": False, "is_area_outlier": False,
           "amenities": ["so_hong"]}
    row.update(extra)
    return row


def test_payload_keys_and_null_dropping():
    payload = qdrant_upload.listing_payload(_row())
    assert set(payload) <= ALLOWED
    assert "ward" not in payload and "street" not in payload and "bathrooms" not in payload
    assert None not in payload.values()
    assert len(payload["description_excerpt"]) == qdrant_upload.DESCRIPTION_EXCERPT_CHARS
    assert payload["published_month"] == "2025-12"
    assert payload["price_m2_mil"] == pytest.approx(100.0)
    assert payload["amenities"] == ["so_hong"]


@pytest.mark.parametrize("extra", [{"is_price_outlier": True}, {"is_area_outlier": True}, {"price_vnd": None},
                                   {"area_m2": None}])
def test_price_m2_null_when_outlier_or_missing(extra):
    assert "price_m2_mil" not in qdrant_upload.listing_payload(_row(**extra))


def test_month_uses_utc():
    tz = dt.timezone(dt.timedelta(hours=7))
    payload = qdrant_upload.listing_payload(_row(published_at=dt.datetime(2026, 2, 1, 3, 0, tzinfo=tz)))
    assert payload["published_month"] == "2026-01"
    assert "published_month" not in qdrant_upload.listing_payload(_row(published_at=None))


def _tables(point_ids):
    chunks = pa.table({"point_id": point_ids, "listing_id": [p // 64 for p in point_ids],
                       "chunk_index": [p % 64 for p in point_ids], "text": [f"t{p}" for p in point_ids]})
    bm25 = pa.table({"point_id": point_ids, "token_ids": [[3, 8], []], "tfs": [[1, 2], []], "doc_len": [3, 0]})
    return chunks, bm25


def test_build_points_merges_rows():
    point_ids = [11 * 64, 11 * 64 + 1]
    chunks, bm25 = _tables(point_ids)
    emb = np.eye(2, 4, dtype=np.float16)
    listings = {11: qdrant_upload.listing_payload(_row())}
    points = list(qdrant_upload.build_points(chunks, emb, bm25, listings, avgdl=3.0))
    assert [p.id for p in points] == point_ids
    assert points[0].vector["dense"] == [1.0, 0.0, 0.0, 0.0]
    assert points[0].vector["bm25"].indices == [3, 8]
    assert "bm25" not in points[1].vector  # chunk không có token BM25: chỉ dense
    assert points[1].payload["chunk_index"] == 1 and points[1].payload["text"] == f"t{point_ids[1]}"
    assert set(points[0].payload) <= ALLOWED


def test_build_points_rejects_misaligned_rows():
    chunks, bm25 = _tables([11 * 64, 11 * 64 + 1])
    swapped = bm25.take([1, 0])
    listings = {11: qdrant_upload.listing_payload(_row())}
    with pytest.raises(ValueError):
        list(qdrant_upload.build_points(chunks, np.zeros((2, 4), np.float16), swapped, listings, 3.0))
    with pytest.raises(ValueError):
        list(qdrant_upload.build_points(chunks, np.zeros((3, 4), np.float16), bm25, listings, 3.0))
