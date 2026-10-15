"""Thẻ tin / chi tiết tin từ bảng sạch qua `v_listings` (DuckDB). File sort theo `listing_id` ⇒ row-group pruning."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from api.deps import ApiSettings
from api.schemas import MAX_DESCRIPTION_EXCERPT
from api.services import db
from vre.common import scalar

DETAIL_SQL = """
SELECT listing_id, dup_cluster_id, n_raw_rows, title, description, property_type, deal_type,
       province, district, ward, street, project, price_vnd, price_m2_mil, area_m2, bedrooms, bathrooms,
       floors, frontage_m, road_width_m, house_direction, balcony_direction, published_at, month, amenities
FROM v_listings WHERE listing_id = ?
"""
CLUSTER_SQL = "SELECT count(*) - 1 AS n FROM read_parquet(?) WHERE dup_cluster_id = ?"


def _int(value: Any) -> int | None:
    value = scalar(value)
    return None if value is None else int(value)


def _float(value: Any) -> float | None:
    value = scalar(value)
    return None if value is None else float(value)


def _iso_utc(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")  # bảng sạch lưu TIMESTAMP không múi giờ = UTC
    text = scalar(value)
    return None if text is None else str(text)


def card_from_row(row: dict[str, Any], score: float | None = None, cited: bool = False) -> dict[str, Any]:
    """Dòng bảng sạch (hoặc payload Qdrant, cùng tên khoá) → `ListingCard` (api-contract §2)."""
    description = scalar(row.get("description")) or ""
    excerpt = scalar(row.get("description_excerpt"))
    return {
        "listing_id": int(row["listing_id"]),
        "dup_cluster_id": int(row["dup_cluster_id"]),
        "title": scalar(row.get("title")),
        "property_type": row["property_type"],
        "deal_type": row["deal_type"],
        "price_vnd": _float(row.get("price_vnd")),
        "price_m2_mil": _float(row.get("price_m2_mil")),
        "area_m2": _float(row.get("area_m2")),
        "bedrooms": _int(row.get("bedrooms")),
        "bathrooms": _int(row.get("bathrooms")),
        "address": {k: scalar(row.get(k)) for k in ("street", "ward", "district", "province")},
        "project": scalar(row.get("project")),
        "published_month": scalar(row.get("month") if "month" in row else row.get("published_month")),
        "description_excerpt": (excerpt if excerpt is not None else description)[:MAX_DESCRIPTION_EXCERPT],
        "amenities": [str(a) for a in (row.get("amenities") or [])],
        "score": score,
        "cited": cited,
    }


def get_detail(settings: ApiSettings, listing_id: int) -> dict[str, Any] | None:
    rows = db.query(settings, DETAIL_SQL, [listing_id])
    if not rows:
        return None
    row = rows[0]
    from vre.config import load_settings
    from vre.data.duck import sql_path

    clean = sql_path(load_settings().clean_path)
    similar = db.query(settings, CLUSTER_SQL, [clean, int(row["dup_cluster_id"])])[0]["n"]
    return {
        **card_from_row(row),
        "description": scalar(row.get("description")),
        "floors": _int(row.get("floors")),
        "frontage_m": _float(row.get("frontage_m")),
        "road_width_m": _float(row.get("road_width_m")),
        "house_direction": scalar(row.get("house_direction")),
        "balcony_direction": scalar(row.get("balcony_direction")),
        "published_at": _iso_utc(row.get("published_at")),
        "n_raw_rows": int(row["n_raw_rows"]),
        "similar_in_cluster": max(0, int(similar)),
    }
