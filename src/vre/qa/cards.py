"""`ListingCard` (api-contract §2) từ dòng kết quả Qdrant (`vre.rag.qdrant_search.HIT_COLUMNS`)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from vre.common import scalar

EXCERPT_CHARS = 400


def _num(value: Any) -> float | None:
    value = scalar(value)
    return None if value is None else float(value)


def _int(value: Any) -> int | None:
    value = scalar(value)
    return None if value is None else int(value)


def _str(value: Any) -> str | None:
    value = scalar(value)
    return value if isinstance(value, str) and value else None


def listing_card(row: dict[str, Any], score: Any = None, cited: bool = False) -> dict[str, Any]:
    amenities = row.get("amenities")
    return {
        "listing_id": int(row["listing_id"]),
        "dup_cluster_id": int(scalar(row.get("dup_cluster_id")) or row["listing_id"]),
        "title": _str(row.get("title")),
        "property_type": row.get("property_type"),
        "deal_type": row.get("deal_type"),
        "price_vnd": _num(row.get("price_vnd")),
        "price_m2_mil": _num(row.get("price_m2_mil")),
        "area_m2": _num(row.get("area_m2")),
        "bedrooms": _int(row.get("bedrooms")),
        "bathrooms": _int(row.get("bathrooms")),
        "address": {k: _str(row.get(k)) for k in ("street", "ward", "district", "province")},
        "project": _str(row.get("project")),
        "published_month": _str(row.get("published_month")),
        "description_excerpt": (_str(row.get("description_excerpt")) or "")[:EXCERPT_CHARS],
        "amenities": [str(a) for a in amenities] if amenities is not None and not isinstance(amenities, float) else [],
        "score": _num(score),
        "cited": cited,
    }


def cards(hits: pd.DataFrame, score_column: str = "score") -> list[dict[str, Any]]:
    """Thẻ tin theo thứ tự `hits`; cột điểm ưu tiên `rerank` > `rrf` > `score`."""
    if hits is None or hits.empty:
        return []
    column = next((c for c in ("rerank", "rrf", score_column) if c in hits), None)
    records: list[dict[str, Any]] = hits.to_dict("records")  # type: ignore[assignment]
    return [listing_card(rec, rec.get(column) if column else None) for rec in records]


def order_cards(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Bỏ trùng theo `listing_id`, giữ thứ tự, tối đa `limit` (= top_k·2)."""
    seen: set[int] = set()
    out = []
    for card in items:
        if card["listing_id"] not in seen:
            seen.add(card["listing_id"])
            out.append(card)
    return out[:limit]
