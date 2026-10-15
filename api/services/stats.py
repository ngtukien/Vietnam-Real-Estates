"""Thống kê: tổng quan + tuỳ chọn bộ lọc (DuckDB trên `v_listings`, cache theo mtime bảng sạch) và thị trường
(MarketStat trong Neo4j, data-contract §6.1). Median giá/m² chỉ trên `deal_type='sale'` như MarketStat."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from api.deps import ApiSettings
from api.schemas import PropertyType
from api.services import db
from api.services.neo import cypher

TOP_PROVINCES = 10


def _clean_mtime() -> float:
    from vre.config import load_settings

    return load_settings().clean_path.stat().st_mtime


def _results_dir() -> Path:
    from vre.config import load_settings

    return load_settings().results_dir


@lru_cache(maxsize=2)
def _overview(settings: ApiSettings, mtime: float) -> dict[str, Any]:
    report_path = _results_dir() / "ingest_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dedup = report["dedup"]
    by_type = db.query(settings, """
        SELECT property_type, count(*) AS n, median(price_m2_mil) FILTER (WHERE deal_type = 'sale') AS med
        FROM v_listings GROUP BY property_type ORDER BY n DESC""")
    by_prov = db.query(settings, f"""
        SELECT province, count(*) AS n, median(price_m2_mil) FILTER (WHERE deal_type = 'sale') AS med
        FROM v_listings WHERE province IS NOT NULL GROUP BY province ORDER BY n DESC LIMIT {TOP_PROVINCES}""")
    return {
        "n_listings": int(dedup["kept"]),
        "n_raw_rows": int(report["raw_rows"]),
        "n_duplicates_removed": int(dedup["exact_dup"]) + int(dedup["prefix_dup"]),
        "by_property_type": [{"property_type": r["property_type"], "n": int(r["n"]), "median_price_m2_mil": r["med"]}
                             for r in by_type],
        "by_province_top": [{"province": r["province"], "n": int(r["n"]), "median_price_m2_mil": r["med"]} for r in by_prov],
    }


def overview(settings: ApiSettings) -> dict[str, Any]:
    return _overview(settings, _clean_mtime())


@lru_cache(maxsize=2)
def _filter_options(settings: ApiSettings, mtime: float) -> dict[str, Any]:
    rows = db.query(settings, """
        SELECT province, district FROM v_listings WHERE province IS NOT NULL
        GROUP BY province, district ORDER BY province, district""")
    districts: dict[str, list[str]] = {}
    for r in rows:
        bucket = districts.setdefault(r["province"], [])
        if r["district"] is not None:
            bucket.append(r["district"])
    months = [r["month"] for r in db.query(settings, "SELECT DISTINCT month FROM v_listings WHERE month IS NOT NULL ORDER BY month")]
    return {"provinces": sorted(districts), "districts": districts, "property_types": [p.value for p in PropertyType],
            "months": months}


def filter_options(settings: ApiSettings) -> dict[str, Any]:
    return _filter_options(settings, _clean_mtime())


def data_period(settings: ApiSettings) -> dict[str, str]:
    months = filter_options(settings)["months"]
    if not months:
        raise RuntimeError("Bảng sạch không có tháng nào (published_at null toàn bộ)")
    return {"min_month": months[0], "max_month": months[-1]}


MARKET_SQL = """
MATCH (m:MarketStat {scope: $scope, area_key: $key, property_type: $ptype, deal_type: $deal})
WHERE m.month <> 'ALL'
RETURN m.month AS month, m.n AS n, m.median_price_m2_mil AS med ORDER BY month
"""
AREA_SQL = """
MATCH (m:MarketStat {scope: $scope, property_type: $ptype, month: 'ALL', deal_type: $deal})
WHERE m.area_key STARTS WITH $prefix
RETURN m.area_key AS key, m.n AS n, m.median_price_m2_mil AS med ORDER BY n DESC
"""
EXISTS_SQL = {
    "province": "MATCH (p:Province {key: $key}) RETURN count(p) AS n",
    "district": "MATCH (d:District {key: $key}) RETURN count(d) AS n",
}


def market(province: str, district: str | None, property_type: str | None, deal_type: str) -> dict[str, Any] | None:
    """None ⇒ tỉnh/quận không có trong đồ thị (404)."""
    scope, key = ("district", f"{province}|{district}") if district else ("province", province)
    if not cypher(EXISTS_SQL[scope], {"key": key})[0]["n"]:
        return None
    ptype = property_type or "ALL"
    monthly = cypher(MARKET_SQL, {"scope": scope, "key": key, "ptype": ptype, "deal": deal_type})
    child = "ward" if district else "district"
    areas = cypher(AREA_SQL, {"scope": child, "prefix": key + "|", "ptype": ptype, "deal": deal_type})
    return {
        "scope": {"province": province, "district": district, "property_type": property_type, "deal_type": deal_type},
        "monthly": [{"month": r["month"], "n": int(r["n"]), "median_price_m2_mil": r["med"]} for r in monthly],
        "by_area": [{"key": r["key"], "name": str(r["key"]).rsplit("|", 1)[-1], "n": int(r["n"]),
                     "median_price_m2_mil": r["med"]} for r in areas],
    }
