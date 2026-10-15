"""Khu vực cho bản đồ: số tin + median từ node Province/District/Ward (Neo4j), toạ độ từ `data/geo/centroids.json`
(Lane C sinh, data-contract §8). Thiếu file ⇒ lat/lon null, coverage 0 (ghi log một lần)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from api.services.neo import cypher

log = logging.getLogger("vre.api")

LEVEL_QUERY = {
    "province": "MATCH (a:Province) RETURN a.key AS key, a.name AS name, a.n_listings AS n, a.median_price_m2_mil AS med",
    "district": ("MATCH (a:District) WHERE a.key STARTS WITH $prefix "
                 "RETURN a.key AS key, a.name AS name, a.n_listings AS n, a.median_price_m2_mil AS med"),
    "ward": ("MATCH (a:Ward) WHERE a.key STARTS WITH $prefix "
             "RETURN a.key AS key, a.name AS name, a.n_listings AS n, a.median_price_m2_mil AS med"),
}
_cache: dict[str, Any] = {"mtime": None, "coords": {}}
_warned = False


def centroids_path() -> Path:
    from vre.config import load_settings

    return load_settings().data_root / "data" / "geo" / "centroids.json"


def _coords() -> dict[tuple[str, str], tuple[float, float]]:
    global _warned
    path = centroids_path()
    if not path.is_file():
        if not _warned:
            log.warning("Chưa có %s (Lane C: scripts/build_geo.py) — lat/lon = null", path.name)
            _warned = True
        return {}
    mtime = path.stat().st_mtime
    if _cache["mtime"] != mtime:
        doc = json.loads(path.read_text(encoding="utf-8"))
        _cache["coords"] = {(i["level"], i["key"]): (float(i["lat"]), float(i["lon"])) for i in doc.get("items", [])}
        _cache["mtime"] = mtime
    coords: dict[tuple[str, str], tuple[float, float]] = _cache["coords"]
    return coords


def areas(level: str, province: str | None) -> dict[str, Any]:
    params = {} if level == "province" else {"prefix": f"{province}|"}
    rows = cypher(LEVEL_QUERY[level], params)
    coords = _coords()
    items = []
    for r in sorted(rows, key=lambda r: -(r["n"] or 0)):
        lat, lon = coords.get((level, str(r["key"])), (None, None))
        items.append({"key": str(r["key"]), "name": str(r["name"] or str(r["key"]).rsplit("|", 1)[-1]), "lat": lat, "lon": lon,
                      "n_listings": int(r["n"] or 0), "median_price_m2_mil": r["med"]})
    located = sum(1 for i in items if i["lat"] is not None)
    return {"level": level, "items": items, "coverage": located / len(items) if items else 0.0}
