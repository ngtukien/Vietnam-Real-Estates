"""graph_export trên parquet tổng hợp nhỏ: header neo4j-admin, khoá node duy nhất, quan hệ trỏ tới node có thật."""

import csv
from datetime import datetime

import pandas as pd
import pytest

from vre.config import load_settings
from vre.index import graph_export

WARDS = [("Hồ Chí Minh", "Quận 7", f"Phường {i}") for i in range(1, 8)]


def _row(i, province, district, ward, **kw):
    base = {"listing_id": i, "dup_cluster_id": i, "n_raw_rows": 1, "source_shard": 0, "source_row": i,
            "title": f"Bán nhà {i}", "description": "mô tả", "property_type": "Nhà" if i % 3 else "Đất", "deal_type": "sale",
            "province": province, "province_2025": province, "district": district, "ward": ward, "street": "Nguyễn Văn Linh",
            "project": None, "price_vnd": 5e9 + i * 1e7, "price_fixed_thousand": False, "area_m2": 50.0 + i % 7,
            "floors": 2, "frontage_m": 4.0, "depth_m": 12.0, "road_width_m": 6.0, "bedrooms": 3, "bathrooms": 2,
            "house_direction": "Đông", "balcony_direction": None, "published_at": datetime(2025, 6 + i % 3, 1),
            "is_price_outlier": False, "is_area_outlier": False, "amenities": ["so_hong"] if i % 2 else []}
    return {**base, **kw}


@pytest.fixture
def exported(tmp_dir):
    rows = [_row(w * 10 + j, *WARDS[w], project="Sunrise City" if j < 3 else None) for w in range(7) for j in range(9)]
    rows += [
        _row(900, "Hồ Chí Minh", "Quận 7", None, title='Nhà "đẹp"\nhẻm\x07xe hơi', amenities=["hem_xe_hoi", "khong_co"]),
        _row(901, "Hồ Chí Minh", None, None, street=None, deal_type="rent", price_vnd=1.5e7, project="sunrise city"),
        _row(902, None, None, None, street=None, published_at=None),
    ]
    settings = load_settings(data_root=tmp_dir, duckdb_threads=2, duckdb_memory_gb=1)
    settings.clean_path.parent.mkdir(parents=True)
    df = pd.DataFrame(rows)
    df["bedrooms"] = df["bedrooms"].astype("int16")
    df.to_parquet(settings.clean_path, index=False)
    out = tmp_dir / "graph_import"
    return graph_export.export(settings, out_dir=out), out, len(rows)


def _read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def test_headers_follow_neo4j_admin_format(exported):
    info, out, _ = exported
    for label, name in info["nodes"].items():
        header = _read(out / name)[0]
        assert header[0] == f":ID({label})" and header[-1] == ":LABEL"
    listing_header = _read(out / info["nodes"]["Listing"])[0]
    assert {"listing_id:long", "price_vnd:double", "price_m2_mil:double", "bedrooms:long", "month"} <= set(listing_header)
    for name in info["relationships"]:
        header = _read(out / name)[0]
        assert header[0].startswith(":START_ID(") and header[1].startswith(":END_ID(") and header[2] == ":TYPE"


def test_node_keys_unique_and_relationships_resolve(exported):
    info, out, n_rows = exported
    ids = {}
    for label, name in info["nodes"].items():
        keys = [r[0] for r in _read(out / name)[1:]]
        assert len(keys) == len(set(keys)), label
        ids[label] = set(keys)
    assert len(ids["Listing"]) == n_rows == info["n_listings"]
    assert "Hồ Chí Minh|Quận 7|Phường 1" in ids["Ward"] and "Hồ Chí Minh|quận 7" not in ids["District"]
    assert {"Hồ Chí Minh|sunrise city"} == ids["Project"]  # khoá dự án hạ chữ thường, gộp hai cách viết
    for name in info["relationships"]:
        rows = _read(out / name)
        start, end = (h.split("(")[1].rstrip(")") for h in rows[0][:2])
        for s, e, _ in rows[1:]:
            assert s in ids[start] and e in ids[end], (name, s, e)


def test_listing_location_and_amenity_rules(exported):
    info, out, _ = exported
    in_district = {r[0] for r in _read(out / "rel_listing_in_district.csv")[1:]}
    in_province = {r[0] for r in _read(out / "rel_listing_in_province.csv")[1:]}
    assert in_district == {"900"} and in_province == {"901"}  # 902 thiếu cả tỉnh: không nối địa giới
    amen = {(r[0], r[1]) for r in _read(out / "rel_listing_has_amenity.csv")[1:]}
    assert ("900", "hem_xe_hoi") in amen and all(a != "khong_co" for _, a in amen)


def test_strings_escaped_and_single_line(exported):
    info, out, _ = exported
    text = (out / info["nodes"]["Listing"]).read_text(encoding="utf-8")
    assert '"Nhà ""đẹp"" hẻm xe hơi"' in text and "\x07" not in text
    rows = _read(out / info["nodes"]["Listing"])
    assert len(text.strip().splitlines()) == len(rows)


def test_market_stats_and_communities(exported):
    info, out, _ = exported
    rows = _read(out / info["nodes"]["MarketStat"])
    header, data = rows[0], [dict(zip(rows[0], r, strict=True)) for r in rows[1:]]
    assert "n:long" in header and all(int(r["n:long"]) >= graph_export.MARKET_MIN_N for r in data)
    q7 = [r for r in data if r["key"] == "district|Hồ Chí Minh|Quận 7|ALL|ALL|sale"]
    assert len(q7) == 1 and int(q7[0]["n:long"]) == 64  # 63 tin phường + 1 tin thiếu phường, đều bán
    assert {r["deal_type"] for r in data} == {"sale"}
    comm = info["communities"]
    assert comm["n_wards"] == 7 and comm["n_clusters"] >= 1 and comm["k"] == 5 and comm["seed"] == 42
    assert len(_read(out / "rel_ward_in_cluster.csv")) == 8
