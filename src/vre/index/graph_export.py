"""Xuất bảng sạch thành CSV cho `neo4j-admin database import full` (data-contract §6.1–6.2).

Mọi tổng hợp chạy trong DuckDB trên `v_listings`; Python chỉ giữ bảng phường đã tổng hợp (cho Leiden).
Header CSV theo neo4j-admin: `:ID(<Label>)`, `<prop>:long|double|int`, `:LABEL`, `:START_ID(..)`, `:END_ID(..)`, `:TYPE`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from vre.config import Settings
from vre.data.amenities import amenity_table
from vre.data.duck import connect, one_row, sql_path
from vre.graph.communities import ward_communities, ward_profiles

COMMUNITY_K, COMMUNITY_SEED, COMMUNITY_MIN_LISTINGS = 5, 42, 8
MARKET_MIN_N = 5
MARKET_DEAL_TYPES = ("sale",)
SALE_MEDIAN = "median(price_m2_mil) FILTER (WHERE deal_type = 'sale')"
# Ký tự điều khiển + ngắt dòng Unicode: bỏ để mỗi bản ghi CSV nằm trên đúng một dòng.
CLEAN_MACRO = r"""CREATE OR REPLACE TEMP MACRO cl(s) AS
  nullif(trim(regexp_replace(s, '[[:cntrl:]\x{85}\x{2028}\x{2029}]+', ' ', 'g')), '')"""

BASE_SQL = """CREATE OR REPLACE TEMP TABLE g AS
SELECT *,
  CASE WHEN province IS NOT NULL AND district IS NOT NULL THEN province || '|' || district END AS dk,
  CASE WHEN province IS NOT NULL AND district IS NOT NULL AND ward IS NOT NULL
       THEN province || '|' || district || '|' || ward END AS wk,
  CASE WHEN province IS NOT NULL AND district IS NOT NULL AND street IS NOT NULL
       THEN province || '|' || district || '|' || street END AS sk,
  CASE WHEN province IS NOT NULL AND project IS NOT NULL THEN province || '|' || lower(project) END AS prk
FROM (SELECT listing_id, dup_cluster_id, cl(title) AS title, property_type, deal_type, price_vnd, area_m2, price_m2_mil,
        bedrooms, bathrooms, floors, frontage_m, road_width_m, cl(house_direction) AS direction,
        strftime(published_at, '%Y-%m-%dT%H:%M:%S') AS published_at, month, amenities,
        cl(province) AS province, cl(province_2025) AS province_2025, cl(district) AS district, cl(ward) AS ward,
        cl(street) AS street, cl(project) AS project
      FROM v_listings {limit})"""

NODES: dict[str, str] = {
    "Listing": """SELECT listing_id AS ":ID(Listing)", listing_id AS "listing_id:long", title, property_type, deal_type,
        price_vnd AS "price_vnd:double", area_m2 AS "area_m2:double", price_m2_mil AS "price_m2_mil:double",
        bedrooms AS "bedrooms:long", bathrooms AS "bathrooms:long", floors AS "floors:long",
        frontage_m AS "frontage_m:double", road_width_m AS "road_width_m:double", direction, published_at, month,
        dup_cluster_id AS "dup_cluster_id:long", 'Listing' AS ":LABEL" FROM g""",
    "PropertyType": """SELECT DISTINCT property_type AS ":ID(PropertyType)", property_type AS name,
        'PropertyType' AS ":LABEL" FROM g""",
    "Province": f"""SELECT province AS ":ID(Province)", province AS key, province AS name, mode(province_2025) AS name_2025,
        count(*) AS "n_listings:long", {SALE_MEDIAN} AS "median_price_m2_mil:double", 'Province' AS ":LABEL"
        FROM g WHERE province IS NOT NULL GROUP BY province""",
    "District": f"""SELECT dk AS ":ID(District)", dk AS key, any_value(district) AS name, count(*) AS "n_listings:long",
        {SALE_MEDIAN} AS "median_price_m2_mil:double", 'District' AS ":LABEL" FROM g WHERE dk IS NOT NULL GROUP BY dk""",
    "Ward": f"""SELECT wk AS ":ID(Ward)", wk AS key, any_value(ward) AS name, count(*) AS "n_listings:long",
        {SALE_MEDIAN} AS "median_price_m2_mil:double", 'Ward' AS ":LABEL" FROM g WHERE wk IS NOT NULL GROUP BY wk""",
    "Street": """SELECT sk AS ":ID(Street)", sk AS key, any_value(street) AS name, count(*) AS "n_listings:long",
        'Street' AS ":LABEL" FROM g WHERE sk IS NOT NULL GROUP BY sk""",
    "Project": f"""SELECT prk AS ":ID(Project)", prk AS key, mode(project) AS name, count(*) AS "n_listings:long",
        {SALE_MEDIAN} AS "median_price_m2_mil:double", 'Project' AS ":LABEL" FROM g WHERE prk IS NOT NULL GROUP BY prk""",
    "Month": """SELECT DISTINCT month AS ":ID(Month)", month AS key, CAST(left(month, 4) AS INTEGER) AS "year:int",
        CAST(right(month, 2) AS INTEGER) AS "month:int", 'Month' AS ":LABEL" FROM g WHERE month IS NOT NULL""",
    "Amenity": """SELECT key AS ":ID(Amenity)", key, name, 'Amenity' AS ":LABEL" FROM amenity_df""",
    "Cluster": """SELECT c.cluster_id AS ":ID(Cluster)", c.cluster_id AS "cluster_id:long", count(DISTINCT c.ward_key) AS "n_wards:long",
        median(g.price_m2_mil) AS "median_price_m2_mil:double",
        mode(g.province) || ' · ' || CAST(round(quantile_cont(g.price_m2_mil, 0.25)) AS BIGINT) || '–'
          || CAST(round(quantile_cont(g.price_m2_mil, 0.75)) AS BIGINT) || ' tr/m²' AS label, 'Cluster' AS ":LABEL"
        FROM cluster_df c JOIN g ON g.wk = c.ward_key AND g.deal_type = 'sale' GROUP BY c.cluster_id""",
    "MarketStat": """SELECT key AS ":ID(MarketStat)", key, scope, area_key, property_type, month, deal_type, n AS "n:long",
        med AS "median_price_m2_mil:double", p25 AS "p25_price_m2_mil:double", p75 AS "p75_price_m2_mil:double",
        med_vnd AS "median_price_vnd:double", med_area AS "median_area_m2:double", 'MarketStat' AS ":LABEL" FROM ms""",
}

# (tệp, kiểu, nhãn đầu, nhãn cuối, biểu thức đầu, biểu thức cuối, FROM ... WHERE ...)
RELS: list[tuple[str, str, str, str, str, str, str]] = [
    ("rel_listing_of_type", "OF_TYPE", "Listing", "PropertyType", "listing_id", "property_type", "g"),
    ("rel_listing_in_ward", "IN_WARD", "Listing", "Ward", "listing_id", "wk", "g WHERE wk IS NOT NULL"),
    ("rel_listing_in_district", "IN_DISTRICT", "Listing", "District", "listing_id", "dk", "g WHERE wk IS NULL AND dk IS NOT NULL"),
    ("rel_listing_in_province", "IN_PROVINCE", "Listing", "Province", "listing_id", "province",
     "g WHERE dk IS NULL AND province IS NOT NULL"),
    ("rel_listing_on_street", "ON_STREET", "Listing", "Street", "listing_id", "sk", "g WHERE sk IS NOT NULL"),
    ("rel_listing_in_project", "IN_PROJECT", "Listing", "Project", "listing_id", "prk", "g WHERE prk IS NOT NULL"),
    ("rel_listing_posted_in", "POSTED_IN", "Listing", "Month", "listing_id", "month", "g WHERE month IS NOT NULL"),
    ("rel_listing_has_amenity", "HAS_AMENITY", "Listing", "Amenity", "listing_id", "a",
     "(SELECT DISTINCT listing_id, unnest(amenities) AS a FROM g) WHERE a IN (SELECT key FROM amenity_df)"),
    ("rel_ward_in_district", "IN_DISTRICT", "Ward", "District", "DISTINCT wk", "dk", "g WHERE wk IS NOT NULL"),
    ("rel_street_in_district", "IN_DISTRICT", "Street", "District", "DISTINCT sk", "dk", "g WHERE sk IS NOT NULL"),
    ("rel_project_in_district", "IN_DISTRICT", "Project", "District", "prk", "arg_max(dk, n)",
     "(SELECT prk, dk, count(*) AS n FROM g WHERE prk IS NOT NULL AND dk IS NOT NULL GROUP BY ALL) GROUP BY prk"),
    ("rel_district_in_province", "IN_PROVINCE", "District", "Province", "DISTINCT dk", "province", "g WHERE dk IS NOT NULL"),
    ("rel_ward_in_cluster", "IN_CLUSTER", "Ward", "Cluster", "ward_key", "cluster_id", "cluster_df"),
] + [(f"rel_stat_of_{scope}", "STAT_OF", "MarketStat", label, "key", "area_key", f"ms WHERE scope = '{scope}'")
     for scope, label in (("province", "Province"), ("district", "District"), ("ward", "Ward"), ("project", "Project"))]

MARKET_SCOPES = {"province": "province", "district": "dk", "ward": "wk", "project": "prk"}
MARKET_SQL = """SELECT '{scope}' AS scope, ak AS area_key,
    CASE WHEN grouping(property_type) = 1 THEN 'ALL' ELSE property_type END AS pt,
    CASE WHEN grouping(month) = 1 THEN 'ALL' ELSE month END AS mo, deal_type, count(*) AS n,
    median(price_m2_mil) AS med, quantile_cont(price_m2_mil, 0.25) AS p25, quantile_cont(price_m2_mil, 0.75) AS p75,
    median(price_vnd) AS med_vnd, median(area_m2) AS med_area
  FROM (SELECT {key} AS ak, * FROM g WHERE {key} IS NOT NULL AND price_m2_mil IS NOT NULL AND deal_type IN ({deals}))
  GROUP BY GROUPING SETS ((ak, deal_type, property_type, month), (ak, deal_type, property_type), (ak, deal_type, month),
                          (ak, deal_type))"""

WARD_AGG_SQL = """WITH s AS (SELECT * FROM g WHERE deal_type = 'sale' AND price_m2_mil IS NOT NULL AND wk IS NOT NULL),
  w AS (SELECT wk, median(price_m2_mil) AS median_price_m2 FROM s GROUP BY wk)
SELECT province, district, ward, property_type, count(*) AS n, any_value(w.median_price_m2) AS median_price_m2
FROM s JOIN w USING (wk) GROUP BY province, district, ward, property_type"""


def _market_stats(con) -> None:
    deals = ", ".join(f"'{d}'" for d in MARKET_DEAL_TYPES)
    parts = " UNION ALL ".join(MARKET_SQL.format(scope=s, key=k, deals=deals) for s, k in MARKET_SCOPES.items())
    con.execute(f"""CREATE OR REPLACE TEMP TABLE ms AS
        SELECT concat_ws('|', scope, area_key, pt, mo, deal_type) AS key, scope, area_key, pt AS property_type,
               mo AS month, deal_type, n, med, p25, p75, med_vnd, med_area
        FROM ({parts}) WHERE n >= {MARKET_MIN_N} AND mo IS NOT NULL""")


def _communities(con) -> dict[str, Any]:
    prof = ward_profiles(con.execute(WARD_AGG_SQL).df(), min_listings=COMMUNITY_MIN_LISTINGS)
    info: dict[str, Any] = {"n_wards": len(prof), "n_clusters": 0, "modularity": None, "k": COMMUNITY_K, "seed": COMMUNITY_SEED}
    cluster_df = pd.DataFrame({"ward_key": pd.Series(dtype="string"), "cluster_id": pd.Series(dtype="int64")})
    if len(prof) > COMMUNITY_K:  # kNN cần ít nhất k+1 phường
        prof, q = ward_communities(prof, k=COMMUNITY_K, seed=COMMUNITY_SEED)
        cluster_df = pd.DataFrame({"ward_key": ["|".join(map(str, i)) for i in prof.index],
                                   "cluster_id": prof["community"].astype("int64").to_numpy()})
        info.update(n_clusters=int(cluster_df["cluster_id"].nunique()), modularity=float(q))
    con.register("cluster_df", cluster_df)
    return info


def export(settings: Settings, out_dir: Path | None = None, limit: int | None = None) -> dict[str, Any]:
    """Ghi mọi CSV vào `out_dir` (mặc định `<data_root>/data/graph_import`). `limit`: chỉ `limit` tin đầu (thử import).

    Trả `{nodes:{Label:file}, relationships:[file], n_listings, price_vnd_sum, communities}`."""
    out = out_dir or settings.data_root / "data" / "graph_import"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.csv"):
        old.unlink()
    con = connect(settings)
    try:
        con.execute(CLEAN_MACRO)
        con.execute(BASE_SQL.format(limit=f"ORDER BY listing_id LIMIT {int(limit)}" if limit else ""))
        table = amenity_table()
        con.register("amenity_df", pd.DataFrame({"key": list(table), "name": [v["name"] for v in table.values()]}))
        _market_stats(con)
        info = _communities(con)

        def copy(name: str, sql: str) -> str:
            path = out / f"{name}.csv"
            con.execute(f"COPY ({sql}) TO '{sql_path(path)}' (HEADER, DELIMITER ',', QUOTE '\"', ESCAPE '\"')")
            return path.name

        nodes = {label: copy(f"node_{label.lower()}", sql) for label, sql in NODES.items()}
        rels = [copy(f, f'SELECT {s} AS ":START_ID({sl})", {e} AS ":END_ID({el})", \'{t}\' AS ":TYPE" FROM {src}')
                for f, t, sl, el, s, e, src in RELS]
        n, total = one_row(con.execute("SELECT count(*), kahan_sum(price_vnd) FROM g"))
    finally:
        con.close()
    return {"nodes": nodes, "relationships": rels, "n_listings": int(n), "price_vnd_sum": float(total or 0.0),
            "communities": info}
