"""Tầng toàn cục của ingest (data-contract §2.2, §3, §3.3): va chạm id, dedup tầng 1+2, outlier, cụm SimHash, ghi file.

Chạy trên DuckDB (threads/memory/temp theo Settings) đọc `data/interim/shard_*.parquet`.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from vre.data import simhash
from vre.data.duck import sql_path

ORDER_REP = "published_at DESC NULLS LAST, n_nonnull DESC"
PRICE_GROUP_MIN = 200  # nhóm ≥ 200 tin dùng phân vị; nhóm nhỏ hơn dùng ngưỡng cố định
PRICE_P = (0.01, 0.99)
PRICE_FIXED = (0.05, 2000.0)  # tr/m²
AREA_P = (0.005, 0.995)
AREA_FIXED = (10.0, 10000.0)  # m²
ROW_GROUP = 100_000
CLEAN_COLUMNS = """listing_id, dup_cluster_id, n_raw_rows::INTEGER AS n_raw_rows, source_shard::SMALLINT AS source_shard,
  source_row::INTEGER AS source_row, title, description, property_type, deal_type, province, province_2025, district, ward,
  street, project, price_vnd, price_fixed_thousand, area_m2, floors::SMALLINT AS floors, frontage_m, depth_m, road_width_m,
  bedrooms::SMALLINT AS bedrooms, bathrooms::SMALLINT AS bathrooms, house_direction, balcony_direction,
  published_at::TIMESTAMP AS published_at, is_price_outlier, is_area_outlier, amenities"""


class ListingIdCollision(RuntimeError):
    """Hai key_exact khác nhau cho cùng listing_id — dừng, không salt ngầm (data-contract §2.1)."""


def check_collisions(con: duckdb.DuckDBPyConnection) -> int:
    bad = con.execute("""SELECT listing_id, count(DISTINCT key_hash2) AS n FROM interim
                         GROUP BY listing_id HAVING count(DISTINCT key_hash2) > 1 LIMIT 20""").fetchall()
    if bad:
        pairs = con.execute("""SELECT listing_id, source_shard, source_row, left(title, 60) FROM interim
                               WHERE listing_id IN (SELECT unnest(?)) ORDER BY 1""", [[b[0] for b in bad]]).fetchall()
        raise ListingIdCollision(f"{len(bad)} listing_id va chạm: {pairs}")
    return 0


def build_dup_map(con: duckdb.DuckDBPyConnection) -> None:
    """Bảng `dup_map` (mọi dòng thô) + `kept` (đại diện, kèm n_raw_rows)."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE rep1 AS
        SELECT source_shard, source_row, listing_id, prefix_key, published_at, n_nonnull,
               row_number() OVER (PARTITION BY listing_id ORDER BY {ORDER_REP}, source_shard, source_row) AS r1
        FROM interim WHERE drop_reason IS NULL""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE rep2 AS
        SELECT listing_id, prefix_key,
               row_number() OVER (PARTITION BY prefix_key ORDER BY {ORDER_REP}, listing_id) AS r2
        FROM rep1 WHERE r1 = 1""")
    con.execute("""CREATE OR REPLACE TEMP TABLE lid_map AS
        SELECT r.listing_id, w.listing_id AS kept_listing_id, r.r2 = 1 AS is_winner
        FROM rep2 r JOIN (SELECT prefix_key, listing_id FROM rep2 WHERE r2 = 1) w USING (prefix_key)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE dup_map AS
        SELECT i.source_shard::SMALLINT AS source_shard, i.source_row::INTEGER AS source_row,
               i.listing_id AS row_listing_id,
               CASE WHEN i.drop_reason IS NULL THEN m.kept_listing_id END AS kept_listing_id,
               CASE WHEN i.drop_reason IS NOT NULL THEN i.drop_reason
                    WHEN r.r1 = 1 AND m.is_winner THEN 'kept'
                    WHEN r.r1 = 1 THEN 'prefix_dup' ELSE 'exact_dup' END AS reason
        FROM interim i
        LEFT JOIN rep1 r USING (source_shard, source_row)
        LEFT JOIN lid_map m ON m.listing_id = i.listing_id""")  # điều kiện drop_reason để ở CASE: giữ hash join
    con.execute("""CREATE OR REPLACE TEMP VIEW kept AS
        SELECT i.*, n.n_raw_rows FROM interim i
        JOIN dup_map d USING (source_shard, source_row)
        JOIN (SELECT kept_listing_id, count(*) AS n_raw_rows FROM dup_map
              WHERE kept_listing_id IS NOT NULL GROUP BY 1) n ON n.kept_listing_id = i.listing_id
        WHERE d.reason = 'kept'""")


def flag_outliers(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""CREATE OR REPLACE TEMP TABLE flagged AS
        WITH k AS (SELECT *, CASE WHEN price_vnd IS NOT NULL AND area_m2 IS NOT NULL
                                  THEN price_vnd / area_m2 / 1e6 END AS pm2 FROM kept),
        pg AS (SELECT deal_type, property_type, province, count(pm2) AS n,
                      quantile_cont(pm2, {PRICE_P[0]}) AS lo, quantile_cont(pm2, {PRICE_P[1]}) AS hi
               FROM k GROUP BY ALL),
        ag AS (SELECT property_type, province, count(area_m2) AS n,
                      quantile_cont(area_m2, {AREA_P[0]}) AS lo, quantile_cont(area_m2, {AREA_P[1]}) AS hi
               FROM k GROUP BY ALL)
        SELECT k.* EXCLUDE (pm2),
          coalesce(CASE WHEN pg.n >= {PRICE_GROUP_MIN} THEN k.pm2 < pg.lo OR k.pm2 > pg.hi
                        ELSE k.pm2 < {PRICE_FIXED[0]} OR k.pm2 > {PRICE_FIXED[1]} END, false) AS is_price_outlier,
          coalesce(k.area_m2 < {AREA_FIXED[0]} OR k.area_m2 > {AREA_FIXED[1]}
                   OR (ag.n >= {PRICE_GROUP_MIN} AND (k.area_m2 < ag.lo OR k.area_m2 > ag.hi)), false) AS is_area_outlier
        FROM k
        LEFT JOIN pg ON pg.deal_type = k.deal_type AND pg.property_type = k.property_type
                    AND pg.province IS NOT DISTINCT FROM k.province
        LEFT JOIN ag ON ag.property_type = k.property_type AND ag.province IS NOT DISTINCT FROM k.province""")


def assign_clusters(con: duckdb.DuckDBPyConnection) -> dict:
    cols = con.execute("""SELECT listing_id, dense_rank() OVER (ORDER BY province, property_type) AS grp,
                                 simhash, simhash_valid FROM flagged""").fetchnumpy()
    ids = np.asarray(cols["listing_id"], dtype=np.int64)
    cluster_ids, stats = simhash.cluster(ids, np.asarray(cols["grp"], dtype=np.int64),
                                         np.asarray(cols["simhash"], dtype=np.uint64),
                                         np.asarray(cols["simhash_valid"], dtype=bool))
    con.register("clusters_df", pd.DataFrame({"listing_id": ids, "dup_cluster_id": cluster_ids}))
    return stats


def _copy(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    con.execute(f"COPY ({query}) TO '{sql_path(tmp)}' "
                f"(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE {ROW_GROUP})")
    os.replace(tmp, path)


def write_outputs(con: duckdb.DuckDBPyConnection, clean_path: Path, dup_map_path: Path) -> None:
    _copy(con, "SELECT * FROM dup_map ORDER BY source_shard, source_row", dup_map_path)
    _copy(con, f"""SELECT {CLEAN_COLUMNS} FROM flagged f JOIN clusters_df c USING (listing_id)
                   ORDER BY listing_id""", clean_path)


def run_global(con: duckdb.DuckDBPyConnection, interim_dir: Path, clean_path: Path, dup_map_path: Path) -> dict:
    """Toàn bộ tầng toàn cục; trả thống kê cho ingest_report."""
    con.execute(f"CREATE OR REPLACE TEMP VIEW interim AS SELECT * FROM read_parquet('{sql_path(interim_dir)}/shard_*.parquet')")
    check_collisions(con)
    build_dup_map(con)
    flag_outliers(con)
    stats = assign_clusters(con)
    write_outputs(con, clean_path, dup_map_path)
    counts = dict(con.execute("SELECT reason, count(*) FROM dup_map GROUP BY 1").fetchall())
    return {"reasons": counts, "simhash": stats}
