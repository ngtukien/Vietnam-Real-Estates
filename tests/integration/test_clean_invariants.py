"""Bất biến bảng sạch + dup_map (data-contract §3, §3.3). Bỏ qua khi chưa chạy ingest."""

import duckdb
import pytest

from vre.config import load_settings

RAW_ROWS = 3_500_744
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def con():
    settings = load_settings()
    if not (settings.clean_path.is_file() and settings.dup_map_path.is_file()):
        pytest.skip("chưa có data/clean (chạy vre_pipeline.py ingest)")
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '2GB'")
    connection.execute(f"CREATE VIEW clean AS SELECT * FROM read_parquet('{settings.clean_path.as_posix()}', file_row_number = true)")
    connection.execute(f"CREATE VIEW dup AS SELECT * FROM read_parquet('{settings.dup_map_path.as_posix()}')")
    yield connection
    connection.close()


def test_dup_map_covers_every_raw_row(con):
    assert con.execute("SELECT count(*) FROM dup").fetchone()[0] == RAW_ROWS


def test_kept_rows_equal_clean_table(con):
    kept = con.execute("SELECT count(*) FROM dup WHERE reason = 'kept'").fetchone()[0]
    n, distinct = con.execute("SELECT count(*), count(DISTINCT listing_id) FROM clean").fetchone()
    assert kept == n == distinct


def test_raw_rows_add_up(con):
    merged = con.execute("SELECT sum(n_raw_rows) FROM clean").fetchone()[0]
    dropped = con.execute("SELECT count(*) FROM dup WHERE kept_listing_id IS NULL").fetchone()[0]
    assert merged + dropped == RAW_ROWS


def test_ids_are_53_bit_and_sorted(con):
    max_id, ordered = con.execute(
        "SELECT max(listing_id), bool_and(listing_id > prev) FROM "
        "(SELECT listing_id, lag(listing_id, 1, -1) OVER (ORDER BY file_row_number) AS prev FROM clean)").fetchone()
    assert max_id <= 2**53 - 1
    assert ordered


def test_cluster_id_is_member_minimum(con):
    bad = con.execute("""SELECT count(*) FROM (
        SELECT dup_cluster_id, min(listing_id) AS m FROM clean GROUP BY 1) WHERE dup_cluster_id <> m""").fetchone()[0]
    assert bad == 0
