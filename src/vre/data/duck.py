"""Kết nối DuckDB theo cấu hình chia máy (coordination §8) và view chuẩn `v_listings` (data-contract §3.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from vre.config import Settings

VIEWS_SQL = Path(__file__).with_name("views.sql")


def sql_path(path: Path) -> str:
    """Đường dẫn tuyệt đối dạng DuckDB chấp nhận (dấu /, escape nháy đơn)."""
    return path.resolve().as_posix().replace("'", "''")


def one_row(cursor: duckdb.DuckDBPyConnection) -> tuple[Any, ...]:
    """Dòng đầu của kết quả (aggregate luôn trả đúng một dòng); không có dòng nào ⇒ lỗi rõ thay vì `None[0]`."""
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("truy vấn DuckDB không trả dòng nào")
    return row


def connect(settings: Settings, views: bool = True) -> duckdb.DuckDBPyConnection:
    tmp = settings.cache_dir / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {int(settings.duckdb_threads)}")
    con.execute(f"SET memory_limit = '{int(settings.duckdb_memory_gb)}GB'")
    con.execute(f"SET temp_directory = '{sql_path(tmp)}'")
    con.execute("SET preserve_insertion_order = false")
    if views:
        con.execute(VIEWS_SQL.read_text(encoding="utf-8").replace("{clean_path}", sql_path(settings.clean_path)))
    return con
