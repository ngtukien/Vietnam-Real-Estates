"""Một kết nối DuckDB / tiến trình (giới hạn RAM/luồng theo lane), mỗi truy vấn dùng `cursor()` riêng (thread-safe)."""

from __future__ import annotations

import threading
from typing import Any

import duckdb

from api.deps import ApiSettings

_lock = threading.Lock()
_con: duckdb.DuckDBPyConnection | None = None


def connection(settings: ApiSettings) -> duckdb.DuckDBPyConnection:
    global _con
    with _lock:
        if _con is None:
            from vre.config import load_settings
            from vre.data.duck import connect

            vs = load_settings(duckdb_memory_gb=settings.duckdb_memory_gb, duckdb_threads=settings.duckdb_threads)
            if not vs.clean_path.is_file():
                raise FileNotFoundError(f"Thiếu bảng sạch {vs.clean_path.name}")
            _con = connect(vs)
        return _con


def query(settings: ApiSettings, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    cur = connection(settings).cursor()
    try:
        rel = cur.execute(sql, params or [])
        cols = [d[0] for d in rel.description]
        return [dict(zip(cols, row, strict=True)) for row in rel.fetchall()]
    finally:
        cur.close()
