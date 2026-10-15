"""Kết nối Neo4j và chạy Cypher (chuyển nguyên từ `src/graph.py`). Hợp đồng: `use`, `driver`, `run_cypher`."""

from __future__ import annotations

from itertools import islice

import pandas as pd

from vre.config import get_settings

_driver = None
_uri: str | None = None  # None: dùng `Settings.neo4j_uri`


def current_uri() -> str:
    """Neo4j đích đang dùng: URI đặt bằng `use()`, nếu chưa đặt thì `Settings.neo4j_uri`."""
    return _uri or get_settings().neo4j_uri


def use(uri: str) -> None:
    """Đổi Neo4j đích (đóng driver cũ nếu có)."""
    global _driver, _uri
    if _driver is not None:
        _driver.close()
    _driver, _uri = None, uri


def driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase

        s = get_settings()
        _driver = GraphDatabase.driver(current_uri(), auth=(s.neo4j_user, s.neo4j_password), connection_timeout=10)
        _driver.verify_connectivity()
    return _driver


def run_cypher(query: str, params: dict | None = None, write: bool = False, timeout: float | None = None,
               max_rows: int | None = None) -> pd.DataFrame:
    """Chạy Cypher, trả DataFrame. Mặc định mở giao dịch CHỈ ĐỌC: Neo4j từ chối mọi lệnh ghi
    trong giao dịch đọc, đây là lớp bảo vệ ở phía cơ sở dữ liệu cho Text2Cypher.
    `timeout` (giây): Neo4j huỷ giao dịch chạy quá lâu. `max_rows`: chỉ đọc chừng ấy dòng đầu."""
    def work(tx):
        result = tx.run(query, params or {})
        records = list(islice(result, max_rows)) if max_rows else list(result)
        return pd.DataFrame([r.data() for r in records], columns=result.keys())

    if timeout:
        from neo4j import unit_of_work

        work = unit_of_work(timeout=timeout)(work)
    with driver().session() as session:
        return session.execute_write(work) if write else session.execute_read(work)
