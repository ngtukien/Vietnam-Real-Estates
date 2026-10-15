"""Cypher chỉ đọc qua `vre.graph.client.run_cypher`; lỗi kết nối Neo4j ⇒ 503 `system_unavailable` (có lý do)."""

from __future__ import annotations

import logging
from typing import Any

from api.errors import ApiError
from api.schemas import ErrorCode
from vre.common import scalar

log = logging.getLogger("vre.api")
TIMEOUT_S = 10.0


def cypher(query: str, params: dict[str, Any] | None = None, max_rows: int | None = None) -> list[dict[str, Any]]:
    from vre.graph.client import run_cypher

    try:
        frame = run_cypher(query, params, timeout=TIMEOUT_S, max_rows=max_rows)
    except Exception as error:  # driver/kết nối/timeout: không lộ chi tiết, ghi log server
        log.warning("Neo4j lỗi (%s): %s", type(error).__name__, error)
        raise ApiError(ErrorCode.SYSTEM_UNAVAILABLE, f"Neo4j không sẵn sàng ({type(error).__name__})") from error
    return [{str(k): scalar(v) for k, v in row.items()} for row in frame.to_dict(orient="records")]
