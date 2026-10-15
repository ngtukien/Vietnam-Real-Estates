"""Chặn Cypher ghi/quản trị và ép LIMIT cho Cypher do LLM sinh (chuyển nguyên từ `src/graph.py`)."""

from __future__ import annotations

import re

from vre.config import get_settings

FORBIDDEN = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL|FOREACH|USE|SHOW|"
                       r"GRANT|DENY|REVOKE|ALTER|RENAME)\b", re.I)


def is_read_only(cypher: str) -> bool:
    """Lớp chặn đầu bằng regex (Phụ lục A.5); lớp thứ hai là giao dịch chỉ đọc của run_cypher."""
    return FORBIDDEN.search(cypher) is None


def ensure_limit(cypher: str, n: int | None = None) -> str:
    """Cypher do LLM sinh: thêm LIMIT n nếu mệnh đề RETURN cuối chưa có LIMIT.
    `n=None`: dùng `Settings.cypher_max_rows`."""
    if n is None:
        n = get_settings().cypher_max_rows
    cypher = cypher.strip().rstrip(";").rstrip()
    last_return = cypher.upper().rfind("RETURN")
    tail = cypher[last_return:] if last_return >= 0 else cypher
    return cypher if re.search(r"\bLIMIT\b", tail, re.I) else f"{cypher}\nLIMIT {n}"
