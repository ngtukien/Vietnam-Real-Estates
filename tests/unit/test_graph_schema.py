"""KG mới: schema prompt + few-shot (A1 bước 4), B7 (cypher_system log khi lỗi), B3/B5 (grep)."""

import logging
import re
from pathlib import Path
from unittest import mock

import pandas as pd

from vre.graph import schema
from vre.graph import text2cypher as t2c
from vre.graph.guard import is_read_only

SRC = Path(__file__).resolve().parents[2] / "src" / "vre"


def test_few_shots_are_read_only_and_cover_new_kg():
    assert 8 <= len(schema.FEW_SHOT) <= 10
    joined = "\n".join(c for _, c in schema.FEW_SHOT)
    assert all(is_read_only(c) for _, c in schema.FEW_SHOT)
    for label in ("MarketStat", "Project", "Street", "Amenity", "Cluster"):
        assert f":{label}" in joined
    assert "m.month <> 'ALL'" in joined  # xu hướng tháng
    assert re.search(r"\bn >= 20\b", joined)


def test_aggregate_few_shots_never_scan_listing():
    for question, cypher in schema.FEW_SHOT:
        if re.search(r"\b(avg|median|percentileCont|percentileDisc)\s*\(", cypher, re.I):
            raise AssertionError(f"few-shot tự tính tổng hợp: {question}")
        if "median_price_m2_mil" in cypher and "Listing" in cypher:
            raise AssertionError(f"tổng hợp giá phải đọc MarketStat: {question}")


def test_system_prompt_lists_types_and_amenity_keys():
    text = schema.build_system(["Đất", "Nhà"])
    assert "giá trị: 'Đất', 'Nhà'" in text and "'thang_may'" in text and "MarketStat" in text


def test_cypher_system_logs_fallback(caplog):
    t2c._cypher_systems.clear()
    with caplog.at_level(logging.WARNING, logger="vre.graph.text2cypher"), \
            mock.patch.object(t2c, "run_cypher", side_effect=ConnectionError("down")):
        assert t2c.cypher_system() == t2c.CYPHER_SYSTEM
    assert any("ConnectionError" in r.getMessage() for r in caplog.records)
    assert t2c._cypher_systems == {}  # lỗi không bị lưu: lần sau đọc lại


def test_cypher_result_shape():
    rows = pd.DataFrame({"district": ["Tây Hồ"], "median": [107.1], "n": [6379], "node": [{"a": 1}]})
    out = t2c.cypher_result({"cypher": "MATCH (m) RETURN m LIMIT 50", "rows": rows, "error": None,
                             "attempts": [{"cypher": "x", "error": None}]})
    assert out["columns"] == ["district", "median", "n", "node"]
    assert out["rows"][0] == {"district": "Tây Hồ", "median": 107.1, "n": 6379, "node": '{"a": 1}'}
    assert (out["error"], out["attempts"]) == (None, 1)


def test_no_duplicate_scalar_helpers():
    """B3: một hàm `vre.common.scalar`, không định nghĩa `_value`/`_clean` riêng trong rag/graph/hybrid/qa."""
    for pkg in ("rag", "graph", "hybrid", "qa", "llm"):
        for path in (SRC / pkg).glob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"def (_value|_clean|scalar)\b", text), path


def test_no_cosine_threshold_in_rag():
    """B5: ngưỡng cosine vô tác dụng đã bỏ khỏi vre/rag."""
    for path in (SRC / "rag").glob("*.py"):
        assert "threshold" not in path.read_text(encoding="utf-8"), path
