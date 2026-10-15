"""Facade `vre.qa` (api-contract §4/§7): thứ tự sự kiện từng hệ thống, lỗi → error + done, khoá answers.jsonl.

Không mạng: retrieve/router/Text2Cypher/LLM stream/Qdrant được thay bằng mock.
"""

import pandas as pd
import pytest

import vre.qa as qa
from vre.llm.client import LLMResult
from vre.qa import pipeline as P
from vre.rag.filters import QueryFilters

LID = 8353835631106358
ANSWER_KEYS = {"system", "answer", "route", "sources", "contexts", "cypher", "error", "fallback",
               "unsupported_citations", "latency_ms", "input_tokens", "output_tokens", "cached"}
ORDER = ["meta", "route", "filters", "cypher", "valuation", "sources", "token", "answer", "usage", "done"]


def call(i=10, o=5, cached=False):
    return LLMResult("", {}, i, o, cached)


class FakeStream:
    def __init__(self, text):
        self.text, self.result = text, None

    def __iter__(self):
        yield self.text[:5]
        yield self.text[5:]
        self.result = LLMResult(self.text, None, 100, 20, False)


class BrokenStream:
    result = None

    def __iter__(self):
        raise RuntimeError("x")


def hits():
    return pd.DataFrame([{"listing_id": LID, "dup_cluster_id": LID, "text": "Căn hộ 2PN", "title": "Căn hộ",
                          "property_type": "Căn hộ chung cư", "deal_type": "sale", "province": "Hồ Chí Minh",
                          "district": "Quận 7", "ward": None, "street": None, "project": None, "price_vnd": 3e9,
                          "area_m2": 70.0, "price_m2_mil": 42.8, "bedrooms": 2, "bathrooms": None,
                          "published_month": "2026-02", "amenities": ["ho_boi"], "description_excerpt": "x",
                          "chunk_index": 0, "score": 0.9, "rrf": 0.03}])


def found():
    return {"hits": hits(), "filters": QueryFilters(district="Quận 7"), "source": "llm", "note": None, "calls": [call()]}


def t2c(error=None):
    rows = pd.DataFrame({"district": ["Tây Hồ"], "median_price_m2_mil": [107.1]})
    return {"cypher": "MATCH (m:MarketStat) RETURN m LIMIT 50", "rows": rows, "attempts": [{"cypher": "", "error": error}],
            "error": error, "usage": [1, 1], "cypher_ms": 12.0, "calls": [call()]}


@pytest.fixture
def mocks(monkeypatch):
    state = {"route": "graph", "t2c": t2c()}
    monkeypatch.setattr(P, "retrieve", lambda *a, **k: found())
    monkeypatch.setattr(P, "stream_llm", lambda prompt, system="": FakeStream(f"Kết quả [Tin#{LID}] [Tin#99]"))
    monkeypatch.setattr(P, "router", lambda q: {"route": state["route"], "reason": "r", "call": call()})
    monkeypatch.setattr(P, "safe_text2cypher", lambda q: state["t2c"])
    monkeypatch.setattr(P, "text2cypher", lambda q: state["t2c"])
    monkeypatch.setattr(P.qdrant_search, "cards_by_ids", lambda ids: hits().assign(score=None))
    monkeypatch.setattr(P, "search_hybrid", lambda *a, **k: hits())
    return state


def names(events):
    return [n for n, _ in events]


@pytest.mark.parametrize(("system", "expected"), [
    ("llm_only", ["meta", "sources", "token", "token", "answer", "usage", "done"]),
    ("basic_rag", ["meta", "filters", "sources", "token", "token", "answer", "usage", "done"]),
    ("graph", ["meta", "cypher", "sources", "token", "token", "answer", "usage", "done"]),
    ("hybrid", ["meta", "route", "cypher", "sources", "token", "token", "answer", "usage", "done"]),
])
def test_event_order_per_system(mocks, system, expected):
    events = list(qa.stream_answer(system, "Quận nào ở Hà Nội đắt nhất?"))
    seq = names(events)
    assert seq == expected
    ranks = [ORDER.index(n) for n in seq]
    assert ranks == sorted(ranks)
    payload = dict(events)["answer"]
    assert payload["cited"] == ([LID] if system == "basic_rag" else [])
    assert 99 in payload["unsupported_citations"]
    usage = dict(events)["usage"]
    assert usage["input_tokens"] > 0 and usage["usd"] >= 0 and usage["cached"] is False
    assert dict(events)["done"] == {"status": "ok"}


def test_hybrid_falls_back_to_rag_when_cypher_fails(mocks):
    mocks["t2c"] = t2c(error="Neo.ClientError: bad")
    events = list(qa.stream_answer("hybrid", "q", stream_tokens=False))
    assert names(events) == ["meta", "route", "filters", "cypher", "sources", "answer", "usage", "done"]
    assert dict(events)["cypher"]["error"] == "Neo.ClientError: bad"  # lỗi Cypher vẫn gửi để hiển thị
    answer = dict(events)["answer"]
    assert answer["fallback"] is True and answer["route"] == "graph" and answer["cited"] == [LID]


def test_hybrid_valuation_route(mocks, monkeypatch):
    from vre.valuation import predict

    mocks["route"] = "valuation"
    payload = {"property_type": "Nhà", "province": "Hồ Chí Minh", "district": "Gò Vấp", "area_m2": 60.0}
    monkeypatch.setattr(P.hv, "extract_input", lambda q: (payload, call()))
    result = {"predicted_price_m2_mil": 100.0, "predicted_price_vnd": 6e9, "interval_price_vnd": {"low": 5e9, "high": 7e9},
              "baseline_ward_median_price_m2_mil": None, "comparables_n": 3, "verdict": None, "shap": [],
              "model": {"variant": "full", "path": "m", "trained_at": "t", "test_mape": 0.2}}
    monkeypatch.setattr(predict, "valuate", lambda p: result)
    events = list(qa.stream_answer("hybrid", "Nhà 60m² Gò Vấp giá bao nhiêu?", stream_tokens=False))
    assert names(events) == ["meta", "route", "filters", "valuation", "sources", "answer", "usage", "done"]
    f = dict(events)["filters"]["filters"]
    assert (f["district"], f["min_area_m2"], f["max_area_m2"]) == ("Gò Vấp", 48.0, 72.0)
    assert dict(events)["valuation"] is result


def test_valuation_route_without_input_falls_back(mocks, monkeypatch):
    from vre.valuation import predict

    mocks["route"] = "valuation"
    monkeypatch.setattr(P.hv, "extract_input", lambda q: ({}, call()))

    def missing(p):
        raise ValueError("Thiếu trường bắt buộc 'area_m2'")

    monkeypatch.setattr(predict, "valuate", missing)
    events = list(qa.stream_answer("hybrid", "định giá giúp", stream_tokens=False))
    assert "valuation" not in names(events) and dict(events)["answer"]["fallback"] is True


def test_error_after_meta_becomes_error_then_done(mocks, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("qdrant")

    monkeypatch.setattr(P, "retrieve", boom)
    events = list(qa.stream_answer("basic_rag", "q"))
    assert names(events) == ["meta", "error", "done"]
    assert dict(events)["error"]["code"] == "system_unavailable"
    assert dict(events)["done"] == {"status": "error"}


def test_invalid_input_raises_before_first_event():
    with pytest.raises(ValueError):
        qa.stream_answer("nope", "q")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        qa.stream_answer("hybrid", "q", filters={"property_type": "Kho"})
    with pytest.raises(ValueError):
        qa.answer("graph", "   ")


def test_answer_has_jsonl_keys(mocks):
    out = qa.answer("hybrid", "q")
    assert set(out) == ANSWER_KEYS
    assert out["route"] == "graph" and out["cypher"].startswith("MATCH") and out["error"] is None
    assert out["sources"] == [] and out["contexts"]  # kết quả đồ thị không có listing_id ⇒ không có nguồn tin


def test_answer_records_runtime_error(mocks, monkeypatch):
    monkeypatch.setattr(P, "stream_llm", lambda *a, **k: BrokenStream())
    out = qa.answer("llm_only", "q")
    assert out["error"].startswith("internal_error") and out["answer"] == ""


def test_listing_card_shape(mocks):
    events = dict(qa.stream_answer("basic_rag", "q", stream_tokens=False))
    card = events["sources"]["sources"][0]
    assert card["listing_id"] == LID
    assert card["address"] == {"street": None, "ward": None, "district": "Quận 7", "province": "Hồ Chí Minh"}
    assert card["bathrooms"] is None and card["amenities"] == ["ho_boi"] and card["score"] == 0.03
    assert card["cited"] is False


def test_valuate_contract_errors(tmp_path):
    from vre.config import load_settings
    from vre.valuation import predict

    settings = load_settings(data_root=tmp_path)
    with pytest.raises(RuntimeError):
        predict.valuate({"property_type": "Nhà", "province": "Hà Nội", "district": "Ba Đình", "area_m2": 50}, settings)
    with pytest.raises(ValueError):
        predict.valuate("not a dict")  # type: ignore[arg-type]


def _api_schemas():
    """`api/schemas.py` của Lane B (sau khi merge: cùng repo; trong lúc làm song song: worktree anh em)."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for path in (root / "api" / "schemas.py", root.parent / "api" / "api" / "schemas.py"):
        if path.is_file():
            spec = importlib.util.spec_from_file_location("vre_api_schemas", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module  # annotation hoãn (`from __future__`) cần module trong sys.modules
            spec.loader.exec_module(module)
            return module
    pytest.skip("không tìm thấy api/schemas.py (Lane B)")


@pytest.mark.parametrize("route", ["graph", "basic_rag", "valuation", "fallback"])
def test_events_validate_against_api_schemas(mocks, monkeypatch, route):
    from vre.valuation import predict

    schemas = _api_schemas()
    mocks["route"] = "graph" if route == "fallback" else route
    if route == "fallback":
        mocks["t2c"] = t2c(error="Neo.ClientError: bad")
    monkeypatch.setattr(P.hv, "extract_input", lambda q: ({"property_type": "Nhà", "province": "Hồ Chí Minh",
                                                           "district": "Gò Vấp", "area_m2": 60.0}, call()))
    monkeypatch.setattr(predict, "valuate", lambda p: {
        "predicted_price_m2_mil": 100.0, "predicted_price_vnd": 6e9, "interval_price_vnd": {"low": 5e9, "high": 7e9},
        "baseline_ward_median_price_m2_mil": None, "comparables_n": 3, "verdict": None,
        "shap": [{"feature": "area_m2", "value": 60.0, "contribution_log": 0.1}],
        "model": {"variant": "full", "path": "m", "trained_at": "t", "test_mape": 0.2}})
    for system in ("llm_only", "basic_rag", "graph", "hybrid"):
        for name, payload in qa.stream_answer(system, "q"):
            schemas.EVENT_MODELS[name].model_validate({**payload, "request_id": "abcd1234", "system": system})
