"""Mọi fixture của chế độ fake hợp lệ theo schema Pydantic (api-contract §6) và chuỗi SSE đúng thứ tự §4."""

from __future__ import annotations

import json

import pytest
from api.services.fake import FIXTURES
from api_helpers import assert_order
from pydantic import BaseModel

from api import schemas as S

REST: dict[str, type[BaseModel]] = {
    "health": S.HealthResponse, "info": S.InfoResponse, "filters_options": S.FilterOptions, "valuation": S.ValuationResult,
    "stats_overview": S.StatsOverview, "market_province": S.MarketStats, "market_district": S.MarketStats,
    "subgraph": S.Subgraph, "benchmark": S.BenchmarkResponse,
}
CHAT = ["chat_llm_only", "chat_basic_rag", "chat_graph", "chat_hybrid", "chat_hybrid_valuation", "chat_error"]


def load(name: str) -> object:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def test_every_fixture_is_covered() -> None:
    names = {p.stem for p in FIXTURES.glob("*.json")}
    assert names == {*REST, *CHAT, "listings", "geo_province", "geo_district", "geo_ward"}


@pytest.mark.parametrize("name", sorted(REST))
def test_rest_fixture(name: str) -> None:
    REST[name].model_validate(load(name))


def test_listing_fixtures() -> None:
    data = load("listings")
    assert isinstance(data, dict) and data
    for key, detail in data.items():
        assert S.ListingDetail.model_validate(detail).listing_id == int(key)


@pytest.mark.parametrize("level", ["province", "district", "ward"])
def test_geo_fixture(level: str) -> None:
    doc = dict(load(f"geo_{level}"))  # type: ignore[call-overload]
    doc.pop("province", None)  # khoá chọn fixture theo tỉnh, không thuộc response
    assert S.GeoAreas.model_validate(doc).level == level


@pytest.mark.parametrize("name", CHAT)
def test_chat_sequence(name: str) -> None:
    seq = load(name)
    assert isinstance(seq, list)
    for event, payload in seq:
        S.EVENT_MODELS[event].model_validate({**payload, "request_id": "abcd1234", "system": "hybrid"})
    assert_order([event for event, _ in seq])


def test_sources_reference_known_listings() -> None:
    """Thẻ nguồn trong chuỗi SSE mở được chi tiết ở `/api/listings/{id}` (E2E bấm vào thẻ)."""
    known = set(load("listings"))  # type: ignore[arg-type]
    for name in CHAT:
        for event, payload in load(name):  # type: ignore[attr-defined]
            if event == "sources":
                assert {str(c["listing_id"]) for c in payload["sources"]} <= known, name
