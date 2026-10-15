"""Endpoint REST ở chế độ fake: 200 đúng schema, 404/400 đúng mã."""

from __future__ import annotations

from typing import Any

import pytest
from api_helpers import request
from pydantic import BaseModel

from api import schemas as S

OK: list[tuple[str, type[BaseModel]]] = [
    ("/api/health", S.HealthResponse), ("/api/info", S.InfoResponse), ("/api/filters/options", S.FilterOptions),
    ("/api/listings/531409273736", S.ListingDetail), ("/api/stats/overview", S.StatsOverview),
    ("/api/stats/market?province=Hồ Chí Minh", S.MarketStats), ("/api/stats/market?province=Hồ Chí Minh&district=Quận 7", S.MarketStats),
    ("/api/geo/areas?level=province", S.GeoAreas), ("/api/geo/areas?level=district&province=Hồ Chí Minh", S.GeoAreas),
    ("/api/graph/subgraph?listing_id=531409273736", S.Subgraph),
    ("/api/graph/subgraph?node_label=District&key=Hồ Chí Minh|Quận 7&limit=10", S.Subgraph), ("/api/benchmark", S.BenchmarkResponse),
]


@pytest.mark.parametrize(("url", "model"), OK)
def test_ok(app: Any, url: str, model: type[BaseModel]) -> None:
    resp = request(app, "GET", url)
    assert resp.status_code == 200, resp.text
    model.model_validate(resp.json())


def test_info_limits_from_settings(app: Any) -> None:
    assert request(app, "GET", "/api/info").json()["limits"] == {"max_question_chars": 500, "rate_per_min": 20}


@pytest.mark.parametrize(("url", "status"), [
    ("/api/listings/42", 404), ("/api/listings/0", 400), ("/api/stats/market?province=Atlantis", 404), ("/api/stats/market", 400),
    ("/api/geo/areas?level=district", 400), ("/api/geo/areas?level=city", 400), ("/api/graph/subgraph", 400),
    ("/api/graph/subgraph?listing_id=1&node_label=Ward&key=x", 400), ("/api/graph/subgraph?node_label=Ward", 400),
    ("/api/graph/subgraph?node_label=Listing&key=abc", 400), ("/api/graph/subgraph?listing_id=1&limit=101", 400),
    ("/api/graph/subgraph?listing_id=7", 404),
])
def test_errors(app: Any, url: str, status: int) -> None:
    resp = request(app, "GET", url)
    assert resp.status_code == status, resp.text
    assert resp.json()["error"]["code"] == ("not_found" if status == 404 else "invalid_request")


def test_null_fields_present(app: Any) -> None:
    detail = request(app, "GET", "/api/listings/531409273736").json()
    assert set(detail) == set(S.ListingDetail.model_fields)
    assert detail["floors"] is None and detail["score"] is None and detail["cited"] is False
