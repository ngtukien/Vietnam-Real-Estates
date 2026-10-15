"""`POST /api/valuate`: validate input, verdict, ánh xạ lỗi facade (ValueError → 400, chưa có model → 503)."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from api.schemas import ValuationResult
from api.services.backend import RealBackend
from api_helpers import make_app, request

BODY = {"property_type": "Căn hộ chung cư", "province": "Hồ Chí Minh", "district": "Quận 7", "ward": "Phú Mỹ", "area_m2": 70,
        "bedrooms": 2}


def test_valuate_ok_and_verdict(app: Any) -> None:
    resp = request(app, "POST", "/api/valuate", json=BODY)
    assert resp.status_code == 200
    result = ValuationResult.model_validate(resp.json())
    assert result.verdict is None and len(result.shap) <= 8
    low = result.interval_price_vnd.low
    assert request(app, "POST", "/api/valuate", json={**BODY, "asking_price_vnd": low / 2}).json()["verdict"] == "below_market"
    assert request(app, "POST", "/api/valuate", json={**BODY, "asking_price_vnd": result.predicted_price_vnd}).json()["verdict"] == "fair"


@pytest.mark.parametrize("patch", [{"area_m2": 0}, {"area_m2": 200_000}, {"property_type": "Kho"}, {"month": "2025-13"},
                                   {"district": ""}, {"asking_price_vnd": -1}])
def test_valuate_validation(app: Any, patch: dict[str, Any]) -> None:
    resp = request(app, "POST", "/api/valuate", json={**BODY, **patch})
    assert resp.status_code == 400 and resp.json()["error"]["code"] == "invalid_request"


def test_rent_without_model_is_400(app: Any) -> None:
    resp = request(app, "POST", "/api/valuate", json={**BODY, "deal_type": "rent"})
    assert resp.status_code == 400


@pytest.mark.parametrize(("exc", "status", "code"), [(ValueError("ward lạ"), 400, "invalid_request"),
                                                    (RuntimeError("chưa có model"), 503, "system_unavailable"),
                                                    (FileNotFoundError("xgb_full_sale.json"), 503, "system_unavailable")])
def test_real_backend_maps_facade_errors(monkeypatch: pytest.MonkeyPatch, exc: Exception, status: int, code: str) -> None:
    module = types.ModuleType("vre.valuation.predict")

    def valuate(payload: dict[str, Any]) -> dict[str, Any]:
        raise exc

    module.valuate = valuate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vre.valuation.predict", module)
    app = make_app()
    app.state.backend = RealBackend(app.state.settings)
    resp = request(app, "POST", "/api/valuate", json=BODY)
    assert resp.status_code == status and resp.json()["error"]["code"] == code
