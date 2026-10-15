"""`/api/stats/overview`, `/api/stats/market`."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from api.errors import ApiError
from api.routers.common import backend, errors
from api.schemas import DealType, ErrorCode, MarketStats, PropertyType, StatsOverview
from api.services.backend import Backend

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/overview", response_model=StatsOverview, responses=errors(401, 429))
def overview(b: Backend = Depends(backend)) -> dict[str, Any]:
    return b.stats_overview()


@router.get("/market", response_model=MarketStats, responses=errors(400, 401, 404, 429, 503))
def market(province: str = Query(min_length=1, max_length=100),
           district: str | None = Query(None, min_length=1, max_length=100),
           property_type: PropertyType | None = None, deal_type: DealType = DealType.SALE,
           b: Backend = Depends(backend)) -> dict[str, Any]:
    out = b.market(province, district, property_type.value if property_type else None, deal_type.value)
    if out is None:
        where = f"{province} / {district}" if district else province
        raise ApiError(ErrorCode.NOT_FOUND, f"Không có thống kê cho {where}")
    return out
