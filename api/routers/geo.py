"""`GET /api/geo/areas?level=&province=`."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from api.errors import ApiError
from api.routers.common import backend, errors
from api.schemas import ErrorCode, GeoAreas, GeoLevel
from api.services.backend import Backend

router = APIRouter(prefix="/api/geo", tags=["geo"])


@router.get("/areas", response_model=GeoAreas, responses=errors(400, 401, 429, 503))
def areas(level: GeoLevel, province: str | None = Query(None, min_length=1, max_length=100),
          b: Backend = Depends(backend)) -> dict[str, Any]:
    if level != GeoLevel.PROVINCE and not province:
        raise ApiError(ErrorCode.INVALID_REQUEST, "province là bắt buộc khi level khác province",
                       {"fields": [{"loc": ["query", "province"], "msg": "bắt buộc khi level khác province"}]})
    return b.geo_areas(level.value, None if level == GeoLevel.PROVINCE else province)
