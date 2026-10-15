"""`GET /api/listings/{listing_id}`."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Path

from api.errors import ApiError
from api.routers.common import backend, errors
from api.schemas import ErrorCode, ListingDetail
from api.services.backend import Backend

router = APIRouter(prefix="/api", tags=["listings"])
MAX_SAFE_INT = 2**53 - 1


@router.get("/listings/{listing_id}", response_model=ListingDetail, responses=errors(400, 401, 404, 429))
def listing(listing_id: int = Path(ge=1, le=MAX_SAFE_INT), b: Backend = Depends(backend)) -> dict[str, Any]:
    found = b.listing(listing_id)
    if found is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"Không có tin {listing_id}")
    return found
