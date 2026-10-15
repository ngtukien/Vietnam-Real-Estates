"""`POST /api/valuate` → facade `vre.valuation.predict.valuate` (ValueError → 400, chưa có model → 503)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from api.routers.common import backend, errors
from api.schemas import ValuationInput, ValuationResult
from api.services.backend import Backend

router = APIRouter(prefix="/api", tags=["valuation"])


@router.post("/valuate", response_model=ValuationResult, responses=errors(400, 401, 413, 415, 429, 503))
async def valuate(body: ValuationInput, b: Backend = Depends(backend)) -> dict[str, Any]:
    return await run_in_threadpool(b.valuate, body.model_dump(mode="json"))
