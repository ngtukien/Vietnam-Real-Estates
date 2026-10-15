"""`/api/health`, `/api/info`, `/api/filters/options`, `/api/benchmark` (api-contract §3)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from api.deps import ApiSettings
from api.errors import ApiError
from api.routers.common import backend, errors, settings
from api.schemas import MAX_QUESTION_CHARS, BenchmarkResponse, ErrorCode, FilterOptions, HealthResponse, InfoResponse
from api.services.backend import Backend

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(b: Backend = Depends(backend)) -> dict[str, Any]:
    return b.health()


@router.get("/info", response_model=InfoResponse, responses=errors(401, 429, 503))
def info(b: Backend = Depends(backend), s: ApiSettings = Depends(settings)) -> dict[str, Any]:
    out = b.info()
    out["limits"] = {"max_question_chars": MAX_QUESTION_CHARS, "rate_per_min": s.chat_rate_per_min}
    return out


@router.get("/filters/options", response_model=FilterOptions, responses=errors(401, 429))
def filter_options(b: Backend = Depends(backend)) -> dict[str, Any]:
    return b.filter_options()


@router.get("/benchmark", response_model=BenchmarkResponse, responses=errors(401, 404, 429))
def benchmark(b: Backend = Depends(backend)) -> dict[str, Any]:
    out = b.benchmark()
    if out is None:
        raise ApiError(ErrorCode.NOT_FOUND, "Chưa chạy eval (thiếu results/eval_summary.json)")
    return out
