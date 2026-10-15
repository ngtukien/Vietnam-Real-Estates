"""Phụ thuộc dùng chung của router: backend, settings, bảng response lỗi cho OpenAPI."""

from __future__ import annotations

from typing import Any

from fastapi import Request

from api.deps import ApiSettings
from api.schemas import ErrorEnvelope
from api.services.backend import Backend

_DESCRIPTIONS = {
    400: "invalid_request", 401: "unauthorized", 404: "not_found", 413: "payload_too_large",
    415: "unsupported_media_type", 429: "rate_limited (kèm Retry-After)", 500: "internal_error",
    502: "upstream_llm_error", 503: "busy | system_unavailable", 504: "timeout",
}


def errors(*codes: int) -> dict[int | str, dict[str, Any]]:
    return {c: {"model": ErrorEnvelope, "description": _DESCRIPTIONS[c]} for c in codes}


def backend(request: Request) -> Backend:
    b: Backend = request.app.state.backend
    return b


def settings(request: Request) -> ApiSettings:
    s: ApiSettings = request.app.state.settings
    return s
