"""Envelope lỗi (api-contract §5) và exception handler. Mọi lỗi HTTP của API đi qua `error_response`."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response

from api.schemas import ErrorCode

log = logging.getLogger("vre.api")

STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.PAYLOAD_TOO_LARGE: 413,
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: 415,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.UPSTREAM_LLM_ERROR: 502,
    ErrorCode.BUSY: 503,
    ErrorCode.SYSTEM_UNAVAILABLE: 503,
    ErrorCode.TIMEOUT: 504,
    ErrorCode.INTERNAL_ERROR: 500,
}


class ApiError(Exception):
    def __init__(self, code: ErrorCode, message: str, details: dict[str, Any] | None = None,
                 headers: dict[str, str] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details
        self.headers = headers or {}

    @property
    def status(self) -> int:
        return STATUS_BY_CODE[self.code]


def envelope(code: ErrorCode, message: str, request_id: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code.value, "message": message, "request_id": request_id, "details": details}}


def envelope_bytes(code: ErrorCode, message: str, request_id: str, details: dict[str, Any] | None = None) -> bytes:
    return json.dumps(envelope(code, message, request_id, details), ensure_ascii=False).encode("utf-8")


def internal_message(request_id: str) -> str:
    return f"Lỗi khi xử lý (mã {request_id})"


def request_id_of(request: Request) -> str:
    return str(request.scope.get("state", {}).get("request_id", "00000000"))


def error_response(request: Request, error: ApiError) -> Response:
    return JSONResponse(envelope(error.code, error.message, request_id_of(request), error.details),
                        status_code=error.status, headers=error.headers)


def _validation_fields(exc: RequestValidationError) -> list[dict[str, Any]]:
    return [{"loc": [str(p) for p in e.get("loc", ())], "msg": str(e.get("msg", ""))} for e in exc.errors()]


def install_handlers(app: FastAPI) -> None:
    async def on_api_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, ApiError)
        return error_response(request, exc)

    async def on_validation(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, RequestValidationError)
        return error_response(request, ApiError(ErrorCode.INVALID_REQUEST, "Dữ liệu yêu cầu không hợp lệ",
                                                {"fields": _validation_fields(exc)}))

    async def on_http(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, StarletteHTTPException)
        if exc.status_code == 404:
            err = ApiError(ErrorCode.NOT_FOUND, "Không tìm thấy")
        elif exc.status_code == 405:
            err = ApiError(ErrorCode.INVALID_REQUEST, "Phương thức không được hỗ trợ")
        else:
            log.warning("HTTPException %s ngoài bảng mã lỗi", exc.status_code)
            err = ApiError(ErrorCode.INTERNAL_ERROR, internal_message(request_id_of(request)))
        return error_response(request, err)

    app.add_exception_handler(ApiError, on_api_error)
    app.add_exception_handler(RequestValidationError, on_validation)
    app.add_exception_handler(StarletteHTTPException, on_http)
