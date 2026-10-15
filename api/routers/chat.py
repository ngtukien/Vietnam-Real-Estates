"""`POST /api/chat` (SSE, api-contract §4). Lỗi trước khi mở stream ⇒ HTTP + ErrorEnvelope; sau đó ⇒ sự kiện."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from api.deps import ApiSettings, Busy, ChatSlots
from api.errors import ApiError, request_id_of
from api.routers.common import backend, errors, settings
from api.schemas import ChatRequest, ErrorCode, SystemId
from api.services.backend import Backend
from api.sse import Event, Lease, Settle, sse_response, starter, system_events

log = logging.getLogger("vre.api")
router = APIRouter(prefix="/api", tags=["chat"])
SSE_DOC: dict[int | str, dict[str, Any]] = {
    200: {"description": "text/event-stream; payload mỗi sự kiện theo các schema Event*", "content": {"text/event-stream": {}}},
}


async def unavailable(b: Backend, systems: list[SystemId]) -> dict[str, str]:
    """{system: lý do} cho các hệ thống chưa sẵn sàng. Rỗng ⇒ tất cả chạy được."""
    status = await run_in_threadpool(b.system_status)
    out: dict[str, str] = {}
    for s in systems:
        ok, reason = status.get(s.value, (False, None))
        if not ok:
            out[s.value] = reason or "chưa sẵn sàng"
    return out


def unavailable_error(missing: dict[str, str]) -> ApiError:
    reasons = "; ".join(f"{s}: {r}" for s, r in missing.items())
    return ApiError(ErrorCode.SYSTEM_UNAVAILABLE, f"Hệ thống chưa sẵn sàng — {reasons}")


async def lease(request: Request, n: int, holders: int) -> Lease:
    slots: ChatSlots = request.app.state.slots
    try:
        got = await slots.acquire(n)
    except Busy:
        raise ApiError(ErrorCode.BUSY, "Server đang bận, thử lại sau") from None
    return Lease(slots, got, holders)


@router.post("/chat", responses={**SSE_DOC, **errors(400, 401, 413, 415, 429, 503)})
async def chat(body: ChatRequest, request: Request, b: Backend = Depends(backend),
               s: ApiSettings = Depends(settings)) -> Response:
    rid = request_id_of(request)
    missing = await unavailable(b, [body.system])
    if missing:
        raise unavailable_error(missing)
    held = await lease(request, 1, 1)
    filters = body.filters.model_dump(mode="json") if body.filters else None
    # Không ghi nội dung câu hỏi (có thể chứa thông tin cá nhân), chỉ độ dài.
    log.info("request=%s chat system=%s chars=%d top_k=%d rerank=%s", rid, body.system.value, len(body.question),
             body.top_k, body.use_rerank)
    settle = Settle(held.settle)
    events: AsyncGenerator[Event, None] = system_events(
        starter(b, body.system.value, body.question, filters, body.top_k, body.use_rerank, body.stream_tokens),
        body.system.value, body.question, rid, s.chat_timeout_s, settle)
    return sse_response(events, s.sse_ping_s, [settle])
