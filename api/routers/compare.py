"""`POST /api/compare` (SSE): các hệ thống chạy song song, sự kiện xen kẽ, cuối cùng `compare_done`."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Iterator

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response

from api.deps import ApiSettings
from api.errors import request_id_of
from api.routers.chat import SSE_DOC, lease, unavailable, unavailable_error
from api.routers.common import backend, errors, settings
from api.schemas import CompareRequest, EventCompareDone
from api.services.backend import Backend
from api.sse import Event, Lease, Settle, Unavailable, sse_response, starter, system_events

log = logging.getLogger("vre.api")
router = APIRouter(prefix="/api", tags=["chat"])
_FINISHED = object()


async def merged(b: Backend, body: CompareRequest, rid: str, timeout_s: float,
                 settles: dict[str, Settle], missing: dict[str, str]) -> AsyncGenerator[Event, None]:
    queue: asyncio.Queue[object] = asyncio.Queue()
    filters = body.filters.model_dump(mode="json") if body.filters else None
    status: dict[str, str] = {}

    def refuse(reason: str) -> Iterator[Event]:
        raise Unavailable(f"Hệ thống chưa sẵn sàng: {reason}")

    async def pump(system: str) -> None:
        start = (lambda: refuse(missing[system])) if system in missing else starter(
            b, system, body.question, filters, body.top_k, False, True)
        gen = system_events(start,
                            system, body.question, rid, timeout_s, settles[system])
        try:
            async for name, payload in gen:
                if name == "done":
                    status[system] = payload["status"]
                await queue.put((name, payload))
        finally:
            await gen.aclose()
            settles[system].abandon()
            await queue.put(_FINISHED)

    tasks = [asyncio.create_task(pump(s.value)) for s in body.systems]
    try:
        remaining = len(tasks)
        while remaining:
            item = await queue.get()
            if item is _FINISHED:
                remaining -= 1
                continue
            yield item  # type: ignore[misc]
        done = EventCompareDone(request_id=rid, systems_ok=[s for s in body.systems if status.get(s.value) == "ok"],
                                systems_failed=[s for s in body.systems if status.get(s.value) != "ok"])
        yield "compare_done", done.model_dump(mode="json")
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@router.post("/compare", responses={**SSE_DOC, **errors(400, 401, 413, 415, 429, 503)})
async def compare(body: CompareRequest, request: Request, b: Backend = Depends(backend),
                  s: ApiSettings = Depends(settings)) -> Response:
    rid = request_id_of(request)
    missing = await unavailable(b, body.systems)
    if len(missing) == len(body.systems):  # không hệ thống nào chạy được ⇒ lỗi trước khi mở stream
        raise unavailable_error(missing)
    held: Lease = await lease(request, len(body.systems), len(body.systems))
    settles = {x.value: Settle(held.settle) for x in body.systems}
    log.info("request=%s compare systems=%s chars=%d", rid, ",".join(x.value for x in body.systems), len(body.question))
    return sse_response(merged(b, body, rid, s.chat_timeout_s, settles, missing), s.sse_ping_s, list(settles.values()))
