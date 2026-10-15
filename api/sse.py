"""Bơm sự kiện SSE từ facade đồng bộ `stream_answer` (chạy trong thread) sang luồng async (api-contract §4).

Bảo đảm: thêm `request_id`/`system` vào mọi payload và validate theo model Pydantic; mỗi hệ thống bắt đầu bằng
`meta` và kết thúc bằng đúng một `done`; exception hoặc quá `VRE_CHAT_TIMEOUT_S` ⇒ `error` → `done{status:error}`.
Slot câu hỏi chỉ được trả khi thread của facade đã thật sự dừng (kể cả khi client ngắt hoặc quá giờ).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from api.deps import ChatSlots
from api.errors import internal_message
from api.schemas import EVENT_MODELS, ErrorCode
from api.services.backend import Backend

log = logging.getLogger("vre.api")
SSE_HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
Event = tuple[str, dict[str, Any]]
_END = object()


class Lease:
    """n slot dùng chung cho k hệ thống; trả slot khi hệ thống cuối cùng `settle()`."""

    def __init__(self, slots: ChatSlots, n: int, holders: int):
        self.slots, self.n, self.pending = slots, n, holders

    def settle(self) -> None:
        self.pending -= 1
        if self.pending == 0:
            self.slots.release(self.n)


class Unavailable(Exception):
    """Hệ thống không sẵn sàng (vd thiếu Neo4j) — trong stream thành sự kiện `error{system_unavailable}`."""


class Settle:
    """Gọi `fn` đúng một lần. `entered` = generator sự kiện đã chạy (khi đó chính nó chịu trách nhiệm gọi)."""

    def __init__(self, fn: Callable[[], None]):
        self.fn, self.entered, self.done = fn, False, False

    def __call__(self) -> None:
        if not self.done:
            self.done = True
            self.fn()

    def abandon(self) -> None:
        """Generator chưa từng chạy (bị huỷ trước sự kiện đầu) ⇒ tự trả slot."""
        if not self.entered:
            self()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _next(it: Iterator[Event]) -> object:
    return next(it, _END)


def _close(it: Iterator[Event] | None) -> None:
    close = getattr(it, "close", None)
    if close is not None:
        try:
            close()
        except Exception:  # generator đã hỏng: chỉ ghi log, slot vẫn được trả
            log.exception("Đóng iterator facade lỗi")


async def system_events(start: Callable[[], Iterator[Event]], system: str, question: str, request_id: str,
                        timeout_s: float, on_settled: Settle) -> AsyncGenerator[Event, None]:
    """Sinh (event, payload đã validate) cho MỘT hệ thống."""
    on_settled.entered = True
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    started = False
    it: Iterator[Event] | None = None
    pending: asyncio.Future[object] | None = None

    def stamp(name: str, payload: dict[str, Any]) -> Event:
        model = EVENT_MODELS[name]
        return name, model.model_validate({**payload, "request_id": request_id, "system": system}).model_dump(mode="json")

    def fail(code: ErrorCode, message: str) -> list[Event]:
        out = [] if started else [stamp("meta", {"question": question, "started_at": _now()})]
        return [*out, stamp("error", {"code": code.value, "message": message}), stamp("done", {"status": "error"})]

    try:
        try:
            it = await loop.run_in_executor(None, start)
        except Unavailable as error:
            for ev in fail(ErrorCode.SYSTEM_UNAVAILABLE, str(error)):
                yield ev
            return
        except Exception:
            log.exception("request=%s system=%s facade lỗi khi khởi động", request_id, system)
            for ev in fail(ErrorCode.INTERNAL_ERROR, internal_message(request_id)):
                yield ev
            return
        last = ""
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            pending = loop.run_in_executor(None, _next, it)
            item = await asyncio.wait_for(asyncio.shield(pending), remaining)
            pending = None
            if item is _END:
                break
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(f"facade phải sinh (event, payload), nhận {type(item).__name__}")
            name: str = item[0]
            payload: dict[str, Any] = item[1]
            if name not in EVENT_MODELS:
                raise ValueError(f"sự kiện lạ {name!r}")
            if not started and name != "meta":
                yield stamp("meta", {"question": question, "started_at": _now()})
            started = True
            yield stamp(name, payload)
            last = name
            if name == "done":
                return
        if last == "error":
            yield stamp("done", {"status": "error"})
        else:
            log.error("request=%s system=%s facade kết thúc thiếu `done`", request_id, system)
            for ev in fail(ErrorCode.INTERNAL_ERROR, internal_message(request_id)):
                yield ev
    except TimeoutError:
        log.warning("request=%s system=%s quá %ss", request_id, system, timeout_s)
        for ev in fail(ErrorCode.TIMEOUT, f"Quá thời gian trả lời ({timeout_s:g}s)"):
            yield ev
    except (ValidationError, ValueError, TypeError) as error:
        log.error("request=%s system=%s payload sai hợp đồng: %s", request_id, system, error)
        for ev in fail(ErrorCode.INTERNAL_ERROR, internal_message(request_id)):
            yield ev
    except Exception:
        log.exception("request=%s system=%s facade lỗi", request_id, system)
        for ev in fail(ErrorCode.INTERNAL_ERROR, internal_message(request_id)):
            yield ev
    finally:
        if pending is not None and not pending.done():
            # thread vẫn đang chạy next(): đóng iterator và trả slot khi nó xong
            def _later(_: asyncio.Future[object], it_: Iterator[Event] | None = it) -> None:
                _close(it_)
                on_settled()

            pending.add_done_callback(_later)
        else:
            _close(it)
            on_settled()


def encode(name: str, payload: dict[str, Any]) -> ServerSentEvent:
    return ServerSentEvent(event=name, data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")), sep="\n")


def sse_response(events: AsyncGenerator[Event, None], ping_s: float, settles: list[Settle]) -> EventSourceResponse:
    async def body() -> AsyncIterator[ServerSentEvent]:
        try:
            async for name, payload in events:
                yield encode(name, payload)
        finally:
            await events.aclose()
            for settle in settles:
                settle.abandon()

    return EventSourceResponse(body(), headers=SSE_HEADERS, ping=ping_s, sep="\n",
                               ping_message_factory=lambda: ServerSentEvent(comment="ping", sep="\n"))


def starter(backend: Backend, system: str, question: str, filters: dict[str, Any] | None, top_k: int, use_rerank: bool,
            stream_tokens: bool) -> Callable[[], Iterator[Event]]:
    return lambda: backend.stream_answer(system, question, filters, top_k, use_rerank, stream_tokens)
