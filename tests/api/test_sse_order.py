"""`POST /api/chat`: thứ tự sự kiện §4, `request_id`/`system` trong mọi payload, lỗi trong stream ⇒ `error` → `done`."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from api_helpers import ScriptedBackend, assert_order, make_app, parse_sse, request

Q = "căn hộ 2pn q7 dưới 4 tỷ"
META = ("meta", {"question": Q, "started_at": "2026-10-15T00:00:00Z"})


def chat(app: Any, **body: Any) -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    resp = request(app, "POST", "/api/chat", json={"question": Q, **body})
    assert resp.status_code == 200, resp.text
    return resp, parse_sse(resp.text)


@pytest.mark.parametrize("system", ["llm_only", "basic_rag", "graph", "hybrid"])
def test_fake_systems_stream_in_contract_order(app: Any, system: str) -> None:
    resp, events = chat(app, system=system)
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-store" and resp.headers["x-accel-buffering"] == "no"
    names = [n for n, _ in events]
    assert_order(names)
    rid = resp.headers["x-request-id"]
    assert all(p["request_id"] == rid and p["system"] == system for _, p in events)
    assert events[0][1]["question"] == Q
    assert ("route" in names) == (system == "hybrid")
    assert ("cypher" in names) == (system in ("graph", "hybrid"))
    assert "token" in names and events[-1][1]["status"] == "ok"


def test_default_system_is_hybrid(app: Any) -> None:
    _, events = chat(app)
    assert {p["system"] for _, p in events} == {"hybrid"}


def test_stream_tokens_false_has_no_token(app: Any) -> None:
    _, events = chat(app, system="basic_rag", stream_tokens=False)
    assert "token" not in [n for n, _ in events]
    assert_order([n for n, _ in events])


def test_valuation_route(app: Any) -> None:
    resp = request(app, "POST", "/api/chat", json={"question": "định giá căn hộ 70m2 Phú Mỹ quận 7", "system": "hybrid"})
    events = parse_sse(resp.text)
    names = [n for n, _ in events]
    assert_order(names)
    assert "valuation" in names and dict(events)["route"]["route"] == "valuation"


def test_fake_error_sequence(app: Any) -> None:
    resp = request(app, "POST", "/api/chat", json={"question": "thử lỗi [fake:error]", "system": "graph"})
    events = parse_sse(resp.text)
    assert [n for n, _ in events] == ["meta", "error", "done"]
    assert events[1][1]["code"] == "upstream_llm_error" and events[2][1]["status"] == "error"


def scripted(*items: tuple[str, dict[str, Any]], raise_after: bool = False, delay: float = 0.0) -> ScriptedBackend:
    def stream(*_: Any, **__: Any) -> Iterator[tuple[str, dict[str, Any]]]:
        for item in items:
            if delay:
                time.sleep(delay)
            yield item
        if raise_after:
            raise RuntimeError("boom with internals")

    return ScriptedBackend(stream=stream)


def test_exception_mid_stream_becomes_error_done() -> None:
    resp = request(make_app(scripted(META, ("sources", {"sources": []}), raise_after=True)), "POST", "/api/chat", json={"question": Q})
    events = parse_sse(resp.text)
    assert [n for n, _ in events] == ["meta", "sources", "error", "done"]
    assert events[2][1]["code"] == "internal_error" and "boom" not in resp.text


def test_exception_before_first_event_synthesizes_meta() -> None:
    def explode(*_: Any, **__: Any) -> Iterator[tuple[str, dict[str, Any]]]:
        raise RuntimeError("boom")

    events = parse_sse(request(make_app(ScriptedBackend(stream=explode)), "POST", "/api/chat", json={"question": Q}).text)
    assert [n for n, _ in events] == ["meta", "error", "done"] and events[0][1]["question"] == Q


def test_facade_error_event_passthrough_and_done_appended() -> None:
    backend = scripted(META, ("error", {"code": "upstream_llm_error", "message": "Groq 503"}))
    events = parse_sse(request(make_app(backend), "POST", "/api/chat", json={"question": Q}).text)
    assert [n for n, _ in events] == ["meta", "error", "done"] and events[-1][1]["status"] == "error"


def test_missing_done_is_reported() -> None:
    events = parse_sse(request(make_app(scripted(META, ("sources", {"sources": []}))), "POST", "/api/chat",
                               json={"question": Q}).text)
    assert [n for n, _ in events][-2:] == ["error", "done"]


def test_payload_violating_contract_is_internal_error() -> None:
    backend = scripted(META, ("answer", {"answer": "x"}))  # thiếu khoá bắt buộc
    events = parse_sse(request(make_app(backend), "POST", "/api/chat", json={"question": Q}).text)
    assert [n for n, _ in events] == ["meta", "error", "done"] and events[1][1]["code"] == "internal_error"


def test_timeout_emits_timeout_error_and_frees_slot() -> None:
    release = threading.Event()

    def stuck(*_: Any, **__: Any) -> Iterator[tuple[str, dict[str, Any]]]:
        yield META
        release.wait(5)
        yield "done", {"status": "ok"}

    app = make_app(ScriptedBackend(stream=stuck), chat_timeout_s=0.3, chat_max_concurrent=1)
    slots = app.state.slots

    async def go() -> tuple[list[str], str, bool, bool]:
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 1))
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            events = parse_sse((await c.post("/api/chat", json={"question": Q})).text)
        held = slots._sem.locked()  # thread facade còn chạy ⇒ slot chưa trả
        release.set()
        for _ in range(60):
            if not slots._sem.locked():
                break
            await asyncio.sleep(0.05)
        return [n for n, _ in events], events[1][1]["code"], held, slots._sem.locked()

    names, code, held, still_held = asyncio.run(go())
    assert names == ["meta", "error", "done"] and code == "timeout"
    assert held and not still_held


def test_slot_released_after_normal_stream(app: Any) -> None:
    for _ in range(6):  # > chat_max_concurrent: lần nào cũng phải lấy được slot
        chat(app, system="basic_rag")
    assert app.state.slots._sem._value == app.state.slots.capacity


def test_heartbeat_ping_comment() -> None:
    backend = scripted(META, ("sources", {"sources": []}), delay=0.4)
    resp = request(make_app(backend, sse_ping_s=0.1), "POST", "/api/chat", json={"question": Q})
    assert ": ping\n\n" in resp.text
