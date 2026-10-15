"""Tiện ích test API: app ASGI qua httpx `ASGITransport` với `VRE_FAKE_BACKENDS=1` (không cần dịch vụ)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx

os.environ["VRE_FAKE_BACKENDS"] = "1"
os.environ.setdefault("VRE_API_TOKEN", "")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.deps import ApiSettings  # noqa: E402
from api.main import create_app  # noqa: E402
from api.services.fake import FakeBackend  # noqa: E402

CONTRACT = ROOT / "plans" / "261015-vre-full-retrain" / "api-contract.md"
EVENT_RANK = {"meta": 0, "route": 1, "filters": 2, "cypher": 3, "valuation": 4, "sources": 5, "token": 6, "answer": 7, "usage": 8,
              "done": 9}


def settings(**overrides: Any) -> ApiSettings:
    base: dict[str, Any] = {"fake_backends": True, "api_token": "", "chat_queue_timeout_s": 2.0, "chat_timeout_s": 10.0}
    return ApiSettings(**{**base, **overrides})


def make_app(backend: Any = None, **overrides: Any) -> Any:
    return create_app(settings(**overrides), backend or FakeBackend())


def request(app: Any, method: str, url: str, client: tuple[str, int] = ("127.0.0.1", 5000), **kwargs: Any) -> httpx.Response:
    async def go() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, client=client)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            return await c.request(method, url, **kwargs)

    return asyncio.run(go())


def requests_parallel(app: Any, calls: list[tuple[str, str, dict[str, Any]]]) -> list[httpx.Response]:
    async def go() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 5000))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            return list(await asyncio.gather(*(c.request(m, u, **kw) for m, u, kw in calls)))

    return asyncio.run(go())


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Tách stream thành (event, payload); bỏ comment `: ping`. Kiểm `data` luôn một dòng."""
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        lines = [ln for ln in block.split("\n") if ln and not ln.startswith(":")]
        if not lines:
            continue
        name = next(ln[7:] for ln in lines if ln.startswith("event: "))
        data = [ln[6:] for ln in lines if ln.startswith("data: ")]
        assert len(data) == 1, f"data phải đúng một dòng: {block!r}"
        events.append((name, json.loads(data[0])))
    return events


def assert_order(names: list[str]) -> None:
    """Bất biến §4 cho MỘT hệ thống: meta → [route] → [filters] → [cypher] → [valuation] → sources → token* → answer
    → usage → done; hoặc bất kỳ lúc nào sau meta: error → done."""
    assert names and names[0] == "meta", names
    assert names[-1] == "done" and names.count("done") == 1, names
    body = names[:-1]
    if "error" in body:
        assert body.count("error") == 1 and body[-1] == "error", names
        body = body[:-1]
    else:
        for required in ("sources", "answer", "usage"):
            assert body.count(required) == 1, names
    ranks = [EVENT_RANK[n] for n in body]
    assert ranks == sorted(ranks), names
    assert all(body.count(n) <= 1 for n in body if n != "token"), names


class ScriptedBackend(FakeBackend):
    """Fake có thể thay `stream_answer` / `system_status` cho từng test."""

    def __init__(self, stream: Callable[..., Iterator[tuple[str, dict[str, Any]]]] | None = None,
                 status: dict[str, tuple[bool, str | None]] | None = None):
        self._stream = stream
        self._status_override = status

    def system_status(self) -> dict[str, tuple[bool, str | None]]:
        return self._status_override or super().system_status()

    def stream_answer(self, *args: Any, **kwargs: Any) -> Iterator[tuple[str, dict[str, Any]]]:
        if self._stream is None:
            return super().stream_answer(*args, **kwargs)
        return self._stream(*args, **kwargs)
