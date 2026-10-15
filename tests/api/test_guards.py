"""Bug B8 + api-contract §0/§5: thứ tự middleware, mã lỗi, envelope, Retry-After, slot bận, loopback-only."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterator
from typing import Any

from api_helpers import ScriptedBackend, make_app, request, requests_parallel

CHAT = {"question": "căn hộ 2pn q7 dưới 4 tỷ", "system": "basic_rag"}
TOKEN = "s3cret-token"


def envelope(resp: Any, status: int, code: str) -> dict[str, Any]:
    assert resp.status_code == status, resp.text
    body = resp.json()["error"]
    assert body["code"] == code
    assert body["request_id"] == resp.headers["x-request-id"]
    assert set(body) == {"code", "message", "request_id", "details"}
    return dict(body)


def test_request_id_and_security_headers_on_every_response(app: Any) -> None:
    for resp in (request(app, "GET", "/api/info"), request(app, "GET", "/api/listings/1"), request(app, "GET", "/api/nope")):
        assert re.fullmatch(r"[0-9a-f]{8}", resp.headers["x-request-id"])
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert "default-src 'none'" in resp.headers["content-security-policy"]


def test_415_when_not_json(app: Any) -> None:
    resp = request(app, "POST", "/api/chat", content=b"question=x", headers={"Content-Type": "application/x-www-form-urlencoded"})
    envelope(resp, 415, "unsupported_media_type")


def test_413_declared_and_streamed_body(app: Any) -> None:
    big = b'{"question":"' + b"a" * 9000 + b'"}'
    envelope(request(app, "POST", "/api/chat", content=big, headers={"Content-Type": "application/json"}), 413, "payload_too_large")

    async def chunks() -> Any:  # không có Content-Length: đếm khi đọc
        for _ in range(10):
            yield b"a" * 1000

    envelope(request(app, "POST", "/api/chat", content=chunks(), headers={"Content-Type": "application/json"}), 413, "payload_too_large")


def test_400_validation_maps_422(app: Any) -> None:
    for body in ({"question": "   "}, {"question": "x" * 501}, {"question": "ok", "system": "nope"}, {"question": "ok", "top_k": 0},
                 {"question": "ok", "extra": 1}):
        err = envelope(request(app, "POST", "/api/chat", json=body), 400, "invalid_request")
        assert err["details"]["fields"] and all(isinstance(f["loc"], list) and f["msg"] for f in err["details"]["fields"])
    envelope(request(app, "POST", "/api/chat", content=b"{bad", headers={"Content-Type": "application/json"}), 400, "invalid_request")
    envelope(request(app, "GET", "/api/listings/abc"), 400, "invalid_request")


def test_401_with_token_and_header_only() -> None:
    app = make_app(api_token=TOKEN)
    envelope(request(app, "GET", "/api/info"), 401, "unauthorized")
    envelope(request(app, "GET", f"/api/info?api_key={TOKEN}"), 401, "unauthorized")  # không nhận token qua query
    assert request(app, "GET", "/api/info", headers={"X-API-Key": TOKEN}).status_code == 200
    assert request(app, "GET", "/api/health").status_code == 200  # health không cần auth


def test_wrong_key_is_still_rate_limited() -> None:
    """Bug cũ: limiter đứng sau auth nên spam sai key không bao giờ bị chặn."""
    app = make_app(api_token=TOKEN, chat_rate_per_min=2)
    bad = {"X-API-Key": "wrong"}
    for _ in range(2):
        envelope(request(app, "POST", "/api/chat", json=CHAT, headers=bad), 401, "unauthorized")
    resp = request(app, "POST", "/api/chat", json=CHAT, headers=bad)
    envelope(resp, 429, "rate_limited")
    assert 1 <= int(resp.headers["retry-after"]) <= 60
    # đúng key cũng bị chặn: lượt đã tính theo IP
    envelope(request(app, "POST", "/api/chat", json=CHAT, headers={"X-API-Key": TOKEN}), 429, "rate_limited")
    # IP khác không bị ảnh hưởng
    other = request(app, "POST", "/api/chat", json=CHAT, headers={"X-API-Key": TOKEN}, client=("127.0.0.2", 1))
    assert other.status_code == 200


def test_middleware_order_body_size_before_rate_limit_before_auth() -> None:
    app = make_app(api_token=TOKEN, chat_rate_per_min=1)
    big = b"x" * 9000
    envelope(request(app, "POST", "/api/chat", content=big, headers={"Content-Type": "application/json"}), 413, "payload_too_large")
    envelope(request(app, "POST", "/api/chat", json=CHAT), 401, "unauthorized")  # lượt 1 (sai key) được tính
    envelope(request(app, "POST", "/api/chat", json=CHAT), 429, "rate_limited")
    # auth trước media type: sai key + sai content-type ⇒ 401 (IP khác để không dính limit)
    envelope(request(app, "POST", "/api/chat", content=b"x", headers={"Content-Type": "text/plain"}, client=("10.0.0.9", 1)),
             401, "unauthorized")


def test_get_bucket_separate_from_chat() -> None:
    app = make_app(chat_rate_per_min=1, api_rate_per_min=3)
    assert request(app, "POST", "/api/chat", json=CHAT).status_code == 200
    for _ in range(3):
        assert request(app, "GET", "/api/info").status_code == 200
    envelope(request(app, "GET", "/api/info"), 429, "rate_limited")
    assert request(app, "GET", "/api/health").status_code == 200  # health không bị limit


def test_loopback_only_without_token() -> None:
    app = make_app()
    envelope(request(app, "GET", "/api/info", client=("192.168.1.20", 4000)), 401, "unauthorized")
    assert request(app, "GET", "/api/info", client=("::1", 4000)).status_code == 200
    remote_ok = make_app(api_allow_remote_without_token=True)
    assert request(remote_ok, "GET", "/api/info", client=("172.18.0.3", 4000)).status_code == 200


def test_cors_preflight_and_origin() -> None:
    app = make_app(api_token=TOKEN)
    pre = request(app, "OPTIONS", "/api/chat", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
                                                         "Access-Control-Request-Headers": "content-type,x-api-key"})
    assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == "http://localhost:5173"
    evil = request(app, "GET", "/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in evil.headers


def test_busy_after_queue_timeout() -> None:
    release = threading.Event()

    def slow(*_: Any, **__: Any) -> Iterator[tuple[str, dict[str, Any]]]:
        yield "meta", {"question": "q", "started_at": "2026-10-15T00:00:00Z"}
        release.wait(5)
        yield "done", {"status": "ok"}

    app = make_app(ScriptedBackend(stream=slow), chat_max_concurrent=1, chat_queue_timeout_s=0.3)

    def unblock() -> None:
        time.sleep(1.0)
        release.set()

    threading.Thread(target=unblock, daemon=True).start()
    first, second = requests_parallel(app, [("POST", "/api/chat", {"json": CHAT}), ("POST", "/api/chat", {"json": CHAT})])
    codes = sorted([first.status_code, second.status_code])
    assert codes == [200, 503], codes
    busy = first if first.status_code == 503 else second
    envelope(busy, 503, "busy")


def test_system_unavailable_before_stream() -> None:
    app = make_app(ScriptedBackend(status={"llm_only": (True, None), "basic_rag": (True, None), "graph": (False, "Neo4j chưa chạy"),
                                           "hybrid": (False, "Neo4j chưa chạy")}))
    err = envelope(request(app, "POST", "/api/chat", json={"question": "q", "system": "graph"}), 503, "system_unavailable")
    assert "Neo4j" in err["message"]


def test_500_envelope_hides_details() -> None:
    class Broken(ScriptedBackend):
        def info(self) -> dict[str, Any]:
            raise RuntimeError("secret stack detail")

    resp = request(make_app(Broken()), "GET", "/api/info")
    err = envelope(resp, 500, "internal_error")
    assert "secret" not in resp.text and err["request_id"] in err["message"]
