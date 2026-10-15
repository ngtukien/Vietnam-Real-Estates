"""Middleware ASGI thuần, đúng thứ tự api-contract §0:
request-id → (CORS) → body-size → rate limit theo IP → auth → media type → handler.

Rate limit đứng TRƯỚC auth nên sai key vẫn bị tính lượt (bug B8). CORS nằm ngoài guard để preflight
OPTIONS (không mang X-API-Key) được trả lời mà không bị chặn.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import secrets
from time import perf_counter
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.deps import ApiSettings, RateLimiter
from api.errors import envelope_bytes, internal_message
from api.schemas import ErrorCode

log = logging.getLogger("vre.api")

SECURITY_HEADERS: dict[str, str] = {
    # API chỉ trả JSON/SSE, không phục vụ HTML ⇒ CSP chặn hết.
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
CHAT_PATHS = frozenset({"/api/chat", "/api/compare"})
PUBLIC_PATHS = frozenset({"/api/health"})
BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return bytes(value).decode("latin-1")
    return None


def _client_host(scope: Scope) -> str:
    client = scope.get("client")
    return str(client[0]) if client else "unknown"


async def send_error(send: Send, scope: Scope, code: ErrorCode, status: int, message: str,
                     headers: dict[str, str] | None = None, details: dict[str, Any] | None = None) -> None:
    rid = str(scope.get("state", {}).get("request_id", "00000000"))
    body = envelope_bytes(code, message, rid, details)
    raw = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
           (b"cache-control", b"no-store")]
    raw += [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    await send({"type": "http.response.start", "status": status, "headers": raw})
    await send({"type": "http.response.body", "body": body})


class RequestIdMiddleware:
    """Gắn `X-Request-Id` (8 hex) + security headers cho MỌI response; bắt exception chưa xử lý → 500 envelope.
    Log truy cập không có nội dung câu hỏi (chỉ method, path, status, thời gian)."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = secrets.token_hex(4)
        scope.setdefault("state", {})["request_id"] = rid
        started = False
        status = 500
        t0 = perf_counter()
        extra = [(b"x-request-id", rid.encode())] + [(k.lower().encode(), v.encode()) for k, v in SECURITY_HEADERS.items()]

        async def wrapped(message: Message) -> None:
            nonlocal started, status
            if message["type"] == "http.response.start":
                started = True
                status = int(message["status"])
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        try:
            await self.app(scope, receive, wrapped)
        except Exception:
            log.exception("request=%s %s %s lỗi không xử lý", rid, scope.get("method"), scope.get("path"))
            if started:
                raise
            await send_error(wrapped, scope, ErrorCode.INTERNAL_ERROR, 500, internal_message(rid))
        finally:
            log.info("request=%s %s %s status=%s ms=%.0f", rid, scope.get("method"), scope.get("path"), status,
                     (perf_counter() - t0) * 1000)


class GuardMiddleware:
    """body-size → rate limit → auth → media type, chỉ cho đường dẫn `/api/*`."""

    def __init__(self, app: ASGIApp, settings: ApiSettings, chat_limiter: RateLimiter, api_limiter: RateLimiter):
        self.app = app
        self.settings = settings
        self.chat_limiter = chat_limiter
        self.api_limiter = api_limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = str(scope.get("path", ""))
        method = str(scope.get("method", "GET")).upper()
        if scope["type"] != "http" or not path.startswith("/api/") or method == "OPTIONS":
            await self.app(scope, receive, send)
            return

        # 1. body-size: đọc hết body (≤ giới hạn) rồi phát lại cho handler
        if method in BODY_METHODS:
            body = await self._read_body(scope, receive, send)
            if body is None:
                return
            receive = _replay(body, receive)

        public = path in PUBLIC_PATHS
        # 2. rate limit theo IP (trước auth: sai key vẫn bị tính lượt)
        if not public:
            limiter = self.chat_limiter if path in CHAT_PATHS else self.api_limiter
            wait = limiter.check(_client_host(scope))
            if wait is not None:
                await send_error(send, scope, ErrorCode.RATE_LIMITED, 429, "Quá nhiều yêu cầu, thử lại sau",
                                 headers={"Retry-After": str(wait)})
                return
            # 3. auth
            denied = self._auth_error(scope)
            if denied:
                await send_error(send, scope, ErrorCode.UNAUTHORIZED, 401, denied)
                return
        # 4. media type
        if method in BODY_METHODS:
            ctype = (_header(scope, b"content-type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                await send_error(send, scope, ErrorCode.UNSUPPORTED_MEDIA_TYPE, 415, "Content-Type phải là application/json")
                return
        await self.app(scope, receive, send)

    def _auth_error(self, scope: Scope) -> str | None:
        token = self.settings.api_token
        if token:
            given = _header(scope, b"x-api-key") or ""
            if hmac.compare_digest(given.encode(), token.encode()):
                return None
            return "Thiếu hoặc sai X-API-Key"
        if self.settings.api_allow_remote_without_token or is_loopback(_client_host(scope)):
            return None
        return "Server chưa đặt VRE_API_TOKEN nên chỉ nhận kết nối loopback"

    async def _read_body(self, scope: Scope, receive: Receive, send: Send) -> bytes | None:
        limit = self.settings.max_body_bytes
        too_large = f"Body tối đa {limit} byte"
        declared = _header(scope, b"content-length")
        if declared is not None:
            try:
                if int(declared) > limit:
                    await send_error(send, scope, ErrorCode.PAYLOAD_TOO_LARGE, 413, too_large)
                    return None
            except ValueError:
                await send_error(send, scope, ErrorCode.INVALID_REQUEST, 400, "Content-Length không hợp lệ")
                return None
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return None
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > limit:
                await send_error(send, scope, ErrorCode.PAYLOAD_TOO_LARGE, 413, too_large)
                return None
            chunks.append(chunk)
            if not message.get("more_body", False):
                return b"".join(chunks)


def _replay(body: bytes, original: Receive) -> Receive:
    """Trả body đã đọc một lần, sau đó uỷ cho `receive` gốc (để sse-starlette thấy `http.disconnect`)."""
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await original()

    return receive
