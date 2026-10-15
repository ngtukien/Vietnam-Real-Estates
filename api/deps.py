"""Cấu hình API (pydantic-settings riêng, prefix `VRE_`), rate limiter có dọn khoá rảnh, slot câu hỏi đồng thời.

Khoá cấu hình của API không nằm trong `vre.config` (phase-03-lanes §2): chỉ module này đọc chúng.
"""

from __future__ import annotations

import asyncio
import math
import threading
from collections import deque
from collections.abc import Callable
from functools import lru_cache
from time import monotonic

from pydantic_settings import BaseSettings, SettingsConfigDict

from vre.config import DEFAULT_DATA_ROOT, dotenv_enabled

DEFAULT_CORS = "http://127.0.0.1:5173,http://localhost:5173"
WINDOW_S = 60.0


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VRE_", extra="ignore", frozen=True)

    data_root: str = str(DEFAULT_DATA_ROOT)
    api_token: str = ""
    cors_origins: str = DEFAULT_CORS
    chat_max_concurrent: int = 4
    chat_rate_per_min: int = 20
    # Bucket riêng cho các GET (/api/info, /api/stats/...): web tải nhiều endpoint mỗi lần mở trang, dùng chung
    # 20/phút của chat sẽ tự chặn chính nó. Chat/compare vẫn dùng `chat_rate_per_min`.
    api_rate_per_min: int = 300
    chat_timeout_s: float = 120.0
    chat_queue_timeout_s: float = 60.0
    max_body_bytes: int = 8 * 1024
    sse_ping_s: float = 15.0
    fake_backends: bool = False
    # Mặc định chỉ client loopback được gọi khi chưa đặt VRE_API_TOKEN. Compose (port publish 127.0.0.1) đặt =1
    # vì client trong mạng Docker không bao giờ là loopback.
    api_allow_remote_without_token: bool = False
    duckdb_memory_gb: int = 3
    duckdb_threads: int = 4

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def load_api_settings() -> ApiSettings:
    import os
    from pathlib import Path

    root = Path(os.environ.get("VRE_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
    env_file = root / ".env" if dotenv_enabled() else None
    return ApiSettings(_env_file=env_file, _env_file_encoding="utf-8")  # type: ignore[call-arg]


@lru_cache(maxsize=1)
def get_api_settings() -> ApiSettings:
    return load_api_settings()


class RateLimiter:
    """Cửa sổ trượt 60 giây theo khoá (IP). `per_minute <= 0` = không giới hạn.
    Dọn khoá rảnh (không có lượt nào trong 60s) mỗi `sweep_every` giây để dict không phình (bug B8)."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = monotonic, sweep_every: float = WINDOW_S):
        self.per_minute = per_minute
        self.clock = clock
        self.sweep_every = sweep_every
        self.hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = clock()

    def _sweep(self, now: float) -> None:
        idle = [k for k, q in self.hits.items() if not q or now - q[-1] > WINDOW_S]
        for key in idle:
            del self.hits[key]
        self._last_sweep = now

    def check(self, key: str) -> int | None:
        """Ghi một lượt; trả None nếu được phép, ngược lại số giây phải chờ (Retry-After, ≥ 1)."""
        if self.per_minute <= 0:
            return None
        now = self.clock()
        with self._lock:
            if now - self._last_sweep >= self.sweep_every:
                self._sweep(now)
            hits = self.hits.setdefault(key, deque())
            while hits and now - hits[0] > WINDOW_S:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return max(1, math.ceil(WINDOW_S - (now - hits[0])))
            hits.append(now)
            return None


class Busy(Exception):
    """Hết slot câu hỏi đồng thời sau thời gian chờ."""


class ChatSlots:
    """Giới hạn số câu hỏi chạy đồng thời; `acquire(n)` chờ tối đa `timeout` cho đủ n slot (all-or-nothing)."""

    def __init__(self, capacity: int, timeout: float):
        self.capacity = max(1, capacity)
        self.timeout = timeout
        self._sem = asyncio.Semaphore(self.capacity)

    async def acquire(self, n: int = 1) -> int:
        n = max(1, min(n, self.capacity))
        got = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        try:
            while got < n:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise Busy
                await asyncio.wait_for(self._sem.acquire(), remaining)
                got += 1
        except (TimeoutError, Busy):
            self.release(got)
            raise Busy from None
        except BaseException:
            self.release(got)
            raise
        return n

    def release(self, n: int) -> None:
        for _ in range(n):
            self._sem.release()
