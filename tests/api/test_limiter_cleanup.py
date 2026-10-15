"""RateLimiter: cửa sổ trượt 60s, Retry-After, dọn khoá rảnh để dict không phình (bug B8)."""

from __future__ import annotations

from api.deps import RateLimiter


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_sliding_window_and_retry_after() -> None:
    clock = Clock()
    lim = RateLimiter(2, clock=clock)
    assert lim.check("a") is None and lim.check("a") is None
    clock.t += 10
    assert lim.check("a") == 50
    clock.t += 51
    assert lim.check("a") is None


def test_idle_keys_are_swept() -> None:
    clock = Clock()
    lim = RateLimiter(5, clock=clock, sweep_every=60)
    for i in range(1000):
        lim.check(f"10.0.{i // 256}.{i % 256}")
    assert len(lim.hits) == 1000
    clock.t += 61
    lim.check("fresh")
    assert set(lim.hits) == {"fresh"}


def test_active_keys_survive_sweep() -> None:
    clock = Clock()
    lim = RateLimiter(5, clock=clock)
    lim.check("old")
    clock.t += 30
    lim.check("busy")
    clock.t += 35  # old rảnh 65s, busy mới 35s
    lim.check("x")
    assert set(lim.hits) == {"busy", "x"}


def test_unlimited() -> None:
    lim = RateLimiter(0)
    assert all(lim.check("a") is None for _ in range(100)) and not lim.hits
