"""Xoay khoá Gemini (`GEMINI_API_KEYS`): hết hạn mức ngày ⇒ khoá kế; 429 theo phút ⇒ giữ khoá, raise cho backoff.
Mọi client là giả: không gọi mạng; khoá không được xuất hiện trong log."""

import logging

import httpx
import openai
import pytest

from vre.config import load_settings
from vre.llm import client

KEYS = ["k-alpha", "k-beta", "k-gamma"]
DAILY = "RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
MINUTE = "RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 30s"


def _rate_limit(text: str) -> openai.RateLimitError:
    request = httpx.Request("POST", "https://example.invalid/chat/completions")
    return openai.RateLimitError(text, response=httpx.Response(429, request=request), body={"message": text})


class FakeClient:
    def __init__(self, key: str, failures: dict[str, list[str]], calls: list[str]):
        self.key, self.failures, self.calls = key, failures, calls
        outer = self

        class _Completions:
            @staticmethod
            def create(model, **_):
                outer.calls.append(outer.key)
                queue = outer.failures.get(outer.key, [])
                if queue:
                    raise _rate_limit(queue.pop(0))
                return f"ok:{outer.key}"

        self.chat = type("Chat", (), {"completions": _Completions})


@pytest.fixture
def gemini(monkeypatch):
    settings = load_settings(llm_provider="gemini", llm_model="gemini-3.8-flash", llm_fallback_models="",
                             gemini_api_keys=",".join(KEYS[:2]) + ", ," + KEYS[0], gemini_api_key=KEYS[2])
    monkeypatch.setattr(client, "get_settings", lambda: settings)
    monkeypatch.delenv(client.OFFLINE_ENV, raising=False)
    monkeypatch.setattr(client, "_gemini_clients", {})
    monkeypatch.setattr(client, "_gemini_key_index", 0)
    monkeypatch.setattr(client, "_gemini_rotations", 0)
    failures: dict[str, list[str]] = {}
    calls: list[str] = []
    monkeypatch.setattr(client, "_new_client", lambda key=None: FakeClient(key, failures, calls))
    return failures, calls


def test_key_list_order_and_dedup(gemini):
    assert client.get_settings().gemini_key_list == KEYS


def test_daily_quota_rotates_to_next_key(gemini, caplog):
    failures, calls = gemini
    failures["k-alpha"] = [DAILY]
    with caplog.at_level(logging.WARNING):
        assert client._create_with_fallback("gemini-3.8-flash", {}) == "ok:k-beta"
    assert calls == ["k-alpha", "k-beta"]
    assert client.gemini_key_rotations() == 1
    assert client.client().key == "k-beta"  # lần sau dùng luôn khoá mới
    assert not any(key in caplog.text for key in KEYS)  # chỉ ghi chỉ số khoá


def test_per_minute_429_keeps_key_and_raises(gemini):
    failures, calls = gemini
    failures["k-alpha"] = [MINUTE]
    with pytest.raises(openai.RateLimitError):
        client._create_with_fallback("gemini-3.8-flash", {})
    assert calls == ["k-alpha"] and client.gemini_key_rotations() == 0
    assert client._create_with_fallback("gemini-3.8-flash", {}) == "ok:k-alpha"


def test_all_keys_exhausted_raises(gemini):
    failures, calls = gemini
    for key in KEYS:
        failures[key] = [DAILY]
    with pytest.raises(openai.RateLimitError):
        client._create_with_fallback("gemini-3.8-flash", {})
    assert calls == KEYS and client.gemini_key_rotations() == 2


def test_daily_classifier():
    assert client.is_daily_quota(_rate_limit(DAILY))
    assert not client.is_daily_quota(_rate_limit(MINUTE))
    assert client.is_daily_quota(_rate_limit("You exceeded your current quota"))
    assert not client.is_daily_quota(_rate_limit("Rate limit reached, slow down"))


def test_missing_keys_errors_without_network(monkeypatch):
    settings = load_settings(llm_provider="gemini", gemini_api_keys="", gemini_api_key="")
    monkeypatch.setattr(client, "get_settings", lambda: settings)
    monkeypatch.delenv(client.OFFLINE_ENV, raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEYS"):
        client.client()
