"""Chế độ chỉ-cache của vre.llm.client: VRE_LLM_OFFLINE=1 / cache_only() trả từ cache đĩa, thiếu ⇒ LLMCacheMiss, không gọi mạng."""

import pytest

from vre.config import load_settings
from vre.llm import cache, client


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    settings = load_settings(data_root=tmp_path, llm_provider="groq", groq_api_key="x")
    monkeypatch.setattr(client, "get_settings", lambda: settings)
    monkeypatch.setattr(cache, "get_settings", lambda: settings)
    monkeypatch.delenv(client.OFFLINE_ENV, raising=False)

    def network(*_a, **_k):
        pytest.fail("không được gọi mạng ở chế độ chỉ-cache")

    monkeypatch.setattr(client, "_ask_openai", network)
    monkeypatch.setattr(client, "_new_client", network)
    return settings


def _seed(settings, prompt: str, text: str) -> None:
    key = cache.cache_key(settings.llm_model, settings.llm_effort, "", prompt, None)
    cache.write(key, {"text": text, "data": None, "input_tokens": 3, "output_tokens": 2})


def test_env_offline_serves_cache_and_raises_on_miss(isolated, monkeypatch):
    monkeypatch.setenv(client.OFFLINE_ENV, "1")
    _seed(isolated, "có", "trả lời")
    hit = client.ask_llm("có")
    assert hit.cached and hit.text == "trả lời"
    with pytest.raises(client.LLMCacheMiss):
        client.ask_llm("chưa có")
    with pytest.raises(client.LLMCacheMiss):
        list(client.stream_llm("chưa có"))
    assert list(client.stream_llm("có")) == ["trả lời"]
    with pytest.raises(client.LLMCacheMiss):
        client.client()  # chốt cuối: không tạo được client


def test_cache_only_context_is_scoped(isolated):
    assert not client.offline()
    with client.cache_only():
        assert client.offline()
        with pytest.raises(client.LLMCacheMiss):
            client.ask_llm("chưa có")
    assert not client.offline()


def test_cache_miss_is_runtime_error():
    assert issubclass(client.LLMCacheMiss, RuntimeError)
