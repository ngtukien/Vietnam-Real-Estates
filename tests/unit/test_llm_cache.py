"""B1: `llm.cache.fresh()` chỉ tác dụng trong ngữ cảnh (luồng) gọi nó; stream dùng chung cache với ask_llm."""

import threading

from vre.llm import cache
from vre.llm import client as llm


def test_fresh_is_context_local():
    inside, entered, release = {}, threading.Event(), threading.Event()

    def measuring():
        with cache.fresh():
            inside["measuring"] = cache.reading()
            entered.set()
            release.wait(5)

    def chatbot():
        entered.wait(5)
        inside["chatbot"] = cache.reading()
        release.set()

    threads = [threading.Thread(target=measuring), threading.Thread(target=chatbot)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert inside == {"measuring": False, "chatbot": True}
    assert cache.reading() is True


def test_fresh_restores_after_exception():
    try:
        with cache.fresh():
            raise RuntimeError("x")
    except RuntimeError:
        pass
    assert cache.reading() is True


def test_cached_stream_yields_one_piece(monkeypatch):
    saved = {"text": "Giá 5 tỷ [Tin#12]", "data": None, "input_tokens": 7, "output_tokens": 3}
    monkeypatch.setattr(cache, "read", lambda key: saved)
    stream = llm.stream_llm("câu hỏi", system="sys")
    assert list(stream) == ["Giá 5 tỷ [Tin#12]"]
    assert stream.result is not None and stream.result.cached and stream.result.input_tokens == 7


def test_stream_accumulates_chunks_and_usage(monkeypatch):
    from types import SimpleNamespace as NS

    def chunk(text, finish=None, usage=None):
        return NS(choices=[NS(delta=NS(content=text), finish_reason=finish)] if text is not None else [], usage=usage)

    chunks = [chunk("Xin "), chunk("chào", "stop"), chunk(None, usage=NS(prompt_tokens=11, completion_tokens=2))]
    written = {}
    monkeypatch.setattr(cache, "read", lambda key: None)
    monkeypatch.setattr(cache, "write", lambda key, payload: written.update(payload))
    monkeypatch.setattr(llm, "_create_with_fallback", lambda model, request: iter(chunks))
    monkeypatch.setattr(llm.get_settings(), "llm_provider", "groq")
    stream = llm.stream_llm("q")
    assert "".join(stream) == "Xin chào"
    assert (stream.result.text, stream.result.input_tokens, stream.result.output_tokens) == ("Xin chào", 11, 2)
    assert written["text"] == "Xin chào"
