"""Gọi LLM (mặc định Groq qua OpenAI SDK; tuỳ chọn OpenAI hoặc Claude, chọn bằng `Settings.llm_provider`),
có cache đĩa (`vre.llm.cache`). Chuyển nguyên từ `src/llm.py`.

Khoá API đọc từ Settings (`GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`); Claude còn nhận
`ANTHROPIC_AUTH_TOKEN` (profile `ant auth login`).
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from vre.config import get_settings
from vre.llm import cache

_client = None
_client_lock = threading.Lock()  # chatbot gọi từ nhiều luồng: chỉ tạo một client
OFFLINE_ENV = "VRE_LLM_OFFLINE"
_cache_only: ContextVar[bool] = ContextVar("vre_llm_cache_only", default=False)


class LLMCacheMiss(RuntimeError):
    """Chế độ chỉ-cache (`VRE_LLM_OFFLINE=1` hoặc khối `cache_only()`): câu chưa có trong cache đĩa LLM — không gọi mạng."""


def offline() -> bool:
    """True khi mọi lời gọi LLM phải trả từ cache đĩa (biến môi trường `VRE_LLM_OFFLINE=1`, hoặc trong `cache_only()`)."""
    return os.getenv(OFFLINE_ENV, "") == "1" or _cache_only.get()


@contextmanager
def cache_only() -> Iterator[None]:
    """Trong khối này (theo ngữ cảnh luồng) LLM chỉ trả từ cache đĩa; thiếu ⇒ `LLMCacheMiss`. Dùng cho notebook/eval offline."""
    token = _cache_only.set(True)
    try:
        yield
    finally:
        _cache_only.reset(token)


def _refuse_network(what: str) -> None:
    if offline():
        raise LLMCacheMiss(f"{what}: chưa có trong cache LLM và đang ở chế độ chỉ-cache ({OFFLINE_ENV}=1)")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass
class LLMResult:
    text: str
    data: dict | None
    input_tokens: int
    output_tokens: int
    cached: bool


def client():
    global _client
    _refuse_network("tạo client LLM")  # chốt cuối: không đường nào tới mạng khi offline
    with _client_lock:
        if _client is None:
            _client = _new_client()
    return _client


def _new_client():
    s = get_settings()
    if s.llm_provider == "anthropic":
        if not (s.anthropic_api_key or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError("Thiếu xác thực Claude: đặt ANTHROPIC_API_KEY trong .env hoặc chạy `ant auth login`.")
        import anthropic
        return anthropic.Anthropic(api_key=s.anthropic_api_key or None, timeout=s.llm_timeout, max_retries=2)
    import openai
    if s.llm_provider == "openai":
        if not s.openai_api_key:
            raise RuntimeError("Thiếu xác thực OpenAI: đặt OPENAI_API_KEY trong .env.")
        return openai.OpenAI(api_key=s.openai_api_key, timeout=s.llm_timeout, max_retries=2)
    if not s.groq_api_key:
        raise RuntimeError("Thiếu xác thực Groq: đặt GROQ_API_KEY trong .env.")
    # Groq giới hạn token/phút theo từng mô hình: SDK tự chờ và thử lại khi gặp 429.
    return openai.OpenAI(api_key=s.groq_api_key, base_url=GROQ_BASE_URL, timeout=s.llm_timeout, max_retries=4)


def _close_objects(node):
    """Structured output cần mọi object có additionalProperties=false và liệt kê đủ required;
    trường tuỳ chọn vẫn nhận null qua anyOf của pydantic."""
    if isinstance(node, dict):
        node.pop("default", None)  # giá trị mặc định do pydantic xử lý khi validate
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        for value in node.values():
            _close_objects(value)
    elif isinstance(node, list):
        for value in node:
            _close_objects(value)
    return node


def _schema(output: type[BaseModel] | dict | None) -> dict | None:
    if output is None or isinstance(output, dict):
        return output
    return _close_objects(output.model_json_schema())


def ask_llm(prompt: str, system: str = "", output: type[BaseModel] | dict | None = None,
            effort: str | None = None, max_tokens: int = 4000, use_cache: bool = True,
            model: str | None = None) -> LLMResult:
    """Một lượt hỏi LLM. `output` (pydantic model hoặc JSON schema) bật structured output;
    khi đó `data` là dict đã parse. Cùng (model, system, prompt, schema) trả lại kết quả đã lưu.
    `model` mặc định `Settings.llm_model` (eval dùng `Settings.judge_model`); `effort` mặc định `Settings.llm_effort`."""
    s = get_settings()
    model = model or s.llm_model
    effort = effort or s.llm_effort
    schema = _schema(output)
    key = cache.cache_key(model, effort, system, prompt, schema)
    if use_cache:
        saved = cache.read(key)
        if saved is not None:
            return LLMResult(**{**saved, "cached": True})
    _refuse_network(f"ask_llm(model={model})")

    ask = _ask_anthropic if s.llm_provider == "anthropic" else _ask_openai
    text, input_tokens, output_tokens, truncated = ask(model, prompt, system, schema, effort, max_tokens)
    if truncated and schema:
        raise RuntimeError(f"Đầu ra có cấu trúc bị cắt ở max_tokens={max_tokens}")
    data: dict | None = json.loads(text) if schema else None
    if isinstance(output, type) and issubclass(output, BaseModel):
        data = output.model_validate(data).model_dump()
    result = LLMResult(text, data, input_tokens, output_tokens, False)
    cache.write(key, {k: v for k, v in result.__dict__.items() if k != "cached"})
    return result


def _ask_anthropic(model: str, prompt: str, system: str, schema: dict | None, effort: str, max_tokens: int):
    request = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": effort, **({"format": {"type": "json_schema", "schema": schema}}
                                            if schema else {})},
        # Khi bộ lọc an toàn từ chối, API tự chạy lại trên mô hình dự phòng được khuyến nghị.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if system:
        request["system"] = system
    response = client().beta.messages.create(**request)
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude từ chối yêu cầu này")
    text = "".join(block.text for block in response.content if block.type == "text")
    return (text, response.usage.input_tokens, response.usage.output_tokens,
            response.stop_reason == "max_tokens")


def _openai_request(prompt: str, system: str, schema: dict | None, effort: str, max_tokens: int) -> dict[str, Any]:
    messages = ([{"role": "developer", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    request: dict[str, Any] = dict(messages=messages, max_completion_tokens=max_tokens, reasoning_effort=effort)
    if schema:
        request["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": "output", "schema": schema, "strict": True}}
    return request


def _create_with_fallback(model: str, request: dict[str, Any]) -> Any:
    """`chat.completions.create` thử lần lượt model chính rồi model dự phòng khi gặp 429/5xx (lỗi lúc tạo
    request; với stream, lỗi giữa chừng không được chuyển model vì đã phát token)."""
    import openai
    models = list(dict.fromkeys([model, *get_settings().fallbacks_for(model)]))
    for i, name in enumerate(models):
        for attempt in range(3):
            try:
                return client().chat.completions.create(model=name, **request)
            except openai.BadRequestError as error:
                # Groq đôi khi trả JSON rỗng cho structured output (json_validate_failed): thử lại cùng model.
                if "json_validate_failed" not in str(error) or (attempt == 2 and i == len(models) - 1):
                    raise
            except (openai.InternalServerError, openai.RateLimitError):
                if i == len(models) - 1:
                    raise
                break
    raise RuntimeError("LLM không trả lời sau khi thử mọi model")


def _ask_openai(model: str, prompt: str, system: str, schema: dict | None, effort: str, max_tokens: int):
    response = _create_with_fallback(model, _openai_request(prompt, system, schema, effort, max_tokens))
    choice = response.choices[0]
    if choice.message.refusal:
        raise RuntimeError(f"Mô hình từ chối yêu cầu này: {choice.message.refusal}")
    return (choice.message.content or "", response.usage.prompt_tokens, response.usage.completion_tokens,
            choice.finish_reason == "length")


class LLMStream:
    """Câu trả lời dạng stream: lặp để nhận từng mảnh văn bản; sau khi lặp hết, `result` là `LLMResult` đầy đủ
    (token lấy từ usage cuối stream). Bản cache phát cả câu trả lời như MỘT mảnh. Chỉ lặp được một lần."""

    def __init__(self, prompt: str, system: str, max_tokens: int, use_cache: bool, model: str | None,
                 effort: str | None):
        s = get_settings()
        self._prompt, self._system, self._max_tokens, self._use_cache = prompt, system, max_tokens, use_cache
        self._model, self._effort = model or s.llm_model, effort or s.llm_effort
        self._started = False
        self.result: LLMResult | None = None

    def __iter__(self) -> Iterator[str]:
        if self._started:
            raise RuntimeError("LLMStream chỉ lặp được một lần")
        self._started = True
        key = cache.cache_key(self._model, self._effort, self._system, self._prompt, None)
        saved = cache.read(key) if self._use_cache else None
        if saved is not None:
            self.result = LLMResult(**{**saved, "cached": True})
            if self.result.text:
                yield self.result.text
            return
        _refuse_network(f"stream_llm(model={self._model})")
        if get_settings().llm_provider == "anthropic":  # chưa stream với Claude: một mảnh duy nhất
            result = ask_llm(self._prompt, self._system, None, self._effort, self._max_tokens, self._use_cache, self._model)
            self.result = result
            if result.text:
                yield result.text
            return
        request = _openai_request(self._prompt, self._system, None, self._effort, self._max_tokens)
        request.update(stream=True, stream_options={"include_usage": True})
        parts: list[str] = []
        usage = None
        truncated = False
        for chunk in _create_with_fallback(self._model, request):
            usage = _chunk_usage(chunk) or usage
            for choice in chunk.choices or []:
                piece = getattr(choice.delta, "content", None)
                if piece:
                    parts.append(piece)
                    yield piece
                truncated = truncated or choice.finish_reason == "length"
        text = "".join(parts)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        self.result = LLMResult(text, None, input_tokens, output_tokens, False)
        if not truncated:  # câu bị cắt ở max_tokens: không lưu, lần sau gọi lại
            cache.write(key, {k: v for k, v in self.result.__dict__.items() if k != "cached"})


def _chunk_usage(chunk: Any) -> Any:
    """Usage ở chunk cuối: chuẩn OpenAI (`chunk.usage`) hoặc Groq cũ (`chunk.x_groq.usage`)."""
    usage = getattr(chunk, "usage", None)
    if usage is not None:
        return usage
    extra = getattr(chunk, "x_groq", None)
    if isinstance(extra, dict):
        return SimpleNamespace(**extra["usage"]) if extra.get("usage") else None
    return getattr(extra, "usage", None)


def stream_llm(prompt: str, system: str = "", max_tokens: int = 4000, use_cache: bool = True,
               model: str | None = None, effort: str | None = None) -> LLMStream:
    """Bản stream của `ask_llm` (văn bản tự do, không structured output). Cùng khoá cache với `ask_llm`."""
    return LLMStream(prompt, system, max_tokens, use_cache, model, effort)
