"""Gọi LLM (mặc định Groq qua OpenAI SDK; tuỳ chọn OpenAI hoặc Claude, chọn bằng LLM_PROVIDER), có cache đĩa
để cell CACHE không gọi lại mô hình trên lớp.

Khoá API đọc từ GROQ_API_KEY, OPENAI_API_KEY hoặc ANTHROPIC_API_KEY (hoặc profile `ant auth login`);
không ghi trong notebook.
"""

import hashlib
import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass

from pydantic import BaseModel

from src.config import CACHE_DIR, LLM_EFFORT, LLM_FALLBACK_MODELS, LLM_MODEL, LLM_PROVIDER, LLM_TIMEOUT

LLM_CACHE = CACHE_DIR / "llm"
_client = None
_client_lock = threading.Lock()  # chatbot gọi từ nhiều luồng: chỉ tạo một client
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_read_cache = True


@dataclass
class LLMResult:
    text: str
    data: dict | None
    input_tokens: int
    output_tokens: int
    cached: bool


def client():
    global _client
    with _client_lock:
        if _client is None:
            _client = _new_client()
    return _client


def _new_client():
    if LLM_PROVIDER == "anthropic":
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError("Thiếu xác thực Claude: đặt ANTHROPIC_API_KEY trong .env hoặc chạy `ant auth login`.")
        import anthropic
        return anthropic.Anthropic(timeout=LLM_TIMEOUT, max_retries=2)
    import openai
    if LLM_PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Thiếu xác thực OpenAI: đặt OPENAI_API_KEY trong .env.")
        return openai.OpenAI(timeout=LLM_TIMEOUT, max_retries=2)
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Thiếu xác thực Groq: đặt GROQ_API_KEY trong .env.")
    # Groq giới hạn token/phút theo từng mô hình: SDK tự chờ và thử lại khi gặp 429.
    return openai.OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL, timeout=LLM_TIMEOUT,
                         max_retries=4)


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


@contextmanager
def fresh():
    """Trong khối này mọi lượt gọi đều đến API (vẫn ghi cache): dùng khi đo độ trễ và token thật."""
    global _read_cache
    _read_cache = False
    try:
        yield
    finally:
        _read_cache = True


def ask_llm(prompt: str, system: str = "", output: type[BaseModel] | dict | None = None,
            effort: str = LLM_EFFORT, max_tokens: int = 4000, use_cache: bool = True,
            model: str | None = None) -> LLMResult:
    """Một lượt hỏi LLM. `output` (pydantic model hoặc JSON schema) bật structured output;
    khi đó `data` là dict đã parse. Cùng (model, system, prompt, schema) trả lại kết quả đã lưu.
    `model` mặc định là LLM_MODEL (eval.py dùng JUDGE_MODEL để chấm)."""
    model = model or LLM_MODEL
    schema = _schema(output)
    key = hashlib.sha256(json.dumps([model, effort, system, prompt, schema],
                                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
    path = LLM_CACHE / f"{key}.json"
    if use_cache and _read_cache and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        return LLMResult(**{**saved, "cached": True})

    ask = _ask_anthropic if LLM_PROVIDER == "anthropic" else _ask_openai
    text, input_tokens, output_tokens, truncated = ask(model, prompt, system, schema, effort, max_tokens)
    if truncated and schema:
        raise RuntimeError(f"Đầu ra có cấu trúc bị cắt ở max_tokens={max_tokens}")
    data = json.loads(text) if schema else None
    if isinstance(output, type) and issubclass(output, BaseModel):
        data = output.model_validate(data).model_dump()
    result = LLMResult(text, data, input_tokens, output_tokens, False)
    LLM_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({k: v for k, v in result.__dict__.items() if k != "cached"},
                              ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # ghi nguyên tử: luồng khác không bao giờ đọc phải file dở
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


def _ask_openai(model: str, prompt: str, system: str, schema: dict | None, effort: str, max_tokens: int):
    messages = ([{"role": "developer", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    request = dict(messages=messages, max_completion_tokens=max_tokens, reasoning_effort=effort)
    if schema:
        request["response_format"] = {"type": "json_schema",
                                      "json_schema": {"name": "output", "schema": schema, "strict": True}}
    import openai
    models = list(dict.fromkeys([model, *LLM_FALLBACK_MODELS]))
    response = None
    for i, name in enumerate(models):
        for attempt in range(3):
            try:
                response = client().chat.completions.create(model=name, **request)
                break
            except openai.BadRequestError as error:
                # Groq đôi khi trả JSON rỗng cho structured output (json_validate_failed): thử lại cùng model.
                if "json_validate_failed" not in str(error) or (attempt == 2 and i == len(models) - 1):
                    raise
            except (openai.InternalServerError, openai.RateLimitError):
                if i == len(models) - 1:
                    raise
                break
        if response is not None:
            break
    choice = response.choices[0]
    if choice.message.refusal:
        raise RuntimeError(f"Mô hình từ chối yêu cầu này: {choice.message.refusal}")
    return (choice.message.content or "", response.usage.prompt_tokens, response.usage.completion_tokens,
            choice.finish_reason == "length")
