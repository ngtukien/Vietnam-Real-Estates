"""Gọi Claude qua Anthropic SDK, có cache đĩa để cell CACHE không gọi lại mô hình trên lớp.

Khoá API đọc từ ANTHROPIC_API_KEY (hoặc profile `ant auth login`); không ghi trong notebook.
"""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel

from src.config import CACHE_DIR, LLM_EFFORT, LLM_MODEL

LLM_CACHE = CACHE_DIR / "llm"
_client = None
_read_cache = True


@dataclass
class LLMResult:
    text: str
    data: dict | None
    input_tokens: int
    output_tokens: int
    cached: bool


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


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
            effort: str = LLM_EFFORT, max_tokens: int = 4000, use_cache: bool = True) -> LLMResult:
    """Một lượt hỏi Claude. `output` (pydantic model hoặc JSON schema) bật structured output;
    khi đó `data` là dict đã parse. Cùng (model, system, prompt, schema) trả lại kết quả đã lưu."""
    schema = _schema(output)
    key = hashlib.sha256(json.dumps([LLM_MODEL, effort, system, prompt, schema],
                                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
    path = LLM_CACHE / f"{key}.json"
    if use_cache and _read_cache and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        return LLMResult(**{**saved, "cached": True})

    request = dict(
        model=LLM_MODEL,
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
    data = json.loads(text) if schema else None
    if isinstance(output, type) and issubclass(output, BaseModel):
        data = output.model_validate(data).model_dump()
    result = LLMResult(text, data, response.usage.input_tokens, response.usage.output_tokens, False)
    LLM_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v for k, v in result.__dict__.items() if k != "cached"},
                               ensure_ascii=False), encoding="utf-8")
    return result
