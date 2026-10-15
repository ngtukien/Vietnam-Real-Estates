"""Cache đĩa cho lượt gọi LLM (chuyển nguyên từ `src/llm.py`): khoá, đọc/ghi nguyên tử, `fresh()`.

Thư mục: `Settings.results_dir / "cache" / "llm"` (= `results/cache/llm/` cũ, giữ để tận dụng cache có sẵn).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from vre.config import get_settings

# Theo ngữ cảnh (luồng / task), không phải biến toàn cục: `fresh()` ở một luồng đo không làm luồng chatbot khác
# bỏ qua cache (bug B1). ThreadPoolExecutor KHÔNG chép ngữ cảnh sang luồng con — dùng `contextvars.copy_context().run`.
_read_cache: ContextVar[bool] = ContextVar("vre_llm_read_cache", default=True)


def cache_dir() -> Path:
    return get_settings().results_dir / "cache" / "llm"


def cache_key(model: str, effort: str, system: str, prompt: str, schema: dict | None) -> str:
    """Cùng (model, effort, system, prompt, schema) cho cùng khoá."""
    return hashlib.sha256(json.dumps([model, effort, system, prompt, schema],
                                     ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


def read(key: str) -> dict | None:
    """Kết quả đã lưu cho `key`; None nếu chưa có hoặc đang trong khối `fresh()`."""
    path = cache_dir() / f"{key}.json"
    if _read_cache.get() and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def write(key: str, payload: dict) -> None:
    """Ghi nguyên tử: luồng khác không bao giờ đọc phải file dở."""
    directory = cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.json"
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def reading() -> bool:
    """False khi đang trong khối `fresh()` của ngữ cảnh hiện tại."""
    return _read_cache.get()


@contextmanager
def fresh() -> Iterator[None]:
    """Trong khối này mọi lượt gọi CỦA NGỮ CẢNH HIỆN TẠI đều đến API (vẫn ghi cache): dùng khi đo độ trễ và token thật."""
    token = _read_cache.set(False)
    try:
        yield
    finally:
        _read_cache.reset(token)
