"""Cache đĩa cho lượt gọi LLM (chuyển nguyên từ `src/llm.py`): khoá, đọc/ghi nguyên tử, `fresh()`.

Thư mục: `Settings.results_dir / "cache" / "llm"` (= `results/cache/llm/` cũ, giữ để tận dụng cache có sẵn).
"""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from pathlib import Path

from vre.config import get_settings

_read_cache = True


def cache_dir() -> Path:
    return get_settings().results_dir / "cache" / "llm"


def cache_key(model: str, effort: str, system: str, prompt: str, schema: dict | None) -> str:
    """Cùng (model, effort, system, prompt, schema) cho cùng khoá."""
    return hashlib.sha256(json.dumps([model, effort, system, prompt, schema],
                                     ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


def read(key: str) -> dict | None:
    """Kết quả đã lưu cho `key`; None nếu chưa có hoặc đang trong khối `fresh()`."""
    path = cache_dir() / f"{key}.json"
    if _read_cache and path.exists():
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


@contextmanager
def fresh():
    """Trong khối này mọi lượt gọi đều đến API (vẫn ghi cache): dùng khi đo độ trễ và token thật."""
    global _read_cache
    _read_cache = False
    try:
        yield
    finally:
        _read_cache = True
