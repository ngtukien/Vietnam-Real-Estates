"""Từ điển amenity (data-contract §3.2): khoá → regex trên lower(NFC(title + mô tả))."""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

AMENITIES_YAML = Path(__file__).with_name("amenities.yaml")


@lru_cache(maxsize=1)
def amenity_table() -> dict[str, dict]:
    """`{key: {"name": str, "patterns": [str]}}` đúng thứ tự trong file."""
    return yaml.safe_load(AMENITIES_YAML.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def compiled() -> dict[str, re.Pattern[str]]:
    return {key: re.compile("|".join(f"(?:{p})" for p in spec["patterns"]))
            for key, spec in amenity_table().items()}


def match_amenities(text: str | None) -> list[str]:
    """Khoá amenity khớp trong một đoạn văn bản đã chuẩn hoá NFC (hàm tự hạ chữ thường)."""
    if not text:
        return []
    lowered = text.lower()
    return [key for key, pattern in compiled().items() if pattern.search(lowered)]


def amenity_lists(texts: pd.Series) -> list[list[str]]:
    """Vector hoá cho cả shard: mỗi phần tử là danh sách khoá (có thể rỗng), theo thứ tự trong file."""
    lowered = texts.fillna("").str.lower()
    with warnings.catch_warnings():  # pattern có nhóm bắt — chỉ cần có/không khớp
        warnings.simplefilter("ignore", UserWarning)
        hits = {key: lowered.str.contains(pattern, regex=True).to_numpy() for key, pattern in compiled().items()}
    keys = list(hits)
    return [[k for k in keys if hits[k][i]] for i in range(len(lowered))]
