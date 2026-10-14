"""Chuẩn hoá chữ và địa danh (giữ quy tắc của `src/data.py` cũ) + map tỉnh 63 → 34 sau sáp nhập 07/2025."""

from __future__ import annotations

import csv
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

ADMIN_2025_CSV = Path(__file__).with_name("admin_2025.csv")
PROVINCE_ALIASES = {"hcm": "Hồ Chí Minh", "tphcm": "Hồ Chí Minh", "tp hcm": "Hồ Chí Minh",
                    "sài gòn": "Hồ Chí Minh", "hn": "Hà Nội"}
_BLANK = {"", "None", "nan"}
_WS = re.compile(r"\s+")
_PROVINCE_PREFIX = re.compile(r"^(tp\.?|thành phố|tỉnh)\s*", re.I)
_DISTRICT_NUMBER = re.compile(r"(?:q\.?|quận)?\s*0*(\d+)", re.I)
_DISTRICT_PREFIX = re.compile(r"^(quận|huyện|thị xã|thành phố|q\.|h\.|tx\.?|tp\.?)\s*", re.I)
_WARD_NUMBER = re.compile(r"(?:p\.?|phường)?\s*0*(\d+)", re.I)
_WARD_PREFIX = re.compile(r"^(phường|xã|thị trấn|p\.|x\.|tt\.?)\s*", re.I)


def normalize_text(value: object) -> str | None:
    """NFC + gộp khoảng trắng; chuỗi rỗng/"None"/"nan"/NaN → None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    out = unicodedata.normalize("NFC", _WS.sub(" ", str(value))).strip()
    return None if out in _BLANK else out


def text(value: object) -> str | None:
    """Giá trị chữ hoặc None: pandas đọc ô chữ bị thiếu thành NaN (float, và NaN là truthy)."""
    return value if isinstance(value, str) and value else None


def normalize_place(value: object, kind: str) -> str | None:
    """Entity resolution cho địa danh: 'Q.7', 'q7', '7' -> 'Quận 7'; 'P.5' -> 'Phường 5';
    'TP. Hồ Chí Minh' -> 'Hồ Chí Minh'; 'Huyện Củ Chi' -> 'Củ Chi'."""
    out = normalize_text(value)
    if not out:
        return None
    if kind == "province":
        out = _PROVINCE_PREFIX.sub("", out).strip()
        return PROVINCE_ALIASES.get(out.lower(), out)
    if kind == "district":
        number = _DISTRICT_NUMBER.fullmatch(out)
        if number:
            return f"Quận {int(number.group(1))}"
        return _DISTRICT_PREFIX.sub("", out).strip() or None
    if kind == "ward":
        number = _WARD_NUMBER.fullmatch(out)
        if number:
            return f"Phường {int(number.group(1))}"
        return _WARD_PREFIX.sub("", out).strip() or None
    return out


@lru_cache(maxsize=1)
def admin_2025() -> dict[str, str]:
    """`province_old` (sau normalize_place) → `province_2025`."""
    with ADMIN_2025_CSV.open(encoding="utf-8") as handle:
        return {row["province_old"]: row["province_2025"] for row in csv.DictReader(handle)}


def province_2025(province: str | None) -> str | None:
    return admin_2025().get(province) if province else None
