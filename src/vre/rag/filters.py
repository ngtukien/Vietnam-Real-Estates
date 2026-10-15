"""Bộ lọc câu hỏi (api-contract §2 `QueryFilters`): đổi tiền, trích bằng LLM, gộp với bộ lọc chỉnh tay.

MỘT danh sách điều kiện (`conditions`) biên dịch ra ba dạng tương đương (bug B4):
Qdrant `Filter` (pre-filter), SQL WHERE trên `v_listings` (DuckDB) và mặt nạ pandas (post-filter / test).
Tên cột chuẩn = khoá payload Qdrant (data-contract §5) = cột `v_listings` (trừ tháng: payload `published_month`,
view `month`).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vre.data.places import normalize_place, normalize_text
from vre.llm.client import LLMResult, ask_llm

log = logging.getLogger(__name__)

# Giá trị `note` của sự kiện `filters` (api-contract §4) khi bộ lọc cho 0 kết quả nên đã bỏ lọc.
FILTER_DROPPED = "filter_dropped"
PROPERTY_TYPES = ("Nhà", "Đất", "Căn hộ chung cư", "Biệt thự/Nhà liền kề", "Shophouse")
DEAL_TYPES = ("sale", "rent")
# Khoảng tháng có dữ liệu (data-contract §7.7: ngoài 2025-06…2026-03 là câu không trả lời được). Qdrant chỉ khớp
# keyword nên khoảng tháng được liệt kê thành MatchAny trong khoảng này; ngoài khoảng không có point nào.
DATA_PERIOD = ("2025-06", "2026-03")
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
FilterSource = Literal["llm", "manual", "merged"]

_NUM = r"\d+(?:[.,]\d+)*"


def _number(token: str) -> float:
    """'8,5' -> 8.5; '1.200.000.000' hay '1,200,000' (dấu ngăn nghìn: ≥ 2 nhóm 3 chữ số) -> số nguyên."""
    parts = re.split(r"[.,]", token)
    if len(parts) > 2:
        return float("".join(parts)) if all(len(p) == 3 for p in parts[1:]) else float(f"{parts[0]}.{parts[1]}")
    return float(".".join(parts))


def parse_money_mil(value: Any) -> float | None:
    """Đổi tiền về triệu đồng: '5 tỷ' -> 5000; '1 tỷ 2' -> 1200; '1 tỷ 50 triệu' -> 1050; '850 triệu' -> 850;
    '3.000.000.000' (VND) -> 3000. Số trần dưới 1 triệu coi là triệu. Không có chữ số -> None."""
    if value is None or isinstance(value, (int, float)):
        return value
    text = str(value).lower().strip()
    match = re.search(rf"({_NUM})\s*(?:tỷ|tỉ|ty)(?!\w)(?:\s*(\d+)\s*(triệu|tr)?(?!\w))?", text)
    if match:
        whole, extra, unit = match.groups()
        rest = 0.0
        if extra and unit:  # '1 tỷ 50 triệu'
            rest = float(extra)
        elif extra and len(extra) <= 3:  # '1 tỷ 2' = 1,2 tỷ; '2 tỷ 05' = 2,05 tỷ
            rest = float(extra) * 10 ** (3 - len(extra))
        return _number(whole) * 1000 + rest
    match = re.search(rf"({_NUM})\s*(?:triệu|tr)(?!\w)", text)
    if match:
        return _number(match.group(1))
    match = re.search(rf"({_NUM})\s*(?:nghìn|ngàn|k)(?!\w)", text)
    if match:
        return _number(match.group(1)) / 1000
    match = re.search(_NUM, text)
    if not match:
        return None
    number = _number(match.group())
    return number / 1e6 if number >= 1e6 else number


class ExtractedFilters(BaseModel):
    """Đầu ra LLM: giữ nguyên cách viết tiền trong câu hỏi ('5 tỷ'), code sẽ đổi đơn vị."""

    province: str | None = Field(None, description="Tỉnh/thành, ví dụ 'Hồ Chí Minh', 'Hà Nội'")
    district: str | None = Field(None, description="Quận/huyện/thành phố thuộc tỉnh, ví dụ 'Thủ Đức', 'Quận 7'")
    ward: str | None = Field(None, description="Phường/xã, ví dụ 'Phường 5', 'Dịch Vọng'")
    property_type: str | None = Field(None, description="Một trong: " + ", ".join(PROPERTY_TYPES))
    deal_type: str | None = Field(None, description="'sale' nếu mua/bán, 'rent' nếu thuê/cho thuê; null nếu không nói")
    bedrooms: int | None = Field(None, description="Số phòng ngủ; '2PN' -> 2")
    min_price: str | None = Field(None, description="Giá tối thiểu đúng như câu hỏi viết, ví dụ '3 tỷ'")
    max_price: str | None = Field(None, description="Giá tối đa đúng như câu hỏi viết, ví dụ '5 tỷ'")
    min_area_m2: float | None = Field(None, description="Diện tích tối thiểu, m²")
    max_area_m2: float | None = Field(None, description="Diện tích tối đa, m²")
    project: str | None = Field(None, description="Tên dự án/khu căn hộ nếu câu hỏi nêu, ví dụ 'Vinhomes Grand Park'")
    street: str | None = Field(None, description="Tên đường nếu câu hỏi nêu, ví dụ 'Nguyễn Văn Linh'")
    month_from: str | None = Field(None, description="Tháng bắt đầu dạng YYYY-MM nếu câu hỏi nêu thời gian")
    month_to: str | None = Field(None, description="Tháng kết thúc dạng YYYY-MM (bao gồm)")


class QueryFilters(BaseModel):
    """Bộ lọc đã kiểm tra (api-contract §2): tiền theo triệu đồng (giá tổng), địa danh chuẩn hoá như dữ liệu sạch.
    Mọi trường None = không lọc. Giá trị sai (loại hình lạ, tháng sai dạng, khoảng ngược) ⇒ ValueError."""

    model_config = ConfigDict(extra="forbid")

    province: str | None = None
    district: str | None = None
    ward: str | None = None
    property_type: str | None = None
    deal_type: str | None = None
    bedrooms: int | None = None
    min_price_mil: float | None = None
    max_price_mil: float | None = None
    min_area_m2: float | None = None
    max_area_m2: float | None = None
    project: str | None = None
    street: str | None = None
    month_from: str | None = None
    month_to: str | None = None

    @field_validator("min_price_mil", "max_price_mil", mode="before")
    @classmethod
    def _money(cls, value: Any) -> Any:
        return parse_money_mil(value)

    @field_validator("province", "district", "ward", mode="before")
    @classmethod
    def _place(cls, value: Any, info: Any) -> str | None:
        return normalize_place(value, info.field_name)

    @field_validator("project", "street", mode="before")
    @classmethod
    def _name(cls, value: Any) -> str | None:
        return normalize_text(value)

    @field_validator("property_type")
    @classmethod
    def _type(cls, value: str | None) -> str | None:
        if value is not None and value not in PROPERTY_TYPES:
            raise ValueError(f"property_type phải là một trong {PROPERTY_TYPES}, nhận {value!r}")
        return value

    @field_validator("deal_type")
    @classmethod
    def _deal(cls, value: str | None) -> str | None:
        if value is not None and value not in DEAL_TYPES:
            raise ValueError(f"deal_type phải là 'sale' hoặc 'rent', nhận {value!r}")
        return value

    @field_validator("month_from", "month_to")
    @classmethod
    def _month(cls, value: str | None) -> str | None:
        if value is not None and not MONTH_RE.match(value):
            raise ValueError(f"tháng phải dạng YYYY-MM, nhận {value!r}")
        return value

    @model_validator(mode="after")
    def _ranges(self) -> QueryFilters:
        for lo, hi in (("min_price_mil", "max_price_mil"), ("min_area_m2", "max_area_m2"), ("month_from", "month_to")):
            a, b = getattr(self, lo), getattr(self, hi)
            if a is not None and b is not None and a > b:
                raise ValueError(f"{lo} > {hi} ({a} > {b})")
        return self

    def is_empty(self) -> bool:
        return all(v is None for v in self.model_dump().values())

    @classmethod
    def from_extracted(cls, raw: dict[str, Any]) -> QueryFilters:
        """Từ đầu ra LLM: giá trị không hợp lệ (loại hình lạ, tháng sai dạng…) bị bỏ, có log — LLM đoán sai
        không được làm hỏng cả câu hỏi; bộ lọc chỉnh tay thì validate chặt (ValueError)."""
        data = {k: v for k, v in raw.items() if k not in ("min_price", "max_price")}
        data.update(min_price_mil=raw.get("min_price"), max_price_mil=raw.get("max_price"))
        for key in list(data):
            try:
                cls.model_validate({key: data[key]})
            except ValueError as error:
                log.warning("bỏ bộ lọc LLM %s không hợp lệ: %s", key, type(error).__name__)
                data[key] = None
        try:
            return cls.model_validate(data)
        except ValueError:  # khoảng ngược (min > max): bỏ cả cặp thay vì đoán chiều
            log.warning("bỏ khoảng giá/diện tích/tháng ngược từ LLM")
            for key in ("min_price_mil", "max_price_mil", "min_area_m2", "max_area_m2", "month_from", "month_to"):
                data[key] = None
            return cls.model_validate(data)


def merge(extracted: QueryFilters | None, manual: dict[str, Any] | QueryFilters | None) -> tuple[QueryFilters, FilterSource]:
    """Trường khác None của bộ lọc chỉnh tay GHI ĐÈ trường LLM trích (api-contract §4 `source`)."""
    llm = extracted or QueryFilters()
    hand = manual if isinstance(manual, QueryFilters) else QueryFilters.model_validate(manual or {})
    overrides = {k: v for k, v in hand.model_dump().items() if v is not None}
    if not overrides:
        return llm, "llm"
    merged = QueryFilters.model_validate({**llm.model_dump(), **overrides})
    return merged, ("manual" if llm.is_empty() else "merged")


FILTER_PROMPT = """Trích bộ lọc tìm kiếm bất động sản từ câu hỏi. Chỉ điền trường được nói rõ trong câu hỏi,
còn lại để null. Không suy đoán tỉnh từ tên quận nếu câu hỏi không nói. Viết tắt thường gặp: 'q7' = Quận 7,
'sg'/'hcm' = Hồ Chí Minh, 'hn' = Hà Nội, '2pn' = 2 phòng ngủ, 'cc' = căn hộ chung cư.

Câu hỏi: {question}"""


def parse_filters(question: str) -> tuple[QueryFilters, LLMResult]:
    result = ask_llm(FILTER_PROMPT.format(question=question), output=ExtractedFilters)
    return QueryFilters.from_extracted(result.data or {}), result


# ---- một danh sách điều kiện, ba trình biên dịch (B4)

Op = Literal["eq", "gte", "lte", "in"]


@dataclass(frozen=True)
class Condition:
    field: str  # tên cột chuẩn (payload Qdrant); tháng = "published_month"
    op: Op
    value: Any


_EQ_FIELDS = ("province", "district", "ward", "property_type", "deal_type", "bedrooms", "project", "street")
_RANGES: tuple[tuple[str, str, Op, float], ...] = (("min_price_mil", "price_vnd", "gte", 1e6), ("max_price_mil", "price_vnd", "lte", 1e6),
           ("min_area_m2", "area_m2", "gte", 1.0), ("max_area_m2", "area_m2", "lte", 1.0))
_SQL_COLUMN = {"published_month": "month"}


def month_span(lo: str | None, hi: str | None) -> list[str]:
    """Các tháng `YYYY-MM` trong [lo, hi] ∩ DATA_PERIOD (đầu thiếu = mép khoảng dữ liệu)."""
    start, end = max(lo or DATA_PERIOD[0], DATA_PERIOD[0]), min(hi or DATA_PERIOD[1], DATA_PERIOD[1])
    year, month = int(start[:4]), int(start[5:])
    out: list[str] = []
    while f"{year:04d}-{month:02d}" <= end:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def conditions(f: QueryFilters | None) -> list[Condition]:
    if f is None:
        return []
    out = [Condition(name, "eq", getattr(f, name)) for name in _EQ_FIELDS if getattr(f, name) is not None]
    for name, column, op, scale in _RANGES:
        value = getattr(f, name)
        if value is not None:
            out.append(Condition(column, op, float(value) * scale))
    if f.month_from is not None or f.month_to is not None:
        months = month_span(f.month_from, f.month_to)
        # Khoảng nằm ngoài dữ liệu: khớp đúng tháng biên (không point nào có) ⇒ 0 kết quả ở cả ba dạng.
        out.append(Condition("published_month", "in", months or [f.month_from or f.month_to]))
    return out


def to_qdrant(f: QueryFilters | None) -> Any:
    """`qdrant_client.models.Filter` hoặc None khi không có điều kiện."""
    conds = conditions(f)
    if not conds:
        return None
    from qdrant_client import models

    must: list[Any] = []
    for c in conds:
        if c.op == "eq" and isinstance(c.value, (int, float)):  # bedrooms: payload index float ⇒ dùng Range
            must.append(models.FieldCondition(key=c.field, range=models.Range(gte=c.value, lte=c.value)))
        elif c.op == "eq":
            must.append(models.FieldCondition(key=c.field, match=models.MatchValue(value=c.value)))
        elif c.op == "in":
            must.append(models.FieldCondition(key=c.field, match=models.MatchAny(any=list(c.value))))
        else:
            must.append(models.FieldCondition(key=c.field, range=models.Range(**{c.op: c.value})))
    return models.Filter(must=must)


def to_sql(f: QueryFilters | None) -> tuple[str, list[Any]]:
    """(`WHERE` không kèm từ khoá, tham số `?`) cho `v_listings`; không có điều kiện ⇒ ('TRUE', [])."""
    parts, params = [], []
    for c in conditions(f):
        column = _SQL_COLUMN.get(c.field, c.field)
        if c.op == "in":
            parts.append(f"{column} IN ({', '.join('?' for _ in c.value)})")
            params.extend(c.value)
        else:
            parts.append(f"{column} {'=' if c.op == 'eq' else ('>=' if c.op == 'gte' else '<=')} ?")
            params.append(c.value)
    return (" AND ".join(parts) or "TRUE"), params


def mask(df: pd.DataFrame, f: QueryFilters | None) -> pd.Series:
    """Mặt nạ pandas trên các cột chuẩn (giá trị thiếu ⇒ không khớp, như Qdrant/SQL)."""
    out = pd.Series(True, index=df.index)
    for c in conditions(f):
        column = df[c.field]
        if c.op == "eq":
            hit = column == c.value
        elif c.op == "in":
            hit = column.isin(list(c.value))
        else:
            hit = column >= c.value if c.op == "gte" else column <= c.value
        out &= hit.fillna(False).astype(bool)
    return out


def post_filter(hits: pd.DataFrame, f: QueryFilters | None) -> pd.DataFrame:
    return hits[mask(hits, f)].reset_index(drop=True)
