"""Hiểu câu hỏi: đổi tiền, bộ lọc LLM, lọc pandas (chuyển nguyên từ `src/rag.py`)."""

from __future__ import annotations

import re

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from vre.data.places import normalize_place
from vre.llm.client import LLMResult, ask_llm

# Ghi vào 'note' khi bộ lọc quá chặt (0 kết quả) nên đã bỏ lọc; generate báo điều này cho LLM.
FILTER_DROPPED = "bộ lọc không có kết quả, đã bỏ lọc"

_NUM = r"\d+(?:[.,]\d+)*"


def _number(token: str) -> float:
    """'8,5' -> 8.5; '1.200.000.000' hay '1,200,000' (dấu ngăn nghìn: ≥ 2 nhóm 3 chữ số) -> số nguyên."""
    parts = re.split(r"[.,]", token)
    if len(parts) > 2:
        return float("".join(parts)) if all(len(p) == 3 for p in parts[1:]) else float(f"{parts[0]}.{parts[1]}")
    return float(".".join(parts))


def parse_money_mil(value) -> float | None:
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
    property_type: str | None = Field(None, description="Một trong: Căn hộ chung cư, Nhà, Đất, Biệt thự/Nhà liền kề, Shophouse")
    bedrooms: int | None = Field(None, description="Số phòng ngủ; '2PN' -> 2")
    min_price: str | None = Field(None, description="Giá tối thiểu đúng như câu hỏi viết, ví dụ '3 tỷ'")
    max_price: str | None = Field(None, description="Giá tối đa đúng như câu hỏi viết, ví dụ '5 tỷ'")
    min_area: float | None = Field(None, description="Diện tích tối thiểu, m²")
    max_area: float | None = Field(None, description="Diện tích tối đa, m²")


class QueryFilters(BaseModel):
    """Bộ lọc đã kiểm tra: tiền theo triệu đồng, địa danh đã chuẩn hoá như dữ liệu sạch."""

    province: str | None = None
    district: str | None = None
    property_type: str | None = None
    bedrooms: int | None = None
    min_price_mil: float | None = None
    max_price_mil: float | None = None
    min_area: float | None = None
    max_area: float | None = None

    @field_validator("min_price_mil", "max_price_mil", mode="before")
    @classmethod
    def _money(cls, value):
        return parse_money_mil(value)

    @field_validator("province", "district", mode="before")
    @classmethod
    def _place(cls, value, info):
        return normalize_place(value, info.field_name)

    @classmethod
    def from_extracted(cls, raw: dict) -> QueryFilters:
        return cls.model_validate({**raw, "min_price_mil": raw.get("min_price"),
                                   "max_price_mil": raw.get("max_price")})


FILTER_PROMPT = """Trích bộ lọc tìm kiếm bất động sản từ câu hỏi. Chỉ điền trường được nói rõ trong câu hỏi,
còn lại để null. Không suy đoán tỉnh từ tên quận nếu câu hỏi không nói.

Câu hỏi: {question}"""


def parse_filters(question: str) -> tuple[QueryFilters, LLMResult]:
    result = ask_llm(FILTER_PROMPT.format(question=question), output=ExtractedFilters)
    return QueryFilters.from_extracted(result.data), result


def matches(df: pd.DataFrame, filters: QueryFilters | None) -> pd.Series:
    """Mặt nạ pandas tương đương bộ lọc Qdrant (qdrant_store.to_filter), dùng cho post-filter."""
    mask = pd.Series(True, index=df.index)
    if filters is None:
        return mask
    f = filters
    for key, value in (("province", f.province), ("district", f.district), ("property_type", f.property_type)):
        if value:
            mask &= df[key] == value
    if f.bedrooms is not None:
        mask &= df["bedrooms"] == f.bedrooms
    if f.min_price_mil is not None:
        mask &= df["price"] >= f.min_price_mil * 1e6
    if f.max_price_mil is not None:
        mask &= df["price"] <= f.max_price_mil * 1e6
    if f.min_area is not None:
        mask &= df["area"] >= f.min_area
    if f.max_area is not None:
        mask &= df["area"] <= f.max_area
    return mask.fillna(False)


def post_filter(hits: pd.DataFrame, filters: QueryFilters) -> pd.DataFrame:
    return hits[matches(hits, filters)].reset_index(drop=True)
