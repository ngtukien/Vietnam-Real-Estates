"""Tuyến `valuation` của hệ hybrid: LLM trích `ValuationInput` (structured output) → `valuate()` →
tin so sánh (cùng quận, loại hình, diện tích ±20%) → ngữ cảnh cho câu trả lời có số dự báo + khoảng + [Tin#id]."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from vre.data.amenities import amenity_table
from vre.data.places import normalize_place
from vre.llm.client import LLMResult, ask_llm
from vre.rag.filters import PROPERTY_TYPES, QueryFilters, parse_money_mil

AREA_TOLERANCE = 0.20  # tin so sánh: diện tích ±20%

VALUATION_SYSTEM_NOTE = """
Phần ĐỊNH GIÁ là kết quả mô hình XGBoost: nêu giá dự báo (tổng và theo m²), khoảng dự báo, so với giá chào nếu có,
và ghi nguồn [Định giá]. Các tin so sánh ghi nguồn [Tin#ID] (chép nguyên ID). Đây là ước lượng, không phải thẩm định."""


class ExtractedValuation(BaseModel):
    property_type: str | None = Field(None, description="Một trong: " + ", ".join(PROPERTY_TYPES))
    deal_type: str | None = Field(None, description="'sale' hoặc 'rent'; null nếu không nói")
    province: str | None = Field(None, description="Tỉnh/thành, ví dụ 'Hồ Chí Minh'")
    district: str | None = Field(None, description="Quận/huyện, ví dụ 'Quận 7', 'Gò Vấp'")
    ward: str | None = Field(None, description="Phường/xã nếu có")
    area_m2: float | None = Field(None, description="Diện tích m²")
    bedrooms: int | None = None
    bathrooms: int | None = None
    floors: int | None = None
    frontage_m: float | None = Field(None, description="Mặt tiền, m")
    road_width_m: float | None = Field(None, description="Đường trước nhà rộng, m")
    house_direction: str | None = Field(None, description="Hướng nhà, ví dụ 'Đông Nam'")
    amenities: list[str] = Field(default_factory=list, description="Khoá tiện ích: " + ", ".join(amenity_table()))
    asking_price: str | None = Field(None, description="Giá chào đúng như câu hỏi viết, ví dụ '4,5 tỷ'")


EXTRACT_PROMPT = """Trích mô tả bất động sản cần định giá từ câu hỏi. Chỉ điền trường được nói rõ, còn lại null.
Viết tắt: 'q7' = Quận 7, 'sg'/'hcm' = Hồ Chí Minh, 'pn' = phòng ngủ, 'cc' = Căn hộ chung cư.

Câu hỏi: {question}"""


def extract_input(question: str) -> tuple[dict[str, Any], LLMResult]:
    """Payload `ValuationInput` (api-contract §3) từ câu hỏi; trường thiếu để `valuate()` báo ValueError."""
    result = ask_llm(EXTRACT_PROMPT.format(question=question), output=ExtractedValuation)
    raw = dict(result.data or {})
    asking = parse_money_mil(raw.pop("asking_price", None))
    payload: dict[str, Any] = {k: v for k, v in raw.items() if v not in (None, "")}
    for kind in ("province", "district", "ward"):
        if kind in payload:
            payload[kind] = normalize_place(payload[kind], kind)
    if payload.get("deal_type") not in ("sale", "rent"):
        payload.pop("deal_type", None)
    known = set(amenity_table())
    payload["amenities"] = [a for a in payload.get("amenities", []) if a in known]
    if asking is not None:
        payload["asking_price_vnd"] = float(asking) * 1e6
    return payload, result


def comparable_filters(payload: dict[str, Any]) -> QueryFilters:
    area = float(payload["area_m2"])
    return QueryFilters(province=payload.get("province"), district=payload.get("district"),
                        property_type=payload.get("property_type"), deal_type=payload.get("deal_type") or "sale",
                        min_area_m2=area * (1 - AREA_TOLERANCE), max_area_m2=area * (1 + AREA_TOLERANCE))


def valuation_context(payload: dict[str, Any], result: dict[str, Any]) -> str:
    keep = {k: result[k] for k in ("predicted_price_m2_mil", "predicted_price_vnd", "interval_price_vnd",
                                    "baseline_ward_median_price_m2_mil", "comparables_n", "verdict")}
    return (f"ĐỊNH GIÁ [Định giá]\nĐầu vào: {json.dumps(payload, ensure_ascii=False)}\n"
            f"Kết quả: {json.dumps(keep, ensure_ascii=False)}\nModel: {result['model']['variant']}, "
            f"MAPE kiểm tra {result['model']['test_mape']}")
