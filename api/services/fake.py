"""Chế độ fake (`VRE_FAKE_BACKENDS=1`, api-contract §6): dữ liệu cố định từ `api/fixtures/*.json`, phát lại chuỗi SSE
ghi sẵn. Không cần Qdrant/Neo4j/LLM/GPU. Chỉ dùng cho test, CI và Playwright E2E.

Chọn chuỗi SSE theo câu hỏi (để E2E thử được mọi nhánh giao diện):
- chứa `[fake:error]` ⇒ `chat_error.json` (meta → error → done)
- `system="hybrid"` và chứa `định giá` ⇒ `chat_hybrid_valuation.json` (route valuation)
- còn lại ⇒ `chat_<system>.json`
`VRE_FAKE_DELAY_MS` (mặc định 0): nghỉ giữa các sự kiện để thấy token chảy trên giao diện.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from api.errors import ApiError
from api.schemas import ErrorCode, SystemId

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ERROR_MARK = "[fake:error]"
VALUATION_MARK = "định giá"


@lru_cache(maxsize=64)
def load(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def sequence_name(system: str, question: str) -> str:
    if ERROR_MARK in question:
        return "chat_error"
    if system == "hybrid" and VALUATION_MARK in question.lower():
        return "chat_hybrid_valuation"
    return f"chat_{system}"


class FakeBackend:
    def health(self) -> dict[str, Any]:
        return dict(load("health"))

    def info(self) -> dict[str, Any]:
        return json.loads(json.dumps(load("info")))

    def filter_options(self) -> dict[str, Any]:
        return dict(load("filters_options"))

    def listing(self, listing_id: int) -> dict[str, Any] | None:
        found: dict[str, Any] | None = load("listings").get(str(listing_id))
        return found

    def valuate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("deal_type") == "rent":
            raise ApiError(ErrorCode.INVALID_REQUEST, "Chưa có model thuê — chỉ định giá được deal_type='sale'")
        result = json.loads(json.dumps(load("valuation")))
        asking = payload.get("asking_price_vnd")
        low, high = result["interval_price_vnd"]["low"], result["interval_price_vnd"]["high"]
        result["verdict"] = None if asking is None else "below_market" if asking < low else "above_market" if asking > high else "fair"
        return dict(result)

    def stats_overview(self) -> dict[str, Any]:
        return dict(load("stats_overview"))

    def market(self, province: str, district: str | None, property_type: str | None, deal_type: str) -> dict[str, Any] | None:
        doc: dict[str, Any] = json.loads(json.dumps(load("market_district" if district else "market_province")))
        scope = doc["scope"]
        if scope["province"] != province or scope["district"] != district:
            return None
        scope["property_type"], scope["deal_type"] = property_type, deal_type
        return doc

    def geo_areas(self, level: str, province: str | None) -> dict[str, Any]:
        doc: dict[str, Any] = load(f"geo_{level}")
        if level != "province" and doc.get("province") != province:
            return {"level": level, "items": [], "coverage": 0.0}
        return {k: v for k, v in doc.items() if k != "province"}

    def subgraph(self, label: str, key: str | int, limit: int) -> dict[str, Any] | None:
        doc: dict[str, Any] = load("subgraph")
        if f"{label}:{key}" not in {n["id"] for n in doc["nodes"]}:
            return None
        return dict(doc)

    def benchmark(self) -> dict[str, Any] | None:
        return dict(load("benchmark"))

    def system_status(self) -> dict[str, tuple[bool, str | None]]:
        return {s.value: (True, None) for s in SystemId}

    def stream_answer(self, system: str, question: str, filters: dict[str, Any] | None, top_k: int, use_rerank: bool,
                      stream_tokens: bool) -> Iterator[tuple[str, dict[str, Any]]]:
        delay = float(os.environ.get("VRE_FAKE_DELAY_MS", "0")) / 1000
        for name, payload in load(sequence_name(system, question)):
            if name == "token" and not stream_tokens:
                continue
            data = json.loads(json.dumps(payload))
            if name == "meta":
                data = {"question": question, "started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
            if delay:
                time.sleep(delay)
            yield name, data
