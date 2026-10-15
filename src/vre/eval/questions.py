"""Câu hỏi benchmark (data-contract §7.7): đọc `benchmark/questions.jsonl`, validate, kiểm SQL chỉ-đọc.

`benchmark/questions.jsonl` là SSOT viết tay, KHÔNG bao giờ bị code ghi đè (bug B6): đáp án chuẩn chỉ ghi vào
`results/benchmark_gt.json` (`vre.eval.ground_truth`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from vre.common import CODE_ROOT

Category = Literal["lookup", "constraint", "aggregate", "multihop", "temporal", "comparison", "project_street",
                   "valuation", "unanswerable"]
Route = Literal["basic_rag", "graph", "hybrid", "valuation"]
GtKind = Literal["ids", "value", "refusal"]

QUESTIONS_PATH = CODE_ROOT / "benchmark" / "questions.jsonl"

# Phân bổ chốt ở phase-03 bước A2 (tổng 100).
ALLOCATION: dict[str, int] = {"lookup": 12, "constraint": 12, "aggregate": 12, "multihop": 12, "temporal": 10,
                              "comparison": 10, "project_street": 10, "valuation": 10, "unanswerable": 12}
ROBUSTNESS_TAGS = frozenset({"no_diacritics", "abbreviation"})
MIN_ROBUSTNESS_QUESTIONS = 15

# Khoá của ValuationInput (api-contract §3). Bắt buộc: property_type, province, district, area_m2.
VALUATION_KEYS = frozenset({"property_type", "deal_type", "province", "district", "ward", "area_m2", "bedrooms",
                            "bathrooms", "floors", "frontage_m", "road_width_m", "house_direction", "month",
                            "amenities", "asking_price_vnd"})
VALUATION_REQUIRED = ("property_type", "province", "district", "area_m2")
VALUATION_MAX_AREA_M2 = 100_000

VIEW_NAME = "v_listings"
_SQL_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
# Từ khoá ghi/side-effect và hàm đọc file: GT chỉ được đọc view chuẩn, không tự đọc parquet/csv.
_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|merge|create|drop|alter|truncate|copy|attach|detach|pragma|install|load|export|"
    r"import|set|reset|call|checkpoint|vacuum|grant|revoke|read_parquet|read_csv|read_csv_auto|read_json|"
    r"read_json_auto|parquet_scan|glob)\b",
    re.IGNORECASE)
_SQL_LITERAL = re.compile(r"'(?:[^']|'')*'")


def check_sql(sql: str) -> None:
    """Raise ValueError nếu SQL không phải MỘT câu SELECT/WITH chỉ đọc `v_listings`.

    Chuỗi trong nháy đơn được bỏ qua khi tìm từ khoá cấm (tên đường/dự án có thể chứa chữ như "Set")."""
    if not _SQL_START.match(sql):
        raise ValueError("gt_sql phải bắt đầu bằng SELECT hoặc WITH")
    bare = _SQL_LITERAL.sub("''", sql)
    if ";" in bare.rstrip().rstrip(";"):
        raise ValueError("gt_sql chỉ được chứa một câu lệnh")
    bad = _SQL_FORBIDDEN.search(bare)
    if bad:
        raise ValueError(f"gt_sql chứa từ khoá cấm {bad.group(0)!r}")
    if not re.search(rf"\b{VIEW_NAME}\b", bare):
        raise ValueError(f"gt_sql phải đọc view {VIEW_NAME}")


def _check_valuation_input(payload: dict[str, Any]) -> None:
    unknown = set(payload) - VALUATION_KEYS
    if unknown:
        raise ValueError(f"valuation_input có khoá lạ: {sorted(unknown)}")
    missing = [k for k in VALUATION_REQUIRED if payload.get(k) in (None, "")]
    if missing:
        raise ValueError(f"valuation_input thiếu khoá bắt buộc: {missing}")
    area = payload["area_m2"]
    if not isinstance(area, (int, float)) or isinstance(area, bool) or not 0 < area <= VALUATION_MAX_AREA_M2:
        raise ValueError(f"valuation_input.area_m2 phải trong (0, {VALUATION_MAX_AREA_M2}], nhận {area!r}")


class Question(BaseModel):
    """Một dòng của `benchmark/questions.jsonl` (đúng các khoá data-contract §7.7)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    category: Category
    route_gold: Route
    question: str
    gt_kind: GtKind
    gt_sql: str | None
    value_tolerance: float | None = None
    tags: list[str] = []
    valuation_input: dict | None = None  # chỉ category=valuation: payload ValuationInput (api-contract §3)

    @model_validator(mode="after")
    def _consistent(self) -> Question:
        if not self.id.strip() or not self.question.strip():
            raise ValueError("id và question không được rỗng")
        if (self.category == "unanswerable") != (self.gt_kind == "refusal"):
            raise ValueError("gt_kind=refusal khi và chỉ khi category=unanswerable")
        if self.gt_kind == "refusal":
            if self.gt_sql is not None:
                raise ValueError("câu refusal phải có gt_sql=null")
        else:
            if not self.gt_sql:
                raise ValueError(f"gt_kind={self.gt_kind} cần gt_sql")
            check_sql(self.gt_sql)
        if self.gt_kind == "value":
            if self.value_tolerance is None or not 0 <= self.value_tolerance < 1:
                raise ValueError("câu gt_kind=value cần value_tolerance trong [0, 1)")
        elif self.value_tolerance is not None:
            raise ValueError("value_tolerance chỉ dùng cho gt_kind=value")
        if (self.category == "valuation") != (self.route_gold == "valuation"):
            raise ValueError("route_gold=valuation khi và chỉ khi category=valuation")
        if self.category == "valuation":
            if self.gt_kind != "value" or self.valuation_input is None:
                raise ValueError("câu valuation cần gt_kind=value và valuation_input")
            _check_valuation_input(self.valuation_input)
        elif self.valuation_input is not None:
            raise ValueError("valuation_input chỉ dùng cho category=valuation")
        return self


def load_questions(path: Path | None = None) -> list[Question]:
    """Đọc + validate mọi câu (mặc định `CODE_ROOT/benchmark/questions.jsonl`); id phải duy nhất."""
    path = QUESTIONS_PATH if path is None else Path(path)
    questions: list[Question] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            q = Question.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"{path.name}:{lineno}: câu hỏi không hợp lệ: {exc}") from exc
        if q.id in seen:
            raise ValueError(f"{path.name}:{lineno}: id trùng {q.id!r}")
        seen.add(q.id)
        questions.append(q)
    if not questions:
        raise ValueError(f"{path} không có câu hỏi nào")
    return questions
