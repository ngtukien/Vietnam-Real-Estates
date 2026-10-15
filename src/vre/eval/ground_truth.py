"""Đáp án chuẩn benchmark: chạy mọi `gt_sql` trên `v_listings`, ghi `results/benchmark_gt.json` (data-contract §7.7).

Chỉ GHI `results/benchmark_gt.json`; `benchmark/questions.jsonl` chỉ được đọc (bug B6).
Kiểm (lỗi nào cũng dừng, không ghi file dở): `ids` không rỗng và có cột `listing_id`; lookup ≤ 5 `dup_cluster_id`;
`value` đúng 1 dòng, không ô nào null, mọi cột cỡ nhóm (`n`, `n_*`) ≥ 20; `refusal` có `gt_sql=null`.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from vre.common import listings_fingerprint, result_header, scalar, write_json_atomic
from vre.config import Settings
from vre.eval.questions import Question, load_questions

GT_FILENAME = "benchmark_gt.json"
SCHEMA_VERSION = 1
LOOKUP_MAX_CLUSTERS = 5  # phase-03 A2: lookup GT ≤ 5 dup_cluster_id
MIN_GROUP_N = 20  # phase-03 A2: aggregate/comparison… cần n ≥ 20 mỗi nhóm
GROUP_SIZE_COLUMN = re.compile(r"^n(_|$)")  # quy ước: cột cỡ nhóm tên `n` hoặc `n_<nhóm>`


def gt_path(settings: Settings) -> Path:
    return settings.results_dir / GT_FILENAME


def format_value(value: Any) -> str:
    """Giá trị chính → chuỗi `gt_value`: số thực làm tròn 2 chữ số thập phân, số nguyên/chuỗi giữ nguyên."""
    value = scalar(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _ids_item(con, q: Question) -> dict[str, Any]:
    cursor = con.execute(q.gt_sql)
    columns = [d[0] for d in cursor.description]
    if "listing_id" not in columns:
        raise ValueError(f"{q.id}: SQL gt_kind=ids phải trả cột listing_id (nhận {columns})")
    col = columns.index("listing_id")
    ids = sorted({int(row[col]) for row in cursor.fetchall() if row[col] is not None})
    if not ids:
        raise ValueError(f"{q.id}: tập tin đúng rỗng")
    clusters = sorted(int(c) for (c,) in con.execute(
        "SELECT DISTINCT dup_cluster_id FROM v_listings WHERE listing_id IN (SELECT unnest(?::BIGINT[]))",
        [ids]).fetchall())
    if q.category == "lookup" and len(clusters) > LOOKUP_MAX_CLUSTERS:
        raise ValueError(f"{q.id}: lookup có {len(clusters)} dup_cluster_id (> {LOOKUP_MAX_CLUSTERS})")
    return {"gt_ids": ids, "gt_dup_clusters": clusters, "gt_value": None, "gt_rows": None}


def _value_item(con, q: Question) -> dict[str, Any]:
    cursor = con.execute(q.gt_sql)
    columns = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    if len(rows) != 1:
        raise ValueError(f"{q.id}: SQL gt_kind=value phải trả đúng 1 dòng (nhận {len(rows)})")
    row = {c: scalar(v) for c, v in zip(columns, rows[0], strict=True)}
    nulls = [c for c, v in row.items() if v is None or (isinstance(v, float) and math.isnan(v))]
    if nulls:
        raise ValueError(f"{q.id}: kết quả có ô null ở cột {nulls}")
    small = {c: v for c, v in row.items()
             if GROUP_SIZE_COLUMN.match(c) and isinstance(v, (int, float)) and v < MIN_GROUP_N}
    if small:
        raise ValueError(f"{q.id}: nhóm quá nhỏ (< {MIN_GROUP_N}): {small}")
    return {"gt_ids": None, "gt_dup_clusters": None, "gt_value": format_value(rows[0][0]), "gt_rows": [row]}


def compute_item(con, q: Question) -> dict[str, Any]:
    """Một phần tử `items[]` của benchmark_gt.json cho câu `q`."""
    if q.gt_kind == "refusal":
        return {"id": q.id, "gt_ids": None, "gt_dup_clusters": None, "gt_value": None, "gt_rows": None,
                "sql_seconds": 0.0}
    start = perf_counter()
    body = _ids_item(con, q) if q.gt_kind == "ids" else _value_item(con, q)
    return {"id": q.id, **body, "sql_seconds": round(perf_counter() - start, 3)}


def run(settings: Settings, *, questions_path: Path | None = None) -> dict:
    """Chạy mọi gt_sql trên v_listings, ghi `results/benchmark_gt.json`, trả dict đó.

    Câu nào lỗi ⇒ RuntimeError liệt kê TẤT CẢ câu lỗi; file cũ (nếu có) giữ nguyên."""
    from vre.data.duck import connect

    questions = load_questions(questions_path)
    con = connect(settings)
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        for q in questions:
            try:
                items.append(compute_item(con, q))
            except Exception as exc:  # gom đủ lỗi rồi báo một lần (sửa cả lô, không từng câu)
                errors.append(f"{q.id}: {type(exc).__name__}: {exc}")
    finally:
        con.close()
    if errors:
        raise RuntimeError(f"{len(errors)} câu không tính được đáp án chuẩn:\n" + "\n".join(errors))
    doc = {**result_header(SCHEMA_VERSION), "data_fingerprint": listings_fingerprint(settings.clean_path),
           "items": items}
    write_json_atomic(gt_path(settings), doc)
    return doc


def load_gt(settings: Settings) -> dict[str, dict]:
    """id → item của `results/benchmark_gt.json`."""
    path = gt_path(settings)
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}; chạy `vre_pipeline.py eval gt` trước")
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in doc["items"]}
