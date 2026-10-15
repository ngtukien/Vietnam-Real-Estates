"""Error analysis (A3.5): gán nguyên nhân cho mỗi câu trả lời chưa đạt 2 điểm → `results/error_analysis.json`
(data-contract §7.7; enum `CAUSES`).

Thứ tự luật (luật đầu tiên khớp thắng — nguyên nhân gốc trước triệu chứng):
1. `runtime_error` — có lỗi và không có câu trả lời.
2. `refusal_missing` — câu `unanswerable` mà không từ chối rõ.
3. `router_wrong` — hybrid chọn route khác `route_gold`.
4. `cypher_error` — Cypher lỗi (kể cả khi đã lui về RAG).
5. Câu `gt_kind=ids` thiếu tin đúng: `filter_wrong` khi bộ lọc LLM trích từ câu hỏi loại mất MỌI tin GT (kiểm bằng SQL
   trên `v_listings`), `insufficient_context` cho llm_only (không truy xuất), còn lại `retrieval_miss`.
6. `llm_hallucination` — trích `[Tin#id]` không có trong nguồn; câu trả lời tự nói "không đủ dữ liệu" ⇒ `insufficient_context`;
   còn lại faithfulness < 0,5 (nhận định không có trong ngữ cảnh) ⇒ `llm_hallucination`.
7. `cypher_wrong_result` — Cypher chạy được nhưng câu trả lời sai.
8. `insufficient_context` — còn lại.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from vre.common import result_header, write_json_atomic
from vre.config import Settings
from vre.eval.questions import Question

log = logging.getLogger(__name__)

CAUSES = ("runtime_error", "router_wrong", "cypher_error", "cypher_wrong_result", "filter_wrong", "retrieval_miss",
          "llm_hallucination", "refusal_missing", "insufficient_context")
FILTER_SYSTEMS = ("basic_rag", "hybrid")
HALLUCINATION_FAITHFULNESS = 0.5
EXAMPLES_PER_CAUSE = 3
EXCERPT_CHARS = 300
SCHEMA_VERSION = 1
NO_DATA = re.compile(r"không (đủ|có) (dữ liệu|thông tin)|không tìm thấy", re.IGNORECASE)


def assign_cause(q: Question, answer: dict, score: dict,
                 filters_exclude_gt: Callable[[Question], bool] = lambda _q: False) -> str | None:
    """Nguyên nhân cho một (câu, hệ thống) chưa đạt; None khi đạt 2 điểm hoặc chưa chấm được."""
    points = score.get("points")
    if points is None or points == 2:
        return None
    system, text, error = answer["system"], (answer.get("answer") or "").strip(), answer.get("error")
    if error and not text:
        return "runtime_error"
    if q.category == "unanswerable":
        return "refusal_missing"
    if system == "hybrid" and answer.get("route") and answer["route"] != q.route_gold:
        return "router_wrong"
    cypher = answer.get("cypher")
    if cypher and error:
        return "cypher_error"
    if q.gt_kind == "ids":
        if system == "llm_only":
            return "insufficient_context"
        if not score.get("recall_at_5") and system in FILTER_SYSTEMS and filters_exclude_gt(q):
            return "filter_wrong"
        return "retrieval_miss"
    if answer.get("unsupported_citations"):
        return "llm_hallucination"
    if NO_DATA.search(text):  # tự nói thiếu dữ liệu: nhận định đó không có trong ngữ cảnh nhưng không phải bịa
        return "insufficient_context"
    faithfulness = score.get("faithfulness")
    if faithfulness is not None and faithfulness < HALLUCINATION_FAITHFULNESS:
        return "llm_hallucination"
    if cypher:
        return "cypher_wrong_result"
    return "insufficient_context"


def filter_checker(settings: Settings, gt: dict[str, dict]) -> Callable[[Question], bool]:
    """Hàm (câu) → True khi bộ lọc LLM trích từ câu hỏi (cùng prompt ⇒ trúng cache LLM của lượt eval) loại mọi tin GT.
    Kết nối DuckDB mở lười, một lần; kết quả nhớ theo id câu."""
    from vre.data.duck import connect
    from vre.eval.runner import with_backoff
    from vre.rag.filters import parse_filters, to_sql

    memo: dict[str, bool] = {}
    con: Any = None

    def check(q: Question) -> bool:
        nonlocal con
        if q.id in memo:
            return memo[q.id]
        ids = [int(i) for i in gt.get(q.id, {}).get("gt_ids") or []]
        excluded = False
        if ids:
            filters, _ = with_backoff(lambda: parse_filters(q.question))
            where, params = to_sql(filters)
            if where != "TRUE":
                if con is None:
                    con = connect(settings)
                    con.execute("SET memory_limit = '1GB'")
                    con.execute("SET threads = 2")
                kept = con.execute(f"SELECT count(*) FROM v_listings WHERE listing_id IN (SELECT unnest(?::BIGINT[])) "
                                   f"AND {where}", [ids, *params]).fetchone()
                excluded = kept is not None and int(kept[0]) == 0
        memo[q.id] = excluded
        return excluded

    return check


def build(questions: dict[str, Question], gt: dict[str, dict], answers: dict[tuple[str, str], dict],
          scores: dict[tuple[str, str], dict], filters_exclude_gt: Callable[[Question], bool]) -> dict:
    """{counts:{system:{cause:n}}, examples:[…]} — ví dụ: tối đa 3 mỗi nguyên nhân, ưu tiên hệ thống khác llm_only và
    hệ thống chưa có ví dụ cho nguyên nhân đó (llm_only sai câu tra cứu theo thiết kế, sẽ lấn hết ví dụ)."""
    counts: dict[str, dict[str, int]] = {}
    by_cause: dict[str, list[tuple[Question, dict]]] = {}
    for key in sorted(scores):
        q, answer = questions[key[0]], answers.get(key)
        if answer is None:
            continue
        cause = assign_cause(q, answer, scores[key], filters_exclude_gt)
        if cause is None:
            continue
        counts.setdefault(answer["system"], {}).setdefault(cause, 0)
        counts[answer["system"]][cause] += 1
        by_cause.setdefault(cause, []).append((q, answer))
    examples = []
    for cause in CAUSES:
        rows = sorted(by_cause.get(cause, []), key=lambda qa: (qa[1]["system"] == "llm_only", qa[0].id))
        picked: list[tuple[Question, dict]] = []
        for q, answer in rows:  # lượt 1: mỗi hệ thống một ví dụ; lượt 2: lấp cho đủ
            if len(picked) < EXAMPLES_PER_CAUSE and answer["system"] not in {a["system"] for _, a in picked}:
                picked.append((q, answer))
        for q, answer in rows:
            if len(picked) < EXAMPLES_PER_CAUSE and (q, answer) not in picked:
                picked.append((q, answer))
        examples += [{"id": q.id, "system": a["system"], "category": q.category, "cause": cause, "question": q.question,
                      "answer_excerpt": (a.get("answer") or a.get("error") or "")[:EXCERPT_CHARS],
                      "gt_value": gt.get(q.id, {}).get("gt_value"), "cypher": a.get("cypher")} for q, a in picked]
    return {"counts": counts, "examples": examples}


def run(settings: Settings) -> dict:
    """`eval report` (phần lỗi): đọc answers/scores thô, ghi `results/error_analysis.json`."""
    from vre.eval.ground_truth import load_gt
    from vre.eval.questions import load_questions
    from vre.eval.runner import answers_path, read_jsonl, scores_path

    questions = {q.id: q for q in load_questions()}
    gt = load_gt(settings)
    doc = {**result_header(SCHEMA_VERSION),
           **build(questions, gt, read_jsonl(answers_path(settings)), read_jsonl(scores_path(settings)),
                   filter_checker(settings, gt))}
    write_json_atomic(settings.results_dir / "error_analysis.json", doc)
    return doc
