"""Chấm theo luật, KHÔNG gọi LLM (`eval score --offline`): dùng khi không còn hạn mức/ngân sách cho giám khảo LLM.

Luật (ghi nguyên văn vào `eval_summary.scoring_rules`):
- `gt_kind=ids`: Recall@5 capped theo `dup_cluster_id` trên nguồn của hệ thống; 2 điểm nếu ≥ 0,8, 1 nếu > 0, 0 nếu không.
- `gt_kind=value`, đáp án số: có số nào trong câu trả lời lệch tương đối ≤ `value_tolerance` (mặc định 5%) so với `gt_value`
  ⇒ 2; ≤ max(20%, 2×tolerance) ⇒ 1; không ⇒ 0. Luật dễ dãi (bất kỳ số nào khớp), nêu rõ khi diễn giải.
- `gt_kind=value`, đáp án chữ (tên quận, dự án…): so sau khi bỏ dấu, viết thường: câu trả lời chứa đáp án ⇒ 2, không ⇒ 0.
- định giá (`category=valuation`): số đầu tiên đi kèm "tỷ" (hoặc "triệu", đổi sang tỷ) là giá dự báo; lệch ≤ tolerance ⇒ 2,
  ≤ 2×tolerance ⇒ 1.
- `refusal`: câu trả lời chứa cụm từ chối ("không có dữ liệu", "không đủ thông tin", "ngoài phạm vi"…) ⇒ 2, không ⇒ 0.
- Câu trả lời rỗng ⇒ 0. Không có faithfulness (RAGAS cần LLM) ⇒ `faithfulness = null`.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from vre.eval import metrics as M
from vre.eval.questions import Question

FULL_RECALL = 0.8
DEFAULT_TOLERANCE = 0.05
PARTIAL_TOLERANCE = 0.20
SCORING = "rule_based"
RULES = [
    "ids: capped cluster Recall@5 over the system's sources; 2 if >= 0.8, 1 if > 0, else 0",
    "value (numeric GT): any number in the answer within value_tolerance (default 5%) of gt_value => 2; "
    "within max(20%, 2x tolerance) => 1; else 0 (lenient any-number match)",
    "value (numeric GT < 0, e.g. '% lower'): an answer number matches by absolute value only if the answer also says "
    "thấp hơn/giảm/rẻ hơn/kém",
    "value (text GT not in the question): diacritic-folded, lower-cased answer contains gt_value => 2, else 0",
    "value (text GT that the question itself names, i.e. comparisons): gt_value must appear within 60 characters before "
    "the first 'cao/đắt/lớn/nhiều hơn' in the answer => 2, else 0",
    "valuation: first amount or range ('a đến b tỷ' => midpoint) with unit 'tỷ'/'triệu' (converted to tỷ) "
    "within value_tolerance => 2, within 2x => 1",
    "refusal: answer contains a refusal phrase (no data / insufficient information / out of range) => 2, else 0",
    "empty answer => 0; faithfulness = null (needs an LLM)",
]
REFUSAL_PHRASES = (
    "khong co du lieu", "khong du du lieu", "khong co thong tin", "khong du thong tin", "khong tim thay",
    "ngoai pham vi", "khong the tra loi", "khong co trong du lieu", "chua co du lieu", "khong co so lieu",
    "khong the cung cap", "khong co du thong tin",
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_NUM = r"(\d+(?:[.,]\d+)*)"
_MONEY = re.compile(_NUM + r"(?:\s*(?:-|–|den|toi)\s*" + _NUM + r")?\s*(ty|trieu)\b")
UP_WORDS = re.compile(r"\b(cao hon|dat hon|lon hon|nhieu hon)\b")
DOWN_WORDS = re.compile(r"\b(thap hon|giam|re hon|kem)\b")
COMPARE_WINDOW = 60


def fold(text: str) -> str:
    """Bỏ dấu tiếng Việt (kể cả đ), viết thường, bỏ ký tự nhấn mạnh markdown, gộp khoảng trắng."""
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[*_#`]", "", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def parse_number(token: str) -> float | None:
    """Số kiểu Việt/Anh: nhóm 3 chữ số sau dấu ⇒ phân cách nghìn ('1.234.567', '2.500'); một dấu với 1–2 chữ số sau
    ⇒ thập phân ('91,26', '86.32'); còn lại (vd '1,5,7') ⇒ None."""
    parts = re.split(r"[.,]", token)
    if len(parts) == 1:
        return float(token)
    if all(len(p) == 3 for p in parts[1:]):
        return float("".join(parts))
    if len(parts) == 2:
        return float(f"{parts[0]}.{parts[1]}")
    return None


def numbers(text: str) -> list[float]:
    out = []
    for token in _NUMBER.findall(text):
        value = parse_number(token)
        if value is not None:
            out.append(value)
    return out


def _band(error: float, tolerance: float) -> int:
    return 2 if error <= tolerance else int(error <= max(PARTIAL_TOLERANCE, 2 * tolerance))


def _as_float(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _winner_named(answer: str, target: str) -> bool:
    """So sánh: đáp án đúng đứng trong COMPARE_WINDOW ký tự trước từ so sánh chiều lên ('cao hơn'…) đầu tiên."""
    folded = fold(answer)
    match = UP_WORDS.search(folded)
    if match is None:
        return False
    return target in folded[max(0, match.start() - COMPARE_WINDOW):match.start()]


def value_points(answer: str, gt_value: Any, tolerance: float | None, question: str = "") -> int:
    tol = DEFAULT_TOLERANCE if tolerance is None else float(tolerance)
    truth = _as_float(gt_value)
    if truth is None:  # đáp án chữ
        target = fold(str(gt_value or ""))
        if not target:
            return 0
        if target in fold(question):  # câu hỏi tự nêu đáp án (A hay B): phải chỉ đúng bên thắng
            return 2 if _winner_named(answer, target) else 0
        return 2 if target in fold(answer) else 0
    found = numbers(answer)
    if truth == 0:
        return 2 if any(n == 0 for n in found) else 0
    if truth < 0 and DOWN_WORDS.search(fold(answer)):  # "thấp hơn 13,46%" ⇔ -13,46
        found = found + [-n for n in found]
    errors = [abs(n - truth) / abs(truth) for n in found]
    return max((_band(e, tol) for e in errors), default=0)


def forecast_bn(answer: str) -> float | None:
    """Số tiền (hoặc khoảng 'a đến b' ⇒ điểm giữa) đầu tiên có đơn vị tỷ/triệu, đổi ra tỷ đồng."""
    match = _MONEY.search(fold(answer))
    if not match:
        return None
    low = parse_number(match.group(1))
    high = parse_number(match.group(2)) if match.group(2) else low
    if low is None or high is None:
        return None
    value = (low + high) / 2
    return value if match.group(3) == "ty" else value / 1000


def is_refusal(answer: str) -> bool:
    folded = fold(answer)
    return any(phrase in folded for phrase in REFUSAL_PHRASES)


def score_row(q: Question, gt: dict, row: dict, cluster_of: dict[int, int], digest: str) -> dict:
    """Một dòng scores.jsonl theo luật (cùng khoá với bản chấm LLM của `runner.score_one`, thêm `scoring`)."""
    out: dict[str, Any] = {"id": q.id, "system": row["system"], "category": q.category, "gt_kind": q.gt_kind,
                           "answer_digest": digest, "points": 0, "reason": None, "predicted_value": None,
                           "faithfulness": None, "judge_error": None, "scoring": SCORING}
    answer = (row.get("answer") or "").strip()
    sources = [int(s) for s in row.get("sources") or []]
    if q.gt_kind == "ids":
        out.update(M.retrieval_scores(sources, gt.get("gt_dup_clusters") or [], cluster_of))
        recall = out["recall_at_5"]
        out["points"] = 2 if recall >= FULL_RECALL else int(recall > 0)
        out["reason"] = f"cluster recall@5 = {recall:.2f}"
    elif not answer:
        out["reason"] = "câu trả lời rỗng"
    elif q.gt_kind == "refusal":
        refused = is_refusal(answer)
        out["points"], out["reason"] = (2 if refused else 0), ("có cụm từ chối" if refused else "không từ chối")
    elif q.category == "valuation":
        predicted = forecast_bn(answer)
        out["predicted_value"] = predicted
        out["points"] = M.value_points(predicted, float(gt["gt_value"]), float(q.value_tolerance or 0.15))
        out["reason"] = f"dự báo {predicted} tỷ so với {gt['gt_value']}"
    else:
        out["points"] = value_points(answer, gt.get("gt_value"), q.value_tolerance, q.question)
        out["reason"] = f"so với gt_value {gt.get('gt_value')}"
    out["correct"] = out["points"] == 2
    return out
