"""Chấm bằng LLM (mô hình `Settings.judge_model`): điểm 0–2 theo đáp án chuẩn, việc từ chối, số dự báo định giá
và các chỉ số kiểu RAGAS (chuyển từ `src/eval.py`, mở rộng Phase 3).

Bug B2: eval từ chối chạy khi giám khảo = mô hình sinh (`check_judge_config`), trừ khi `VRE_ALLOW_SAME_JUDGE=1`.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
from pydantic import BaseModel

from vre.config import Settings, get_settings
from vre.eval.metrics import _align
from vre.llm.client import ask_llm

ALLOW_SAME_JUDGE_ENV = "VRE_ALLOW_SAME_JUDGE"


def check_judge_config(settings: Settings | None = None, judge_used: bool = True) -> dict:
    """Mô hình sinh/giám khảo cho `eval_summary.llm`. Raise RuntimeError khi `JUDGE_MODEL == LLM_MODEL` (giám khảo
    tự chấm câu của chính nó — thiên lệch) trừ khi `VRE_ALLOW_SAME_JUDGE=1`; khi được phép thì ghi lại cờ.
    `judge_used=False` (chấm theo luật, không có giám khảo LLM) ⇒ không kiểm, `judge` ghi null."""
    s = settings or get_settings()
    if not judge_used:
        return {"provider": s.llm_provider, "generator": s.llm_model, "judge": None,
                "generator_fallbacks": s.llm_fallback_list, "same_judge_allowed": False}
    same = s.judge_model == s.llm_model
    allowed = os.getenv(ALLOW_SAME_JUDGE_ENV, "") == "1"
    if same and not allowed:
        raise RuntimeError(f"JUDGE_MODEL trùng LLM_MODEL ({s.llm_model!r}): đặt JUDGE_MODEL khác, hoặc "
                           f"{ALLOW_SAME_JUDGE_ENV}=1 nếu chấp nhận giám khảo tự chấm (ghi vào eval_summary.llm)")
    info = {"provider": s.llm_provider, "generator": s.llm_model, "judge": s.judge_model,
            "generator_fallbacks": s.fallbacks_for(s.llm_model), "configured_fallbacks": s.llm_fallback_list,
            "same_judge_allowed": bool(same and allowed)}
    if s.judge_model in s.llm_fallback_list:  # `fallbacks_for` đã bỏ: câu trả lời không bao giờ do giám khảo sinh
        info["note"] = "judge model bị bỏ khỏi generator_fallbacks (429 trên model chính ⇒ chờ/thử lại, không đổi model)"
    return info


class Verdict(BaseModel):
    score: Literal[0, 1, 2]
    reason: str


DEFAULT_TOLERANCE = 0.05
PARTIAL_TOLERANCE = 0.20

JUDGE_TEMPLATE = """Bạn chấm câu trả lời theo ĐÁP ÁN CHUẨN, thang 0–2:
- 2 (đúng): nêu cùng thực thể/giá trị với đáp án chuẩn (cho phép làm tròn, sai số tương đối tối đa {tol}%,
  cách viết khác nhau); với đáp án nhiều phần thì đủ mọi phần.
- 1 (đúng một phần): đúng thực thể nhưng giá trị lệch {tol}–{partial}%, hoặc chỉ đúng một phần của đáp án nhiều phần
  (ví dụ 3/5 quận trong danh sách).
- 0 (sai): nêu thực thể khác, giá trị lệch hơn {partial}%, hoặc nói không đủ dữ liệu.
CHI TIẾT là các cột phụ của đáp án chuẩn (cỡ mẫu n, thành phần) — chỉ để hiểu đáp án, không bắt buộc phải nêu."""


def judge_system(tolerance: float | None) -> str:
    """Prompt giám khảo với ngưỡng sai số của câu hỏi (`value_tolerance`; None ⇒ 5%)."""
    tol = DEFAULT_TOLERANCE if tolerance is None else float(tolerance)
    partial = max(PARTIAL_TOLERANCE, 2 * tol)
    return JUDGE_TEMPLATE.format(tol=f"{tol * 100:g}", partial=f"{partial * 100:g}")


JUDGE_SYSTEM = judge_system(None)

REFUSAL_SYSTEM = """Câu hỏi này cố ý KHÔNG có đáp án trong dữ liệu tin đăng bất động sản (06/2025–03/2026).
Chấm thang 0–2:
- 2: nói rõ không có hoặc không đủ dữ liệu để trả lời, không đưa ra con số hay khẳng định bịa.
- 1: có nói dữ liệu hạn chế nhưng vẫn đưa ra ước đoán, con số hoặc thông tin ngoài dữ liệu.
- 0: trả lời như thể có dữ liệu (đưa con số, tên, khẳng định cụ thể)."""


def _data(data: dict | None) -> dict:
    if data is None:
        raise RuntimeError("giám khảo không trả structured output")
    return data


def judge_answer(question: str, answer: str, gt_value: str | None, tolerance: float | None = None,
                 details: list[dict] | None = None) -> dict:
    """{"score": 0|1|2, "reason": ...}. gt_value=None: câu không có đáp án, chấm việc từ chối đúng.
    `tolerance` = `value_tolerance` của câu; `details` = `gt_rows` (cột phụ của đáp án chuẩn)."""
    if gt_value is None:
        prompt, system = f"CÂU HỎI: {question}\nCÂU TRẢ LỜI: {answer}", REFUSAL_SYSTEM
    else:
        extra = f"\nCHI TIẾT: {details}" if details else ""
        prompt = f"CÂU HỎI: {question}\nĐÁP ÁN CHUẨN: {gt_value}{extra}\nCÂU TRẢ LỜI: {answer}"
        system = judge_system(tolerance)
    return _data(ask_llm(prompt, system=system, output=Verdict, model=get_settings().judge_model).data)


class PriceForecast(BaseModel):
    price_bn: float | None


PRICE_SYSTEM = """Đọc câu trả lời định giá bất động sản và trích GIÁ DỰ BÁO CHÍNH của căn được hỏi, đơn vị TỶ đồng
(ví dụ "5,12 tỷ" → 5.12; "850 triệu" → 0.85). Nếu chỉ có khoảng giá thì lấy điểm giữa. KHÔNG lấy giá của tin so sánh,
giá chào của người hỏi hay đơn giá/m². Không có giá dự báo ⇒ null."""


def extract_price_bn(answer: str) -> float | None:
    """Số dự báo (tỷ đồng) trong câu trả lời định giá; chấm số học so với GT ở `metrics.value_points`."""
    data = _data(ask_llm(f"CÂU TRẢ LỜI:\n{answer}", system=PRICE_SYSTEM, output=PriceForecast,
                         model=get_settings().judge_model).data)
    value = data.get("price_bn")
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


class Claims(BaseModel):
    claims: list[str]


class ClaimCheck(BaseModel):
    supported: list[bool]


class Questions(BaseModel):
    questions: list[str]


class ContextVerdicts(BaseModel):
    relevant: list[bool]


def claims_supported(answer: str, contexts: list[str]) -> list[bool]:
    """Faithfulness theo định nghĩa RAGAS: tách câu trả lời thành nhận định, kiểm từng nhận định
    có được ngữ cảnh hỗ trợ không."""
    judge = get_settings().judge_model
    split = ask_llm(f"Tách câu trả lời sau thành các nhận định độc lập, ngắn:\n{answer}", output=Claims, model=judge)
    claims: list[str] = _data(split.data)["claims"]
    if not claims:
        return []
    listed = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    check = ask_llm(f"NGỮ CẢNH:\n{chr(10).join(contexts)}\n\nNHẬN ĐỊNH:\n{listed}\n\n"
                    "Với mỗi nhận định theo thứ tự, trả true nếu ngữ cảnh hỗ trợ trực tiếp.", output=ClaimCheck,
                    model=judge)
    return _align(_data(check.data)["supported"], len(claims))


def answer_relevancy(question: str, answer: str, n: int = 3) -> float:
    """RAGAS: sinh n câu hỏi ngược từ câu trả lời, lấy cosine trung bình với câu hỏi gốc (encoder truy vấn
    e5-large của RAG, nạp một lần mỗi tiến trình; vector đã chuẩn hoá L2 nên tích vô hướng = cosine)."""
    from vre.rag.query_encoder import embed as embed_query

    reply = ask_llm(f"Viết {n} câu hỏi mà câu trả lời sau trả lời được:\n{answer}",
                    output=Questions, model=get_settings().judge_model)
    generated: list[str] = _data(reply.data)["questions"][:n]
    if not generated:
        return 0.0
    q = np.asarray(embed_query(question))
    return float(np.mean([q @ np.asarray(embed_query(g)) for g in generated]))


def context_precision(question: str, contexts: list[str], reference: str) -> float:
    """RAGAS: trung bình precision@i tại các vị trí ngữ cảnh liên quan (ngữ cảnh đúng nên đứng đầu)."""
    if not contexts:
        return 0.0
    listed = "\n\n".join(f"[{i + 1}] {c[:1500]}" for i, c in enumerate(contexts))
    reply = ask_llm(f"CÂU HỎI: {question}\nĐÁP ÁN THAM CHIẾU: {reference}\n\nNGỮ CẢNH:\n{listed}\n\n"
                    "Với mỗi ngữ cảnh theo thứ tự, trả true nếu nó hữu ích để ra đáp án tham chiếu.",
                    output=ContextVerdicts, model=get_settings().judge_model)
    verdicts = _align(_data(reply.data)["relevant"], len(contexts))
    hits, total = 0, 0.0
    for i, relevant in enumerate(verdicts, start=1):
        if relevant:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0
