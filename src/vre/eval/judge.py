"""Chấm bằng LLM (mô hình `Settings.judge_model`): điểm 0–2 theo đáp án chuẩn và các chỉ số kiểu RAGAS
(chuyển nguyên từ `src/eval.py`)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from vre.config import get_settings
from vre.eval.metrics import _align
from vre.llm.client import ask_llm


class Verdict(BaseModel):
    score: Literal[0, 1, 2]
    reason: str


JUDGE_SYSTEM = """Bạn chấm câu trả lời theo ĐÁP ÁN CHUẨN, thang 0–2:
- 2 (đúng): nêu cùng thực thể/giá trị với đáp án chuẩn (cho phép làm tròn, sai số tương đối tối đa 5%,
  cách viết khác nhau); với đáp án nhiều phần thì đủ mọi phần.
- 1 (đúng một phần): đúng thực thể nhưng giá trị lệch 5–20%, hoặc chỉ đúng một phần của đáp án nhiều phần
  (ví dụ 3/5 quận trong danh sách).
- 0 (sai): nêu thực thể khác, giá trị lệch hơn 20%, hoặc nói không đủ dữ liệu."""

REFUSAL_SYSTEM = """Câu hỏi này cố ý KHÔNG có đáp án trong dữ liệu tin đăng bất động sản (tháng 6/2025).
Chấm thang 0–2:
- 2: nói rõ không có hoặc không đủ dữ liệu để trả lời, không đưa ra con số hay khẳng định bịa.
- 1: có nói dữ liệu hạn chế nhưng vẫn đưa ra ước đoán, con số hoặc thông tin ngoài dữ liệu.
- 0: trả lời như thể có dữ liệu (đưa con số, tên, khẳng định cụ thể)."""


def judge_answer(question: str, answer: str, gt_value: str | None) -> dict:
    """{"score": 0|1|2, "reason": ...}. gt_value=None: câu không có đáp án, chấm việc từ chối đúng."""
    if gt_value is None:
        prompt, system = f"CÂU HỎI: {question}\nCÂU TRẢ LỜI: {answer}", REFUSAL_SYSTEM
    else:
        prompt, system = f"CÂU HỎI: {question}\nĐÁP ÁN CHUẨN: {gt_value}\nCÂU TRẢ LỜI: {answer}", JUDGE_SYSTEM
    return ask_llm(prompt, system=system, output=Verdict, model=get_settings().judge_model).data


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
    claims = ask_llm(f"Tách câu trả lời sau thành các nhận định độc lập, ngắn:\n{answer}", output=Claims,
                     model=judge).data["claims"]
    if not claims:
        return []
    listed = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    check = ask_llm(f"NGỮ CẢNH:\n{chr(10).join(contexts)}\n\nNHẬN ĐỊNH:\n{listed}\n\n"
                    "Với mỗi nhận định theo thứ tự, trả true nếu ngữ cảnh hỗ trợ trực tiếp.", output=ClaimCheck,
                    model=judge)
    return _align(check.data["supported"], len(claims))


def answer_relevancy(question: str, answer: str, n: int = 3) -> float:
    """RAGAS: sinh n câu hỏi ngược từ câu trả lời, lấy cosine trung bình với câu hỏi gốc (E5)."""
    from vre.index.embed import embed_query

    generated = ask_llm(f"Viết {n} câu hỏi mà câu trả lời sau trả lời được:\n{answer}",
                        output=Questions, model=get_settings().judge_model).data["questions"][:n]
    if not generated:
        return 0.0
    q = embed_query(question)
    return float(np.mean([q @ embed_query(g) for g in generated]))


def context_precision(question: str, contexts: list[str], reference: str) -> float:
    """RAGAS: trung bình precision@i tại các vị trí ngữ cảnh liên quan (ngữ cảnh đúng nên đứng đầu)."""
    if not contexts:
        return 0.0
    listed = "\n\n".join(f"[{i + 1}] {c[:1500]}" for i, c in enumerate(contexts))
    verdicts = ask_llm(f"CÂU HỎI: {question}\nĐÁP ÁN THAM CHIẾU: {reference}\n\nNGỮ CẢNH:\n{listed}\n\n"
                       "Với mỗi ngữ cảnh theo thứ tự, trả true nếu nó hữu ích để ra đáp án tham chiếu.",
                       output=ContextVerdicts, model=get_settings().judge_model).data["relevant"]
    verdicts = _align(verdicts, len(contexts))
    hits, total = 0, 0.0
    for i, relevant in enumerate(verdicts, start=1):
        if relevant:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0
