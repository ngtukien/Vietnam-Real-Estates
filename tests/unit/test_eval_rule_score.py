"""vre.eval.rule_score: chấm theo luật không LLM — số kiểu Việt, đáp án chữ bỏ dấu, từ chối, định giá, ids theo cụm."""

import pytest

from vre.eval import rule_score as R
from vre.eval.questions import load_questions

QUESTIONS = load_questions()


def _q(predicate):
    return next(q for q in QUESTIONS if predicate(q))


@pytest.mark.parametrize(("token", "value"), [("91,26", 91.26), ("86.32", 86.32), ("1.234.567", 1234567.0),
                                              ("2.500", 2500.0), ("12", 12.0), ("1,5,7", None)])
def test_parse_number(token, value):
    assert R.parse_number(token) == value


def test_fold_removes_diacritics():
    assert R.fold("Quận  Tây Hồ, Đà Nẵng") == "quan tay ho, da nang"


def test_value_points_numeric_and_text():
    assert R.value_points("khoảng 107,14 triệu/m²", "107.14", 0.05) == 2
    assert R.value_points("khoảng 120 triệu/m²", "107.14", 0.05) == 1  # lệch 12% ≤ 20%
    assert R.value_points("khoảng 200 triệu/m²", "107.14", 0.05) == 0
    assert R.value_points("Cao nhất là quận Tay Ho", "Tây Hồ", None) == 2
    assert R.value_points("Cao nhất là Ba Đình", "Tây Hồ", None) == 0


def test_forecast_bn_units():
    assert R.forecast_bn("Giá dự báo khoảng 5,12 tỷ (khoảng 4,8–5,4 tỷ)") == 5.12
    assert R.forecast_bn("Khoảng 850 triệu") == 0.85
    assert R.forecast_bn("không rõ") is None


def test_score_row_kinds():
    ids_q = _q(lambda q: q.gt_kind == "ids")
    row = {"system": "basic_rag", "answer": "x", "sources": [1, 2, 3, 4, 9]}
    out = R.score_row(ids_q, {"gt_dup_clusters": [1, 2, 3, 4, 5]}, row, {}, "d")
    assert out["recall_at_5"] == 0.8 and out["points"] == 2 and out["scoring"] == "rule_based"
    assert R.score_row(ids_q, {"gt_dup_clusters": [7]}, row, {}, "d")["points"] == 0
    refusal_q = _q(lambda q: q.gt_kind == "refusal")
    assert R.score_row(refusal_q, {}, {"system": "hybrid", "answer": "Không có dữ liệu cho năm 2024."}, {}, "d")["points"] == 2
    assert R.score_row(refusal_q, {}, {"system": "hybrid", "answer": "Có 1.200 tin."}, {}, "d")["points"] == 0
    val_q = _q(lambda q: q.category == "valuation")
    out = R.score_row(val_q, {"gt_value": "5.12"}, {"system": "hybrid", "answer": "Ước tính 5,3 tỷ đồng."}, {}, "d")
    assert out["points"] == 2 and out["predicted_value"] == 5.3 and out["faithfulness"] is None
    empty = R.score_row(_q(lambda q: q.category == "aggregate"), {"gt_value": "1"}, {"system": "graph", "answer": ""}, {}, "d")
    assert empty["points"] == 0 and empty["correct"] is False


def test_comparison_needs_winner_before_comparative():
    q = "Giá/m² căn hộ ở Cầu Giấy hay Nam Từ Liêm cao hơn?"
    assert R.value_points("Nhìn chung, giá/m² tại **Cầu Giấy cao hơn** so với Nam Từ Liêm.", "Cầu Giấy", None, q) == 2
    assert R.value_points("Nam Từ Liêm cao hơn Cầu Giấy.", "Cầu Giấy", None, q) == 0
    assert R.value_points("Cầu Giấy và Nam Từ Liêm tương đương.", "Cầu Giấy", None, q) == 0


def test_negative_truth_needs_down_word():
    assert R.value_points("thấp hơn 13,46% so với toàn thành phố", "-13.46", 0.05) == 2
    assert R.value_points("cao hơn 13,46% so với toàn thành phố", "-13.46", 0.05) == 0


def test_forecast_range_midpoint():
    assert R.forecast_bn("dao động khoảng **3,5 đến 6,5 tỷ đồng**") == 5.0
    assert R.forecast_bn("từ 4,68 - 8,59 tỷ") == pytest.approx(6.635)


@pytest.mark.parametrize("answer", ["Tháng 07/2025 có nhiều tin nhất.", "Nhiều nhất là 7/2025 (312 tin).",
                                    "Đó là **tháng 7/2025**.", "tháng 7 năm 2025 dẫn đầu", "Tháng 2025-07."])
def test_month_answers_normalised(answer):  # TM02: GT "2025-07"
    assert R.value_points(answer, "2025-07", 0.05) == 2


def test_month_answer_wrong_month_scores_zero():
    assert R.value_points("Tháng 8/2025 có nhiều tin nhất.", "2025-07", 0.05) == 0
    assert R.value_points("Tháng 17/2025", "2025-07", 0.05) == 0
    assert R.answer_months("từ 12/2024 đến tháng 3 năm 2025") == {(2024, 12), (2025, 3)}


def test_refusal_scores_zero_on_answerable_question():  # CP02: nhắc tên đáp án nhưng từ chối
    cp = _q(lambda q: q.category == "comparison")
    refusal = {"system": "llm_only", "answer": "Tôi không đủ dữ liệu để so sánh giá/m² giữa Quận 3 và Phú Nhuận."}
    out = R.score_row(cp, {"gt_value": "Quận 3"}, refusal, {}, "d")
    assert out["points"] == 0 and out["reason"] == "từ chối ở câu có đáp án"
    val_q = _q(lambda q: q.category == "valuation")
    hedge = {"system": "llm_only", "answer": "Không có dữ liệu để định giá, có thể khoảng 5,3 tỷ."}
    assert R.score_row(val_q, {"gt_value": "5.3"}, hedge, {}, "d")["points"] == 0


def test_comparison_category_uses_winner_rule_with_abbreviated_question():  # CP02 hỏi "q3 hay phu nhuan"
    assert R.value_points("Phú Nhuận đắt hơn Quận 3.", "Quận 3", None, "nha o q3 hay phu nhuan dat hon",
                          comparison=True) == 0
    assert R.value_points("Quận 3 đắt hơn Phú Nhuận.", "Quận 3", None, "nha o q3 hay phu nhuan dat hon",
                          comparison=True) == 2
