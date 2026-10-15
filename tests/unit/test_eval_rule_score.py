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
