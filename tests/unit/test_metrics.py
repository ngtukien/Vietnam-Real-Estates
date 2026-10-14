"""vre.eval: chỉ số thuần, đáp án chuẩn, chấm điểm, ca lỗi, modularity (port từ tests/test_presentation_notebooks.py).

Không gọi LLM thật: giám khảo được thay bằng mock.
"""

from unittest import mock

import pandas as pd
import pytest

from vre.eval import judge, metrics, runner
from vre.graph import communities
from vre.llm.client import LLMResult


def test_recall_mrr_faithfulness():
    assert metrics.recall_at_k(["d4", "d2", "d1", "d9", "d3"], {"d1", "d4", "d7"}, 5) == pytest.approx(2 / 3)
    runs = [(["a", "x", "y"], {"a"}), (["x", "y", "b"], {"b"}), (["x", "c", "y"], {"c"})]
    assert metrics.mrr(runs) == pytest.approx((1 + 1 / 3 + 1 / 2) / 3)
    assert metrics.faithfulness([True, True, True, False]) == 0.75
    # capped: 10 tin đúng, lấy được 5 trong top-5 -> 1,0
    assert metrics.recall_at_k(list(range(5)), set(range(10)), 5, capped=True) == 1.0


def test_modularity_toy():
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    assert communities.modularity(edges, [{0, 1, 2}, {3, 4, 5}]) == pytest.approx(5 / 14)


def test_community_examples_come_from_main_province():
    index = pd.MultiIndex.from_tuples([("Bình Dương", "Dĩ An", "An Bình"), ("Hà Nội", "Ba Đình", "Giảng Võ"),
                                       ("Hà Nội", "Tây Hồ", "Nhật Tân")], names=["province", "district", "ward"])
    prof = pd.DataFrame({"median_price_m2": [40.0, 250.0, 240.0], "community": [0, 0, 0]}, index=index)
    row = communities.community_table(prof).iloc[0]
    assert (row["tỉnh chính"], row["ví dụ"]) == ("Hà Nội", "Giảng Võ (Ba Đình), Nhật Tân (Tây Hồ)")


def test_ground_truth_rules():
    df = pd.DataFrame({"listing_id": [1, 2, 3, 4], "district": ["A", "A", "B", "B"],
                       "price_m2_mil": [10.0, 20.0, 50.0, None], "type": ["x", "y", "x", "x"]})
    assert runner.ground_truth(df, {"query": "type == 'x'"}) == ([1, 3, 4], None)
    assert runner.ground_truth(df, {"query": "type == 'x'", "agg": "count"})[1] == "3"
    rule = {"groupby": ["district"], "value": "price_m2_mil", "agg": "mean", "pick": "min"}
    assert runner.ground_truth(df, rule)[1] == "A (15.00)"
    assert runner.ground_truth(df, {**rule, "min_count": 2})[1] == "A (15.00)"
    assert runner.ground_truth(df, {"groupby": ["district"], "agg": "count", "pick": "top5"})[1] == "A (2); B (2)"
    assert runner.ground_truth(df, {"unanswerable": True}) == ([], None)


def test_score_points():
    questions = pd.DataFrame({"id": ["L", "A", "U"], "type": ["lookup", "aggregate", "unanswerable"],
                              "gt_ids": [[1, 2], [], []], "gt_value": [None, "5", None]})
    results = pd.DataFrame({"id": ["L", "L", "A", "U", "U"], "system": ["a", "b", "a", "a", "b"],
                            "type": ["lookup", "lookup", "aggregate", "unanswerable", "unanswerable"],
                            "question": "q", "answer": ["x", "x", "gần 5", "không đủ dữ liệu", ""],
                            "sources": [[1, 2], [2, 9], [], [], []]})
    calls = []

    def fake_judge(question, answer, gt_value):
        calls.append(gt_value)
        return {"score": 1, "reason": ""}

    with mock.patch.object(runner, "judge_answer", fake_judge):
        scored = runner.score(results, questions, k=5)
    assert scored["points"].tolist() == [2, 1, 1, 1, 0]  # câu trả lời rỗng: 0, không gọi giám khảo
    assert scored["correct"].tolist() == [True, False, False, False, False]
    assert calls == ["5", None]  # câu không có đáp án được chấm theo việc từ chối
    assert runner.accuracy_table(scored, "points").index.tolist() == ["lookup", "aggregate", "unanswerable"]


def test_error_cases():
    scored = pd.DataFrame({"id": ["U", "U", "A"], "type": ["unanswerable", "unanswerable", "aggregate"],
                           "system": ["llm_only", "graph", "basic_rag"], "question": "q", "route": None,
                           "route_gold": "graph", "answer": ["bịa", "", "sai"], "cypher": None,
                           # dtype=object: pandas 3 suy ra dtype str và đổi None thành NaN (NaN là truthy trong
                           # error_cases). Lỗi logic cũ, Phase 3 sửa; ở đây giữ đúng ý test gốc.
                           "error": pd.Series([None, "TimeoutError: x", None], dtype=object), "correct": False})
    out = runner.error_cases(scored, n=5)
    assert out["system"].tolist() == ["graph", "basic_rag"]  # llm_only bị loại
    assert out["nguyên nhân"].tolist() == ["lỗi khi chạy: TimeoutError: x",
                                           "truy xuất trượt hoặc ngữ cảnh không đủ để tính"]


def test_judge_verdicts_are_aligned():
    assert metrics._align([True], 3) == [True, False, False]
    assert metrics._align([True, True, True], 2) == [True, True]
    replies = iter([{"claims": ["a", "b", "c"]}, {"supported": [True]}])

    def fake(*_a, **_k):
        return LLMResult("", next(replies), 1, 1, False)

    with mock.patch.object(judge, "ask_llm", fake):
        assert metrics.faithfulness(judge.claims_supported("x", ["ctx"])) == pytest.approx(1 / 3)
