"""vre.eval.report: README chỉ thay khối giữa hai dấu mốc; bảng sinh từ results/*.json; ví dụ câu trả lời từ eval thật."""

import json

import pytest

from vre.config import load_settings
from vre.eval import report
from vre.eval.questions import load_questions


def _summary():
    cell = {"n": 2, "accuracy": 0.5, "mean_points": 1.0, "ci95": [0.0, 1.0], "n_unscored": 0}
    return {"schema_version": 1, "git_sha": "abc1234", "systems": ["basic_rag", "hybrid"], "n_questions": 2,
            "n_questions_total": 100, "llm": {"generator": "g", "judge": "j"},
            "by_category": {"aggregate": {"basic_rag": cell, "hybrid": cell}},
            "overall": {"basic_rag": cell, "hybrid": {**cell, "accuracy": 0.75}},
            "pairwise": [{"a": "hybrid", "b": "basic_rag", "scope": "overall", "metric": "accuracy", "diff": 0.25,
                          "ci95": [-0.1, 0.5], "p_value": 0.2, "n_boot": 10000, "n": 2}],
            "cost": {"hybrid": {"latency_ms_p50": 1200.0, "latency_ms_p95": 3000.0, "input_tokens_mean": 900.0,
                                "output_tokens_mean": 80.0, "usd_per_question": 0.0002, "bad_citation_rate": 0.0}},
            "router": {"accuracy": 0.5, "n": 2, "accuracy_excl_unanswerable": 1.0}, "notes": ["chưa đủ"]}


def test_replace_block_only_between_markers():
    text = f"đầu\n{report.START}\ncũ\n{report.END}\ncuối\n"
    out = report.replace_block(text, "mới\n")
    assert out == f"đầu\n{report.START}\nmới\n{report.END}\ncuối\n"
    with pytest.raises(ValueError):
        report.replace_block("không có dấu mốc", "x")
    with pytest.raises(ValueError):
        report.replace_block(f"{report.START}{report.END}{report.START}{report.END}", "x")


def test_readme_block_generated_from_results(tmp_path):
    settings = load_settings(data_root=tmp_path)
    settings.results_dir.mkdir(parents=True)
    (settings.results_dir / "eval_summary.json").write_text(json.dumps(_summary()), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_bytes(f"# Tiêu đề\r\n{report.START}\r\n{report.END}\r\nđuôi\r\n".encode())
    result = report.readme(settings, readme)
    text = readme.read_bytes().decode()
    assert result["changed"] and text.startswith("# Tiêu đề\r\n") and text.endswith("đuôi\r\n")
    assert "75.0% [0–100%]" in text and "hybrid − basic_rag" in text and "abc1234" in text
    assert "\n" not in text.replace("\r\n", "")  # giữ CRLF
    assert report.readme(settings, readme)["changed"] is False  # idempotent


def test_answer_examples_pairs_and_contrast():
    questions = {q.id: q for q in load_questions()}
    answers = {("LK01", "llm_only"): {"answer": "không rõ"}, ("LK01", "basic_rag"): {"answer": "Có 2 WC [Tin#123]"},
               ("AG01", "basic_rag"): {"answer": "sai"}, ("AG01", "graph"): {"answer": "91,26", "route": None},
               ("AG01", "hybrid"): {"answer": "Đáp [Tin#5]", "route": "graph"}}
    scores = {("LK01", "llm_only"): {"points": 0}, ("LK01", "basic_rag"): {"points": 2},
              ("AG01", "basic_rag"): {"points": 0}, ("AG01", "graph"): {"points": 2}, ("AG01", "hybrid"): {"points": 2}}
    notes: list[str] = []
    out = report.answer_examples(questions, answers, scores, notes)
    pair = [e for e in out if e["pair"] == "llm_only_vs_basic_rag"]
    assert [(e["id"], e["system"]) for e in pair] == [("LK01", "llm_only"), ("LK01", "basic_rag")]
    assert pair[1]["cited"] == [123]
    contrast = [(e["id"], e["system"], e["points"]) for e in out if e["pair"] == "basic_rag_vs_graph"]
    assert contrast == [("AG01", "basic_rag", 0), ("AG01", "graph", 2)]
    assert not notes


def test_answer_examples_notes_missing_cases():
    questions = {q.id: q for q in load_questions()}
    notes: list[str] = []
    out = report.answer_examples(questions, {("AG01", "hybrid"): {"answer": "x"}}, {("AG01", "hybrid"): {"points": 2}},
                                 notes)
    assert [(e["id"], e["pair"]) for e in out] == [("AG01", None)] and len(notes) == 2
