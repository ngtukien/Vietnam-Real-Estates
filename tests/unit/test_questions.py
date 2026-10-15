"""benchmark/questions.jsonl (data-contract §7.7) + vre.eval.questions: schema, phân bổ, id, SQL chỉ đọc, bug B6."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from test_ground_truth import synthetic_settings  # noqa: F401  (fixture: parquet giả nhỏ)

from vre.eval.questions import (
    ALLOCATION,
    MIN_ROBUSTNESS_QUESTIONS,
    QUESTIONS_PATH,
    ROBUSTNESS_TAGS,
    VIEW_NAME,
    check_sql,
    load_questions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def questions():
    return load_questions()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_path_is_repo_benchmark():
    assert QUESTIONS_PATH == REPO_ROOT / "benchmark" / "questions.jsonl"
    assert not (REPO_ROOT / "benchmark" / "questions.csv").exists(), "questions.csv cũ phải bị xoá (B6)"


def test_allocation_matches_phase_plan(questions):
    assert sum(ALLOCATION.values()) == 100
    assert dict(Counter(q.category for q in questions)) == ALLOCATION


def test_ids_unique(questions):
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids)) == 100


def test_robustness_tags(questions):
    tagged = [q.id for q in questions if ROBUSTNESS_TAGS & set(q.tags)]
    assert len(tagged) >= MIN_ROBUSTNESS_QUESTIONS


def test_every_sql_is_readonly_select_on_view(questions):
    for q in questions:
        if q.gt_kind == "refusal":
            assert q.gt_sql is None, q.id
            continue
        sql = q.gt_sql.strip().lower()
        assert sql.startswith(("select", "with")), q.id
        assert VIEW_NAME in sql, q.id
        assert "read_parquet" not in sql, q.id
        check_sql(q.gt_sql)


def test_kind_and_route_consistency(questions):
    for q in questions:
        if q.category == "unanswerable":
            assert q.gt_kind == "refusal", q.id
        if q.category in ("lookup", "constraint"):
            assert q.gt_kind == "ids", q.id
        if q.category in ("aggregate", "multihop", "temporal", "comparison"):
            assert q.gt_kind == "value", q.id
        if q.category == "valuation":
            assert q.route_gold == "valuation" and q.value_tolerance == pytest.approx(0.15), q.id
            assert q.valuation_input is not None, q.id


def test_file_lines_have_exact_contract_keys():
    allowed = {"id", "category", "route_gold", "question", "gt_kind", "gt_sql", "value_tolerance", "tags",
               "valuation_input"}
    for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert set(json.loads(line)) <= allowed


@pytest.mark.parametrize("sql", [
    "DELETE FROM v_listings",
    "SELECT * FROM read_parquet('x.parquet')",
    "SELECT 1",  # không đọc view
    "SELECT * FROM v_listings; DROP TABLE x",
    "WITH a AS (SELECT 1) INSERT INTO t SELECT * FROM v_listings",
    "COPY (SELECT * FROM v_listings) TO 'out.csv'",
])
def test_check_sql_rejects(sql):
    with pytest.raises(ValueError):
        check_sql(sql)


def test_check_sql_ignores_keywords_inside_literals():
    check_sql("SELECT listing_id FROM v_listings WHERE street = 'Set Drop Update'")


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


BASE = {"id": "X1", "category": "lookup", "route_gold": "basic_rag", "question": "q", "gt_kind": "ids",
        "gt_sql": "SELECT listing_id FROM v_listings", "value_tolerance": None, "tags": []}


def test_load_rejects_duplicate_id(tmp_path):
    with pytest.raises(ValueError, match="trùng"):
        load_questions(_write(tmp_path, [BASE, BASE]))


@pytest.mark.parametrize("patch", [
    {"extra_key": 1},
    {"category": "unanswerable"},  # refusal bắt buộc
    {"gt_kind": "value"},  # thiếu value_tolerance
    {"gt_sql": None},
    {"route_gold": "valuation"},  # chỉ cho category=valuation
    {"valuation_input": {"property_type": "Nhà"}},
])
def test_load_rejects_invalid_rows(tmp_path, patch):
    with pytest.raises(ValueError):
        load_questions(_write(tmp_path, [{**BASE, **patch}]))


def test_valuation_input_validated(tmp_path):
    row = {**BASE, "category": "valuation", "route_gold": "valuation", "gt_kind": "value", "value_tolerance": 0.15,
           "valuation_input": {"property_type": "Nhà", "province": "Hà Nội", "district": "Đống Đa", "area_m2": 0}}
    with pytest.raises(ValueError, match="area_m2"):
        load_questions(_write(tmp_path, [row]))
    row["valuation_input"]["area_m2"] = 45
    assert load_questions(_write(tmp_path, [row]))[0].valuation_input["area_m2"] == 45


def test_questions_file_untouched(tmp_path, synthetic_settings):  # noqa: F811
    """B6: `eval gt` chỉ ghi results/benchmark_gt.json, không bao giờ ghi file câu hỏi."""
    from vre.eval.ground_truth import gt_path, run

    real_before = _sha256(QUESTIONS_PATH)
    small = _write(tmp_path, [{**BASE, "gt_sql": "SELECT listing_id FROM v_listings WHERE listing_id = 1000"}])
    small_before = _sha256(small)
    run(synthetic_settings, questions_path=small)
    assert gt_path(synthetic_settings).exists()
    with pytest.raises(RuntimeError):  # bộ thật trên dữ liệu giả: nhiều câu không có đáp án ⇒ dừng
        run(synthetic_settings)
    assert _sha256(small) == small_before
    assert _sha256(QUESTIONS_PATH) == real_before
