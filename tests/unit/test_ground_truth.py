"""vre.eval.ground_truth trên parquet giả nhỏ: ids không rỗng, value đúng 1 dòng, refusal gt_sql null, file kết quả."""

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from vre.config import load_settings
from vre.eval.ground_truth import LOOKUP_MAX_CLUSTERS, MIN_GROUP_N, compute_item, format_value, gt_path, load_gt, run
from vre.eval.questions import Question


def _frame() -> pd.DataFrame:
    rows = []
    for i in range(60):
        district = "Cầu Giấy" if i < 30 else "Đống Đa"
        rows.append({
            "listing_id": 1000 + i,
            "dup_cluster_id": 1000 + (i // 2) * 2,  # cặp tin repost chung cụm
            "deal_type": "sale",
            "property_type": "Căn hộ chung cư",
            "province": "Hà Nội",
            "district": district,
            "price_vnd": float(3e9 + i * 1e8),
            "area_m2": 60.0 + i,
            "bedrooms": 2 if i % 3 else 3,
            "is_price_outlier": False,
            "is_area_outlier": False,
            "amenities": ["ho_boi"] if i % 5 == 0 else [],
            "published_at": pd.Timestamp("2026-03-01") if i % 2 else pd.Timestamp("2026-02-01"),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_settings(tmp_path: Path):
    settings = load_settings(data_root=tmp_path, duckdb_threads=1, duckdb_memory_gb=1)
    settings.clean_path.parent.mkdir(parents=True, exist_ok=True)
    df = _frame()  # noqa: F841 — DuckDB đọc DataFrame theo tên biến
    con = duckdb.connect()
    con.execute(f"COPY (SELECT * FROM df) TO '{settings.clean_path.as_posix()}' (FORMAT parquet)")
    con.close()
    return settings


@pytest.fixture
def con(synthetic_settings):
    from vre.data.duck import connect

    c = connect(synthetic_settings)
    yield c
    c.close()


def q(**kw) -> Question:
    base = {"id": "T1", "category": "constraint", "route_gold": "basic_rag", "question": "?", "gt_kind": "ids",
            "gt_sql": "SELECT listing_id FROM v_listings WHERE list_contains(amenities, 'ho_boi')"}
    return Question.model_validate({**base, **kw})


def write_questions(path: Path, questions: list[Question]) -> Path:
    path.write_text("\n".join(x.model_dump_json() for x in questions) + "\n", encoding="utf-8")
    return path


def test_ids_item_non_empty_with_clusters(con):
    item = compute_item(con, q())
    assert item["gt_ids"] == [1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035, 1040, 1045, 1050, 1055]
    assert item["gt_dup_clusters"] == sorted({1000 + (i // 2) * 2 for i in range(0, 60, 5)})
    assert item["gt_value"] is None and item["gt_rows"] is None
    assert item["sql_seconds"] >= 0


def test_ids_empty_is_error(con):
    with pytest.raises(ValueError, match="rỗng"):
        compute_item(con, q(gt_sql="SELECT listing_id FROM v_listings WHERE district = 'Không có'"))


def test_ids_needs_listing_id_column(con):
    with pytest.raises(ValueError, match="listing_id"):
        compute_item(con, q(gt_sql="SELECT dup_cluster_id FROM v_listings"))


def test_lookup_cluster_cap(con):
    sql = f"SELECT listing_id FROM v_listings WHERE listing_id < {1000 + 2 * (LOOKUP_MAX_CLUSTERS + 1)}"
    with pytest.raises(ValueError, match="dup_cluster_id"):
        compute_item(con, q(category="lookup", gt_sql=sql))
    ok = compute_item(con, q(category="lookup", gt_sql="SELECT listing_id FROM v_listings WHERE listing_id IN (1000, 1001)"))
    assert ok["gt_dup_clusters"] == [1000]


def value_q(sql: str) -> Question:
    return q(category="aggregate", route_gold="graph", gt_kind="value", value_tolerance=0.05, gt_sql=sql)


def test_value_exactly_one_row(con):
    item = compute_item(con, value_q(
        "SELECT median(price_m2_mil) AS median_price_m2_mil, count(*) AS n FROM v_listings WHERE district = 'Cầu Giấy'"))
    expected = _frame().head(30)
    median = float((expected.price_vnd / expected.area_m2 / 1e6).median())
    assert item["gt_rows"] == [{"median_price_m2_mil": pytest.approx(median), "n": 30}]
    assert item["gt_value"] == f"{median:.2f}"
    assert item["gt_ids"] is None and item["gt_dup_clusters"] is None


def test_value_multiple_rows_is_error(con):
    with pytest.raises(ValueError, match="1 dòng"):
        compute_item(con, value_q("SELECT district, count(*) AS n FROM v_listings GROUP BY district"))


def test_value_null_is_error(con):
    with pytest.raises(ValueError, match="null"):
        compute_item(con, value_q("SELECT median(price_m2_mil) AS m FROM v_listings WHERE district = 'Không có'"))


def test_value_small_group_is_error(con):
    with pytest.raises(ValueError, match="nhóm quá nhỏ"):
        compute_item(con, value_q(
            f"SELECT median(price_m2_mil) AS m, count(*) AS n FROM v_listings WHERE listing_id < {1000 + MIN_GROUP_N - 1}"))


def test_refusal_has_null_sql_and_null_answer(con):
    item = compute_item(con, q(id="U1", category="unanswerable", gt_kind="refusal", gt_sql=None))
    assert item == {"id": "U1", "gt_ids": None, "gt_dup_clusters": None, "gt_value": None, "gt_rows": None,
                    "sql_seconds": 0.0}


def test_format_value():
    assert format_value(3.14159) == "3.14"
    assert format_value(42) == "42"
    assert format_value("Tây Hồ") == "Tây Hồ"


def test_run_writes_results_and_load_gt(tmp_path, synthetic_settings):
    path = write_questions(tmp_path / "q.jsonl", [
        q(),
        value_q("SELECT count(*) AS n_listings FROM v_listings").model_copy(update={"id": "T2"}),
        q(id="U1", category="unanswerable", gt_kind="refusal", gt_sql=None),
    ])
    doc = run(synthetic_settings, questions_path=path)
    on_disk = json.loads(gt_path(synthetic_settings).read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(doc))
    assert gt_path(synthetic_settings) == synthetic_settings.results_dir / "benchmark_gt.json"
    assert {"schema_version", "generated_at", "git_sha", "data_fingerprint", "items"} <= set(doc)
    assert doc["data_fingerprint"]["n_listings"] == 60
    assert [i["id"] for i in doc["items"]] == ["T1", "T2", "U1"]
    gt = load_gt(synthetic_settings)
    assert gt["T2"]["gt_value"] == "60"
    assert gt["U1"]["gt_value"] is None


def test_run_failure_reports_all_and_keeps_old_file(tmp_path, synthetic_settings):
    good = write_questions(tmp_path / "good.jsonl", [q()])
    run(synthetic_settings, questions_path=good)
    before = gt_path(synthetic_settings).read_bytes()
    bad = write_questions(tmp_path / "bad.jsonl", [
        q(id="B1", gt_sql="SELECT listing_id FROM v_listings WHERE district = 'Không có'"),
        value_q("SELECT district FROM v_listings").model_copy(update={"id": "B2"}),
    ])
    with pytest.raises(RuntimeError) as exc:
        run(synthetic_settings, questions_path=bad)
    assert "B1" in str(exc.value) and "B2" in str(exc.value)
    assert gt_path(synthetic_settings).read_bytes() == before


def test_load_gt_missing_file(synthetic_settings):
    with pytest.raises(FileNotFoundError):
        load_gt(synthetic_settings)
