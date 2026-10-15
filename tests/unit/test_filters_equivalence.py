"""B4: một danh sách điều kiện → Qdrant Filter, SQL WHERE (DuckDB), mặt nạ pandas cho CÙNG tập tin.

Qdrant chạy chế độ local trong RAM (`QdrantClient(":memory:")`) — không cần server.
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

from vre.rag import filters as F

N_ROWS = 200


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "listing_id": np.arange(1, N_ROWS + 1),
        "province": rng.choice(["Hồ Chí Minh", "Hà Nội"], N_ROWS),
        "district": rng.choice(["Quận 7", "Cầu Giấy", "Thủ Đức"], N_ROWS),
        "ward": rng.choice(["Tân Phong", "Dịch Vọng", None], N_ROWS),
        "property_type": rng.choice(list(F.PROPERTY_TYPES), N_ROWS),
        "deal_type": rng.choice(["sale", "rent"], N_ROWS, p=[0.8, 0.2]),
        "bedrooms": rng.choice([1.0, 2.0, 3.0, np.nan], N_ROWS),
        "price_vnd": rng.choice([8e8, 2.5e9, 4e9, 7.5e9, np.nan], N_ROWS),
        "area_m2": rng.uniform(20, 200, N_ROWS).round(1),
        "project": rng.choice(["Vinhomes Grand Park", None], N_ROWS),
        "street": rng.choice(["Nguyễn Văn Linh", "Cầu Giấy", None], N_ROWS),
        "published_month": rng.choice(F.month_span(None, None), N_ROWS),
    })
    return df


@pytest.fixture(scope="module")
def qdrant(frame):
    from qdrant_client import QdrantClient, models

    qc = QdrantClient(":memory:")
    qc.create_collection("t", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    points = []
    for rec in frame.to_dict("records"):
        payload = {k: (int(v) if k == "bedrooms" else v) for k, v in rec.items() if not pd.isna(v)}
        points.append(models.PointStruct(id=int(rec["listing_id"]), vector=[1.0, 0.5], payload=payload))
    qc.upsert("t", points)
    return qc


CASES = [
    {},
    {"province": "Hà Nội"},
    {"district": "Q7", "property_type": "Căn hộ chung cư"},
    {"bedrooms": 2, "max_price_mil": 4000},
    {"min_price_mil": "2 tỷ", "max_price_mil": "8 tỷ", "deal_type": "sale"},
    {"min_area_m2": 50, "max_area_m2": 120, "ward": "Dịch Vọng"},
    {"project": "Vinhomes Grand Park", "street": "Nguyễn Văn Linh"},
    {"project": "khu đô thị vinhomes grand park", "street": "đường nguyễn văn linh"},  # phân giải tên → `in`
    {"project": "Dự án không có", "district": "Thủ Đức"},  # tên không khớp ⇒ bỏ riêng điều kiện tên
    {"month_from": "2025-09", "month_to": "2025-12"},
    {"month_from": "2026-02"},
    {"month_to": "2024-12"},  # ngoài khoảng dữ liệu ⇒ 0 tin ở cả ba dạng
    {"province": "Hồ Chí Minh", "bedrooms": 3, "min_area_m2": 60, "month_to": "2025-10"},
]


@pytest.mark.parametrize("raw", CASES)
def test_three_compilers_select_same_rows(frame, qdrant, raw):
    f = F.QueryFilters.model_validate(raw)
    by_mask = set(frame.loc[F.mask(frame, f), "listing_id"].tolist())

    con = duckdb.connect()
    con.register("df", frame.rename(columns={"published_month": "month"}))
    where, params = F.to_sql(f)
    by_sql = {int(r[0]) for r in con.execute(f"SELECT listing_id FROM df WHERE {where}", params).fetchall()}

    points, _ = qdrant.scroll("t", scroll_filter=F.to_qdrant(f), limit=N_ROWS + 1)
    by_qdrant = {int(p.id) for p in points}
    assert by_mask == by_sql == by_qdrant
    if raw == {}:
        assert len(by_mask) == N_ROWS
    if raw == {"month_to": "2024-12"}:
        assert by_mask == set()


def test_bad_manual_filters_raise_but_llm_values_are_dropped():
    with pytest.raises(ValueError):
        F.QueryFilters(property_type="Kho xưởng")
    with pytest.raises(ValueError):
        F.QueryFilters(month_from="2026-13")
    with pytest.raises(ValueError):
        F.QueryFilters(min_area_m2=100, max_area_m2=50)
    f = F.QueryFilters.from_extracted({"property_type": "Kho xưởng", "month_from": "tháng 3", "district": "q7",
                                       "max_price": "4 tỷ"})
    assert (f.property_type, f.month_from, f.district, f.max_price_mil) == (None, None, "Quận 7", 4000)


def test_merge_manual_overrides_llm_fields():
    llm = F.QueryFilters(province="Hồ Chí Minh", district="Quận 7", bedrooms=2)
    merged, source = F.merge(llm, {"district": "Thủ Đức", "max_price_mil": 3000})
    assert (merged.province, merged.district, merged.bedrooms, merged.max_price_mil) == ("Hồ Chí Minh", "Thủ Đức", 2, 3000)
    assert source == "merged"
    assert F.merge(llm, None) == (llm, "llm")
    assert F.merge(F.QueryFilters(), {"province": "Hà Nội"})[1] == "manual"


def test_month_span():
    assert F.month_span("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]
    assert F.month_span(None, None)[0] == F.DATA_PERIOD[0] and F.month_span(None, None)[-1] == F.DATA_PERIOD[1]
