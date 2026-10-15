"""EDA trên parquet tổng hợp nhỏ theo schema data-contract §3 (không cần dữ liệu thật, không nạp model)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from vre.config import load_settings
from vre.data import eda
from vre.data.eda_plots import FIGURE_FILES

TYPES = ["Nhà", "Đất", "Căn hộ chung cư", "Biệt thự/Nhà liền kề", "Shophouse"]
PROVINCES = ["Hồ Chí Minh", "Hồ Chí Minh", "Hà Nội", "Đà Nẵng", None]
SUMMARY_KEYS = {"schema_version", "generated_at", "git_sha", "n_listings", "period", "by_property_type", "top_provinces",
                "hn_hcm_share", "monthly", "missing_pct", "description_chars", "tokens_per_listing", "figures"}
N = 50


def _frame() -> pd.DataFrame:
    rows = []
    for i in range(N):
        prov = PROVINCES[i % 5]
        area = 50.0 + i
        rows.append({
            "listing_id": i + 1, "dup_cluster_id": i + 1, "n_raw_rows": 1 + (i % 4 == 0) + 2 * (i == 7),
            "source_shard": 0, "source_row": i, "title": f"Bán nhà số {i}", "description": None if i % 10 == 0 else "mô tả " * (i + 1),
            "property_type": TYPES[i % 5], "deal_type": "rent" if i % 9 == 0 else "sale", "province": prov, "province_2025": prov,
            "district": None if prov is None else f"Quận {i % 3}", "ward": None, "street": None, "project": None,
            "price_vnd": None if i % 7 == 0 else area * 50e6, "price_fixed_thousand": False, "area_m2": area,
            "floors": None, "frontage_m": None, "depth_m": None, "road_width_m": None, "bedrooms": 2, "bathrooms": 1,
            "house_direction": None, "balcony_direction": None, "published_at": datetime(2025, 6 + i % 3, 1),
            "is_price_outlier": i == 3, "is_area_outlier": i == 5, "amenities": ["so_hong"] if i % 2 else [],
        })
    return pd.DataFrame(rows)


@pytest.fixture
def settings(tmp_dir: Path, monkeypatch: pytest.MonkeyPatch):
    s = load_settings(data_root=tmp_dir, duckdb_threads=2, duckdb_memory_gb=1)
    s.clean_path.parent.mkdir(parents=True)
    df = _frame()
    con = duckdb.connect()
    try:
        con.register("df", df)
        con.execute(f"""COPY (SELECT * REPLACE (listing_id::BIGINT AS listing_id, n_raw_rows::INTEGER AS n_raw_rows,
            source_shard::SMALLINT AS source_shard, bedrooms::SMALLINT AS bedrooms, floors::SMALLINT AS floors)
            FROM df) TO '{s.clean_path.as_posix()}' (FORMAT parquet)""")
    finally:
        con.close()
    monkeypatch.setattr(eda, "load_tokenizer", lambda _s: lambda texts: [len(t.split()) for t in texts])
    return s


def test_run_writes_summary_and_eight_figures(settings) -> None:
    out = eda.run(settings)
    assert set(out) == SUMMARY_KEYS
    assert out["n_listings"] == N
    on_disk = json.loads((settings.results_dir / "eda_summary.json").read_text(encoding="utf-8"))
    assert on_disk["n_listings"] == N
    pngs = sorted(p.name for p in (settings.results_dir / "eda").glob("*.png"))
    assert pngs == sorted(FIGURE_FILES) and len(pngs) == 8
    assert [f["file"] for f in out["figures"]] == list(FIGURE_FILES)
    assert all(f["title"] and f["caption"] for f in out["figures"])
    from matplotlib import image

    assert image.imread(settings.results_dir / "eda" / "monthly_trend.png").shape[:2] == (900, 1600)


def test_summary_values(settings) -> None:
    out = eda.run(settings)
    assert out["period"] == {"min_month": "2025-06", "max_month": "2025-08"}
    assert sum(r["n"] for r in out["by_property_type"]) == N
    assert sum(r["n"] for r in out["monthly"]) == N
    assert out["hn_hcm_share"] == pytest.approx(30 / N)
    top = out["top_provinces"][0]
    assert top["province"] == "Hồ Chí Minh" and top["n"] == 20 and top["share"] == pytest.approx(0.4)
    assert out["missing_pct"]["ward"] == 100.0
    assert out["missing_pct"]["description"] == pytest.approx(10.0)
    assert out["missing_pct"]["province"] == pytest.approx(20.0)
    # giá/m² = 50 tr/m² cho mọi tin bán không outlier
    assert [r["median_price_m2_mil"] for r in out["by_property_type"]] == pytest.approx([50.0] * 5)
    tok = out["tokens_per_listing"]
    assert tok["p50"] <= tok["p90"] <= tok["p99"]
    assert out["description_chars"]["p50"] > 0


def test_idempotent_second_run(settings, capsys: pytest.CaptureFixture[str]) -> None:
    first = eda.run(settings)
    png = settings.results_dir / "eda" / "outliers.png"
    mtime = png.stat().st_mtime_ns
    capsys.readouterr()
    second = eda.run(settings)
    assert "đã xong" in capsys.readouterr().out
    assert second == json.loads(json.dumps(first, ensure_ascii=False))
    assert png.stat().st_mtime_ns == mtime


def test_rerun_when_figure_missing(settings, capsys: pytest.CaptureFixture[str]) -> None:
    eda.run(settings)
    (settings.results_dir / "eda" / "duplicates.png").unlink()
    capsys.readouterr()
    eda.run(settings)
    assert "đã xong" not in capsys.readouterr().out
    assert (settings.results_dir / "eda" / "duplicates.png").exists()


def test_rerun_when_row_count_changes(settings, capsys: pytest.CaptureFixture[str]) -> None:
    eda.run(settings)
    path = settings.results_dir / "eda_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["n_listings"] = N + 1
    path.write_text(json.dumps(data), encoding="utf-8")
    capsys.readouterr()
    assert eda.run(settings)["n_listings"] == N
    assert "đã xong" not in capsys.readouterr().out


E5_CACHE = load_settings().cache_dir / "huggingface" / "hub" / "models--intfloat--multilingual-e5-large"


@pytest.mark.skipif(not E5_CACHE.exists(), reason="tokenizer e5-large chưa có trong cache HF")
def test_real_tokenizer_counts_without_model(tmp_dir: Path) -> None:
    count = eda.load_tokenizer(load_settings())
    a, b = count(["Bán nhà Quận 7", "Bán nhà Quận 7\nNhà đẹp hẻm xe hơi, sổ hồng chính chủ"])
    assert 0 < a < b
