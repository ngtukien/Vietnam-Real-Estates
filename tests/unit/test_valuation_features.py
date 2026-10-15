"""vre.valuation: split theo tháng, target encoding K-fold không rò rỉ fold, join PCA giả cho `full`, run/valuate trên CPU."""

import json

import numpy as np
import pandas as pd
import pytest

from vre.config import load_settings
from vre.valuation import features as F

MONTHS = ["2025-05", "2025-06", "2025-09", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]
PTYPES = ["Nhà", "Đất", "Căn hộ chung cư"]


def synthetic_clean(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Bảng sạch giả theo schema data-contract §3 (chỉ các cột view + valuation đọc)."""
    rng = np.random.default_rng(seed)
    prov = rng.choice(["Hồ Chí Minh", "Hà Nội", "Đà Nẵng"], n)
    dist = np.array([f"Quận {i}" for i in rng.integers(1, 4, n)])
    ward = np.array([f"Phường {i}" for i in rng.integers(1, 4, n)])
    ptype = rng.choice(PTYPES, n)
    area = rng.uniform(30, 300, n)
    level = {"Hồ Chí Minh": 120.0, "Hà Nội": 140.0, "Đà Nẵng": 60.0}
    pm2 = np.array([level[p] for p in prov]) * (1 + 0.1 * (dist == "Quận 1")) * rng.lognormal(0, 0.15, n)
    keys = list(F.amenity_keys())
    return pd.DataFrame({
        "listing_id": np.arange(1, n + 1, dtype=np.int64),
        "deal_type": np.where(rng.random(n) < 0.02, "rent", "sale"),
        "property_type": ptype, "province": prov, "district": dist, "ward": ward,
        "street": [f"Đường {i}" for i in rng.integers(1, 30, n)], "project": [None] * n,
        "price_vnd": pm2 * area * 1e6, "area_m2": area,
        "floors": rng.integers(1, 6, n).astype("int16"), "frontage_m": rng.uniform(3, 10, n), "depth_m": None,
        "road_width_m": rng.uniform(2, 20, n), "bedrooms": rng.integers(1, 6, n).astype("int16"),
        "bathrooms": rng.integers(1, 4, n).astype("int16"), "house_direction": rng.choice(["Đông", "Tây", None], n),
        "published_at": pd.to_datetime([f"{m}-15" for m in rng.choice(MONTHS, n)]),
        "is_price_outlier": rng.random(n) < 0.01, "is_area_outlier": False,
        "amenities": [list(rng.choice(keys, 2, replace=False)) for _ in range(n)],
    })


@pytest.fixture
def settings(tmp_dir):
    (tmp_dir / "data" / "clean").mkdir(parents=True)
    synthetic_clean().to_parquet(tmp_dir / "data" / "clean" / "listings.parquet", index=False)
    return load_settings(data_root=tmp_dir, duckdb_threads=2, duckdb_memory_gb=1, xgb_nthread=2,
                         xgb_device_full="cpu", rent_min_rows=20000)


def write_fake_pca(settings, ids: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    pca = pd.DataFrame({"listing_id": ids, **{f"pca_{i:02d}": rng.normal(size=len(ids)).astype("float32")
                                              for i in range(settings.pca_dim)}}).sample(frac=1.0, random_state=3)
    F.pca_path(settings).parent.mkdir(parents=True, exist_ok=True)
    pca.to_parquet(F.pca_path(settings), index=False)
    return pca


def test_time_split_months():
    df = F.add_keys(pd.DataFrame({"month": MONTHS, "province": "A", "district": "B", "ward": None,
                                  "street": None, "project": None, "price_m2_mil": 1.0}))
    m = F.split(df)
    assert df["month"][m["fit"]].tolist() == ["2025-06", "2025-09", "2025-12"]
    assert df["month"][m["valid"]].tolist() == ["2026-01"]
    assert df["month"][m["test"]].tolist() == ["2026-02", "2026-03"]
    assert not (m["train"] & m["test"]).any()
    assert df["month_idx"].tolist()[1:5] == [0, 3, 6, 7]
    assert df["ward_key"].isna().all() and df["district_key"].tolist()[0] == "A|B"


def test_load_frame_filters_sale_window_and_outliers(settings):
    df = F.load_frame(settings)
    raw = synthetic_clean()
    assert set(df["listing_id"]).isdisjoint(set(raw.loc[raw["deal_type"] == "rent", "listing_id"]))
    assert set(df["listing_id"]).isdisjoint(set(raw.loc[raw["is_price_outlier"], "listing_id"]))
    assert df["month"].between("2025-06", "2026-03").all() and len(df) > 1000
    assert {f"am_{k}" for k in F.amenity_keys()} <= set(df.columns)


def test_target_encoding_oof_has_no_fold_leakage():
    n = 500
    y = np.random.default_rng(0).normal(5, 1, n)
    train = pd.DataFrame({c: [f"{c}{i}" for i in range(n)] for c in F.TE_COLS})  # khoá duy nhất mỗi dòng
    train[F.TARGET] = y
    oof, enc = F.target_encode_oof(train, seed=42)
    folds = F.fold_ids(n, 42)
    for k in range(F.K_FOLDS):
        held = folds == k
        expected_prior = y[~held].mean()  # dòng chỉ có giá trị riêng ⇒ chỉ còn prior của 4 fold kia
        assert np.allclose(oof.loc[held, "te_ward_key"], expected_prior)
    leaky = (y + F.SMOOTHING_M * y.mean()) / (1 + F.SMOOTHING_M)
    assert not np.allclose(oof["te_ward_key"], leaky)
    # Mapping toàn train cho test: có dùng target của chính khoá
    assert enc["maps"]["ward_key"]["ward_key0"] == pytest.approx((y[0] + F.SMOOTHING_M * y.mean()) / (1 + F.SMOOTHING_M))


def test_target_encoding_shared_key_uses_only_other_folds():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({c: rng.choice(["a", "b"], 400) for c in F.TE_COLS})
    train[F.TARGET] = rng.normal(0, 1, 400) + (train["province"] == "a") * 3
    oof, _ = F.target_encode_oof(train, seed=7)
    folds = F.fold_ids(400, 7)
    row = int(np.flatnonzero(folds == 2)[0])
    others = train[folds != 2]
    same = others[others["province"] == train.loc[row, "province"]][F.TARGET]
    prior = others[F.TARGET].mean()
    expected = (same.sum() + F.SMOOTHING_M * prior) / (len(same) + F.SMOOTHING_M)
    assert oof.loc[row, "te_province"] == pytest.approx(expected)
    unseen = F.target_encode_apply(pd.DataFrame({c: ["zzz"] for c in F.TE_COLS}), F.target_encode_oof(train, 7)[1])
    assert unseen["te_province"].iloc[0] == pytest.approx(train[F.TARGET].mean())


def test_target_encoding_train_ignores_validation_targets():
    rng = np.random.default_rng(3)
    n = 600
    train = pd.DataFrame({c: rng.choice(["a", "b", "c"], n) for c in F.TE_COLS})
    train["month"] = rng.choice(["2025-06", "2025-09", F.VALID_MONTH], n)
    train[F.TARGET] = rng.normal(5, 1, n)
    train.index = train.index + 1000  # chỉ số không bắt đầu từ 0
    valid = F.split(train)["valid"]
    te, enc = F.target_encode_train(train, valid, seed=11)
    perturbed = train.copy()
    perturbed.loc[valid, F.TARGET] += 100.0  # đổi mọi target tháng validation
    te_p, enc_p = F.target_encode_train(perturbed, valid, seed=11)
    pd.testing.assert_frame_equal(te, te_p)  # dòng fit (OOF) lẫn dòng valid không đổi
    fit_oof, fit_enc = F.target_encode_oof(train[~valid], seed=11)
    pd.testing.assert_frame_equal(te[~valid], fit_oof)
    pd.testing.assert_frame_equal(te[valid], F.target_encode_apply(train[valid], fit_enc))
    # Encoder lưu cho test/predict: fit trên cả cửa sổ train (fit ∪ valid)
    assert enc["prior"] == pytest.approx(train[F.TARGET].mean()) and enc_p["prior"] != enc["prior"]


def test_count_rent_respects_limit(settings):
    from vre.valuation import train

    full = train.count_rent(settings)
    assert full > 5 and train.count_rent(settings, 5) == 5 and train.count_rent(settings, 10**9) == full


def test_full_variant_joins_fake_pca_by_listing_id(settings):
    df = F.load_frame(settings)
    pca = write_fake_pca(settings, df["listing_id"].to_numpy()[:-5])  # 5 tin cuối không có embedding
    joined = F.join_pca(df, F.pca_path(settings), settings.pca_dim)
    by_id = pca.set_index("listing_id")
    lid = int(df["listing_id"].iloc[10])
    assert joined.loc[df.index[10], "pca_07"] == pytest.approx(by_id.loc[lid, "pca_07"])
    assert joined.iloc[-5:].isna().all().all()
    x = F.build_matrix(df, F.target_encode_apply(df, {"prior": 0.0, "maps": {c: {} for c in F.TE_COLS}}),
                       F.categories(df), "full", joined)
    assert [c for c in x.columns if c.startswith("pca_")] == [f"pca_{i:02d}" for i in range(32)]
    tab = F.build_matrix(df, F.target_encode_apply(df, {"prior": 0.0, "maps": {c: {} for c in F.TE_COLS}}),
                         F.categories(df), "tabular")
    assert not any(c.startswith("pca_") for c in tab.columns)


def test_join_pca_missing_file_raises(settings):
    with pytest.raises(FileNotFoundError):
        F.join_pca(pd.DataFrame({"listing_id": [1]}), F.pca_path(settings), settings.pca_dim)


def test_run_tabular_then_full_preserves_entries_and_valuate(settings, monkeypatch):
    from vre.valuation import predict, train
    from vre.valuation import search_cache as SC

    cleared, real_clear = [], SC.clear
    monkeypatch.setattr(SC, "clear", lambda d: (cleared.append((d.name, len(list(d.glob("*.meta.json"))))), real_clear(d)))
    out = train.run(settings, "tabular")
    assert cleared == [("tabular_sale", SC.N_CONFIGS), ("tabular_rent", 0)]  # cache đã có rồi mới được dọn
    v = out["variants"]
    assert v["full"] is None and v["tabular"]["mape"] is not None and v["baseline_ward_median"]["mape"] is not None
    assert v["tabular"]["model_path"] == "models/valuation/xgb_tabular_sale.json"
    assert out["rent_model"]["trained"] is False and "rent_min_rows" in out["rent_model"]["reason"]
    assert len(v["tabular"]["shap_top"]) > 0 and v["tabular"]["by_property_type"]
    assert not (settings.data_root / "models" / "valuation" / "_search").exists()  # cache dò tham số đã dọn
    write_fake_pca(settings, F.load_frame(settings)["listing_id"].to_numpy())
    out_full = train.run(settings, "full")
    assert out_full["variants"]["tabular"] == v["tabular"] and out_full["variants"]["full"]["mape"] is not None
    stored = json.loads((settings.results_dir / "valuation_metrics.json").read_text(encoding="utf-8"))
    assert stored["variants"]["tabular"] and stored["variants"]["full"]
    assert train.run(settings, "tabular") == stored  # idempotent: "đã xong"

    res = predict.valuate({"property_type": "Nhà", "province": "Hà Nội", "district": "Quận 1", "ward": "Phường 2",
                           "area_m2": 80, "bedrooms": 3, "amenities": ["thang_may"], "asking_price_vnd": 1e9}, settings)
    assert tuple(res) == predict.RESULT_KEYS
    assert res["interval_price_vnd"]["low"] <= res["predicted_price_vnd"] * 1.0001
    assert len(res["shap"]) == 8 and set(res["shap"][0]) == {"feature", "value", "contribution_log"}
    assert res["verdict"] == "below_market" and res["comparables_n"] > 0
    assert res["model"]["variant"] == "tabular" and res["model"]["test_mape"] == v["tabular"]["mape"]
    with pytest.raises(ValueError):
        predict.valuate({"property_type": "Nhà", "province": "Hà Nội", "district": "Quận 1", "area_m2": 0}, settings)
    with pytest.raises(ValueError):
        predict.valuate({"property_type": "Nhà", "deal_type": "rent", "province": "Hà Nội", "district": "Q", "area_m2": 5},
                        settings)
