"""Đặc trưng định giá (Step 7): đọc `v_listings`, chia theo tháng, target encoding K-fold, ma trận cho XGBoost."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from vre.config import EMBED_MODEL_SLUG, Settings
from vre.data.amenities import amenity_table

Variant = Literal["tabular", "full"]
TRAIN_MONTHS = ("2025-06", "2026-01")  # gồm cả tháng validation
VALID_MONTH = "2026-01"                # early stopping
TEST_MONTHS = ("2026-02", "2026-03")
MONTH_ORIGIN = (2025, 6)
K_FOLDS = 5
SMOOTHING_M = 20.0
TE_COLS = ("province", "district_key", "ward_key", "project_key", "street_key")
CAT_COLS = ("property_type", "house_direction")
NUM_COLS = ("area_m2", "bedrooms", "bathrooms", "floors", "frontage_m", "road_width_m", "month_idx")
TARGET = "log_price_m2"


def amenity_keys() -> list[str]:
    return list(amenity_table())


def pca_path(settings: Settings) -> Path:
    return settings.features_dir / f"emb_pca{settings.pca_dim}_{EMBED_MODEL_SLUG}.parquet"


def load_frame(settings: Settings, deal_type: str = "sale", limit: int | None = None) -> pd.DataFrame:
    """Tin có `price_m2_mil` (đã loại outlier giá + diện tích ở view) trong cửa sổ train ∪ test."""
    from vre.data.duck import connect

    amen = ", ".join(f"CAST(list_contains(amenities, '{k}') AS TINYINT) AS am_{k}" for k in amenity_keys())
    sql = f"""SELECT listing_id, month, property_type, house_direction, province, district, ward, street, project,
                     area_m2, bedrooms, bathrooms, floors, frontage_m, road_width_m, price_m2_mil, {amen}
              FROM v_listings
              WHERE deal_type = ? AND price_m2_mil IS NOT NULL AND month BETWEEN ? AND ?
              {f'LIMIT {int(limit)}' if limit else ''}"""
    con = connect(settings)
    try:
        df = con.execute(sql, [deal_type, TRAIN_MONTHS[0], TEST_MONTHS[1]]).df()
    finally:
        con.close()
    return add_keys(df)


def _key(df: pd.DataFrame, *cols: str) -> pd.Series:
    """Khoá ghép `a|b|c`; null nếu thành phần cuối null (không có giá trị riêng để mã hoá)."""
    parts = [df[c].astype("string").fillna("") for c in cols]
    key = parts[0]
    for part in parts[1:]:
        key = key + "|" + part
    return key.where(parts[-1] != "", None)


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["district_key"] = _key(df, "province", "district")
    df["ward_key"] = _key(df, "province", "district", "ward")
    df["project_key"] = _key(df, "province", "project")
    df["street_key"] = _key(df, "province", "district", "street")
    year, mon = df["month"].str.slice(0, 4).astype(int), df["month"].str.slice(5, 7).astype(int)
    df["month_idx"] = (year - MONTH_ORIGIN[0]) * 12 + (mon - MONTH_ORIGIN[1])
    if "price_m2_mil" in df:
        df[TARGET] = np.log(df["price_m2_mil"].astype(float))
    return df


def split(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Mặt nạ: `train` (cả cửa sổ train), `fit` (train trừ tháng validation), `valid`, `test`."""
    train = df["month"].between(*TRAIN_MONTHS)
    valid = df["month"] == VALID_MONTH
    return {"train": train, "fit": train & ~valid, "valid": valid, "test": df["month"].between(*TEST_MONTHS)}


def _smoothed(keys: pd.Series, y: pd.Series, prior: float) -> dict[str, float]:
    stats = y.groupby(keys).agg(["sum", "count"])
    return ((stats["sum"] + SMOOTHING_M * prior) / (stats["count"] + SMOOTHING_M)).to_dict()


def fold_ids(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n) % K_FOLDS


def target_encode_oof(train: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """OOF cho tập train (mỗi fold chỉ thấy 4 fold kia) + mapping toàn train cho test/predict."""
    y = train[TARGET]
    folds = fold_ids(len(train), seed)
    out = pd.DataFrame(index=train.index)
    for col in TE_COLS:
        enc = np.empty(len(train))
        for k in range(K_FOLDS):
            held = folds == k
            prior = float(y[~held].mean())
            mapping = _smoothed(train[col][~held], y[~held], prior)
            enc[held] = train[col][held].map(mapping).astype(float).fillna(prior).to_numpy()
        out[f"te_{col}"] = enc
    return out, fit_encoder(train)


def fit_encoder(df: pd.DataFrame) -> dict[str, Any]:
    """Mapping làm trơn trên toàn bộ `df` (prior = trung bình target của `df`)."""
    y = df[TARGET]
    prior = float(y.mean())
    return {"prior": prior, "maps": {col: _smoothed(df[col], y, prior) for col in TE_COLS}}


def target_encode_train(train: pd.DataFrame, valid: pd.Series, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """te_* cho cửa sổ train không rò tháng validation vào chọn model.

    Dòng `fit`: OOF K-fold chỉ trên `fit`; dòng `valid`: mapping fit-only. Encoder trả về (lưu, dùng cho test/predict)
    fit trên cả cửa sổ train (fit ∪ valid) — không bao giờ thấy target test.
    """
    valid = valid.loc[train.index].to_numpy(bool)
    fit = train[~valid]
    te_fit, enc_fit = target_encode_oof(fit, seed)
    te = pd.concat([te_fit, target_encode_apply(train[valid], enc_fit)]).loc[train.index]
    return te, fit_encoder(train)


def target_encode_apply(df: pd.DataFrame, enc: dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in TE_COLS:
        out[f"te_{col}"] = df[col].map(enc["maps"][col]).astype(float).fillna(enc["prior"])
    return out


def categories(train: pd.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(train[c].dropna().astype(str).unique().tolist()) for c in CAT_COLS}


def join_pca(df: pd.DataFrame, path: Path, dim: int) -> pd.DataFrame:
    """Left join `pca_00..pca_{dim-1}` theo `listing_id` (data-contract §4); thiếu file/cột ⇒ lỗi rõ ràng."""
    cols = [f"pca_{i:02d}" for i in range(dim)]
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu file PCA {path} (Phase 2 sinh) — không thể dựng bản 'full'")
    pca = pd.read_parquet(path, columns=["listing_id", *cols])
    if pca["listing_id"].duplicated().any():
        raise ValueError(f"{path}: listing_id trùng lặp")
    return df[["listing_id"]].merge(pca, on="listing_id", how="left").set_index(df.index)[cols]


def build_matrix(df: pd.DataFrame, te: pd.DataFrame, cats: dict[str, list[str]], variant: Variant,
                 pca: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ma trận đặc trưng float32 + 2 cột categorical native; thứ tự cột cố định (= feature_names của model)."""
    x = te.astype("float32")
    for c in CAT_COLS:
        x[c] = pd.Categorical(df[c].astype("string"), categories=cats[c])
    for c in NUM_COLS:
        x[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for k in amenity_keys():
        col = df[f"am_{k}"] if f"am_{k}" in df else pd.Series(0, index=df.index)
        x[f"am_{k}"] = pd.to_numeric(col, errors="coerce").fillna(0).astype("float32")
    if variant == "full":
        if pca is None:
            raise ValueError("Bản 'full' cần bảng PCA (join_pca)")
        x = pd.concat([x, pca.astype("float32")], axis=1)
    return x


def frame_from_input(inp: dict[str, Any], month: str) -> pd.DataFrame:
    """Một dòng thô từ `ValuationInput` (api-contract §3) theo đúng các cột `load_frame`."""
    amen = set(inp.get("amenities") or [])
    unknown = amen - set(amenity_keys())
    if unknown:
        raise ValueError(f"Khoá amenity lạ: {sorted(unknown)}")
    row: dict[str, Any] = {"listing_id": -1, "month": inp.get("month") or month, "street": None, "project": None}
    for c in ("property_type", "house_direction", "province", "district", "ward", *NUM_COLS[:-1]):
        row[c] = inp.get(c)
    row.update({f"am_{k}": int(k in amen) for k in amenity_keys()})
    return add_keys(pd.DataFrame([row]))
