"""`valuate(input)` cho Phase 3 — trả đúng `ValuationResult` (api-contract §3)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from vre.config import Settings, get_settings
from vre.valuation import explain
from vre.valuation import features as F
from vre.valuation.train import baseline_predict, paths

MAX_AREA_M2 = 100_000
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
RESULT_KEYS = ("predicted_price_m2_mil", "predicted_price_vnd", "interval_price_vnd", "baseline_ward_median_price_m2_mil",
               "comparables_n", "verdict", "shap", "model")


class ModelUnavailableError(FileNotFoundError):
    """Model chưa được huấn luyện (API trả 503)."""


@lru_cache(maxsize=4)
def _load(model_path: str, enc_path: str, mtime: float) -> tuple[xgb.Booster, dict[str, Any]]:
    booster = xgb.Booster()
    booster.load_model(model_path)
    booster.set_param({"nthread": 1})
    return booster, json.loads(Path(enc_path).read_text(encoding="utf-8"))


def load_model(settings: Settings, variant: F.Variant, deal: str) -> tuple[xgb.Booster, dict[str, Any], Path]:
    model_path, enc_path = paths(settings.data_root, variant, deal)
    if not (model_path.is_file() and enc_path.is_file()):
        if deal == "rent":
            raise ValueError("Chưa có model thuê (rent_model.trained=false) — chỉ định giá được deal_type='sale'")
        raise ModelUnavailableError(f"Chưa có model định giá {model_path.name} — chạy `vre_pipeline.py price --variant {variant}`")
    booster, enc = _load(str(model_path), str(enc_path), model_path.stat().st_mtime)
    return booster, enc, model_path


def validate(inp: dict[str, Any], enc: dict[str, Any]) -> None:
    for key in ("property_type", "province", "district", "area_m2"):
        if inp.get(key) in (None, ""):
            raise ValueError(f"Thiếu trường bắt buộc {key!r}")
    area = float(inp["area_m2"])
    if not 0 < area <= MAX_AREA_M2:
        raise ValueError(f"area_m2 phải trong (0, {MAX_AREA_M2}], nhận {area}")
    if inp["property_type"] not in enc["categories"]["property_type"]:
        raise ValueError(f"property_type lạ {inp['property_type']!r}")
    if inp.get("month") and not MONTH_RE.match(str(inp["month"])):
        raise ValueError(f"month phải dạng YYYY-MM, nhận {inp['month']!r}")
    asking = inp.get("asking_price_vnd")
    if asking is not None and float(asking) <= 0:
        raise ValueError("asking_price_vnd phải > 0")


def _features(row: pd.DataFrame, enc: dict[str, Any], variant: F.Variant) -> pd.DataFrame:
    te = F.target_encode_apply(row, enc["te"])
    pca = None
    if variant == "full":  # tin mới không có embedding ⇒ PCA thiếu (XGBoost xử lý missing)
        pca = pd.DataFrame(np.nan, index=row.index, columns=[f"pca_{i:02d}" for i in range(int(enc["pca_dim"]))])
    x = F.build_matrix(row, te, enc["categories"], variant, pca)
    if list(x.columns) != enc["feature_names"]:
        raise RuntimeError("Cột đặc trưng không khớp model — encoders và model lệch phiên bản")
    return x


def _verdict(asking: float | None, low: float, high: float) -> str | None:
    if asking is None:
        return None
    if asking < low:
        return "below_market"
    return "above_market" if asking > high else "fair"


def valuate(input: dict[str, Any], settings: Settings | None = None, variant: F.Variant = "tabular") -> dict[str, Any]:
    """Định giá một BĐS; `ValueError` = input sai (400), `ModelUnavailableError` = chưa có model (503)."""
    settings = settings or get_settings()
    inp = dict(input)
    deal = inp.get("deal_type") or "sale"
    if deal not in ("sale", "rent"):
        raise ValueError(f"deal_type lạ {deal!r}")
    booster, enc, model_path = load_model(settings, variant, deal)
    validate(inp, enc)
    row = F.frame_from_input(inp, enc["max_month"])
    x = _features(row, enc, variant)
    price_m2 = float(np.exp(booster.predict(xgb.DMatrix(x, enable_categorical=True))[0]))
    area = float(inp["area_m2"])
    price_vnd = price_m2 * 1e6 * area
    q = enc["interval_ratio_q"].get(str(inp["property_type"]), enc["interval_ratio_q"]["ALL"])
    low, high = price_vnd * q[0], price_vnd * q[1]
    base, count = baseline_predict(row, enc["baseline"])
    raw = {**{c: row.iloc[0][c] for c in (*F.CAT_COLS, *F.NUM_COLS)},
           **{f"te_{c}": row.iloc[0][c] for c in F.TE_COLS}}
    asking = inp.get("asking_price_vnd")
    return {
        "predicted_price_m2_mil": price_m2,
        "predicted_price_vnd": price_vnd,
        "interval_price_vnd": {"low": low, "high": high},
        "baseline_ward_median_price_m2_mil": float(base[0]) if count[0] > 0 else None,
        "comparables_n": int(count[0]),
        "verdict": _verdict(None if asking is None else float(asking), low, high),
        "shap": explain.explain_row(booster, x, raw),
        "model": {"variant": variant, "path": model_path.relative_to(settings.data_root).as_posix(),
                  "trained_at": enc["trained_at"], "test_mape": enc["test_mape"]},
    }
