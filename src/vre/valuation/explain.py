"""SHAP bằng `booster.predict(DMatrix, pred_contribs=True)` của XGBoost (không dùng thư viện `shap`)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

SHAP_MAX_ROWS = 20_000
TOP_ROW_ITEMS = 8


def contributions(booster: xgb.Booster, x: pd.DataFrame) -> np.ndarray:
    """`[n, n_features + 1]` đóng góp vào log(giá/m²); cột cuối là bias."""
    return booster.predict(xgb.DMatrix(x, enable_categorical=True), pred_contribs=True)


def shap_top(booster: xgb.Booster, x: pd.DataFrame, seed: int, max_rows: int = SHAP_MAX_ROWS) -> list[dict[str, Any]]:
    """`[{feature, mean_abs_contribution}]` giảm dần, trên ≤ `max_rows` dòng (mẫu ngẫu nhiên theo seed)."""
    if len(x) == 0:
        return []
    if len(x) > max_rows:
        x = x.sample(n=max_rows, random_state=seed)
    mean_abs = np.abs(contributions(booster, x)[:, :-1]).mean(axis=0)
    order = np.argsort(-mean_abs)
    return [{"feature": str(x.columns[i]), "mean_abs_contribution": float(mean_abs[i])} for i in order]


def _display(value: Any) -> str | float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    if isinstance(value, (np.floating, float, np.integer, int)):
        return float(value)
    return str(value)


def explain_row(booster: xgb.Booster, x: pd.DataFrame, raw: dict[str, Any], top: int = TOP_ROW_ITEMS) -> list[dict[str, Any]]:
    """`ShapItem[]` (api-contract): top `top` theo |contribution_log| cho một dòng; `value` lấy từ đầu vào thô nếu có."""
    if len(x) != 1:
        raise ValueError(f"explain_row cần đúng 1 dòng, nhận {len(x)}")
    contrib = contributions(booster, x)[0, :-1]
    order = np.argsort(-np.abs(contrib))[:top]
    items = []
    for i in order:
        name = str(x.columns[i])
        value = raw.get(name, x.iloc[0, i])
        items.append({"feature": name, "value": _display(value), "contribution_log": float(contrib[i])})
    return items
