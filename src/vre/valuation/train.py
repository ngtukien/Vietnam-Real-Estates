"""Huấn luyện định giá XGBoost (Step 7) → `models/valuation/*` + `results/valuation_metrics.json` (data-contract §7.6)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from vre.common import result_header, utc_now, write_json_atomic
from vre.config import Settings
from vre.valuation import explain
from vre.valuation import features as F

N_CONFIGS = 8
SEARCH_SEED = 42
MAX_ROUNDS = 3000
EARLY_STOPPING = 100
SPACE = {"max_depth": [6, 8, 10, 12], "eta": [0.03, 0.05, 0.1], "subsample": [0.7, 0.8, 0.9, 1.0],
         "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 5, 10, 20], "lambda": [1.0, 3.0, 10.0]}
BASELINE_LEVELS = ("ward_key", "district_key", "province", "property_type")
INTERVAL_Q = (0.10, 0.90)


def paths(root: Path, variant: str, deal: str = "sale") -> tuple[Path, Path]:
    suffix = "" if deal == "sale" else "_rent"
    model_dir = root / "models" / "valuation"
    return model_dir / f"xgb_{variant}_{deal}.json", model_dir / f"encoders_{variant}{suffix}.json"


def sample_configs(seed: int = SEARCH_SEED, n: int = N_CONFIGS) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    return [{k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()} for _ in range(n)]


def baseline_maps(train: pd.DataFrame) -> dict[str, Any]:
    y = train["price_m2_mil"]
    maps = {lvl: {str(k): [float(r["median"]), int(r["count"])] for k, r in y.groupby(train[lvl]).agg(["median", "count"]).iterrows()}
            for lvl in BASELINE_LEVELS}
    return {"levels": maps, "global": float(y.median())}


def baseline_predict(df: pd.DataFrame, base: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Median phường, lùi quận → tỉnh → loại → toàn cục; trả (giá/m² dự báo, số tin so sánh của mức dùng)."""
    pred = np.full(len(df), np.nan)
    count = np.zeros(len(df), dtype=np.int64)
    for lvl in BASELINE_LEVELS:
        hit = df[lvl].map(lambda k, m=base["levels"][lvl]: m.get(str(k)) if pd.notna(k) else None)
        todo = np.isnan(pred) & hit.notna().to_numpy()
        pred[todo] = [v[0] for v in hit[todo]]
        count[todo] = [v[1] for v in hit[todo]]
    pred[np.isnan(pred)] = base["global"]
    return pred, count


def scores(y: np.ndarray, yhat: np.ndarray) -> dict[str, float | None]:
    if len(y) == 0:
        return {"mae_mil_m2": None, "mape": None, "r2": None}
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - yhat) ** 2).sum()) / ss_tot if ss_tot > 0 else None
    return {"mae_mil_m2": float(np.abs(y - yhat).mean()), "mape": float(np.abs((y - yhat) / y).mean()), "r2": r2}


def metrics(test: pd.DataFrame, yhat: np.ndarray, n_train: int) -> dict[str, Any]:
    y = test["price_m2_mil"].to_numpy(float)

    def by(col: str, top: int | None) -> list[dict[str, Any]]:
        groups = test[col].value_counts().index[:top] if top else sorted(test[col].dropna().unique())
        return [{col: g, "n": int(m.sum()), **scores(y[m], yhat[m])} for g in groups for m in [(test[col] == g).to_numpy()]]

    return {"n_train": n_train, "n_test": len(test), **scores(y, yhat),
            "by_property_type": by("property_type", None), "by_province_top10": by("province", 10)}


def intervals(test: pd.DataFrame, yhat: np.ndarray) -> dict[str, list[float]]:
    ratio = pd.Series(test["price_m2_mil"].to_numpy(float) / yhat, index=test.index)
    out = {"ALL": [float(q) for q in ratio.quantile(INTERVAL_Q)]} if len(ratio) else {"ALL": [1.0, 1.0]}
    for pt, r in ratio.groupby(test["property_type"]):
        out[str(pt)] = [float(q) for q in r.quantile(INTERVAL_Q)]
    return out


def search(dfit: xgb.DMatrix, dval: xgb.DMatrix, device: str, settings: Settings) -> tuple[xgb.Booster, dict[str, Any]]:
    base = {"objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist", "device": device,
            "nthread": settings.xgb_nthread, "seed": settings.seed}
    best: tuple[float, xgb.Booster, dict[str, Any]] | None = None
    for i, cfg in enumerate(sample_configs()):
        booster = xgb.train({**base, **cfg}, dfit, MAX_ROUNDS, evals=[(dval, "valid")],
                            early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)
        print(f"price: cấu hình {i + 1}/{N_CONFIGS} {cfg} → rmse_valid={booster.best_score:.4f} @ {booster.best_iteration}")
        if best is None or booster.best_score < best[0]:
            best = (booster.best_score, booster, cfg)
    assert best is not None
    _, booster, cfg = best
    used = json.loads(booster.save_config())["learner"]["generic_param"].get("device")
    print(f"price: XGBoost tree_method=hist device={used} nthread={settings.xgb_nthread}")
    return booster[: booster.best_iteration + 1], {**cfg, "best_iteration": booster.best_iteration, "device": used}


def train_deal(df: pd.DataFrame, variant: F.Variant, settings: Settings, device: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    masks = F.split(df)
    train, test = df[masks["train"]], df[masks["test"]]
    if masks["fit"].sum() == 0 or masks["valid"].sum() == 0 or len(test) == 0:
        raise ValueError(f"Thiếu dữ liệu cho split: fit={int(masks['fit'].sum())} valid={int(masks['valid'].sum())} test={len(test)}")
    te_train, enc = F.target_encode_oof(train, settings.seed)
    cats = F.categories(train)
    pca = F.join_pca(df, F.pca_path(settings), settings.pca_dim) if variant == "full" else None
    x_train = F.build_matrix(train, te_train, cats, variant, None if pca is None else pca.loc[train.index])
    x_test = F.build_matrix(test, F.target_encode_apply(test, enc), cats, variant, None if pca is None else pca.loc[test.index])
    fit, val = masks["fit"][masks["train"]].to_numpy(), masks["valid"][masks["train"]].to_numpy()
    y = train[F.TARGET].to_numpy()
    # QuantileDMatrix: dựng thẳng bin của `hist`, không giữ bản sao float — bộ nhớ (commit) nhỏ hơn nhiều.
    dfit = xgb.QuantileDMatrix(x_train[fit], y[fit], enable_categorical=True)
    dval = xgb.QuantileDMatrix(x_train[val], y[val], ref=dfit, enable_categorical=True)
    feature_names = list(x_train.columns)
    del x_train
    booster, params = search(dfit, dval, device, settings)
    yhat = np.exp(booster.predict(xgb.DMatrix(x_test, enable_categorical=True)))
    m = metrics(test, yhat, len(train))
    m["train_seconds"] = round(time.perf_counter() - t0, 1)
    m["shap_top"] = explain.shap_top(booster, x_test, settings.seed)
    base = baseline_maps(train)
    encoders = {"variant": variant, "trained_at": utc_now(), "te": enc, "categories": cats,
                "feature_names": feature_names, "amenities": F.amenity_keys(), "max_month": str(df["month"].max()),
                "pca_dim": settings.pca_dim if variant == "full" else None, "baseline": base,
                "interval_ratio_q": intervals(test, yhat), "test_mape": m["mape"], "params": params}
    return {"booster": booster, "encoders": encoders, "metrics": m, "test": test, "train": train}


def baseline_metrics(res: dict[str, Any]) -> dict[str, Any]:
    pred, _ = baseline_predict(res["test"], res["encoders"]["baseline"])
    m = metrics(res["test"], pred, len(res["train"]))
    return {**m, "shap_top": [], "model_path": None, "train_seconds": 0.0}


def save(res: dict[str, Any], model_path: Path, enc_path: Path, root: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    res["booster"].save_model(model_path)
    write_json_atomic(enc_path, res["encoders"])
    res["metrics"]["model_path"] = model_path.relative_to(root).as_posix()


def rent_model(settings: Settings, variant: F.Variant, root: Path, n_rent: int, existing: dict[str, Any] | None,
               limit: int | None) -> dict[str, Any]:
    if variant == "full":
        return existing or {"trained": False, "n_rent": n_rent, "reason": "model thuê chỉ huấn luyện ở lượt tabular"}
    if n_rent < settings.rent_min_rows:
        return {"trained": False, "n_rent": n_rent, "reason": f"n_rent={n_rent} < rent_min_rows={settings.rent_min_rows}"}
    res = train_deal(F.load_frame(settings, "rent", limit), variant, settings, settings.xgb_device_tabular)
    save(res, *paths(root, variant, "rent"), root)
    return {"trained": True, "n_rent": n_rent, "reason": None}


def count_rent(settings: Settings) -> int:
    from vre.data.duck import connect

    con = connect(settings)
    try:
        return int(con.execute("SELECT count(*) FROM v_listings WHERE deal_type='rent' AND price_m2_mil IS NOT NULL "
                               "AND month BETWEEN ? AND ?", [F.TRAIN_MONTHS[0], F.TEST_MONTHS[1]]).fetchone()[0])
    finally:
        con.close()


def run(settings: Settings, variant: F.Variant, limit: int | None = None, output_root: Path | None = None) -> dict:
    """`limit` (smoke) bắt buộc đi kèm `output_root` khác data_root để không đè artifact thật."""
    if variant not in ("tabular", "full"):
        raise ValueError(f"variant lạ {variant!r}; hợp lệ: tabular, full")
    if limit is not None and output_root is None:
        raise ValueError("limit chỉ dùng cho smoke — cần output_root riêng")
    root = output_root or settings.data_root
    metrics_path = root / "results" / "valuation_metrics.json"
    existing = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    model_path, enc_path = paths(root, variant)
    if model_path.is_file() and enc_path.is_file() and (existing.get("variants") or {}).get(variant):
        print(f"price: đã xong ({variant})")
        return existing
    device = settings.xgb_device_tabular if variant == "tabular" else settings.xgb_device_full
    res = train_deal(F.load_frame(settings, "sale", limit), variant, settings, device)
    save(res, model_path, enc_path, root)
    rent = rent_model(settings, variant, root, count_rent(settings), existing.get("rent_model"), limit)
    variants = {"tabular": None, "full": None, **(existing.get("variants") or {})}
    variants[variant] = res["metrics"]
    variants["baseline_ward_median"] = baseline_metrics(res)
    out = {**result_header(), "split": {"train_months": list(F.TRAIN_MONTHS), "test_months": list(F.TEST_MONTHS)},
           "rent_model": rent, "variants": variants}
    write_json_atomic(metrics_path, out)
    return out
