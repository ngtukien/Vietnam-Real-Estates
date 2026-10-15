"""Dò siêu tham số XGBoost có thể chạy tiếp: mỗi cấu hình xong được lưu atomic dưới `_search/`, lần chạy sau dùng lại.

`_search/<variant>_<deal>/cfg_XX.ubj` (booster đã cắt tại best_iteration+1) + `cfg_XX.meta.json` (khóa cache, điểm).
Meta ghi SAU booster nên là dấu "đã xong": bị giết giữa chừng chỉ mất cấu hình đang chạy. Khóa = hash của cấu hình,
MAX_ROUNDS, EARLY_STOPPING, tên cột, số dòng fit/valid, tổng nhãn fit/valid (làm tròn 1e-6) và device; nthread không
thuộc khóa. Lệch khóa ⇒ huấn luyện lại và ghi đè.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xgboost as xgb

from vre.common import write_json_atomic

if TYPE_CHECKING:
    from vre.config import Settings

N_CONFIGS = 8
SEARCH_SEED = 42
MAX_ROUNDS = 3000
EARLY_STOPPING = 100
# max_depth ≤ 10, min_child_weight ≥ 5: depth 12 + mcw 1 cho model 750 MB (JSON), quá nặng cho API; lợi ~3% RMSE.
SPACE = {"max_depth": [6, 8, 10], "eta": [0.03, 0.05, 0.1], "subsample": [0.7, 0.8, 0.9, 1.0],
         "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [5, 10, 20], "lambda": [1.0, 3.0, 10.0]}
SEARCH_DIRNAME = "_search"
LABEL_SUM_DECIMALS = 6


def sample_configs(seed: int = SEARCH_SEED, n: int = N_CONFIGS) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    return [{k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()} for _ in range(n)]


def cache_dir(model_dir: Path, variant: str, deal: str) -> Path:
    return model_dir / SEARCH_DIRNAME / f"{variant}_{deal}"


def data_fingerprint(dfit: xgb.DMatrix, dval: xgb.DMatrix, device: str) -> dict[str, Any]:
    """Phần khóa không phụ thuộc cấu hình — tính một lần cho cả lượt dò."""
    return {"max_rounds": MAX_ROUNDS, "early_stopping": EARLY_STOPPING, "features": list(dfit.feature_names or []),
            "n_fit": int(dfit.num_row()), "n_valid": int(dval.num_row()),
            "sum_fit": round(float(np.sum(dfit.get_label(), dtype=np.float64)), LABEL_SUM_DECIMALS),
            "sum_valid": round(float(np.sum(dval.get_label(), dtype=np.float64)), LABEL_SUM_DECIMALS), "device": device}


def cache_key(cfg: dict[str, Any], fingerprint: dict[str, Any]) -> str:
    blob = json.dumps({"config": cfg, **fingerprint}, sort_keys=True, ensure_ascii=True, default=float)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _files(cdir: Path, i: int) -> tuple[Path, Path]:
    return cdir / f"cfg_{i:02d}.ubj", cdir / f"cfg_{i:02d}.meta.json"  # UBJ: nhỏ, lưu ít bộ nhớ hơn JSON


def load(cdir: Path | None, i: int, key: str) -> dict[str, Any] | None:
    """Meta của cấu hình `i` nếu có và khớp khóa (kèm file booster); không thì None."""
    if cdir is None:
        return None
    model_file, meta_file = _files(cdir, i)
    if not (meta_file.is_file() and model_file.is_file()):
        return None
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    return meta if meta.get("key") == key else None


def load_booster(cdir: Path, i: int) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(_files(cdir, i)[0])
    return booster


def store(cdir: Path | None, i: int, key: str, booster: xgb.Booster, meta: dict[str, Any]) -> None:
    """Booster (tmp + os.replace) trước, meta sau — meta có mặt ⇔ booster đầy đủ."""
    if cdir is None:
        return
    cdir.mkdir(parents=True, exist_ok=True)
    model_file, meta_file = _files(cdir, i)
    meta_file.unlink(missing_ok=True)  # khóa cũ không được trỏ vào booster mới đang ghi dở
    tmp = model_file.with_name(model_file.stem + ".tmp.ubj")
    booster.save_model(tmp)
    os.replace(tmp, model_file)
    write_json_atomic(meta_file, {"key": key, **meta})


def clear(cdir: Path) -> None:
    """Xóa cache của một variant/deal sau khi model + metrics đã ghi xong; dọn `_search/` nếu rỗng."""
    shutil.rmtree(cdir, ignore_errors=True)
    try:
        cdir.parent.rmdir()
    except OSError:
        pass  # còn cache của variant/deal khác (hoặc không tồn tại)


def search(dfit: xgb.DMatrix, dval: xgb.DMatrix, device: str, settings: Settings,
           cdir: Path | None = None) -> tuple[xgb.Booster, dict[str, Any]]:
    """Giữ cấu hình đầu tiên có best_score thấp nhất (nghiêm ngặt); `cdir=None` ⇒ không cache."""
    base = {"objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist", "device": device,
            "nthread": settings.xgb_nthread, "seed": settings.seed}
    fingerprint = {**data_fingerprint(dfit, dval, device), "seed": settings.seed}  # seed đổi ⇒ không dùng lại
    best: tuple[float, xgb.Booster, dict[str, Any]] | None = None
    for i, cfg in enumerate(sample_configs()):
        key = cache_key(cfg, fingerprint)
        meta = load(cdir, i, key)
        booster: xgb.Booster | None = None
        tag = " (dùng lại)"
        if meta is None:
            trained = xgb.train({**base, **cfg}, dfit, MAX_ROUNDS, evals=[(dval, "valid")],
                                early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)
            used = json.loads(trained.save_config())["learner"]["generic_param"].get("device")
            meta = {"config": cfg, "best_score": float(trained.best_score),
                    "best_iteration": int(trained.best_iteration), "device": used}
            booster = trained[: meta["best_iteration"] + 1]
            del trained
            store(cdir, i, key, booster, meta)
            tag = ""
        print(f"price: cấu hình {i + 1}/{N_CONFIGS}{tag} {cfg} → rmse_valid={meta['best_score']:.4f} "
              f"@ {meta['best_iteration']}")
        if best is None or meta["best_score"] < best[0]:
            best = (meta["best_score"], booster if booster is not None else load_booster(cdir, i), meta)  # type: ignore[arg-type]
    assert best is not None
    _, booster, meta = best
    print(f"price: XGBoost tree_method=hist device={meta['device']} nthread={settings.xgb_nthread}")
    return booster, {**meta["config"], "best_iteration": meta["best_iteration"], "device": meta["device"]}
