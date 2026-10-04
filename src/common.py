"""Hàm phụ trợ dùng chung: cache cho cell CACHE, hiển thị gọn, seed và phiên bản thư viện."""

import json
import pickle
import random
from importlib import metadata
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, REBUILD, SEED


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def versions(packages=("datasets", "pandas", "numpy", "sentence-transformers", "qdrant-client",
                       "pyvi", "neo4j", "networkx", "leidenalg", "anthropic")) -> dict:
    found = {}
    for name in packages:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = "chưa cài"
    return found


def _load(path: Path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as handle:
        return pickle.load(handle)


def _save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        value.to_parquet(path, index=False)
    elif path.suffix == ".npy":
        np.save(path, value)
    elif path.suffix == ".json":
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    else:
        with path.open("wb") as handle:
            pickle.dump(value, handle)


def cached(path: Path | str, compute, rebuild: bool = REBUILD):
    """Cell CACHE: đọc kết quả từ đĩa nếu có, nếu không thì tính rồi lưu.

    Đường dẫn tương đối nằm trong results/cache/. Định dạng theo đuôi file:
    .parquet (DataFrame), .npy (mảng), .json, còn lại là pickle.
    """
    path = Path(path)
    if not path.is_absolute():
        path = CACHE_DIR / path
    if path.exists() and not rebuild:
        return _load(path)
    value = compute()
    _save(path, value)
    return value


class Timer:
    """with Timer() as t: ...; t.seconds"""

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = perf_counter() - self.start


def show(df: pd.DataFrame | list, columns=None, rows: int = 10, width: int = 60) -> pd.DataFrame:
    """Bảng gọn cho output trên lớp: tối đa `rows` dòng, cắt chuỗi dài."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    out = df.head(rows).copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda v: v if not isinstance(v, str) or len(v) <= width
                                    else v[:width - 1] + "…")
    return out.reset_index(drop=True)


def fmt_vnd(value) -> str:
    """7450000000 -> '7,45 tỷ'; 850000000 -> '850 triệu'."""
    if value is None or pd.isna(value):
        return "?"
    if value >= 1e9:
        return f"{value / 1e9:,.2f} tỷ".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{value / 1e6:,.0f} triệu".replace(",", ".")
