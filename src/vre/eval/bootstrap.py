"""Bootstrap theo câu hỏi (data-contract §7.7): 10.000 mẫu lại, seed 42, CI phân vị 2,5/97,5.

- `mean_ci`: CI 95% của trung bình một hệ thống (độ chính xác / điểm trung bình).
- `paired`: chênh lệch a − b GHÉP CẶP — mỗi mẫu lại chọn cùng một tập câu hỏi cho cả hai hệ thống, nên phương sai
  giữa câu hỏi (câu khó với mọi hệ) bị triệt tiêu. `p_value` = tỷ lệ mẫu có dấu ngược với `diff` (diff = 0 ⇒ 1,0:
  không có bằng chứng chênh lệch).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

N_BOOT = 10_000
SEED = 42
CI_LEVEL = 0.95


def _values(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("bootstrap cần mảng 1 chiều")
    if not np.all(np.isfinite(arr)):
        raise ValueError("bootstrap nhận giá trị không hữu hạn (NaN/inf) — lọc trước khi gọi")
    return arr


def _indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _ci(samples: np.ndarray) -> list[float]:
    alpha = (1 - CI_LEVEL) / 2
    lo, hi = np.quantile(samples, [alpha, 1 - alpha])
    return [round(float(lo), 4), round(float(hi), 4)]


def mean_ci(x: Sequence[float], n_boot: int = N_BOOT, seed: int = SEED) -> list[float] | None:
    """[lo, hi] của trung bình; None khi rỗng (không có câu nào ⇒ không có CI, không phải [0, 0])."""
    arr = _values(x)
    if arr.size == 0:
        return None
    return _ci(arr[_indices(arr.size, n_boot, seed)].mean(axis=1))


def paired(a: Sequence[float], b: Sequence[float], n_boot: int = N_BOOT, seed: int = SEED) -> dict | None:
    """{diff, ci95, p_value, n_boot, n} cho mean(a) − mean(b), a[i] và b[i] là CÙNG câu hỏi. None khi rỗng."""
    xa, xb = _values(a), _values(b)
    if xa.shape != xb.shape:
        raise ValueError(f"bootstrap ghép cặp cần cùng số câu ({xa.size} ≠ {xb.size})")
    if xa.size == 0:
        return None
    delta = xa - xb
    diff = float(delta.mean())
    samples = delta[_indices(delta.size, n_boot, seed)].mean(axis=1)
    if diff > 0:
        p_value = float(np.mean(samples <= 0))
    elif diff < 0:
        p_value = float(np.mean(samples >= 0))
    else:
        p_value = 1.0
    return {"diff": round(diff, 4), "ci95": _ci(samples), "p_value": round(p_value, 4), "n_boot": int(n_boot),
            "n": int(delta.size)}
