"""vre.eval.bootstrap trên dữ liệu giả có chênh lệch biết trước: CI chứa giá trị thật, CI hẹp lại khi n tăng."""

import numpy as np
import pytest

from vre.eval import bootstrap


def _pair(n: int, true_diff: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Điểm 0/1 ghép cặp: b ~ Bernoulli(0.4), a = b nhưng thêm true_diff xác suất thắng ở câu b sai."""
    rng = np.random.default_rng(seed)
    b = (rng.random(n) < 0.4).astype(float)
    flip = rng.random(n) < true_diff / 0.6
    a = np.where(b == 0, flip.astype(float), 1.0)
    return a, b


def test_paired_ci_covers_true_difference():
    a, b = _pair(400, 0.2)
    result = bootstrap.paired(a, b)
    assert result is not None
    lo, hi = result["ci95"]
    assert lo <= 0.2 <= hi
    assert lo > 0  # chênh lệch thật 0,2 trên 400 câu: có ý nghĩa
    assert result["p_value"] < 0.01
    assert result["n_boot"] == 10_000 and result["n"] == 400


def test_ci_narrows_with_n():
    widths = []
    for n in (50, 200, 800):
        a, b = _pair(n, 0.1, seed=n)
        lo, hi = bootstrap.paired(a, b)["ci95"]
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]


def test_deterministic_seed_42():
    a, b = _pair(100, 0.1)
    assert bootstrap.paired(a, b) == bootstrap.paired(a, b)
    assert bootstrap.mean_ci(a) == bootstrap.mean_ci(a)


def test_no_difference_has_high_p():
    a = np.array([1.0, 0.0] * 50)
    result = bootstrap.paired(a, a.copy())
    assert result["diff"] == 0 and result["p_value"] == 1.0 and result["ci95"] == [0.0, 0.0]


def test_mean_ci_contains_mean_and_empty_is_none():
    x = np.array([1.0] * 30 + [0.0] * 70)
    lo, hi = bootstrap.mean_ci(x)
    assert lo <= 0.3 <= hi
    assert bootstrap.mean_ci([]) is None
    assert bootstrap.paired([], []) is None


def test_rejects_mismatch_and_nan():
    with pytest.raises(ValueError):
        bootstrap.paired([1.0, 0.0], [1.0])
    with pytest.raises(ValueError):
        bootstrap.mean_ci([1.0, float("nan")])


def test_negative_diff_p_value_counts_opposite_sign():
    a, b = _pair(300, 0.15, seed=3)
    result = bootstrap.paired(b, a)  # đảo chiều: diff âm
    assert result["diff"] < 0 and result["ci95"][1] < 0 and result["p_value"] < 0.01
