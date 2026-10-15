"""vre.valuation.search_cache: dò tham số bị dừng giữa chừng chạy tiếp được, dữ liệu đổi ⇒ cache mất hiệu lực."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from vre.config import load_settings
from vre.valuation import search_cache as SC


def features(seed: int = 0) -> pd.DataFrame:
    return pd.DataFrame(np.random.default_rng(seed).normal(size=(400, 4)), columns=["a", "b", "c", "d"])


def dmatrices(seed: int = 0, bump: float = 0.0) -> tuple[xgb.DMatrix, xgb.DMatrix]:
    rng = np.random.default_rng(seed + 1)
    x = features(seed)
    y = 2 * x["a"] - x["b"] + rng.normal(scale=0.3, size=len(x))
    y.iloc[0] += bump
    dfit = xgb.QuantileDMatrix(x[:300], y[:300])
    return dfit, xgb.QuantileDMatrix(x[300:], y[300:], ref=dfit)


@pytest.fixture
def settings(tmp_dir):
    return load_settings(data_root=tmp_dir, xgb_nthread=1)


@pytest.fixture
def calls(monkeypatch):
    """Đếm số lần `xgb.train`; `calls["fail_at"] = n` ⇒ lần gọi thứ n ném lỗi (giả lập watchdog giết tiến trình)."""
    state = {"n": 0, "fail_at": None}
    real = xgb.train

    def counting(*args, **kwargs):
        state["n"] += 1
        if state["fail_at"] == state["n"]:
            raise MemoryError("watchdog")
        return real(*args, **kwargs)

    monkeypatch.setattr(xgb, "train", counting)
    monkeypatch.setattr(SC, "MAX_ROUNDS", 60)
    monkeypatch.setattr(SC, "EARLY_STOPPING", 10)
    return state


def test_interrupted_search_resumes_without_retraining_finished_configs(settings, tmp_dir, calls, capsys):
    dfit, dval = dmatrices()
    ref_booster, ref_params = SC.search(dfit, dval, "cpu", settings, None)  # không cache = hành vi cũ
    assert calls["n"] == SC.N_CONFIGS

    cdir, k = tmp_dir / "_search" / "tabular_sale", 3
    calls.update(n=0, fail_at=k + 1)
    with pytest.raises(MemoryError):
        SC.search(dfit, dval, "cpu", settings, cdir)
    assert sorted(p.name for p in cdir.glob("*.meta.json")) == [f"cfg_{i:02d}.meta.json" for i in range(k)]

    calls.update(n=0, fail_at=None)
    capsys.readouterr()
    booster, params = SC.search(dfit, dval, "cpu", settings, cdir)
    assert calls["n"] == SC.N_CONFIGS - k
    assert capsys.readouterr().out.count("(dùng lại)") == k
    assert params == ref_params
    x = features()[300:]
    np.testing.assert_array_equal(booster.predict(xgb.DMatrix(x)), ref_booster.predict(xgb.DMatrix(x)))

    calls.update(n=0)
    assert SC.search(dfit, dval, "cpu", settings, cdir)[1] == ref_params and calls["n"] == 0  # tất cả dùng lại


def test_changed_target_invalidates_cache(settings, tmp_dir, calls):
    cdir = tmp_dir / "_search" / "tabular_sale"
    SC.search(*dmatrices(), "cpu", settings, cdir)
    calls.update(n=0)
    SC.search(*dmatrices(bump=1.0), "cpu", settings, cdir)
    assert calls["n"] == SC.N_CONFIGS
    calls.update(n=0)
    SC.search(*dmatrices(bump=1.0), "cpu", settings, cdir)
    assert calls["n"] == 0  # cache đã ghi đè theo dữ liệu mới


def test_cache_key_covers_inputs_except_nthread():
    dfit, dval = dmatrices()
    fp = SC.data_fingerprint(dfit, dval, "cpu")
    cfg = SC.sample_configs()[0]
    key = SC.cache_key(cfg, fp)
    assert SC.cache_key(dict(cfg), dict(fp)) == key
    assert SC.cache_key({**cfg, "eta": 0.5}, fp) != key
    assert SC.cache_key(cfg, SC.data_fingerprint(dfit, dval, "cuda")) != key
    assert SC.cache_key(cfg, {**fp, "features": ["a", "b", "c", "x"]}) != key
    assert "nthread" not in fp


def test_clear_removes_dir_and_empty_parent(tmp_dir):
    a, b = tmp_dir / "_search" / "tabular_sale", tmp_dir / "_search" / "tabular_rent"
    for d in (a, b):
        d.mkdir(parents=True)
        (d / "cfg_00.json").write_text("{}", encoding="utf-8")
    SC.clear(a)
    assert not a.exists() and b.exists()
    SC.clear(b)
    assert not (tmp_dir / "_search").exists()
