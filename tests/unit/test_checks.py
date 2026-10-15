"""vre.checks: mỗi kiểm tra §9.2 có ca fail và ca pass trên dữ kiện giả; require() raise khi có fail."""

import pytest

from vre import checks

CASES = {
    "raw_shards_present": ({"missing": []}, {"missing": ["shard_0003.parquet"]}),
    "e5_large_cached": ({"repo_dir": "x", "n_snapshots": 1}, {"repo_dir": "x", "n_snapshots": 0}),
    "gpu": ({"cuda_available": True, "device": "RTX"}, {"cuda_available": False, "device": None}),
    "qdrant_healthy": ({"url": "u", "status": 200}, {"url": "u", "status": None, "error": "refused"}),
    "neo4j_healthy": ({"uri": "b", "ok": True}, {"uri": "b", "ok": False, "error": "down"}),
    "d_free_space": ({"drive": "D:", "free_gb": 90.0, "min_gb": 45}, {"drive": "D:", "free_gb": 10.0, "min_gb": 45}),
    "groq_key_present": ({"present": True}, {"present": False}),
    "env_paths_on_d": ({"violations": {}}, {"violations": {"TMP": r"C:\t"}}),
    "clean_data_ready": (
        {"path": "p", "exists": True, "actual": {"n_listings": 10, "price_vnd_sum": 5.0},
         "signal": {"n_listings": 10, "price_vnd_sum": 5.0}},
        {"path": "p", "exists": True, "actual": {"n_listings": 10, "price_vnd_sum": 5.0},
         "signal": {"n_listings": 11, "price_vnd_sum": 5.0}},
    ),
}


def test_every_contract_check_is_registered():
    assert set(checks.CHECKS) == set(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_check_passes_and_fails(name):
    good, bad = CASES[name]
    check = checks.CHECKS[name][1]
    assert check(good)[0] is True
    assert check(bad)[0] is False


def test_d_free_space_fails_on_other_drive():
    assert checks.check_d_free_space({"drive": "C:", "free_gb": 500.0, "min_gb": 45})[0] is False


def test_clean_data_ready_fails_without_file_or_signal():
    assert checks.check_clean_data_ready({"path": "p", "exists": False, "actual": None, "signal": None})[0] is False
    assert checks.check_clean_data_ready({"path": "p", "exists": True, "actual": None, "signal": None})[0] is False


def test_require_raises_on_failure(monkeypatch):
    monkeypatch.setitem(checks.CHECKS, "gpu", (lambda s: {"cuda_available": False}, checks.check_gpu))
    with pytest.raises(checks.PreconditionError, match="gpu"):
        checks.require(object(), "gpu", echo=lambda _: None)


def test_require_passes(monkeypatch):
    monkeypatch.setitem(checks.CHECKS, "gpu", (lambda s: {"cuda_available": True, "device": "x"}, checks.check_gpu))
    assert checks.require(object(), "gpu", echo=lambda _: None)["gpu"][0] is True


def test_unknown_check_name():
    with pytest.raises(KeyError):
        checks.run_checks(object(), "nope")
