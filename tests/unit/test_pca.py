"""Đặc trưng PCA (data-contract §4): trung bình chunk theo tin, fit chỉ trên tin tháng train, đủ mọi tin ở đầu ra."""

import datetime as dt
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vre.pipeline import pca
from vre.pipeline.state import layout_for


def test_listing_means_averages_contiguous_chunks():
    emb = np.array([[1, 0], [3, 0], [0, 2], [5, 5]], dtype=np.float16)
    ids, means = pca.listing_means(emb, np.array([7, 7, 9, 4]))
    assert ids.tolist() == [7, 9, 4]
    assert means.dtype == np.float32 and means.tolist() == [[2, 0], [0, 2], [5, 5]]
    with pytest.raises(ValueError):
        pca.listing_means(emb, np.array([7, 9, 7, 4]))


def test_train_mask_uses_train_months_only():
    assert pca.train_mask(["2025-06", "2026-01", "2026-02", None]).tolist() == [True, True, False, False]


def test_choose_sample_is_global_and_bounded():
    picks = pca.choose_sample([5, 0, 3], size=6, seed=1)
    assert sum(len(p) for p in picks) == 6 and len(picks[1]) == 0
    assert all(p.max(initial=-1) < n for p, n in zip(picks, [5, 0, 3], strict=True))
    assert sum(len(p) for p in pca.choose_sample([2, 2], size=100)) == 4


def _fixture(tmp_path, n_parts=2, part_size=40, dim=8):
    rng = np.random.default_rng(0)
    settings = SimpleNamespace(index_dir=tmp_path / "index", results_dir=tmp_path / "results", collection="c",
                               dryrun_collection="d", clean_path=tmp_path / "listings.parquet", part_size=part_size,
                               pca_dim=3, pca_fit_sample=1000, features_dir=tmp_path / "features")
    layout = layout_for(settings, False)
    n = n_parts * part_size
    months = [dt.datetime(2025, 12, 1) if i % 2 == 0 else dt.datetime(2026, 3, 1) for i in range(n)]
    pq.write_table(pa.table({"listing_id": list(range(n)), "published_at": months}), settings.clean_path,
                   row_group_size=25)
    for part in range(n_parts):
        lids = np.repeat(np.arange(part * part_size, (part + 1) * part_size), rng.integers(1, 4, part_size))
        layout.chunks(part).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"listing_id": lids}), layout.chunks(part))
        layout.emb(part).parent.mkdir(parents=True, exist_ok=True)
        np.save(layout.emb(part), rng.normal(size=(len(lids), dim)).astype(np.float16))
    return settings, layout, n


def test_run_fits_on_train_and_writes_every_listing(tmp_path, monkeypatch):
    from sklearn.decomposition import IncrementalPCA

    fitted = []
    original = IncrementalPCA.partial_fit
    monkeypatch.setattr(IncrementalPCA, "partial_fit",
                        lambda self, X, *a, **k: (fitted.append(np.asarray(X)), original(self, X, *a, **k))[1])
    settings, layout, n = _fixture(tmp_path)
    out = pca.run(settings, layout, 2, log=lambda _: None)
    table = pq.read_table(out["path"])
    assert out["path"].endswith("emb_pca3_e5_large.parquet")
    assert table.num_rows == n == out["n_listings"]
    assert table.column_names == ["listing_id", "pca_00", "pca_01", "pca_02"]
    assert table.column("listing_id").to_pylist() == list(range(n))
    train_means = []
    for part in range(2):
        ids, means = pca._part_vectors(layout, part)
        train_means.append(means[ids % 2 == 0])
    expected = np.concatenate(train_means)
    got = np.concatenate(fitted)
    assert out["n_fit"] == len(expected) == len(got)
    assert {tuple(r) for r in got.round(5)} == {tuple(r) for r in expected.round(5)}  # không có tin tháng test
    assert not list(settings.features_dir.glob("*.tmp"))
