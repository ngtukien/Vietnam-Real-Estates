"""Checkpoint `night_state.json` (data-contract §4.1): resume, ghi nguyên tử, fingerprint lệch ⇒ lỗi, dry-run tách riêng."""

import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vre.pipeline import stages
from vre.pipeline.state import Ledger, StateError, layout_for

FP = {"n_listings": 100, "price_vnd_sum": 1.5e12}


def _settings(tmp_path):
    return SimpleNamespace(index_dir=tmp_path / "index", results_dir=tmp_path / "results", collection="vre_chunks_e5_large",
                           dryrun_collection="vre_dryrun")


def test_layouts_keep_dry_run_separate(tmp_path):
    real, dry = layout_for(_settings(tmp_path), False), layout_for(_settings(tmp_path), True)
    assert real.state_path == tmp_path / "index" / "state" / "night_state.json"
    assert real.report_path == tmp_path / "results" / "night_report.json"
    assert real.emb(3) == tmp_path / "index" / "emb" / "e5_large" / "part_0003.npy"
    assert dry.collection == "vre_dryrun" and dry.root == tmp_path / "index" / "dryrun"
    for path in (dry.state_path, dry.report_path, dry.chunks(0), dry.tok(0), dry.bm25(0), dry.emb(0)):
        assert dry.root in path.parents
    assert real.manifest_path == tmp_path / "index" / "manifest_vre_chunks_e5_large.json"


def test_new_state_shape_and_atomic_write(tmp_path):
    layout = layout_for(_settings(tmp_path), False)
    ledger = Ledger.open(layout, 3, FP, resume=True)
    ledger.mark(0, "chunked", n_chunks=52)
    saved = json.loads(layout.state_path.read_text(encoding="utf-8"))
    assert saved["parts"]["0000"] == {"chunked": True, "bm25": False, "embedded": False, "uploaded": False, "n_chunks": 52}
    for key in ("schema_version", "model_slug", "collection", "n_parts", "listings_fingerprint", "bm25_avgdl",
                "hnsw_enabled", "hnsw_green", "alias_set", "pca_done", "runs"):
        assert key in saved
    assert not list(layout.state_path.parent.glob("*.tmp"))
    with pytest.raises(ValueError):
        ledger.mark(0, "indexed")


def test_resume_keeps_progress_and_runs(tmp_path):
    layout = layout_for(_settings(tmp_path), False)
    first = Ledger.open(layout, 2, FP, resume=True)
    first.start_run()
    first.mark(0, "embedded")
    assert first.running
    again = Ledger.open(layout, 2, FP, resume=True)
    assert again.done(0, "embedded") and not again.done(1, "embedded")
    again.end_run(ok=True)
    assert not again.running and again.state["runs"][-1]["exit"] == "ok"
    fresh = Ledger.open(layout, 2, FP, resume=False)
    assert fresh.state["parts"] == {} and fresh.state["runs"] == []


def test_fingerprint_mismatch_refuses_to_resume(tmp_path):
    layout = layout_for(_settings(tmp_path), False)
    Ledger.open(layout, 2, FP, resume=True)
    with pytest.raises(StateError, match="fingerprint"):
        Ledger.open(layout, 2, {**FP, "price_vnd_sum": FP["price_vnd_sum"] * 1.001}, resume=True)
    with pytest.raises(StateError):
        Ledger.open(layout, 2, {**FP, "n_listings": 99}, resume=True)
    Ledger.open(layout, 2, {**FP, "price_vnd_sum": FP["price_vnd_sum"] * (1 + 1e-12)}, resume=True)


def test_dry_run_state_does_not_touch_real_checkpoint(tmp_path):
    real = Ledger.open(layout_for(_settings(tmp_path), False), 54, FP, resume=True)
    real.mark(0, "uploaded")
    dry = Ledger.open(layout_for(_settings(tmp_path), True), 2, FP, resume=True)
    assert dry.part(0) == {} and dry.state["collection"] == "vre_dryrun"
    assert Ledger.open(layout_for(_settings(tmp_path), False), 54, FP, resume=True).done(0, "uploaded")


def test_reconcile_redoes_only_broken_parts(tmp_path):
    layout = layout_for(_settings(tmp_path), False)
    ledger = Ledger.open(layout, 2, FP, resume=True)
    for part, n in ((0, 3), (1, 2)):
        for stage in ("chunked", "bm25", "embedded"):
            ledger.mark(part, stage, n_chunks=n)
        layout.chunks(part).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"point_id": list(range(n))}), layout.chunks(part))
        layout.bm25(part).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"point_id": list(range(n))}), layout.bm25(part))
    layout.emb(0).parent.mkdir(parents=True, exist_ok=True)
    np.save(layout.emb(0), np.zeros((3, 4), np.float16))
    np.save(layout.emb(1), np.zeros((1, 4), np.float16))  # part 1 bị ghi dở: sai số hàng
    stages.reconcile(ledger, layout, 2, log=lambda _: None)
    assert ledger.done(0, "embedded") and ledger.done(0, "chunked")
    assert not ledger.done(1, "embedded") and not ledger.done(1, "chunked") and ledger.done(1, "bm25")
