"""vre.eval.ablation: chỉ số theo cụm, kế hoạch run, kiểm cache HF, `corpus` từ manifest, resume (không Qdrant/LLM)."""

import json
import math
from unittest import mock

import pandas as pd
import pytest

from vre.config import Settings
from vre.eval import ablation


def test_score_run_cluster_metrics():
    retrieved = [[7, 1, 2], [3, 4, 5, 6, 8, 9]]
    relevant = [{1}, set(range(100, 120))]
    s = ablation.score_run(retrieved, relevant, [10.0, 30.0])
    assert s["recall_at_5"] == pytest.approx(0.5)
    assert s["mrr"] == pytest.approx(0.25)
    assert s["ndcg_at_10"] == pytest.approx((1 / math.log2(3)) / 2)
    assert s["n_questions"] == 2 and s["latency_ms_p50"] == 20.0


def test_ndcg_local_perfect_and_capped():
    assert ablation._ndcg_local([1, 2, 3], {1, 2, 3}, 10) == pytest.approx(1.0)
    assert ablation._ndcg_local(list(range(10)), set(range(50)), 10) == pytest.approx(1.0)
    assert ablation._ndcg_local([9], {1}, 10) == 0.0


def test_dedup_clusters_keeps_order():
    assert ablation.dedup_clusters([5, 3, 5, 1, 3]) == [5, 3, 1]


def test_plan_runs_rerank_only_when_cached():
    assert len(ablation.plan_runs(False)) == 6 and not any(r for _, _, r in ablation.plan_runs(False))
    names = {ablation.run_name(*r) for r in ablation.plan_runs(True)}
    assert len(names) == 12 and "full_hybrid_prefilter_rerank" in names


def test_hf_cached(tmp_path):
    assert not ablation.hf_cached("org/model", tmp_path)
    snap = tmp_path / "hub" / "models--org--model" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    assert not ablation.hf_cached("org/model", tmp_path)
    (snap / "config.json").write_text("{}")
    assert ablation.hf_cached("org/model", tmp_path)


def _fake_env(tmp_path):
    settings = Settings(data_root=tmp_path)
    settings.index_dir.mkdir(parents=True)
    (settings.index_dir / f"manifest_{settings.collection}.json").write_text(json.dumps({"n_listings": 123}))
    items = [ablation.Item("Q1", "căn hộ q7", (11, 12), frozenset({11})),
             ablation.Item("Q2", "nhà hn", (21,), frozenset({21}))]
    hits = pd.DataFrame({"listing_id": [11, 30], "dup_cluster_id": [11, 30]})
    return settings, items, hits


def test_run_shape_and_resume(tmp_path):
    settings, items, hits = _fake_env(tmp_path)
    out = tmp_path / "ablation.json"
    search = mock.Mock(return_value=(hits, False))
    with mock.patch.object(ablation, "load_items", return_value=items), \
            mock.patch.object(ablation, "extract_filters", return_value=({"Q1": None, "Q2": None}, 0)), \
            mock.patch.object(ablation, "hf_cached", return_value=False), \
            mock.patch.object(ablation, "search", search), mock.patch.object(ablation, "release_gpu"):
        doc = ablation.run(settings, out=out, log=lambda m: None)
        assert search.call_count == 12  # 6 run × 2 câu
        ablation.run(settings, out=out, log=lambda m: None)  # resume: không truy vấn lại
        assert search.call_count == 12
    assert doc["corpus"] == {"n_listings": 123, "n_gt_listings": 3, "seed": 42}
    assert len(doc["runs"]) == 6 and any("rerank runs omitted" in n for n in doc["notes"])
    keys = {"name", "embedding", "chunk_size", "retrieval", "prefilter", "rerank", "recall_at_5", "recall_at_10",
            "mrr", "ndcg_at_10", "n_questions", "latency_ms_p50"}
    for r in doc["runs"]:
        assert set(r) == keys and r["embedding"] == "e5_large" and r["chunk_size"] == 400
        assert r["recall_at_5"] == pytest.approx(0.5) and r["n_questions"] == 2
    assert json.loads(out.read_text(encoding="utf-8"))["runs"] == doc["runs"]
