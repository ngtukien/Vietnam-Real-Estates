"""vre.eval.ablation(+_subcorpus): subcorpus chứa 100% tin GT, tất định theo seed; chỉ số theo cụm; kế hoạch run;
kiểm cache HF; `corpus`/`subcorpus`; resume theo scope (không Qdrant/LLM/GPU)."""

import contextlib
import json
import math
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from vre.config import Settings
from vre.eval import ablation
from vre.eval import ablation_subcorpus as sub


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


@pytest.fixture
def all_ids():
    return np.arange(10_000, 60_000, 3, dtype=np.int64)  # bảng sạch giả, id rời rạc


def test_subcorpus_contains_all_gt(all_ids):
    gt = [int(all_ids[5]), int(all_ids[-1]), int(all_ids[12_345])]
    ids = sub.subcorpus_ids(all_ids, gt, n_random=100, seed=42)
    assert set(gt) <= set(ids.tolist())  # 100% tin GT
    assert 100 <= len(ids) <= 103
    assert np.all(np.diff(ids) > 0)  # tăng dần, duy nhất
    assert set(ids.tolist()) <= set(all_ids.tolist())


def test_subcorpus_deterministic_with_seed(all_ids):
    gt = [int(all_ids[7])]
    a = sub.subcorpus_ids(all_ids, gt, 500, seed=42)
    b = sub.subcorpus_ids(all_ids, gt, 500, seed=42)
    c = sub.subcorpus_ids(all_ids, gt, 500, seed=43)
    assert np.array_equal(a, b) and sub.ids_digest(a) == sub.ids_digest(b)
    assert not np.array_equal(a, c)
    assert sub.descriptor(a, 1, 500) == sub.descriptor(b, 1, 500)


def test_subcorpus_caps_at_corpus_and_rejects_unknown_gt(all_ids):
    assert len(sub.subcorpus_ids(all_ids, [], 10**9)) == len(all_ids)
    with pytest.raises(ValueError, match="không có trong bảng sạch"):
        sub.subcorpus_ids(all_ids, [1], 10)


def test_effective_chunk_tokens_fits_e5_max_length():
    assert sub.effective_chunk_tokens(256, 512, 3) == 256
    assert sub.effective_chunk_tokens(400, 512, 3) == 400
    assert sub.effective_chunk_tokens(512, 512, 3) == 507  # [bos] + 3 tiền tố + 507 + [eos] = 512


def test_run_names_unique_and_collection_names():
    order = ablation.canonical_order()
    assert len(order) == len(set(order)) == 12 + 18
    assert ablation.sub_run_name(512, "hybrid", True) == "c512_hybrid_prefilter"
    assert sub.collection_name(256) == "vre_ablation_e5_large_c256"


def test_kept_runs_matches_scope_reference():
    corpus, subc = {"n_listings": 1}, {"digest": "a"}
    old = {"corpus": corpus, "subcorpus": {"digest": "b"},
           "runs": [{"name": "full_bm25", "n_questions": 2}, {"name": "c256_bm25", "n_questions": 2},
                    {"name": "full_dense", "n_questions": 3}]}
    assert set(ablation.kept_runs(old, corpus, subc, 2)) == {"full_bm25"}
    assert set(ablation.kept_runs({**old, "subcorpus": subc}, corpus, subc, 2)) == {"full_bm25", "c256_bm25"}


def _fake_env(tmp_path):
    settings = Settings(data_root=tmp_path)
    settings.index_dir.mkdir(parents=True)
    (settings.index_dir / f"manifest_{settings.collection}.json").write_text(json.dumps({"n_listings": 123}))
    items = [ablation.Item("Q1", "căn hộ q7", (11, 12), frozenset({11})),
             ablation.Item("Q2", "nhà hn", (21,), frozenset({21}))]
    hits = pd.DataFrame({"listing_id": [11, 30], "dup_cluster_id": [11, 30]})
    return settings, items, hits


def _patches(items, hits, search, cached=False):
    return [mock.patch.object(ablation, "load_items", return_value=items),
            mock.patch.object(ablation, "extract_filters", return_value=({"Q1": None, "Q2": None}, 0, [])),
            mock.patch.object(ablation, "hf_cached", return_value=cached),
            mock.patch.object(ablation, "load_reranker"),
            mock.patch.object(sub, "all_listing_ids", return_value=np.array([11, 12, 21, 30, 40], dtype=np.int64)),
            mock.patch.object(ablation, "search", search), mock.patch.object(ablation, "release_gpu")]


RUN_KEYS = {"name", "embedding", "chunk_size", "retrieval", "prefilter", "rerank", "recall_at_5", "recall_at_10",
            "mrr", "ndcg_at_10", "n_questions", "latency_ms_p50"}


def test_run_shape_and_resume(tmp_path):
    settings, items, hits = _fake_env(tmp_path)
    out = tmp_path / "ablation.json"
    search = mock.Mock(return_value=(hits, False))
    with contextlib.ExitStack() as stack:
        for p in _patches(items, hits, search):
            stack.enter_context(p)
        doc = ablation.run(settings, scope="full", subcorpus_size=2, out=out, log=lambda m: None)
        assert search.call_count == 12  # 6 run × 2 câu
        ablation.run(settings, scope="full", subcorpus_size=2, out=out, log=lambda m: None)  # resume
        assert search.call_count == 12
    assert doc["corpus"] == {"n_listings": 123, "n_gt_listings": 3, "seed": 42}
    assert doc["subcorpus"]["n_gt_listings"] == 3 and doc["subcorpus"]["n_random"] == 2
    assert len(doc["runs"]) == 6 and any("rerank runs omitted" in n for n in doc["notes"])
    for r in doc["runs"]:
        assert set(r) == RUN_KEYS and r["embedding"] == "e5_large" and r["chunk_size"] == 400
        assert r["recall_at_5"] == pytest.approx(0.5) and r["n_questions"] == 2
    assert json.loads(out.read_text(encoding="utf-8"))["runs"] == doc["runs"]


def test_sub_scope_builds_measures_drops_and_keeps_full_runs(tmp_path):
    settings, items, hits = _fake_env(tmp_path)
    out = tmp_path / "ablation.json"
    search = mock.Mock(return_value=(hits, False))
    build = mock.Mock(side_effect=lambda s, size, ids, log, **kw: (sub.make_layout(s, size), {0: ids},
                                                             {"chunk_tokens": min(size, 507)}))
    with contextlib.ExitStack() as stack:
        for p in _patches(items, hits, search, cached=True):
            stack.enter_context(p)
        stack.enter_context(mock.patch.object(sub, "build_artifacts", build))
        stack.enter_context(mock.patch.object(sub, "load_collection", return_value={"n_points": 7}))
        drop = stack.enter_context(mock.patch.object(sub, "drop", return_value=True))
        stack.enter_context(mock.patch.object(sub, "use_collection", return_value=contextlib.nullcontext()))
        ablation.run(settings, scope="full", subcorpus_size=2, out=out, log=lambda m: None)
        assert search.call_count == 24  # 12 run full (rerank có trong cache) × 2 câu
        doc = ablation.run(settings, scope="sub", chunk_sizes=(256, 512), subcorpus_size=2, out=out,
                           log=lambda m: None)
        assert search.call_count == 24 + 24  # 2 cỡ × 6 run × 2 câu
        assert drop.call_count == 2 and build.call_count == 2
        again = ablation.run(settings, scope="sub", chunk_sizes=(256,), subcorpus_size=2, out=out, log=lambda m: None)
        assert build.call_count == 2  # resume: không dựng lại
    assert any(n.startswith("c512: header+body capped") for n in again["notes"])  # ghi chú của tiến trình trước
    names = [r["name"] for r in doc["runs"]]
    assert names == ablation.canonical_order()[:12] + [n for n in ablation.canonical_order()[12:]
                                                       if n[:4] in ("c256", "c512")]
    assert all(set(r) == RUN_KEYS for r in doc["runs"])
    assert {r["chunk_size"] for r in doc["runs"] if r["name"].startswith("c512")} == {512}
    assert doc["subcorpus_builds"]["c512"]["collection_deleted"] is True
    assert any("c512: header+body capped at 507" in n for n in doc["notes"])
    assert set(doc["subcorpus"]) == {"n_listings", "n_gt_listings", "n_random", "seed", "digest"}


def test_extract_filters_offline_skips_misses_and_never_calls(monkeypatch):
    from vre.llm import client
    from vre.rag import filters as rf

    def fake_parse(question):
        if question == "miss":
            raise client.LLMCacheMiss("x")
        return rf.QueryFilters(), client.LLMResult("", {}, 1, 1, True)

    monkeypatch.setenv(client.OFFLINE_ENV, "1")
    monkeypatch.setattr(rf, "parse_filters", fake_parse)
    items = [ablation.Item("A", "hit", (1,), frozenset({1})), ablation.Item("B", "miss", (2,), frozenset({2}))]
    out, calls, missing = ablation.extract_filters(items, lambda _m: None)
    assert out == {"A": None} and calls == 0 and missing == ["B"]
    monkeypatch.setattr(rf, "parse_filters", lambda q: (rf.QueryFilters(), client.LLMResult("", {}, 1, 1, False)))
    with pytest.raises(RuntimeError, match="offline"):
        ablation.extract_filters(items[:1], lambda _m: None)


def test_build_only_builds_without_filters_or_json(tmp_path):
    settings, items, hits = _fake_env(tmp_path)
    out = tmp_path / "ablation.json"
    build = mock.Mock(return_value=(None, {}, {"chunk_tokens": 256}))
    with mock.patch.object(ablation, "load_items", return_value=items), \
            mock.patch.object(ablation, "extract_filters") as filters, \
            mock.patch.object(sub, "all_listing_ids", return_value=np.array([11, 12, 21, 30], dtype=np.int64)), \
            mock.patch.object(sub, "build_artifacts", build), mock.patch.object(ablation, "release_gpu"):
        doc = ablation.run(settings, scope="sub", chunk_sizes=(256,), subcorpus_size=1, build_only=True,
                           embed_token_budget=8192, vram_cap_gb=2.0, out=out, log=lambda m: None)
        with pytest.raises(ValueError, match="build_only"):
            ablation.run(settings, scope="full", build_only=True, out=out, log=lambda m: None)
    assert doc["subcorpus_builds"] == {"c256": {"chunk_tokens": 256}} and not out.exists()
    filters.assert_not_called()
    assert build.call_args.kwargs == {"embed_token_budget": 8192, "vram_cap_gb": 2.0}

