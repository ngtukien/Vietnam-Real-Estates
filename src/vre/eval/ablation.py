"""Ablation truy xuất (phase-03 A4) → `results/ablation.json` (data-contract §7.7). Chỉ retrieval, không sinh câu trả lời.

Mọi run trên chỉ mục đầy đủ (alias `vre_chunks`, e5-large, chunk 400 của Phase 2), đo trên các câu `gt_kind=ids`:
dense / bm25 / hybrid × pre-filter có/không, × rerank có/không CHỈ khi `RERANK_MODEL` đã có trong cache HF (không
bao giờ tải; vắng ⇒ bỏ run rerank và ghi `notes`). Quét `chunk_size` trên subcorpus đã bị lead cắt khỏi phạm vi
(2026-10-15, tiết kiệm thời gian) nên mọi run có `chunk_size = Settings.chunk_size` (400).

Bộ lọc = `parse_filters` (LLM, qua cache đĩa của `vre.llm`); số lời gọi không trúng cache được ghi vào `notes`.
Đơn vị chấm = `dup_cluster_id` (data-contract §2.2): danh sách tin truy xuất → cụm theo thứ tự, bỏ trùng; tập đúng =
`gt_dup_clusters`. Recall@k chia cho min(|đúng|, k) (câu ràng buộc có hàng trăm tin đúng). nDCG@10 nhị phân.
Resume: `ablation.json` ghi nguyên tử sau MỖI run; run cùng tên, cùng `n_questions`, cùng `corpus` thì bỏ qua.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vre.common import result_header, write_json_atomic
from vre.config import EMBED_MODEL_SLUG, Settings
from vre.eval import metrics

SCHEMA_VERSION = 1
SEED = 42  # khoá `corpus.seed` của §7.7; không còn lấy mẫu ngẫu nhiên (đo trên toàn chỉ mục)
RETRIEVALS = ("dense", "bm25", "hybrid")
K_RETRIEVE = 10  # đủ cho recall@10 / nDCG@10
FILENAME = "ablation.json"
Log = Callable[[str], None]


def _print(message: str) -> None:
    print(f"[ablation] {message}", flush=True)


# ---- chỉ số (thuần)

def _ndcg_local(retrieved: list, relevant: set, k: int) -> float:
    """nDCG@k nhị phân; IDCG = min(|relevant|, k) vị trí đầu đều đúng."""
    if not relevant:
        return float("nan")
    dcg = sum(1 / math.log2(pos + 1) for pos, doc in enumerate(retrieved[:k], start=1) if doc in relevant)
    ideal = sum(1 / math.log2(pos + 1) for pos in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


ndcg_at_k = getattr(metrics, "ndcg_at_k", _ndcg_local)


def dedup_clusters(clusters: Iterable[int]) -> list[int]:
    return [int(c) for c in dict.fromkeys(int(c) for c in clusters)]


def score_run(retrieved: list[list[int]], relevant: list[set[int]], latencies_ms: list[float]) -> dict[str, Any]:
    """Các khoá chỉ số của một run (data-contract §7.7). `retrieved[i]` = cụm theo thứ tự, đã bỏ trùng."""
    if not retrieved or len(retrieved) != len(relevant):
        raise ValueError("retrieved/relevant rỗng hoặc lệch độ dài")
    pairs = list(zip(retrieved, relevant, strict=True))
    return {
        "recall_at_5": float(np.mean([metrics.recall_at_k(r, g, 5, capped=True) for r, g in pairs])),
        "recall_at_10": float(np.mean([metrics.recall_at_k(r, g, 10, capped=True) for r, g in pairs])),
        "mrr": float(metrics.mrr(pairs)),
        "ndcg_at_10": float(np.mean([ndcg_at_k(r, g, 10) for r, g in pairs])),
        "n_questions": len(retrieved),
        "latency_ms_p50": round(float(np.percentile(latencies_ms, 50)), 1),
    }


def run_name(retrieval: str, prefilter: bool, rerank: bool) -> str:
    return "_".join(["full", retrieval, *(["prefilter"] if prefilter else []), *(["rerank"] if rerank else [])])


def plan_runs(rerank_available: bool) -> list[tuple[str, bool, bool]]:
    reranks = (False, True) if rerank_available else (False,)
    return [(r, p, k) for r in RETRIEVALS for p in (False, True) for k in reranks]


def hf_cached(model_id: str, hf_home: str | os.PathLike | None = None) -> bool:
    """Model có snapshot trong cache HF (`$HF_HOME/hub/models--org--name/snapshots/*/config.json`)? Không gọi mạng."""
    root = Path(hf_home or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
    snapshots = root / f"models--{model_id.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any((s / "config.json").exists() for s in snapshots.iterdir())


# ---- câu hỏi + bộ lọc

@dataclass(frozen=True)
class Item:
    id: str
    question: str
    gt_ids: tuple[int, ...]
    gt_clusters: frozenset[int]


def load_items(settings: Settings, limit: int | None) -> list[Item]:
    from vre.eval.ground_truth import load_gt
    from vre.eval.questions import load_questions

    gt = load_gt(settings)
    items = []
    for q in load_questions():
        if q.gt_kind != "ids":
            continue
        g = gt.get(q.id)
        if not g or not g.get("gt_ids") or not g.get("gt_dup_clusters"):
            raise ValueError(f"{q.id}: thiếu gt_ids/gt_dup_clusters trong benchmark_gt.json; chạy lại `eval gt`")
        items.append(Item(q.id, q.question, tuple(int(i) for i in g["gt_ids"]),
                          frozenset(int(c) for c in g["gt_dup_clusters"])))
    if not items:
        raise ValueError("không có câu gt_kind=ids nào")
    return items[:limit] if limit else items


def extract_filters(items: list[Item], log: Log) -> tuple[dict[str, Any], int]:
    """Bộ lọc LLM mỗi câu (cache đĩa của `vre.llm`). Trả (id → QueryFilters | None nếu rỗng, số lời gọi LLM thật)."""
    from vre.rag.filters import parse_filters

    out, calls = {}, 0
    for it in items:
        filters, result = parse_filters(it.question)
        calls += 0 if result.cached else 1
        out[it.id] = None if filters.is_empty() else filters
    log(f"bộ lọc: {len(items)} câu, {calls} lời gọi LLM ngoài cache")
    return out, calls


# ---- truy xuất

def search(question: str, retrieval: str, filters: Any, rerank: bool, settings: Settings) -> tuple[pd.DataFrame, bool]:
    """Một truy vấn như `vre.rag.retrieve.retrieve` nhưng chọn được dense/bm25/hybrid. Bộ lọc cho 0 kết quả ⇒ bỏ lọc
    (như hệ thống thật). Trả (hits, đã bỏ lọc?)."""
    from vre.rag import retrieve as rt

    fn = {"dense": rt.search_dense, "bm25": rt.search_bm25, "hybrid": rt.search_hybrid}[retrieval]
    pool = max(settings.rerank_candidates, K_RETRIEVE) if rerank else K_RETRIEVE
    hits, dropped = fn(question, pool, filters), False
    if hits.empty and filters is not None:
        hits, dropped = fn(question, pool, None), True
    if rerank and not hits.empty:
        hits = rt.rerank(question, hits, top=K_RETRIEVE)
    return hits, dropped


def measure(items: list[Item], filters: dict[str, Any], retrieval: str, prefilter: bool, rerank: bool,
            settings: Settings) -> tuple[dict[str, Any], int]:
    retrieved, latencies, drops = [], [], 0
    for it in items:
        start = time.perf_counter()
        hits, dropped = search(it.question, retrieval, filters[it.id] if prefilter else None, rerank, settings)
        latencies.append((time.perf_counter() - start) * 1000)
        drops += dropped
        retrieved.append(dedup_clusters(hits["dup_cluster_id"].dropna().astype("int64")) if not hits.empty else [])
    return score_run(retrieved, [set(it.gt_clusters) for it in items], latencies), drops


def release_gpu() -> None:
    """Thả cross-encoder rerank (GPU nếu có) và trả bộ nhớ CUDA."""
    from vre.rag import retrieve

    retrieve._reranker = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---- điều phối

def index_corpus(settings: Settings, n_gt_listings: int) -> dict:
    """`corpus` của §7.7 mô tả chỉ mục đầy đủ (manifest Phase 2 của collection sau alias)."""
    path = Path(settings.index_dir) / f"manifest_{settings.collection}.json"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}; chỉ mục Phase 2 chưa xong")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {"n_listings": int(manifest["n_listings"]), "n_gt_listings": n_gt_listings, "seed": SEED}


def run(settings: Settings, limit: int | None = None, *, out: Path | None = None, log: Log = _print) -> dict:
    """Chạy (tiếp) mọi run ablation, ghi `results/ablation.json` sau mỗi run, trả dict cuối.
    `limit` = trần số câu hỏi (smoke). `out` (chỉ cho smoke/test) ghi đè đường dẫn kết quả."""
    if limit is not None and limit <= 0:
        raise ValueError("limit phải > 0")
    items = load_items(settings, limit)
    corpus = index_corpus(settings, len({i for it in items for i in it.gt_ids}))
    path = out or Path(settings.results_dir) / FILENAME
    previous: dict[str, dict] = {}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("corpus") == corpus:
            previous = {r["name"]: r for r in old.get("runs", []) if r.get("n_questions") == len(items)}
    doc: dict[str, Any] = {**result_header(SCHEMA_VERSION), "corpus": corpus, "runs": [], "notes": []}
    rerank_ok = hf_cached(settings.rerank_model)
    if not rerank_ok:
        doc["notes"].append(f"rerank runs omitted: RERANK_MODEL {settings.rerank_model} not in HF cache (never downloaded)")
    doc["notes"].append("chunk_size sweep (256/400/512 subcorpus) dropped by lead scope cut 2026-10-15")
    filters, calls = extract_filters(items, log)
    doc["notes"].append(f"filter_llm_calls={calls} (questions not served from the LLM cache in this run)")
    try:
        for retrieval, prefilter, rerank in plan_runs(rerank_ok):
            name = run_name(retrieval, prefilter, rerank)
            if name in previous:
                doc["runs"].append(previous[name])
                log(f"{name}: đã có, bỏ qua")
                continue
            start = time.perf_counter()
            scores, drops = measure(items, filters, retrieval, prefilter, rerank, settings)
            doc["runs"].append({"name": name, "embedding": EMBED_MODEL_SLUG, "chunk_size": settings.chunk_size,
                                "retrieval": retrieval, "prefilter": prefilter, "rerank": rerank, **scores})
            if drops:
                doc["notes"].append(f"{name}: {drops} question(s) had 0 hits under the filter ⇒ filter dropped")
            write_json_atomic(path, doc)
            log(f"{name}: recall@5={scores['recall_at_5']:.3f} ndcg@10={scores['ndcg_at_10']:.3f} "
                f"({time.perf_counter() - start:.1f}s)")
    finally:
        release_gpu()
    write_json_atomic(path, doc)
    return doc
