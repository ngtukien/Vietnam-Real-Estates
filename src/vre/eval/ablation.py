"""Ablation truy xuất (phase-03 A4) → `results/ablation.json` (data-contract §7.7). Chỉ retrieval, không sinh câu trả lời.

Hai nhóm run, mọi run embedding `e5_large`, đo trên các câu `gt_kind=ids`:
- **Chỉ mục đầy đủ** (alias `vre_chunks`, chunk 400 của Phase 2, tên `full_*`): dense / bm25 / hybrid × pre-filter
  có/không × rerank có/không (`RERANK_MODEL` phải có trong cache HF — tải trước bằng `--download-rerank`; vắng ⇒ bỏ
  run rerank và ghi `notes`). Cross-encoder chạy trên `rerank_device` (mặc định CPU: không thêm CUDA vào commit).
- **Subcorpus** (`vre.eval.ablation_subcorpus`, tên `c{size}_*`) = 200.000 tin ngẫu nhiên (seed 42) ∪ mọi tin GT:
  `chunk_size` 256/400/512 × dense / bm25 / hybrid × pre-filter có/không, không rerank. Collection
  `vre_ablation_e5_large_c{size}` xoá ngay sau khi đo cỡ đó. Khối `subcorpus` của json mô tả tập tin đã dùng.

Bộ lọc = `parse_filters` (LLM, qua cache đĩa của `vre.llm`); số lời gọi không trúng cache được ghi vào `notes`.
Đơn vị chấm = `dup_cluster_id` (data-contract §2.2): danh sách tin truy xuất → cụm theo thứ tự, bỏ trùng; tập đúng =
`gt_dup_clusters`. Recall@k chia cho min(|đúng|, k) (câu ràng buộc có hàng trăm tin đúng). nDCG@10 nhị phân.
Resume: `ablation.json` ghi nguyên tử sau MỖI run; run cũ được giữ khi cùng `n_questions` và cùng `corpus` (run
`full_*`) / cùng `subcorpus` (run `c*`). Mỗi lần gọi chỉ chạy `scope`/`chunk_sizes` được chọn, các run khác giữ nguyên
(để mỗi cỡ chunk chạy một tiến trình riêng — trả hết bộ nhớ CUDA/commit khi tiến trình thoát).
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
from vre.eval import ablation_subcorpus as sub
from vre.eval import metrics

SCHEMA_VERSION = 1
SEED = sub.SEED  # khoá `corpus.seed` của §7.7 (seed lấy mẫu subcorpus)
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


def sub_run_name(chunk_size: int, retrieval: str, prefilter: bool) -> str:
    return "_".join([f"c{chunk_size}", retrieval, *(["prefilter"] if prefilter else [])])


def plan_runs(rerank_available: bool) -> list[tuple[str, bool, bool]]:
    reranks = (False, True) if rerank_available else (False,)
    return [(r, p, k) for r in RETRIEVALS for p in (False, True) for k in reranks]


def plan_sub_runs() -> list[tuple[str, bool]]:
    return [(r, p) for r in RETRIEVALS for p in (False, True)]


def canonical_order() -> list[str]:
    """Thứ tự run trong json: full (kể cả rerank) rồi từng cỡ chunk."""
    return [run_name(*r) for r in plan_runs(True)] + \
        [sub_run_name(c, r, p) for c in sub.CHUNK_SIZES for r, p in plan_sub_runs()]


def hf_cached(model_id: str, hf_home: str | os.PathLike | None = None) -> bool:
    """Model có snapshot trong cache HF (`$HF_HOME/hub/models--org--name/snapshots/*/config.json`)? Không gọi mạng."""
    root = Path(hf_home or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
    snapshots = root / f"models--{model_id.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any((s / "config.json").exists() for s in snapshots.iterdir())


RERANK_FILES = ("config.json", "model.safetensors", "sentencepiece.bpe.model", "special_tokens_map.json",
                "tokenizer.json", "tokenizer_config.json", "README.md")  # bỏ onnx/openvino/pytorch_model.bin (~3 GB)


def download_rerank_model(settings: Settings, log: Log = _print) -> str:
    """Tải `RERANK_MODEL` vào `$HF_HOME` (chỉ file cần cho `sentence_transformers.CrossEncoder`). Có sẵn ⇒ không tải."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(settings.rerank_model, allow_patterns=list(RERANK_FILES))
    if not hf_cached(settings.rerank_model):
        raise RuntimeError(f"tải {settings.rerank_model} xong nhưng không thấy snapshot trong HF_HOME")
    log(f"rerank model: {path}")
    return str(path)


def load_reranker(settings: Settings, device: str) -> None:
    """Nạp cross-encoder một lần vào `vre.rag.retrieve._reranker` trên `device` (mặc định của ST là CUDA nếu có)."""
    from sentence_transformers import CrossEncoder

    from vre.rag import retrieve

    retrieve._reranker = CrossEncoder(settings.rerank_model, device=device)


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


def extract_filters(items: list[Item], log: Log) -> tuple[dict[str, Any], int, list[str]]:
    """Bộ lọc LLM mỗi câu (cache đĩa của `vre.llm`). Trả (id → QueryFilters | None nếu rỗng, số lời gọi LLM thật,
    id câu chưa có bộ lọc trong cache khi đang ở chế độ chỉ-cache `VRE_LLM_OFFLINE=1`)."""
    from vre.llm.client import LLMCacheMiss, offline
    from vre.rag.filters import parse_filters

    out, calls, missing = {}, 0, []
    for it in items:
        try:
            filters, result = parse_filters(it.question)
        except LLMCacheMiss:
            missing.append(it.id)
            continue
        calls += 0 if result.cached else 1
        out[it.id] = None if filters.is_empty() else filters
    if offline() and calls:
        raise RuntimeError(f"ablation offline nhưng có {calls} lời gọi LLM thật — không được phép")
    log(f"bộ lọc: {len(items)} câu, {calls} lời gọi LLM ngoài cache, {len(missing)} câu chưa có trong cache")
    return out, calls, missing


# ---- truy xuất

def search(question: str, retrieval: str, filters: Any, rerank: bool, settings: Settings) -> tuple[pd.DataFrame, bool]:
    """Một truy vấn như `vre.rag.retrieve.retrieve` nhưng chọn được dense/bm25/hybrid. Bộ lọc cho 0 kết quả ⇒ nới dần
    rồi mới bỏ hết (`retrieve.search_relaxed`, như hệ thống thật). Trả (hits, đã bỏ HẾT lọc?)."""
    from vre.rag import retrieve as rt
    from vre.rag.filters import FILTER_DROPPED

    fn = {"dense": rt.search_dense, "bm25": rt.search_bm25, "hybrid": rt.search_hybrid}[retrieval]
    pool = max(settings.rerank_candidates, K_RETRIEVE) if rerank else K_RETRIEVE
    hits, note = rt.search_relaxed(fn, question, pool, filters)
    dropped = note == FILTER_DROPPED
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
    """Thả encoder GPU (re-embed) và cross-encoder rerank, trả bộ nhớ CUDA."""
    from vre.index import embed
    from vre.rag import retrieve

    embed._encoder = None
    retrieve._reranker = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---- điều phối

SCOPES = ("all", "full", "sub")


def index_corpus(settings: Settings, n_gt_listings: int) -> dict:
    """`corpus` của §7.7 mô tả chỉ mục đầy đủ (manifest Phase 2 của collection sau alias)."""
    path = Path(settings.index_dir) / f"manifest_{settings.collection}.json"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}; chỉ mục Phase 2 chưa xong")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {"n_listings": int(manifest["n_listings"]), "n_gt_listings": n_gt_listings, "seed": SEED}


def kept_runs(old: dict[str, Any], corpus: dict, subcorpus: dict, n_questions: int) -> dict[str, dict]:
    """Run cũ còn hợp lệ: cùng số câu; `full_*` cùng `corpus`, `c*` cùng `subcorpus`."""
    kept = {}
    for r in old.get("runs", []):
        if r.get("n_questions") != n_questions:
            continue
        reference = ("corpus", corpus) if r["name"].startswith("full_") else ("subcorpus", subcorpus)
        if old.get(reference[0]) == reference[1]:
            kept[r["name"]] = r
    return kept


def run(settings: Settings, limit: int | None = None, *, scope: str = "all",
        chunk_sizes: Iterable[int] = sub.CHUNK_SIZES, subcorpus_size: int = sub.SUBCORPUS_SIZE,
        rerank_device: str = "cpu", build_only: bool = False, embed_token_budget: int | None = None,
        vram_cap_gb: float | None = None, out: Path | None = None, log: Log = _print) -> dict:
    """Chạy (tiếp) các run của `scope` (`all|full|sub`), ghi `results/ablation.json` sau mỗi run, trả dict cuối.
    `limit` = trần số câu hỏi; `subcorpus_size` = số tin ngẫu nhiên (smoke: vài nghìn); `out` ghi đè đường dẫn.
    `build_only` (scope sub): chỉ dựng artifact (encoder GPU) rồi thoát, không bộ lọc LLM, không encoder truy vấn,
    không ghi ablation.json — tiến trình đo sau đó chỉ dùng CPU (máy thiếu commit: hai việc không chung tiến trình)."""
    if limit is not None and limit <= 0:
        raise ValueError("limit phải > 0")
    if scope not in SCOPES:
        raise ValueError(f"scope phải thuộc {SCOPES}")
    sizes = tuple(int(c) for c in chunk_sizes)
    if bad := [c for c in sizes if c not in sub.CHUNK_SIZES]:
        raise ValueError(f"chunk_size {bad} ngoài {sub.CHUNK_SIZES}")
    if subcorpus_size <= 0:
        raise ValueError("subcorpus_size phải > 0")
    items = load_items(settings, limit)
    path = out or Path(settings.results_dir) / FILENAME
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    gt_all = {i for it in items for i in it.gt_ids}  # tin GT của MỌI câu ids, kể cả câu thiếu cache bộ lọc
    ids = sub.subcorpus_ids(sub.all_listing_ids(settings), gt_all, subcorpus_size, SEED)
    subcorpus = sub.descriptor(ids, len(gt_all), subcorpus_size)
    if build_only:
        if scope != "sub":
            raise ValueError("build_only chỉ dùng với scope=sub")
        builds = {}
        for size in sizes:
            _, _, builds[f"c{size}"] = sub.build_artifacts(settings, size, ids, log, embed_token_budget=embed_token_budget,
                                                            vram_cap_gb=vram_cap_gb)
            release_gpu()
        return {"subcorpus": subcorpus, "subcorpus_builds": builds}
    notes: list[str] = []
    filters, calls, missing = extract_filters(items, log)
    notes.append(f"filter_llm_calls={calls} (questions not served from the LLM cache in this run)")
    if missing:  # chế độ chỉ-cache: mọi run đo trên CÙNG tập câu có bộ lọc trong cache để so sánh công bằng
        items = [it for it in items if it.id not in missing]
        if not items:
            raise RuntimeError("không câu nào có bộ lọc trong cache LLM (chế độ chỉ-cache)")
        notes.append(f"offline: {len(missing)} question(s) without cached LLM filters excluded from every run: "
                     f"{', '.join(missing)}; filters model = {settings.llm_model}")
    corpus = index_corpus(settings, len({i for it in items for i in it.gt_ids}))
    runs = kept_runs(old, corpus, subcorpus, len(items))
    builds = dict(old.get("subcorpus_builds") or {}) if old.get("subcorpus") == subcorpus else {}
    rerank_ok = hf_cached(settings.rerank_model)
    if not rerank_ok:
        notes.append(f"rerank runs omitted: RERANK_MODEL {settings.rerank_model} not in HF cache "
                     "(download with `ablation --download-rerank`)")
    notes.append(f"rerank: cross-encoder {settings.rerank_model} on {rerank_device}, rescoring the top "
                 f"{max(settings.rerank_candidates, K_RETRIEVE)} candidates down to {K_RETRIEVE}")
    notes.append("c256/c400/c512 runs are measured on the subcorpus (see `subcorpus`), not on the full index: "
                 "compare them with each other, not with full_* runs")
    doc: dict[str, Any] = {**result_header(SCHEMA_VERSION), "corpus": corpus, "subcorpus": subcorpus, "runs": [],
                           "subcorpus_builds": builds, "notes": notes}
    order = {n: i for i, n in enumerate(canonical_order())}

    def flush() -> None:
        doc["runs"] = sorted(runs.values(), key=lambda r: order.get(r["name"], len(order)))
        write_json_atomic(path, doc)

    def record(name: str, chunk_size: int, retrieval: str, prefilter: bool, rerank: bool) -> None:
        if name in runs:
            log(f"{name}: đã có, bỏ qua")
            return
        start = time.perf_counter()
        scores, drops = measure(items, filters, retrieval, prefilter, rerank, settings)
        runs[name] = {"name": name, "embedding": EMBED_MODEL_SLUG, "chunk_size": chunk_size, "retrieval": retrieval,
                      "prefilter": prefilter, "rerank": rerank, **scores}
        if drops:
            notes.append(f"{name}: {drops} question(s) had 0 hits under the filter ⇒ filter dropped")
        flush()
        log(f"{name}: recall@5={scores['recall_at_5']:.3f} ndcg@10={scores['ndcg_at_10']:.3f} "
            f"({time.perf_counter() - start:.1f}s, private {sub.private_gb():.2f} GB)")

    try:
        if scope in ("all", "full"):
            plan = plan_runs(rerank_ok)
            if any(k and run_name(r, p, k) not in runs for r, p, k in plan):
                load_reranker(settings, rerank_device)
            for retrieval, prefilter, rerank in plan:
                record(run_name(retrieval, prefilter, rerank), settings.chunk_size, retrieval, prefilter, rerank)
            release_gpu()
        if scope in ("all", "sub"):
            for size in sizes:
                if all(sub_run_name(size, r, p) in runs for r, p in plan_sub_runs()):
                    log(f"c{size}: mọi run đã có, bỏ qua")
                    continue
                layout, parts, build = sub.build_artifacts(settings, size, ids, log, allow_embed=False)
                release_gpu()  # encoder re-embed xong; run subcorpus không rerank
                build.update(sub.load_collection(settings, layout, parts, log))
                builds[f"c{size}"] = build
                if build["chunk_tokens"] != size:
                    notes.append(f"c{size}: header+body capped at {build['chunk_tokens']} tokens so the whole "
                                 f"sequence fits e5-large max_length {settings.embed_max_length}")
                start = time.perf_counter()
                with sub.use_collection(layout.collection):
                    for retrieval, prefilter in plan_sub_runs():
                        record(sub_run_name(size, retrieval, prefilter), size, retrieval, prefilter, False)
                build["measure_seconds"] = round(time.perf_counter() - start, 1)
                build["collection_deleted"] = sub.drop(settings, size, log)
                flush()
    finally:
        release_gpu()
    flush()
    return doc
