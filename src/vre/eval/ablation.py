"""Ablation truy xuất (phase-03 A4) → `results/ablation.json` (data-contract §7.7). Chỉ retrieval, không sinh câu trả lời.

Hai nhóm run, mọi run embedding `e5_large`, đo trên các câu `gt_kind=ids`:
- **Chỉ mục đầy đủ** (alias `vre_chunks`, chunk 400): dense / bm25 / hybrid × pre-filter có/không × rerank có/không.
  Bộ lọc = `parse_filters` (LLM, qua cache đĩa của `vre.llm`; số lời gọi không trúng cache được đếm vào `notes`).
- **Subcorpus** = `SUBCORPUS_SIZE` tin ngẫu nhiên (seed 42) ∪ mọi tin GT: `chunk_size` 256/400/512, hybrid + pre-filter.
  Chunk 400 tái dùng chunks/bm25/`.npy` của Phase 2; 256/512 chunk lại (`vre.index.chunk`) và embed (`vre.index.embed`).
  Artifact ở `index/ablation/c{size}/` (theo part nguồn, resume được), nạp vào `vre_ablation_e5_large_c{size}`;
  đo xong thì xoá collection và artifact.

Đơn vị chấm = `dup_cluster_id` (data-contract §2.2): danh sách tin truy xuất → cụm theo thứ tự, bỏ trùng; tập đúng =
`gt_dup_clusters`. Recall@k chia cho min(|đúng|, k) (câu ràng buộc có hàng trăm tin đúng). nDCG@10 nhị phân.
Resume: `ablation.json` ghi nguyên tử sau MỖI run; run cùng tên, cùng `n_questions`, cùng `corpus` thì bỏ qua.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vre.common import result_header, write_json_atomic
from vre.config import EMBED_MODEL_SLUG, Settings
from vre.eval import metrics

SCHEMA_VERSION = 1
SEED = 42
SUBCORPUS_SIZE = 200_000
CHUNK_SIZES = (256, 400, 512)
REUSED_CHUNK_SIZE = 400  # cỡ chunk của chỉ mục Phase 2 (data-contract §4.2)
RETRIEVALS = ("dense", "bm25", "hybrid")
K_RETRIEVE = 10  # đủ cho recall@10 / nDCG@10
COLLECTION_PREFIX = "vre_ablation_e5_large_c"
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
    return {
        "recall_at_5": float(np.mean([metrics.recall_at_k(r, g, 5, capped=True) for r, g in zip(retrieved, relevant, strict=True)])),
        "recall_at_10": float(np.mean([metrics.recall_at_k(r, g, 10, capped=True) for r, g in zip(retrieved, relevant, strict=True)])),
        "mrr": float(metrics.mrr(list(zip(retrieved, relevant, strict=True)))),
        "ndcg_at_10": float(np.mean([ndcg_at_k(r, g, 10) for r, g in zip(retrieved, relevant, strict=True)])),
        "n_questions": len(retrieved),
        "latency_ms_p50": round(float(np.percentile(latencies_ms, 50)), 1),
    }


# ---- subcorpus (thuần)

def subcorpus_ids(all_ids: np.ndarray, gt_ids: Iterable[int], n_random: int, seed: int = SEED) -> np.ndarray:
    """`n_random` tin ngẫu nhiên (theo vị trí trong `all_ids`, seed cố định) ∪ mọi tin GT; trả tăng dần, duy nhất.
    Tin GT không có trong `all_ids` ⇒ ValueError (GT và bảng sạch lệch nhau)."""
    all_ids = np.asarray(all_ids, dtype=np.int64)
    gt = np.unique(np.fromiter((int(i) for i in gt_ids), dtype=np.int64))
    missing = np.setdiff1d(gt, all_ids)
    if len(missing):
        raise ValueError(f"{len(missing)} tin GT không có trong bảng sạch (vd {missing[:5].tolist()})")
    rng = np.random.default_rng(seed)
    picked = all_ids[rng.choice(len(all_ids), size=min(n_random, len(all_ids)), replace=False)]
    return np.union1d(picked, gt)


def ids_digest(ids: np.ndarray) -> str:
    return hashlib.blake2b(np.asarray(ids, dtype=np.int64).tobytes(), digest_size=16).hexdigest()


def effective_chunk_tokens(chunk_size: int, max_length: int, prefix_len: int) -> int:
    """Số token header+body của chunk sao cho cả chuỗi `[bos] passage: … [eos]` ≤ `max_length` của e5-large
    (chunk 512 danh nghĩa phải co lại; 256/400 giữ nguyên)."""
    return min(chunk_size, max_length - 2 - prefix_len)


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

@contextmanager
def use_collection(name: str) -> Iterator[None]:
    """Cho `vre.rag` truy vấn collection `name` thay alias (giữ nguyên một đường truy vấn SSOT của `retrieve`)."""
    from vre.rag import qdrant_search as qs

    original = qs.collection
    qs.collection = lambda: name
    try:
        yield
    finally:
        qs.collection = original


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


def run_name(chunk_size: int | None, retrieval: str, prefilter: bool, rerank: bool) -> str:
    scope = "full" if chunk_size is None else f"c{chunk_size}"
    return "_".join([scope, retrieval, *(["prefilter"] if prefilter else []), *(["rerank"] if rerank else [])])


# ---- dựng subcorpus theo cỡ chunk

def _save_npy_atomic(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    os.replace(tmp, path)


def _source_parts(settings: Settings, ids: np.ndarray) -> dict[int, np.ndarray]:
    """Part nguồn (data-contract §4: dòng `[k·part_size, …)` của bảng sạch) → listing_id của subcorpus trong part đó."""
    import pyarrow.parquet as pq

    all_ids = pq.read_table(settings.clean_path, columns=["listing_id"]).column("listing_id").to_numpy()
    positions = np.flatnonzero(np.isin(all_ids, ids))
    if len(positions) != len(ids):
        raise ValueError(f"subcorpus có {len(ids)} tin nhưng chỉ thấy {len(positions)} trong bảng sạch")
    parts = positions // settings.part_size
    return {int(p): all_ids[positions[parts == p]] for p in np.unique(parts)}


def _part_rows(settings: Settings, part: int, ids: np.ndarray, columns: tuple[str, ...]) -> list[dict]:
    import pyarrow as pa
    import pyarrow.compute as pc

    from vre.index.chunk import read_part

    table = read_part(settings.clean_path, part, settings.part_size, columns)
    return table.filter(pc.is_in(table["listing_id"], value_set=pa.array(ids, pa.int64()))).to_pylist()


def _reuse_part(settings: Settings, layout, part: int, ids: np.ndarray) -> None:
    """Chunk 400: lọc chunks/bm25/emb Phase 2 của part về các tin subcorpus (giữ thứ tự hàng)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from vre.index.chunk import write_parquet_atomic
    from vre.pipeline.state import layout_for

    src = layout_for(settings, dry_run=False)
    chunks = pq.read_table(src.chunks(part))
    keep = pc.is_in(chunks["listing_id"], value_set=pa.array(ids, pa.int64()))
    rows = np.flatnonzero(keep.to_numpy(zero_copy_only=False))
    emb = np.load(src.emb(part), mmap_mode="r")
    bm = pq.read_table(src.bm25(part))
    if len(emb) != len(chunks) or len(bm) != len(chunks):
        raise ValueError(f"part {part:04d} Phase 2: chunks/emb/bm25 lệch số hàng")
    write_parquet_atomic(chunks.take(rows), layout.chunks(part))
    _save_npy_atomic(np.ascontiguousarray(emb[rows]), layout.emb(part))
    write_parquet_atomic(bm.take(rows), layout.bm25(part))


def _rechunk_part(settings: Settings, layout, part: int, ids: np.ndarray, tok, special, encoder) -> None:
    from vre.index import bm25
    from vre.index.chunk import PART_COLUMNS, chunk_table, iter_chunk_batches, write_parquet_atomic

    rows = _part_rows(settings, part, ids, PART_COLUMNS)
    records: list[tuple] = []
    sequences: list[list[int]] = []
    for rec, seqs, _ in iter_chunk_batches(rows, tok, special, settings):
        records += rec
        sequences += seqs
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    np.cumsum([len(s) for s in sequences], out=offsets[1:])
    flat = np.fromiter((t for s in sequences for t in s), dtype=np.int64, count=int(offsets[-1]))
    emb = encoder.encode_ids(flat, offsets)
    write_parquet_atomic(chunk_table(records), layout.chunks(part))
    _save_npy_atomic(emb, layout.emb(part))
    bm25.bm25_part(layout, part)


def build_artifacts(settings: Settings, size: int, ids: np.ndarray, log: Log) -> tuple[Any, dict[int, np.ndarray], dict]:
    """Artifact subcorpus cho cỡ `size` dưới `index/ablation/c{size}/`; part đủ 3 file thì bỏ qua (resume).
    Dấu vân tay subcorpus lệch ⇒ xoá thư mục làm lại."""
    from vre.index import tokenize
    from vre.pipeline.state import Layout

    root = Path(settings.index_dir) / "ablation" / f"c{size}"
    layout = Layout(root, root / "state.json", root / "report.json", f"{COLLECTION_PREFIX}{size}", False)
    tok = tokenize.load_fast_tokenizer()
    special = tokenize.special_ids(tok)
    tokens = effective_chunk_tokens(size, settings.embed_max_length, len(special.passage_prefix))
    fingerprint = {"digest": ids_digest(ids), "n_listings": int(len(ids)), "chunk_tokens": tokens,
                   "overlap": settings.chunk_overlap}
    if layout.state_path.exists() and json.loads(layout.state_path.read_text(encoding="utf-8")) != fingerprint:
        log(f"c{size}: artifact cũ của subcorpus khác — xoá {root}")
        shutil.rmtree(root)
    write_json_atomic(layout.state_path, fingerprint)
    parts = _source_parts(settings, ids)
    chunk_settings = settings.model_copy(update={"chunk_size": tokens})
    encoder, start, built = None, time.perf_counter(), 0
    for part, part_ids in parts.items():
        if all(p.exists() for p in (layout.chunks(part), layout.emb(part), layout.bm25(part))):
            continue
        if size == REUSED_CHUNK_SIZE:
            _reuse_part(settings, layout, part, part_ids)
        else:
            if encoder is None:
                from vre.index.embed import get_encoder

                encoder = get_encoder(settings)
            _rechunk_part(chunk_settings, layout, part, part_ids, tok, special, encoder)
        built += 1
        log(f"c{size}: part {part:04d} xong ({built}/{len(parts)} cần dựng, {time.perf_counter() - start:.0f}s)")
    return layout, parts, {"chunk_tokens": tokens, "build_seconds": round(time.perf_counter() - start, 1),
                           "parts_built": built, "reused_phase2": size == REUSED_CHUNK_SIZE}


def load_collection(settings: Settings, layout, parts: dict[int, np.ndarray], log: Log) -> dict:
    """Nạp artifact vào `layout.collection` (avgdl của chính subcorpus). Có sẵn đủ point ⇒ không nạp lại;
    số point lệch ⇒ xoá, tạo lại, nạp; sau cùng chờ HNSW green và kiểm đếm chính xác."""
    import pyarrow.parquet as pq

    from vre.index import bm25, qdrant_store, qdrant_upload

    qc = qdrant_store.client(settings)
    name, start = layout.collection, time.perf_counter()
    avgdl = bm25.corpus_avgdl(layout, list(parts))
    expected = sum(pq.ParquetFile(layout.chunks(p)).metadata.num_rows for p in parts)
    if qc.collection_exists(name) and qc.count(name, exact=True).count != expected:
        log(f"{name}: có sẵn nhưng lệch số point — xoá tạo lại")
        qc.delete_collection(name)
    counters: dict[str, int] = {}
    if qdrant_store.create_collection(settings, qc, name):
        for part, part_ids in parts.items():
            chunks = pq.read_table(layout.chunks(part), columns=["point_id", "listing_id", "chunk_index", "text"])
            table = pq.read_table(layout.bm25(part))
            emb = np.load(layout.emb(part), mmap_mode="r")
            listings = {r["listing_id"]: qdrant_upload.listing_payload(r)
                        for r in _part_rows(settings, part, part_ids, qdrant_upload.PAYLOAD_COLUMNS)}

            def upload(chunks=chunks, emb=emb, table=table, listings=listings) -> None:
                qc.upload_points(name, points=qdrant_upload.build_points(chunks, emb, table, listings, avgdl),
                                 batch_size=settings.qdrant_upload_batch, parallel=settings.qdrant_upload_parallel,
                                 max_retries=5, wait=True)

            qdrant_upload.resilient(settings, upload, counters, log)
        log(f"{name}: nạp {expected} point trong {time.perf_counter() - start:.0f}s")
    state = qdrant_store.wait_green(qc, name, settings.qdrant_green_timeout_s, log)
    n = qc.count(name, exact=True).count
    if n != expected:
        raise RuntimeError(f"{name}: Qdrant có {n} point, artifact có {expected}")
    return {"n_points": n, "avgdl": round(avgdl, 3), "upload_seconds": round(time.perf_counter() - start, 1),
            "upload_retries": counters.get("upload_retries", 0), "status": state["status"]}


def drop_subcorpus(settings: Settings, layout, log: Log) -> None:
    from vre.index import qdrant_store

    qc = qdrant_store.client(settings)
    if qc.collection_exists(layout.collection):
        qc.delete_collection(layout.collection)
    shutil.rmtree(layout.root, ignore_errors=True)
    log(f"đã xoá {layout.collection} và {layout.root}")


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

def _load_previous(path: Path, corpus: dict) -> dict[str, dict]:
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("corpus") != corpus:
        return {}
    return {r["name"]: r for r in doc.get("runs", [])}


FULL_RUNS = tuple((r, p, k) for r in RETRIEVALS for p in (False, True) for k in (False, True))


def run(settings: Settings, limit: int | None = None, *, subcorpus_size: int | None = None,
        full_runs: Iterable[tuple[str, bool, bool]] = FULL_RUNS, chunk_sizes: Iterable[int] = CHUNK_SIZES,
        out: Path | None = None, log: Log = _print) -> dict:
    """Chạy (tiếp) mọi run ablation, ghi `results/ablation.json` sau mỗi run, trả dict cuối.
    `limit` = trần số câu hỏi VÀ số tin ngẫu nhiên của subcorpus (smoke). Tham số keyword chỉ để smoke/test:
    `subcorpus_size` (ghi đè số tin ngẫu nhiên), `full_runs` (retrieval, prefilter, rerank), `chunk_sizes`, `out`."""
    import pyarrow.parquet as pq

    if limit is not None and limit <= 0:
        raise ValueError("limit phải > 0")
    items = load_items(settings, limit)
    gt_ids = {i for it in items for i in it.gt_ids}
    n_random = subcorpus_size or (min(limit, SUBCORPUS_SIZE) if limit else SUBCORPUS_SIZE)
    all_ids = pq.read_table(settings.clean_path, columns=["listing_id"]).column("listing_id").to_numpy()
    ids = subcorpus_ids(all_ids, gt_ids, n_random, SEED)
    corpus = {"n_listings": int(len(ids)), "n_gt_listings": len(gt_ids), "seed": SEED}
    path = out or Path(settings.results_dir) / FILENAME
    previous = _load_previous(path, corpus)
    doc: dict[str, Any] = {**result_header(SCHEMA_VERSION), "corpus": corpus, "runs": [],
                           "notes": [], "subcorpus_builds": {}}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("corpus") == corpus:
            doc["subcorpus_builds"] = old.get("subcorpus_builds", {})
    done = {n: r for n, r in previous.items() if r.get("n_questions") == len(items)}
    filters, calls = extract_filters(items, log)
    doc["notes"].append(f"filter_llm_calls={calls} (câu ngoài cache LLM khi chạy lần này)")

    def record(name: str, chunk_size: int, retrieval: str, prefilter: bool, rerank: bool) -> None:
        if name in done:
            doc["runs"].append(done[name])
            log(f"{name}: đã có, bỏ qua")
            return
        start = time.perf_counter()
        scores, drops = measure(items, filters, retrieval, prefilter, rerank, settings)
        doc["runs"].append({"name": name, "embedding": EMBED_MODEL_SLUG, "chunk_size": chunk_size,
                            "retrieval": retrieval, "prefilter": prefilter, "rerank": rerank, **scores})
        if drops:
            doc["notes"].append(f"{name}: {drops} câu bộ lọc cho 0 kết quả ⇒ bỏ lọc")
        write_json_atomic(path, doc)
        log(f"{name}: recall@5={scores['recall_at_5']:.3f} ndcg@10={scores['ndcg_at_10']:.3f} "
            f"({time.perf_counter() - start:.1f}s)")

    try:
        for retrieval, prefilter, rerank in full_runs:
            record(run_name(None, retrieval, prefilter, rerank), settings.chunk_size, retrieval, prefilter, rerank)
        for size in chunk_sizes:
            name = run_name(size, "hybrid", True, False)
            if name in done:
                record(name, size, "hybrid", True, False)
                continue
            layout, parts, build = build_artifacts(settings, size, ids, log)
            release_gpu()  # encoder re-embed không cần nữa trước khi đo (rerank tắt ở run subcorpus)
            build.update(load_collection(settings, layout, parts, log))
            doc["subcorpus_builds"][f"c{size}"] = build
            if build["chunk_tokens"] != size:
                doc["notes"].append(f"c{size}: header+body {build['chunk_tokens']} token để cả chuỗi vừa max_length "
                                    f"{settings.embed_max_length} của e5-large")
            with use_collection(layout.collection):
                record(name, size, "hybrid", True, False)
            drop_subcorpus(settings, layout, log)
    finally:
        release_gpu()
    write_json_atomic(path, doc)
    return doc
