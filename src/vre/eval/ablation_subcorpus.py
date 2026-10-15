"""Subcorpus của ablation `chunk_size` (phase-03 A4, data-contract §5 `vre_ablation_e5_large_c{size}`).

Subcorpus = `n_random` tin ngẫu nhiên (seed 42) ∪ mọi tin GT của câu `gt_kind=ids`. Với mỗi cỡ chunk:
- 400 (cỡ của Phase 2) lọc lại chunks/bm25/`.npy` của Phase 2 về các tin subcorpus (không embed lại);
- 256/512 chunk lại (`vre.index.chunk`) và embed (`vre.index.embed`, giữ trần CUDA của nó, loader không mmap).
Artifact ở `index/ablation/c{size}/` theo part nguồn (resume được: part đủ 3 file thì bỏ qua), nạp vào
`vre_ablation_e5_large_c{size}`; đo xong thì `drop` xoá collection và artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from vre.common import write_json_atomic
from vre.config import Settings

SEED = 42
SUBCORPUS_SIZE = 200_000
CHUNK_SIZES = (256, 400, 512)
REUSED_CHUNK_SIZE = 400  # cỡ chunk của chỉ mục Phase 2 (data-contract §4.2)
COLLECTION_PREFIX = "vre_ablation_e5_large_c"
Log = Callable[[str], None]


# ---- thuần

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


def collection_name(size: int) -> str:
    return f"{COLLECTION_PREFIX}{size}"


def descriptor(ids: np.ndarray, n_gt: int, n_random: int) -> dict[str, Any]:
    """Khoá `subcorpus` của `ablation.json`: run `c{size}_*` cũ chỉ được tái dùng khi khớp đúng khối này."""
    return {"n_listings": int(len(ids)), "n_gt_listings": int(n_gt), "n_random": int(n_random), "seed": SEED,
            "digest": ids_digest(ids)}


# ---- truy vấn trên collection subcorpus

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


# ---- dựng artifact

def _save_npy_atomic(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    os.replace(tmp, path)


def all_listing_ids(settings: Settings) -> np.ndarray:
    import pyarrow.parquet as pq

    return pq.read_table(settings.clean_path, columns=["listing_id"]).column("listing_id").to_numpy()


def source_parts(settings: Settings, ids: np.ndarray) -> dict[int, np.ndarray]:
    """Part nguồn (data-contract §4: dòng `[k·part_size, …)` của bảng sạch) → listing_id của subcorpus trong part đó."""
    all_ids = all_listing_ids(settings)
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


def _reuse_part(settings: Settings, layout: Any, part: int, ids: np.ndarray) -> int:
    """Chunk 400: lọc chunks/bm25/emb Phase 2 của part về các tin subcorpus (giữ thứ tự hàng). Trả số chunk."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from vre.index.chunk import write_parquet_atomic
    from vre.pipeline.state import layout_for

    src = layout_for(settings, dry_run=False)
    chunks = pq.read_table(src.chunks(part))
    keep = pc.is_in(chunks["listing_id"], value_set=pa.array(ids, pa.int64()))
    rows = np.flatnonzero(keep.to_numpy(zero_copy_only=False))
    emb = np.load(src.emb(part), mmap_mode="r")  # .npy chỉ đọc: ánh xạ file, không tính vào commit
    bm = pq.read_table(src.bm25(part))
    if len(emb) != len(chunks) or len(bm) != len(chunks):
        raise ValueError(f"part {part:04d} Phase 2: chunks/emb/bm25 lệch số hàng")
    write_parquet_atomic(chunks.take(rows), layout.chunks(part))
    _save_npy_atomic(np.ascontiguousarray(emb[rows]), layout.emb(part))
    write_parquet_atomic(bm.take(rows), layout.bm25(part))
    return len(rows)


def _rechunk_part(settings: Settings, layout: Any, part: int, ids: np.ndarray, tok: Any, special: Any,
                  encoder: Any) -> tuple[int, int]:
    """Chunk lại + embed + BM25 một part. Trả (số chunk, số tin bị cắt ở `max_chunks_per_listing`)."""
    from vre.index import bm25
    from vre.index.chunk import PART_COLUMNS, chunk_table, iter_chunk_batches, write_parquet_atomic

    rows = _part_rows(settings, part, ids, PART_COLUMNS)
    records: list[tuple] = []
    sequences: list[list[int]] = []
    truncated = 0
    for rec, seqs, trunc in iter_chunk_batches(rows, tok, special, settings):
        records += rec
        sequences += seqs
        truncated += trunc
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    np.cumsum([len(s) for s in sequences], out=offsets[1:])
    flat = np.fromiter((t for s in sequences for t in s), dtype=np.int64, count=int(offsets[-1]))
    del sequences
    emb = encoder.encode_ids(flat, offsets)
    write_parquet_atomic(chunk_table(records), layout.chunks(part))
    _save_npy_atomic(emb, layout.emb(part))
    bm25.bm25_part(layout, part)
    return len(records), truncated


def make_layout(settings: Settings, size: int) -> Any:
    from vre.pipeline.state import Layout

    root = Path(settings.index_dir) / "ablation" / f"c{size}"
    return Layout(root, root / "state.json", root / "report.json", collection_name(size), False)


def build_artifacts(settings: Settings, size: int, ids: np.ndarray, log: Log) -> tuple[Any, dict[int, np.ndarray], dict]:
    """Artifact subcorpus cho cỡ `size` dưới `index/ablation/c{size}/`; part đủ 3 file thì bỏ qua (resume).
    Dấu vân tay subcorpus lệch ⇒ xoá thư mục làm lại."""
    from vre.index import tokenize

    layout = make_layout(settings, size)
    tok = tokenize.load_fast_tokenizer()
    special = tokenize.special_ids(tok)
    tokens = effective_chunk_tokens(size, settings.embed_max_length, len(special.passage_prefix))
    fingerprint = {"digest": ids_digest(ids), "n_listings": int(len(ids)), "chunk_tokens": tokens,
                   "overlap": settings.chunk_overlap}
    if layout.state_path.exists() and json.loads(layout.state_path.read_text(encoding="utf-8")) != fingerprint:
        log(f"c{size}: artifact cũ của subcorpus khác — xoá {layout.root}")
        shutil.rmtree(layout.root)
    write_json_atomic(layout.state_path, fingerprint)
    parts = source_parts(settings, ids)
    chunk_settings = settings.model_copy(update={"chunk_size": tokens})
    encoder, start, built, n_chunks, truncated = None, time.perf_counter(), 0, 0, 0
    for part, part_ids in parts.items():
        if all(p.exists() for p in (layout.chunks(part), layout.emb(part), layout.bm25(part))):
            continue
        if size == REUSED_CHUNK_SIZE:
            n_chunks += _reuse_part(settings, layout, part, part_ids)
        else:
            if encoder is None:
                from vre.index.embed import get_encoder

                encoder = get_encoder(settings)
            n, t = _rechunk_part(chunk_settings, layout, part, part_ids, tok, special, encoder)
            n_chunks, truncated = n_chunks + n, truncated + t
        built += 1
        elapsed = time.perf_counter() - start
        log(f"c{size}: part {part:04d} xong ({built}/{len(parts)} part, {n_chunks} chunk mới, {elapsed:.0f}s, "
            f"{n_chunks / max(elapsed, 1e-9):.0f} chunk/s)")
    return layout, parts, {"chunk_tokens": tokens, "build_seconds": round(time.perf_counter() - start, 1),
                           "parts_built": built, "n_chunks_built": n_chunks, "truncated_listings": truncated,
                           "reused_phase2": size == REUSED_CHUNK_SIZE}


def load_collection(settings: Settings, layout: Any, parts: dict[int, np.ndarray], log: Log) -> dict:
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


def drop(settings: Settings, size: int, log: Log) -> bool:
    """Xoá collection `vre_ablation_e5_large_c{size}` và artifact của nó. Trả True nếu collection đã tồn tại."""
    from vre.index import qdrant_store

    layout = make_layout(settings, size)
    qc = qdrant_store.client(settings)
    existed = bool(qc.collection_exists(layout.collection))
    if existed:
        qc.delete_collection(layout.collection)
    if qc.collection_exists(layout.collection):
        raise RuntimeError(f"không xoá được {layout.collection}")
    shutil.rmtree(layout.root, ignore_errors=True)
    log(f"đã xoá {layout.collection} ({'có' if existed else 'không có'} trên Qdrant) và {layout.root}")
    return existed
