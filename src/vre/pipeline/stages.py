"""Các bước của lượt chạy dài, tách khỏi phần điều phối luồng (`night.py`).

Hàm `*_task` là điểm vào của tiến trình con (pool `spawn`) nên phải ở cấp module và nhận đối số pickle được.
"""

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from vre.common import listings_fingerprint, write_json_atomic
from vre.config import EMBED_MODEL_SLUG, EMBED_MODELS
from vre.index import chunk, qdrant_store
from vre.pipeline import pca
from vre.pipeline.state import STAGES, Ledger, now_iso

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
SMOKE_QUERIES = ("căn hộ 2 phòng ngủ quận 7 dưới 4 tỷ", "nhà mặt tiền kinh doanh Hà Nội", "đất nền sổ đỏ Bình Dương")
SMOKE_MIN_HITS = 5


def offline_env() -> None:
    """Không bao giờ gọi mạng HF trong lượt chạy (đặt trước khi tạo pool để tiến trình con kế thừa)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Tiến trình con (tokenize/BM25/upload) không dùng BLAS: mặc định OpenBLAS mở 1 luồng + buffer cho mỗi lõi (32)
    # trong MỖI tiến trình ⇒ cạn commit memory ("OpenBLAS error: Memory allocation still failed"). Tiến trình chính đã
    # nạp numpy trước đó nên không bị ảnh hưởng.
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def keep_awake(on: bool) -> None:
    """Giữ máy thức chỉ trong tiến trình này (không đổi thiết lập nguồn của hệ thống); tắt lại trong `finally`."""
    if sys.platform == "win32":
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))


def chunk_task(args) -> dict:
    from vre.index.chunk import chunk_part

    settings, layout, part = args
    return chunk_part(settings, layout, part)


def bm25_task(args) -> dict:
    from vre.index.bm25 import bm25_part

    layout, part = args
    return bm25_part(layout, part)


def _rows(path: Path) -> int | None:
    return pq.ParquetFile(path).metadata.num_rows if path.exists() else None


def reconcile(ledger, layout, n_parts: int, log=print) -> None:
    """Trước khi chạy tiếp: cờ nào không còn khớp artifact trên đĩa thì hạ xuống để làm lại đúng part đó."""
    for part in range(n_parts):
        entry = ledger.part(part)
        n = entry.get("n_chunks")
        if entry.get("chunked") and _rows(layout.chunks(part)) != n:
            log(f"part {part:04d}: chunks hỏng/thiếu ⇒ làm lại từ chunk")
            for stage in STAGES:
                ledger.mark(part, stage, False)
            continue
        if entry.get("chunked") and not entry.get("embedded") and not layout.tok(part).exists():
            log(f"part {part:04d}: thiếu token tạm ⇒ chunk lại")
            ledger.mark(part, "chunked", False)
        if entry.get("bm25") and _rows(layout.bm25(part)) != n:
            log(f"part {part:04d}: bm25 hỏng/thiếu ⇒ làm lại")
            ledger.mark(part, "bm25", False)
        if entry.get("embedded"):
            path = layout.emb(part)
            shape = np.load(path, mmap_mode="r").shape if path.exists() else None
            if shape is None or shape[0] != n:
                log(f"part {part:04d}: emb {shape} ≠ {n} chunk ⇒ chunk + embed lại")
                for stage in ("chunked", "embedded", "uploaded"):
                    ledger.mark(part, stage, False)


def prepare(job, resume: bool, log=print) -> bool:
    """Nạp/tạo checkpoint, đối chiếu artifact, tạo collection. False ⇒ mọi thứ đã xong, không làm gì thêm."""
    s, layout = job.settings, job.layout
    job.report.begin("checks")
    fingerprint = listings_fingerprint(s.clean_path)
    n_parts = chunk.n_parts(fingerprint["n_listings"], s.part_size)
    job.n_parts = min(n_parts, s.dryrun_parts) if job.dry_run else n_parts
    job.ledger = ledger = Ledger.open(layout, job.n_parts, fingerprint, resume)
    reconcile(ledger, layout, job.n_parts, log)
    st = ledger.state
    finished = all(ledger.done(p, "uploaded") for p in range(job.n_parts)) and st["hnsw_green"] \
        and (job.dry_run or (st["alias_set"] and st["pca_done"]))
    if resume and finished and layout.report_path.exists():
        log("Lượt chạy dài đã xong (state + report khớp) — không làm gì thêm.")
        return False
    job.qc = qdrant_store.client(s)
    if not resume and job.qc.collection_exists(layout.collection):
        log(f"--no-resume: xoá collection {layout.collection} để nạp lại từ đầu")
        job.qc.delete_collection(layout.collection)
    qdrant_store.create_collection(s, job.qc, layout.collection)
    ledger.set(hnsw_enabled=True)
    ledger.start_run()
    job.report.end("checks", note="vre.checks.require (CLI) + fingerprint/state")
    return True


def resume_uploads(ledger, n_parts: int, log=print) -> list[int]:
    """In các part được bỏ qua khi tiếp tục; trả các part đã embed nhưng chưa upload (đưa thẳng vào hàng upload)."""
    waiting = []
    for part in range(n_parts):
        if ledger.done(part, "uploaded"):
            log(f"skip part {part:04d} (đã xong)")
        elif ledger.done(part, "embedded"):
            log(f"skip part {part:04d} (đã embed, chỉ còn upload)")
            waiting.append(part)
    return waiting


def embed_part(encoder, layout, part: int, tokens) -> int:
    """Embed token của part, ghi `.npy` nguyên tử (`.tmp` + `os.replace`). Trả số chunk."""
    ids, offsets = tokens
    vectors = encoder.encode_ids(ids, offsets)
    path = layout.emb(part)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, vectors)
    os.replace(tmp, path)
    return len(vectors)


def smoke_all(settings, collection: str) -> dict:
    hits = {q: len(qdrant_store.smoke_search(settings, q, collection=collection, limit=10)) for q in SMOKE_QUERIES}
    weak = {q: n for q, n in hits.items() if n < SMOKE_MIN_HITS}
    if weak:
        raise RuntimeError(f"smoke_search < {SMOKE_MIN_HITS} tin: {weak}")
    return hits


def finalize(job, log=print) -> None:
    """HNSW green → alias → count = Σ n_chunks → smoke 3 truy vấn → manifest → PCA (dry-run: bỏ alias, manifest, PCA)."""
    s, layout, qc, ledger, report = job.settings, job.layout, job.qc, job.ledger, job.report
    job.step = "hnsw"
    report.begin("hnsw")
    report.qdrant = qdrant_store.wait_green(qc, layout.collection, s.qdrant_green_timeout_s, log)
    ledger.set(hnsw_green=True)
    report.end("hnsw")
    job.step = "verify"
    report.begin("verify")
    if not job.dry_run:
        qdrant_store.set_alias(qc, s.collection_alias, layout.collection)
        ledger.set(alias_set=True)
    points, expected = qc.count(layout.collection, exact=True).count, ledger.n_chunks()
    report.counters["points_in_qdrant"] = points
    if points != expected:
        raise RuntimeError(f"Qdrant có {points} point ≠ Σ n_chunks {expected}")
    hits = smoke_all(s, layout.collection)
    if not job.dry_run:
        fp = ledger.state["listings_fingerprint"]
        write_json_atomic(layout.manifest_path, {
            "schema_version": 1, "collection": layout.collection, "model_slug": EMBED_MODEL_SLUG,
            "dim": EMBED_MODELS[EMBED_MODEL_SLUG][1], "n_listings": fp["n_listings"], "n_points": points,
            "price_vnd_sum": fp["price_vnd_sum"], "bm25_avgdl": ledger.state["bm25_avgdl"], "created_at": now_iso()})
    report.end("verify", note=f"smoke hits {hits}")
    if not job.dry_run and not ledger.state["pca_done"]:
        job.step = "pca"
        report.begin("pca", job.n_parts)
        out = pca.run(s, layout, job.n_parts, log)
        ledger.set(pca_done=True)
        report.end("pca", note=f"{out['n_listings']} tin, explained_variance={out['explained_variance']:.3f}")
