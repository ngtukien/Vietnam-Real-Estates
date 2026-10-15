"""Nạp một part vào Qdrant qua gRPC (data-contract §5): point = chunk, payload từ bảng sạch, bỏ khoá null.

`point_id` tất định ⇒ nạp lại một part không nhân bản. Lỗi/timeout ⇒ tenacity 6 lần, backoff mũ 2–120s; Qdrant không trả lời
quá 2 phút ⇒ một lần `docker compose up -d --wait qdrant`. Part chỉ được coi là xong khi `count(listing_id ∈ part) = n_chunks`;
lệch ⇒ upsert lại tối đa 2 lần rồi raise. `wait=True` (không phải `wait=False` như plan): dry-run 2026-10-15 cho thấy khi Qdrant
đang xây HNSW, `wait=False` đẩy nhanh hơn tốc độ áp dụng ⇒ count tụt sau 10 phút ⇒ upsert lại ⇒ quá tải gRPC. Upload chỉ cần
~100 point/s (bám theo GPU) nên backpressure không làm chậm tổng thời gian. Chạy trong tiến trình riêng (`upload_part_task`)
để việc dựng point (Python thuần, giữ GIL) không giành CPU với vòng GPU.
"""

import datetime as dt
import subprocess
import time

import numpy as np
import pyarrow.parquet as pq
from tenacity import Retrying, stop_after_attempt, wait_exponential

from vre.index import bm25, qdrant_store
from vre.index.chunk import read_part

PAYLOAD_COLUMNS = ("listing_id", "dup_cluster_id", "title", "description", "property_type", "deal_type", "province",
                   "district", "ward", "street", "project", "price_vnd", "area_m2", "bedrooms", "bathrooms",
                   "published_at", "is_price_outlier", "is_area_outlier", "amenities")
DESCRIPTION_EXCERPT_CHARS = 400
RETRY_ATTEMPTS, RETRY_MIN_S, RETRY_MAX_S = 6, 2, 120
UNRESPONSIVE_RESTART_S = 120
REUPSERT_ATTEMPTS = 2
COUNT_SETTLE_TIMEOUT_S, COUNT_POLL_S = 600, 5


def _month(value) -> str | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(dt.UTC)
    return value.strftime("%Y-%m")


def listing_payload(row: dict) -> dict:
    """Phần payload theo tin (dùng chung cho mọi chunk của tin). `price_m2_mil` theo đúng định nghĩa `v_listings`."""
    price, area = row.get("price_vnd"), row.get("area_m2")
    clean = not row.get("is_price_outlier") and not row.get("is_area_outlier")
    description = row.get("description")
    payload = {
        "listing_id": row["listing_id"], "dup_cluster_id": row["dup_cluster_id"], "title": row.get("title"),
        "description_excerpt": description[:DESCRIPTION_EXCERPT_CHARS] if description else None,
        **{k: row.get(k) for k in ("property_type", "deal_type", "province", "district", "ward", "street", "project",
                                   "price_vnd", "area_m2", "bedrooms", "bathrooms")},
        "price_m2_mil": price / area / 1e6 if clean and price is not None and area else None,
        "published_month": _month(row.get("published_at")),
        "amenities": list(row.get("amenities") or []),
    }
    return {k: v for k, v in payload.items() if v is not None}


def build_points(chunks, emb: np.ndarray, bm25_table, listings: dict[int, dict], avgdl: float):
    """Sinh PointStruct theo thứ tự hàng của `chunks/part` (↔ hàng của `.npy` và `bm25/part`)."""
    from qdrant_client import models

    point_ids = chunks.column("point_id").to_pylist()
    if bm25_table.column("point_id").to_pylist() != point_ids or len(emb) != len(point_ids):
        raise ValueError("chunks/bm25/emb của part không cùng thứ tự hoặc cùng số hàng")
    columns = zip(point_ids, chunks.column("listing_id").to_pylist(), chunks.column("chunk_index").to_pylist(),
                  chunks.column("text").to_pylist(), bm25_table.column("token_ids").to_pylist(),
                  bm25_table.column("tfs").to_pylist(), bm25_table.column("doc_len").to_pylist(), strict=True)
    for i, (pid, lid, index, text, ids, tfs, doc_len) in enumerate(columns):
        vector = {"dense": emb[i].astype(np.float32).tolist()}
        if ids:
            vector["bm25"] = bm25.document_vector(ids, tfs, doc_len, avgdl)
        yield models.PointStruct(id=pid, vector=vector,
                                 payload={**listings[lid], "chunk_index": index, "text": text})


def compose_up(settings, log=print) -> None:
    log("Qdrant không phản hồi > 2 phút: docker compose up -d --wait qdrant")
    subprocess.run(["docker", "compose", "-f", str(settings.compose_file), "up", "-d", "--wait", "qdrant"],
                   check=False, timeout=600)


def resilient(settings, fn, counters: dict, log=print):
    """Gọi `fn` với retry; đếm `upload_retries`; một lần `compose up` khi lỗi kéo dài quá 2 phút."""
    first_failure, restarted = None, False
    for attempt in Retrying(stop=stop_after_attempt(RETRY_ATTEMPTS), reraise=True,
                            wait=wait_exponential(multiplier=2, min=RETRY_MIN_S, max=RETRY_MAX_S)):
        with attempt:
            try:
                return fn()
            except Exception as exc:
                counters["upload_retries"] = counters.get("upload_retries", 0) + 1
                now = time.monotonic()
                first_failure = first_failure or now
                log(f"Qdrant lỗi ({type(exc).__name__}: {exc}); thử lại")
                if not restarted and now - first_failure > UNRESPONSIVE_RESTART_S:
                    compose_up(settings, log)
                    restarted = True
                raise


def wait_count(qc, collection: str, lo: int, hi: int, expected: int) -> int:
    deadline = time.monotonic() + COUNT_SETTLE_TIMEOUT_S
    while True:
        n = qdrant_store.count_part(qc, collection, lo, hi)
        if n >= expected or time.monotonic() > deadline:
            return n
        time.sleep(COUNT_POLL_S)


def upload_part(settings, qc, collection: str, layout, part: int, avgdl: float, counters: dict, log=print) -> int:
    """Nạp part và kiểm đếm. Trả số point của part trong Qdrant (= n_chunks)."""
    chunks = pq.read_table(layout.chunks(part), columns=["point_id", "listing_id", "chunk_index", "text"])
    table = pq.read_table(layout.bm25(part))
    emb = np.load(layout.emb(part), mmap_mode="r")
    rows = read_part(settings.clean_path, part, settings.part_size, PAYLOAD_COLUMNS).to_pylist()
    listings = {r["listing_id"]: listing_payload(r) for r in rows}
    lo, hi, expected = min(listings), max(listings), len(chunks)
    for attempt in range(1 + REUPSERT_ATTEMPTS):
        resilient(settings, lambda: qc.upload_points(
            collection, points=build_points(chunks, emb, table, listings, avgdl),
            batch_size=settings.qdrant_upload_batch, parallel=settings.qdrant_upload_parallel,
            max_retries=5, wait=True), counters, log)
        n = resilient(settings, lambda: wait_count(qc, collection, lo, hi, expected), counters, log)
        if n == expected:
            return n
        log(f"part {part:04d}: Qdrant có {n} point, cần {expected}; upsert lại ({attempt + 1})")
    raise RuntimeError(f"part {part:04d}: count {n} ≠ n_chunks {expected} sau {REUPSERT_ATTEMPTS} lần upsert lại")


def upload_part_task(args) -> dict:
    """Điểm vào tiến trình upload riêng: tự tạo client, trả số point + số lần retry (để cộng vào report)."""
    settings, layout, part, avgdl = args
    counters = {"upload_retries": 0}
    qc = qdrant_store.client(settings)
    n = upload_part(settings, qc, layout.collection, layout, part, avgdl, counters,
                    log=lambda m: print(m, flush=True))
    return {"n": n, **counters}
