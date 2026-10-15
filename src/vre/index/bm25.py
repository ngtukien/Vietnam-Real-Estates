"""BM25 dạng sparse vector cho Qdrant (data-contract §4, §4.2, §5).

Tách từ bằng pyvi trên `lower(text)`, bỏ token không có chữ/số; id token = `crc32(token) & 0x7FFFFFFF`.
Phía tài liệu lưu `tf·(k1+1) / (tf + k1·(1 − b + b·doc_len/avgdl))` với `avgdl` của toàn corpus; IDF do Qdrant tính
(modifier IDF). Truy vấn: mỗi token trọng số 1.
"""

import multiprocessing as mp
import re
import zlib
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

K1, B = 1.2, 0.75
_WORD = re.compile(r"\w")


def tokenize_vi(text: str) -> list[str]:
    """'Trường học gần chợ' -> ['trường_học', 'gần', 'chợ']."""
    from pyvi import ViTokenizer

    return [t for t in ViTokenizer.tokenize((text or "").lower()).split() if _WORD.search(t)]


def token_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


def doc_terms(tokens: list[str]) -> tuple[list[int], list[int], int]:
    """(id token tăng dần, tf tương ứng, doc_len). Hai token trùng crc32 được cộng gộp."""
    tf = Counter(token_id(t) for t in tokens)
    ids = sorted(tf)
    return ids, [min(tf[i], 65535) for i in ids], len(tokens)


def weights(tfs, doc_len: int, avgdl: float) -> list[float]:
    tf = np.asarray(tfs, dtype=np.float64)
    norm = K1 * (1 - B + B * doc_len / avgdl)
    return (tf * (K1 + 1) / (tf + norm)).tolist()


def document_vector(token_ids, tfs, doc_len: int, avgdl: float):
    from qdrant_client import models

    return models.SparseVector(indices=[int(i) for i in token_ids], values=weights(tfs, doc_len, avgdl))


def query_vector(text: str):
    from qdrant_client import models

    ids = sorted({token_id(t) for t in tokenize_vi(text)})
    return models.SparseVector(indices=ids, values=[1.0] * len(ids))


BM25_SCHEMA = pa.schema([("point_id", pa.int64()), ("token_ids", pa.list_(pa.uint32())),
                         ("tfs", pa.list_(pa.uint16())), ("doc_len", pa.int32())])


def bm25_part(layout, part: int) -> dict:
    """Việc của một tiến trình BM25: đọc `chunks/part`, ghi `bm25/part` nguyên tử. Trả số chunk và tổng doc_len."""
    import os

    chunks = pq.read_table(layout.chunks(part), columns=["point_id", "text"])
    ids_col, tfs_col, lens = [], [], []
    for text in chunks.column("text").to_pylist():
        ids, tfs, n = doc_terms(tokenize_vi(text))
        ids_col.append(ids)
        tfs_col.append(tfs)
        lens.append(n)
    table = pa.table({"point_id": chunks.column("point_id"), "token_ids": ids_col, "tfs": tfs_col,
                      "doc_len": lens}, schema=BM25_SCHEMA)
    path = layout.bm25(part)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return {"part": part, "n_chunks": len(lens), "doc_len_sum": int(sum(lens))}


def corpus_avgdl(layout, parts: list[int]) -> float:
    """Trung bình doc_len trên MỌI chunk của mọi part (đọc lại cột `doc_len`, không giữ trạng thái phụ)."""
    total, count = 0, 0
    for part in parts:
        lens = pq.read_table(layout.bm25(part), columns=["doc_len"]).column("doc_len").to_numpy()
        total += int(lens.sum())
        count += len(lens)
    if not count:
        raise ValueError("không có chunk nào để tính bm25_avgdl")
    return total / count


def make_pool(workers: int):
    """Pool tiến trình `spawn` (tiến trình con không kế thừa trạng thái CUDA; Windows không có fork).

    Dùng `ProcessPoolExecutor` thay `multiprocessing.Pool`: worker chết đột ngột ⇒ `BrokenProcessPool` được raise,
    còn `Pool.imap_unordered` thì treo vĩnh viễn (lượt chạy dài sẽ không bao giờ thoát mã 20 để tự chạy lại).
    """
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=below_normal_priority)


def below_normal_priority() -> None:
    """Worker CPU nhường vòng GPU: dry-run 2026-10-15 chỉ đạt 26 chunk/s (đo riêng 132/s) khi worker giành CPU."""
    import psutil

    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS") else 10)


def terminate_pool(pool) -> None:
    """Dừng ngay: huỷ việc chưa chạy và kill worker đang chạy (không chờ part dở dang; checkpoint lo phần còn lại)."""
    processes = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.kill()
