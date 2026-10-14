"""Đặc trưng embedding cho định giá `full` (data-contract §4): vector tin = trung bình vector các chunk của tin,
`IncrementalPCA` `pca_dim` chiều fit trên `pca_fit_sample` tin thuộc tháng train, transform toàn bộ
→ `data/features/emb_pca32_e5_large.parquet` (`listing_id`, `pca_00` … `pca_31`).
"""

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from vre.config import EMBED_MODEL_SLUG
from vre.index.chunk import read_part

TRAIN_LAST_MONTH = "2026-01"  # data-contract §7.6: train 2025-06 → 2026-01, test 2026-02 → 2026-03
SEED = 42
FIT_BATCH = 20000


def output_path(settings) -> Path:
    return Path(settings.features_dir) / f"emb_pca{settings.pca_dim}_{EMBED_MODEL_SLUG}.parquet"


def listing_means(emb: np.ndarray, listing_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chunk của một tin nằm liền nhau (thứ tự `chunks/part`). Trả (listing_id duy nhất theo thứ tự, trung bình fp32)."""
    if len(listing_ids) == 0:
        return listing_ids, np.zeros((0, emb.shape[1]), dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, listing_ids[1:] != listing_ids[:-1]])
    ids = listing_ids[starts]
    if len(np.unique(ids)) != len(ids):
        raise ValueError("chunk của một tin không liền nhau trong part")
    sums = np.add.reduceat(np.asarray(emb, dtype=np.float32), starts, axis=0)
    counts = np.diff(np.r_[starts, len(listing_ids)])
    return ids, sums / counts[:, None]


def train_mask(months: list[str | None]) -> np.ndarray:
    """Tin thuộc tháng train (`published_month ≤ TRAIN_LAST_MONTH`); thiếu ngày đăng ⇒ không dùng để fit."""
    return np.array([m is not None and m <= TRAIN_LAST_MONTH for m in months], dtype=bool)


def _part_months(settings, part: int) -> tuple[np.ndarray, list[str | None]]:
    table = read_part(settings.clean_path, part, settings.part_size, ("listing_id", "published_at"))
    months = pc.strftime(table.column("published_at"), format="%Y-%m").to_pylist()
    return table.column("listing_id").to_numpy(), months


def _part_vectors(layout, part: int) -> tuple[np.ndarray, np.ndarray]:
    ids = pq.read_table(layout.chunks(part), columns=["listing_id"]).column("listing_id").to_numpy()
    return listing_means(np.load(layout.emb(part), mmap_mode="r"), ids)


def choose_sample(train_counts: list[int], size: int, seed: int = SEED) -> list[np.ndarray]:
    """Chọn ngẫu nhiên `size` tin train trên toàn corpus; trả chỉ số (trong danh sách tin train) theo từng part."""
    total = sum(train_counts)
    picked = np.sort(np.random.default_rng(seed).choice(total, size=min(size, total), replace=False))
    bounds = np.r_[0, np.cumsum(train_counts)]
    return [picked[(picked >= lo) & (picked < hi)] - lo for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)]


def run(settings, layout, n_parts: int, log=print) -> dict:
    from sklearn.decomposition import IncrementalPCA

    masks = []
    for part in range(n_parts):
        listing_ids, months = _part_months(settings, part)
        masks.append((listing_ids, train_mask(months)))
    sample = choose_sample([int(m.sum()) for _, m in masks], settings.pca_fit_sample)
    pca, pending = IncrementalPCA(n_components=settings.pca_dim), []
    for part in range(n_parts):
        ids, means = _part_vectors(layout, part)
        listing_ids, mask = masks[part]
        if not np.array_equal(ids, listing_ids):
            raise ValueError(f"part {part:04d}: tin trong chunks khác bảng sạch")
        pending.append(means[np.flatnonzero(mask)[sample[part]]])
        n_pending = sum(len(p) for p in pending)
        last = part == n_parts - 1
        if n_pending >= FIT_BATCH or (last and n_pending >= settings.pca_dim):
            pca.partial_fit(np.concatenate(pending))
            pending = []
        elif last and n_pending:  # IncrementalPCA cần mỗi lô ≥ n_components hàng
            if not hasattr(pca, "components_"):
                raise ValueError(f"chỉ có {n_pending} tin train để fit PCA {settings.pca_dim} chiều")
            log(f"PCA: bỏ {n_pending} tin train cuối (< {settings.pca_dim} hàng, không đủ một lô partial_fit)")
    path = output_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    names = [f"pca_{i:02d}" for i in range(settings.pca_dim)]
    schema = pa.schema([("listing_id", pa.int64()), *[(n, pa.float32()) for n in names]])
    rows = 0
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for part in range(n_parts):
            ids, means = _part_vectors(layout, part)
            z = pca.transform(means).astype(np.float32)
            writer.write_table(pa.table({"listing_id": ids, **{n: z[:, i] for i, n in enumerate(names)}}, schema=schema))
            rows += len(ids)
    os.replace(tmp, path)
    log(f"PCA {settings.pca_dim} chiều: fit {sum(len(s) for s in sample)} tin train, ghi {rows} tin → {path}")
    return {"path": str(path), "n_listings": rows, "n_fit": int(sum(len(s) for s in sample)),
            "explained_variance": float(pca.explained_variance_ratio_.sum())}
