"""Lệnh `ingest`: 10 shard thô → `data/clean/listings.parquet` + `dup_map.parquet` + `results/ingest_report.json`.

Giai đoạn 1 theo shard (ProcessPoolExecutor, spawn): làm sạch + id + SimHash → `data/interim/shard_NNNN.parquet`.
Giai đoạn 2 toàn cục (DuckDB, `vre.data.dedup`). Cuối lệnh sao chép thẳng số đo embedding 2026-10-15 (data-contract §7.4).
"""

from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from vre.common import result_header, write_json_atomic
from vre.config import Settings
from vre.data import dedup, duck
from vre.data.clean import clean_frame, norm
from vre.data.simhash import simhashes

RAW_COLUMNS = ["name", "description", "property_type_name", "province_name", "district_name", "ward_name", "street_name",
               "project_name", "price", "area", "floor_count", "frontage_width", "house_depth", "road_width", "bedroom_count",
               "bathroom_count", "house_direction", "balcony_direction", "published_at"]
N_SHARDS = 10
BATCH_ROWS = 50_000  # bộ nhớ mỗi worker có giới hạn (8 worker chạy song song)
BENCH_SOURCE = Path(".cache") / "scratch" / "bench_final.jsonl"
MISSING_COLS = ("ward", "street", "project", "bedrooms")
# Logic sinh bảng trung gian/bảng sạch: đổi file nào ⇒ hash đổi ⇒ không dùng lại artifact cũ.
LOGIC_FILES = ("clean.py", "simhash.py", "places.py", "amenities.py", "amenities.yaml", "admin_2025.csv")
META_NAME = "_ingest_logic.json"


def logic_hash() -> str:
    digest = hashlib.sha256()
    for name in LOGIC_FILES:
        digest.update(name.encode() + (Path(__file__).with_name(name)).read_bytes())
    return digest.hexdigest()[:16]


def meta_matches(folder: Path) -> bool:
    meta = folder / META_NAME
    return meta.is_file() and json.loads(meta.read_text(encoding="utf-8")).get("logic") == logic_hash()


def write_meta(folder: Path) -> None:
    write_json_atomic(folder / META_NAME, {"logic": logic_hash(), "files": list(LOGIC_FILES)})


def interim_dir(settings: Settings) -> Path:
    return settings.data_root / "data" / "interim"


def peak_rss_gb() -> float:
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss) / 1e9


def process_shard(shard: int, raw_dir: str, out_dir: str) -> dict:
    """Một shard → parquet trung gian theo lô BATCH_ROWS dòng (bộ nhớ có giới hạn); ghi .tmp rồi os.replace."""
    t0 = time.perf_counter()
    source = pq.ParquetFile(Path(raw_dir) / f"shard_{shard:04d}.parquet")
    path = Path(out_dir) / f"shard_{shard:04d}.parquet"
    tmp = path.with_name(path.name + ".tmp")
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for batch in source.iter_batches(batch_size=BATCH_ROWS, columns=RAW_COLUMNS):
            raw = batch.to_pandas()
            frame = clean_frame(raw, shard, row_offset=rows)
            frame["simhash"], frame["simhash_valid"] = simhashes(norm(raw["description"]))
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                schema = pa.schema([f.with_type(pa.string()) if pa.types.is_null(f.type) else f for f in table.schema])
                writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            writer.write_table(table.cast(writer.schema))
            rows += len(frame)
            del raw, frame, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp, path)
    return {"shard": shard, "rows": rows, "seconds": round(time.perf_counter() - t0, 1), "peak_rss_gb": round(peak_rss_gb(), 2)}


def copy_embedding_benchmark(settings: Settings) -> Path:
    """Bản sao thẳng `.cache/scratch/bench_final.jsonl` → `results/embedding_benchmark.json` (mảng JSON, không lọc)."""
    lines = (settings.data_root / BENCH_SOURCE).read_text(encoding="utf-8").splitlines()
    return write_json_atomic(settings.results_dir / "embedding_benchmark.json", [json.loads(x) for x in lines if x.strip()])


def already_done(settings: Settings) -> bool:
    report = settings.results_dir / "ingest_report.json"
    bench = settings.results_dir / "embedding_benchmark.json"
    if not (settings.clean_path.is_file() and settings.dup_map_path.is_file() and report.is_file() and bench.is_file()):
        return False
    if not meta_matches(settings.clean_path.parent):
        print("ingest: bảng sạch được tạo bởi logic khác (hoặc chưa có dấu logic) ⇒ làm lại", flush=True)
        return False
    kept = json.loads(report.read_text(encoding="utf-8"))["dedup"]["kept"]
    con = duck.connect(settings, views=False)
    try:
        n = duck.one_row(con.execute(f"SELECT count(*) FROM read_parquet('{duck.sql_path(settings.clean_path)}')"))[0]
    finally:
        con.close()
    return int(n) == int(kept)


def build_report(con, stats: dict, seconds: float) -> dict:  # noqa: ANN001 — DuckDB connection
    reasons = stats["reasons"]
    q = con.execute(f"""SELECT count(*) FILTER (price_vnd IS NULL), count(*) FILTER (price_fixed_thousand),
                         count(*) FILTER (is_price_outlier), count(*) FILTER (is_area_outlier),
                         count(*) FILTER (deal_type = 'sale'), count(*) FILTER (deal_type = 'rent'),
                         {", ".join(f"avg(({c} IS NULL)::INT) * 100" for c in MISSING_COLS)}
                         FROM flagged""").fetchone()
    return {**result_header(), "raw_rows": int(sum(reasons.values())), "shards": N_SHARDS, "seconds": round(seconds, 1),
            "dropped": {"no_text": int(reasons.get("dropped_no_text", 0)), "bad_type": int(reasons.get("dropped_bad_type", 0))},
            "dedup": {"exact_dup": int(reasons.get("exact_dup", 0)), "prefix_dup": int(reasons.get("prefix_dup", 0)),
                      "kept": int(reasons.get("kept", 0)), **stats["simhash"]},
            "price": {"null_or_unparsable": int(q[0]), "fixed_thousand": int(q[1]), "outliers": int(q[2])},
            "area_outliers": int(q[3]), "deal_type": {"sale": int(q[4]), "rent": int(q[5])},
            "missing_pct": {c: round(float(v), 2) for c, v in zip(MISSING_COLS, q[6:], strict=True)},
            "listing_id_collisions": 0}


def run(settings: Settings) -> dict:
    if already_done(settings):
        print("ingest: đã xong (bảng sạch khớp ingest_report)")
        return json.loads((settings.results_dir / "ingest_report.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    out = interim_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=settings.ingest_workers, mp_context=mp.get_context("spawn")) as pool:
        # Resume: shard trung gian đã ghi xong được dùng lại CHỈ khi cùng logic (meta); khác logic ⇒ làm lại cả 10 shard.
        reuse = meta_matches(out)
        todo = [s for s in range(N_SHARDS) if not (reuse and (out / f"shard_{s:04d}.parquet").is_file())]
        print(f"ingest: {N_SHARDS - len(todo)} shard trung gian dùng lại, {len(todo)} shard cần xử lý", flush=True)
        futures = [pool.submit(process_shard, s, str(settings.raw_dir), str(out)) for s in todo]
        for fut in futures:
            print("ingest shard:", fut.result(), flush=True)
    write_meta(out)
    con = duck.connect(settings, views=False)
    try:
        stats = dedup.run_global(con, out, settings.clean_path, settings.dup_map_path)
        report = build_report(con, stats, time.perf_counter() - t0)
    finally:
        con.close()
    write_json_atomic(settings.results_dir / "ingest_report.json", report)
    write_meta(settings.clean_path.parent)
    copy_embedding_benchmark(settings)
    print(f"ingest: peak RSS tiến trình chính {peak_rss_gb():.2f} GB, {report['seconds']} s", flush=True)
    return report
