"""Pha A. Lập chỉ mục dữ liệu từ DATA_URL hoặc CSV cục bộ vào PostgreSQL cho chatbot.

Chạy từ thư mục gốc dự án; bị dừng giữa chừng thì chạy lại lệnh cũ để làm tiếp:
    .venv/bin/python app/index.py                  # toàn bộ dataset (~3,5 triệu dòng)
    .venv/bin/python app/index.py --limit 20000    # thử với ít dữ liệu
    .venv/bin/python app/index.py --reset          # xóa chỉ mục cũ, làm lại từ đầu

Các giai đoạn (mỗi giai đoạn ghi tiến độ vào bảng app_state):
  1. tin, thực thể KG, chunk + embedding, theo từng lô (chia chunk song song trên CPU, embedding trên GPU)
  2. bậc của thực thể (cho trọng số KG)
  3. cộng đồng Louvain theo từng quận/huyện + báo cáo cộng đồng có embedding (GraphRAG)
  4. index HNSW cho vector
"""

import argparse
import json
import multiprocessing
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from itertools import islice
from pathlib import Path

import numpy as np
from pgvector import HalfVector
from psycopg.types.json import Jsonb

from data_source import resolve_source
from engine import (DIMENSION, LISTING_COLUMNS, VIEW_KINDS, chunk_worker, community_stats, connect, count_rows,
                    detect_communities, init_chunk_worker, iter_documents, listing_entities, load_encoder)

TABLES = ("app_listings", "app_chunks", "app_entities", "app_listing_entities", "app_listing_communities",
          "app_reports", "app_state")
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS app_state(key text PRIMARY KEY, value jsonb NOT NULL);
CREATE TABLE IF NOT EXISTS app_listings(
    id integer PRIMARY KEY, title text NOT NULL, description text NOT NULL,
    province_name text, district_name text, ward_name text, street_name text, project_name text,
    property_type_name text, price float8, area float8, bedroom_count float8, bathroom_count float8,
    floor_count float8, extra jsonb NOT NULL);
-- Cột lọc lặp lại trong app_chunks để HNSW lọc ngay trong một bảng.
CREATE TABLE IF NOT EXISTS app_chunks(
    id text PRIMARY KEY, listing_id integer NOT NULL, chunk_index integer, token_start integer,
    token_end integer, province_name text, property_type_name text, price float8, area float8,
    content text NOT NULL, embedding halfvec({DIMENSION}) NOT NULL);
CREATE TABLE IF NOT EXISTS app_entities(
    id integer PRIMARY KEY, kind text NOT NULL, key text UNIQUE NOT NULL, label text NOT NULL,
    parent_id integer, degree integer NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS app_listing_entities(
    entity_id integer NOT NULL, listing_id integer NOT NULL, PRIMARY KEY(entity_id, listing_id));
CREATE TABLE IF NOT EXISTS app_listing_communities(listing_id integer PRIMARY KEY, community_id integer NOT NULL);
CREATE TABLE IF NOT EXISTS app_reports(
    id integer PRIMARY KEY, listing_count integer NOT NULL, stats jsonb NOT NULL, content text NOT NULL,
    embedding halfvec({DIMENSION}) NOT NULL);
"""
HNSW_INDEXES = {
    "app_chunks_embedding": "app_chunks",
    "app_reports_embedding": "app_reports",
}


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def get_state(conn, key, default=None):
    row = conn.execute("SELECT value FROM app_state WHERE key=%s", (key,)).fetchone()
    return row[0] if row else default


def set_state(conn, key, value):
    conn.execute("INSERT INTO app_state VALUES(%s,%s) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, Jsonb(value)))


def batches(iterable, size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


def write_batch(conn, documents, chunks, vectors, entity_ids):
    """Ghi một lô trong một transaction: tin, thực thể mới, liên kết KG, chunk + vector, tiến độ."""
    new_entities, links = [], []
    for doc in documents:
        for entity in listing_entities(doc):
            entity_id = entity_ids.get(entity["key"])
            if entity_id is None:
                entity_id = entity_ids[entity["key"]] = len(entity_ids) + 1
                # Cấp trên luôn đứng trước trong cùng tin nên đã có ID.
                new_entities.append((entity_id, entity["kind"], entity["key"], entity["label"],
                                     entity_ids.get(entity["parent"])))
            links.append((entity_id, int(doc["id"])))
    known = set(LISTING_COLUMNS) | {"name", "description"}
    with conn.transaction(), conn.cursor() as cursor:
        with cursor.copy(f"COPY app_listings (id,title,description,{','.join(LISTING_COLUMNS)},extra) FROM STDIN") as copy:
            for doc in documents:
                meta = doc["metadata"]
                copy.write_row((int(doc["id"]), doc["title"], doc["description"],
                                *(meta.get(name) for name in LISTING_COLUMNS),
                                Jsonb({k: v for k, v in meta.items() if k not in known})))
        with cursor.copy("COPY app_entities (id,kind,key,label,parent_id) FROM STDIN") as copy:
            for row in new_entities:
                copy.write_row(row)
        with cursor.copy("COPY app_listing_entities (entity_id,listing_id) FROM STDIN") as copy:
            for row in links:
                copy.write_row(row)
        with cursor.copy("COPY app_chunks (id,listing_id,chunk_index,token_start,token_end,province_name,"
                         "property_type_name,price,area,content,embedding) FROM STDIN") as copy:
            for chunk, vector in zip(chunks, vectors):
                meta = chunk["metadata"]
                copy.write_row((chunk["id"], int(chunk["listing_id"]), meta["chunk_index"], meta["token_start"],
                                meta["token_end"], meta.get("province_name"), meta.get("property_type_name"),
                                meta.get("price"), meta.get("area"), chunk["text"], HalfVector(vector)))
        set_state(conn, "loaded_through", int(documents[-1]["id"]))


def load_listings(conn, encoder, target, batch_size, workers, source=None):
    """Giai đoạn 1. Chia chunk song song (CPU) trong lúc GPU embedding lô trước đó."""
    done = get_state(conn, "loaded_through", 0)
    if (target is not None and done >= target) or (target is None and get_state(conn, "source_complete")):
        log(f"Giai đoạn 1: đã nạp đủ {done:,} dòng")
        return
    # Có dữ liệu mới: index HNSW và cộng đồng cũ không còn đúng, làm lại sau khi nạp.
    for name in HNSW_INDEXES:
        conn.execute(f"DROP INDEX IF EXISTS {name}")
    conn.execute("TRUNCATE app_listing_communities, app_reports")
    for key in ("degrees_done", "communities_through", "ready", "source_complete"):
        conn.execute("DELETE FROM app_state WHERE key=%s", (key,))

    entity_ids = {key: entity_id for entity_id, key in conn.execute("SELECT id, key FROM app_entities")}
    destination = f"{target:,}" if target is not None else "hết nguồn dữ liệu"
    log(f"Giai đoạn 1: nạp dòng {done + 1:,} → {destination} (lô {batch_size:,} tin, {workers} tiến trình chia chunk)")
    started, loaded, chunk_count = time.time(), 0, 0
    context = multiprocessing.get_context("spawn")  # không fork tiến trình đã khởi tạo CUDA
    with closing(iter_documents(source, limit=target, start_after=done)) as stream, \
            ProcessPoolExecutor(workers, mp_context=context, initializer=init_chunk_worker) as pool:
        documents = batches(stream, batch_size)
        pending = deque()

        def submit_next():
            batch = next(documents, None)
            if batch:
                pending.append((batch, pool.submit(chunk_worker, batch)))

        for _ in range(workers + 1):  # giới hạn số lô chờ để không đọc cả CSV vào RAM
            submit_next()
        while pending:
            batch, future = pending.popleft()
            chunks = future.result()
            submit_next()
            vectors = encoder.encode(["passage: " + c["text"] for c in chunks], batch_size=128,
                                     normalize_embeddings=True, convert_to_numpy=True)
            write_batch(conn, batch, chunks, vectors.astype(np.float16), entity_ids)
            loaded, chunk_count = loaded + len(batch), chunk_count + len(chunks)
            position = int(batch[-1]["id"])
            rate = (position - done) / (time.time() - started)
            progress = f"dòng {position:,}"
            remaining = ""
            if target is not None:
                progress += f"/{target:,} ({position / target:.1%})"
                remaining = f" | còn ~{(target - position) / rate / 60:,.0f} phút"
            log(f"  {progress} | {loaded:,} tin, {chunk_count:,} chunk | {rate:,.0f} dòng/s{remaining}")
    if target is not None:
        set_state(conn, "loaded_through", target)
    else:
        set_state(conn, "source_complete", True)


def compute_degrees(conn):
    """Giai đoạn 2. Bậc như trong đồ thị notebook: số tin + số thực thể con + cấp trên."""
    if get_state(conn, "degrees_done"):
        return
    log("Giai đoạn 2: index liên kết và tính bậc thực thể")
    conn.execute("CREATE INDEX IF NOT EXISTS app_listing_entities_listing ON app_listing_entities(listing_id)")
    conn.execute("""
        WITH listings AS (SELECT entity_id, count(*) AS n FROM app_listing_entities GROUP BY entity_id),
             children AS (SELECT parent_id, count(*) AS n FROM app_entities WHERE parent_id IS NOT NULL
                          GROUP BY parent_id)
        UPDATE app_entities e
        SET degree = coalesce(l.n, 0) + coalesce(c.n, 0) + (e.parent_id IS NOT NULL)::int
        FROM app_entities x LEFT JOIN listings l ON l.entity_id = x.id LEFT JOIN children c ON c.parent_id = x.id
        WHERE e.id = x.id
    """)
    set_state(conn, "degrees_done", True)


def build_communities(conn, encoder):
    """Giai đoạn 3. Louvain cho từng quận/huyện (tin không có quận: gom theo tỉnh), mỗi cộng đồng một báo cáo."""
    parents = dict(conn.execute("SELECT id, parent_id FROM app_entities WHERE parent_id IS NOT NULL").fetchall())
    groups = [("district", entity_id) for (entity_id,) in conn.execute(
        "SELECT id FROM app_entities WHERE kind='district_name' ORDER BY id")]
    groups += [("province", province) for (province,) in conn.execute(
        "SELECT DISTINCT coalesce(province_name,'') FROM app_listings WHERE coalesce(district_name,'')='' ORDER BY 1")]
    done = get_state(conn, "communities_through", 0)
    if done >= len(groups):
        return
    log(f"Giai đoạn 3: cộng đồng Louvain cho {len(groups) - done:,}/{len(groups):,} nhóm")
    next_id = conn.execute("SELECT coalesce(max(id), 0) + 1 FROM app_reports").fetchone()[0]
    started = time.time()
    for position, (kind, value) in enumerate(groups[done:], done + 1):
        if kind == "district":
            ids = [r[0] for r in conn.execute(
                "SELECT listing_id FROM app_listing_entities WHERE entity_id=%s", (value,))]
        else:
            ids = [r[0] for r in conn.execute(
                "SELECT id FROM app_listings WHERE coalesce(district_name,'')='' AND coalesce(province_name,'')=%s",
                (value,))]
        links = conn.execute(
            "SELECT le.listing_id, le.entity_id FROM app_listing_entities le JOIN app_entities e "
            "ON e.id = le.entity_id WHERE le.listing_id = ANY(%s) AND e.kind = ANY(%s)",
            (ids, list(VIEW_KINDS))).fetchall()
        rows = {r[0]: dict(zip(("province_name", "district_name", "property_type_name", "project_name",
                                "price", "area"), r[1:])) for r in conn.execute(
            "SELECT id, province_name, district_name, property_type_name, project_name, price, area "
            "FROM app_listings WHERE id = ANY(%s)", (ids,))}
        communities = detect_communities(ids, links, parents)
        reports = []
        for members in communities:
            stats = community_stats([rows[i] for i in members])
            reports.append((next_id, members, stats, json.dumps(stats, ensure_ascii=False)))
            next_id += 1
        vectors = encoder.encode(["passage: " + r[3] for r in reports], batch_size=128,
                                 normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
        with conn.transaction(), conn.cursor() as cursor:
            with cursor.copy("COPY app_listing_communities (listing_id, community_id) FROM STDIN") as copy:
                for report_id, members, _, _ in reports:
                    for listing_id in members:
                        copy.write_row((listing_id, report_id))
            with cursor.copy("COPY app_reports (id, listing_count, stats, content, embedding) FROM STDIN") as copy:
                for (report_id, members, stats, text), vector in zip(reports, vectors):
                    copy.write_row((report_id, len(members), Jsonb(stats), text, HalfVector(vector)))
            set_state(conn, "communities_through", position)
        if position % 50 == 0 or position == len(groups):
            rate = (position - done) / (time.time() - started)
            log(f"  nhóm {position:,}/{len(groups):,} | {next_id - 1:,} cộng đồng "
                f"| còn ~{(len(groups) - position) / rate / 60:,.0f} phút")


def build_indexes(conn):
    """Giai đoạn 4. HNSW cho vector (cosine trên halfvec), rồi cập nhật thống kê cho planner."""
    conn.execute("SET maintenance_work_mem = '2GB'")
    conn.execute("SET max_parallel_maintenance_workers = 4")
    for name, table in HNSW_INDEXES.items():
        log(f"Giai đoạn 4: build HNSW {name} (có thể mất hàng chục phút với hàng triệu vector)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} USING hnsw (embedding halfvec_cosine_ops)")
    conn.execute("ANALYZE")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="chỉ nạp N bản ghi đầu nguồn dữ liệu (mặc định: toàn bộ)")
    parser.add_argument("--batch-size", type=int, default=20000, help="số tin mỗi lô (mặc định 20000)")
    parser.add_argument("--workers", type=int, default=4, help="số tiến trình chia chunk (mặc định 4)")
    parser.add_argument("--reset", action="store_true", help="xóa chỉ mục cũ và làm lại từ đầu")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit phải lớn hơn 0")
    source = resolve_source()

    with connect() as conn:
        if args.reset:
            log("Xóa chỉ mục cũ")
            conn.execute(f"DROP TABLE IF EXISTS {', '.join(TABLES)}, app_index CASCADE")
        conn.execute(SCHEMA)
        # Supabase mở mọi bảng trong schema public qua Data API; bật RLS không kèm policy để chặn
        # truy cập đó. Chủ bảng (user kết nối ở đây) không bị RLS chặn nên vẫn đọc/ghi bình thường.
        for table in TABLES:
            conn.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")

        if args.limit is not None:
            target = args.limit
        elif isinstance(source, Path):
            target = get_state(conn, "csv_rows") or count_rows(source)
            set_state(conn, "csv_rows", target)
        else:
            # Link lớn chỉ đọc một lần, không tải hết chỉ để đếm dòng trước khi lập chỉ mục.
            target = None
        log(f"Mục tiêu: {target:,} bản ghi" if target is not None else "Mục tiêu: toàn bộ dữ liệu qua DATA_URL")

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = load_encoder(device)
        if device == "cuda":
            encoder.half()  # fp16 nhanh gấp ~3 lần, vector vẫn lưu dạng halfvec
        log(f"Embedding trên {device}")

        load_listings(conn, encoder, target, args.batch_size, args.workers, source)
        compute_degrees(conn)
        build_communities(conn, encoder)
        build_indexes(conn)
        stats = dict(zip(("documents", "chunks", "entities", "links", "communities"), conn.execute("""
            SELECT (SELECT count(*) FROM app_listings), (SELECT count(*) FROM app_chunks),
                   (SELECT count(*) FROM app_entities), (SELECT count(*) FROM app_listing_entities),
                   (SELECT count(*) FROM app_reports)""").fetchone()))
        size = conn.execute("SELECT pg_size_pretty(sum(pg_total_relation_size(c.oid))) FROM pg_class c "
                            "WHERE c.relname = ANY(%s)", (list(TABLES),)).fetchone()[0]
        set_state(conn, "ready", stats)
        log(f"Xong: {stats} | dung lượng {size}")


if __name__ == "__main__":
    main()
