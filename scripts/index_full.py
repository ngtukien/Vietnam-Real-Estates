"""Lập chỉ mục toàn bộ dataset cho chatbot: Qdrant (collection listings_full) + Neo4j (graph_db_full).

Đọc dataset theo luồng, mỗi lô: làm sạch → bỏ tin trùng (toàn cục) → chunk → embed E5 → tách từ BM25
→ upsert Qdrant → MERGE vào Neo4j → ghi checkpoint. Bị ngắt thì chạy lại cùng lệnh để tiếp tục.

    docker compose --profile full up -d --wait
    .venv/bin/python scripts/index_full.py                 # toàn bộ (~3,5 triệu tin, nhiều giờ)
    .venv/bin/python scripts/index_full.py --limit 50000   # chạy thử
    .venv/bin/python scripts/index_full.py --reset         # xoá chỉ mục toàn bộ và làm lại từ đầu
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from itertools import islice
from pathlib import Path
from time import perf_counter

for key, value in {"TQDM_DISABLE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_VERBOSITY": "error",
                   "TRANSFORMERS_VERBOSITY": "error", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(key, value)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import data, graph, qdrant_store  # noqa: E402
from src.config import DATASET, FULL_COLLECTION, INDEX_DIR, NEO4J_FULL_URI  # noqa: E402
from src.rag import tokenize_vi  # noqa: E402

STATE_PATH = INDEX_DIR / f"{FULL_COLLECTION}_state.json"
DIGESTS_PATH = INDEX_DIR / f"{FULL_COLLECTION}_digests.bin"  # uint64 của (tiêu đề|mô tả) đã gặp
GRAPH_BLOCKS = ("provinces", "districts", "wards", "property_types", "listings",
                "of_type", "in_ward", "in_district", "in_province")
UPSERT_BATCH = 1024


def digest(title, description) -> int:
    key = f"{title or ''}|{description or ''}".lower().encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"rows_done": 0, "listings": 0, "chunks": 0, "price_sum": 0.0, "avgdl": None, "digests": 0, "done": False}


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)  # ghi nguyên tử: checkpoint không bao giờ dở dang


def load_digests(count: int) -> set[int]:
    """Đọc lại các digest đã ghi; cắt phần thừa nếu lần trước bị ngắt sau khi ghi digest mà chưa lưu state."""
    if not DIGESTS_PATH.exists():
        return set()
    with DIGESTS_PATH.open("r+b") as handle:
        handle.truncate(count * 8)
    return set(np.fromfile(DIGESTS_PATH, dtype=np.uint64).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="chỉ đọc N dòng đầu của dataset")
    parser.add_argument("--batch-size", type=int, default=4096, help="số dòng dataset mỗi lô")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="số tiến trình tách từ pyvi")
    parser.add_argument("--reset", action="store_true", help="xoá collection, đồ thị và checkpoint rồi làm lại")
    args = parser.parse_args()

    # Tạo pool trước khi nạp mô hình/CUDA để tiến trình con không kế thừa trạng thái GPU.
    # Windows không có fork: dùng spawn (tiến trình con tự import src.rag).
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    pool = mp.get_context(method).Pool(args.workers)
    from src import vector  # nạp sau khi fork

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    qdrant_store.use(FULL_COLLECTION)
    graph.use(NEO4J_FULL_URI)
    dim = vector.dimension()
    if args.reset:
        STATE_PATH.unlink(missing_ok=True)
        DIGESTS_PATH.unlink(missing_ok=True)
        qdrant_store.ensure_collection(reset=True, dim=dim)
        graph.reset_graph()
    state = load_state()
    if state["done"] and not args.limit:
        print("Chỉ mục toàn bộ đã xong. Dùng --reset để làm lại.")
        return
    qdrant_store.ensure_collection(dim=dim)
    graph.run_block("constraints")
    seen = load_digests(state["digests"])
    tokenizer = vector.embedder().tokenizer

    from datasets import load_dataset

    stream = load_dataset(DATASET, split="train", streaming=True)
    if state["rows_done"]:
        print(f"Tiếp tục từ dòng {state['rows_done'] + 1:,} ({state['listings']:,} tin, {state['chunks']:,} chunk)")
        stream = stream.skip(state["rows_done"])
    rows = iter(stream)
    start, rows_at_start = perf_counter(), state["rows_done"]
    while not args.limit or state["rows_done"] < args.limit:
        size = args.batch_size if not args.limit else min(args.batch_size, args.limit - state["rows_done"])
        batch = list(islice(rows, size))
        if not batch:
            state["done"] = not args.limit
            break
        raw = pd.DataFrame(batch)
        raw.insert(0, "listing_id", np.arange(state["rows_done"] + 1, state["rows_done"] + len(raw) + 1))
        df, _ = data.clean(raw)
        keys = [digest(t, d) for t, d in zip(df["title"], df["description"])]
        keep = [k not in seen for k in keys]
        df, keys = df[keep].reset_index(drop=True), [k for k, ok in zip(keys, keep) if ok]

        if len(df):
            chunks = data.build_chunks(df, tokenizer)
            # Tách từ BM25 trên CPU chạy song song với embed trên GPU.
            pending = pool.map_async(tokenize_vi, chunks["text"].tolist(), chunksize=64)
            dense = vector.embed_passages(chunks["text"].tolist())
            tokens = pending.get()
            if state["avgdl"] is None:  # cố định từ lô đầu để trọng số BM25 nhất quán giữa các lô
                state["avgdl"] = float(np.mean([len(t) for t in tokens]))
            for i in range(0, len(chunks), UPSERT_BATCH):
                qdrant_store.upsert(chunks.iloc[i:i + UPSERT_BATCH], df, dense[i:i + UPSERT_BATCH],
                                    tokens[i:i + UPSERT_BATCH], state["avgdl"])
            graph_rows = graph.graph_rows(df)
            for name in GRAPH_BLOCKS:
                graph.run_block(name, graph_rows[name])
            with DIGESTS_PATH.open("ab") as handle:
                handle.write(np.asarray(keys, dtype=np.uint64).tobytes())
            seen.update(keys)
            state["chunks"] += len(chunks)
            state["price_sum"] += float(df["price"].sum())
        state["rows_done"] += len(raw)
        state["listings"] += len(df)
        state["digests"] = len(seen)
        save_state(state)
        rate = (state["rows_done"] - rows_at_start) / (perf_counter() - start)
        print(f"{state['rows_done']:>10,} dòng | {state['listings']:,} tin | {state['chunks']:,} chunk | "
              f"{rate:,.0f} dòng/s", flush=True)

    pool.close()
    pool.join()
    if state["done"] or args.limit:
        qdrant_store.finish_indexing()
    save_state(state)
    qdrant_store.write_manifest({k: state[k] for k in ("listings", "chunks", "price_sum")}, FULL_COLLECTION)
    print(f"Xong: {state['listings']:,} tin, {state['chunks']:,} chunk trong Qdrant {FULL_COLLECTION} "
          f"và Neo4j {NEO4J_FULL_URI}" + ("" if state["done"] else " (chưa hết dataset)"))


if __name__ == "__main__":
    main()
