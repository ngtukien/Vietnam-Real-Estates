"""Chunk tin rao theo cửa sổ token (data-contract §2.3, §4, §4.2).

Mỗi tin được tokenize đúng một lần: header một lần, mô tả một lần (kèm offsets ký tự). Chunk được ghép từ id
đã có: `[bos] + "passage: " + header[:header_max_tokens] + nl + body[cửa sổ] + [eos]`; văn bản lưu = header + "\\n" + body[span].
Ghép id rời cho đúng chuỗi id của tokenize cả văn bản (đã kiểm với XLM-R: "\\n" không sinh token).
"""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TOKENIZE_BATCH = 2000  # số tin mỗi lời gọi tokenizer (giới hạn RAM của list Python)
POINT_STRIDE = 64  # point_id = listing_id * 64 + chunk_index (§2.3)
PART_COLUMNS = ("listing_id", "title", "description", "property_type", "province", "district", "ward",
                "price_vnd", "area_m2", "bedrooms")


def chunk_starts(length: int, size: int, overlap: int) -> list[int]:
    """Vị trí bắt đầu của từng cửa sổ khi cắt `length` token, cỡ `size`, chồng lấn `overlap`."""
    starts, s = [0], 0
    while s + size < length:
        s += size - overlap
        starts.append(s)
    return starts


def n_chunks_formula(length: int, size: int, overlap: int) -> int:
    """n = ⌈(L − O) / (S − O)⌉ khi L > S, ngược lại 1."""
    return 1 if length <= size else math.ceil((length - overlap) / (size - overlap))


def point_id(listing_id: int, chunk_index: int) -> int:
    return int(listing_id) * POINT_STRIDE + int(chunk_index)


def listing_of(point: int) -> tuple[int, int]:
    return int(point) // POINT_STRIDE, int(point) % POINT_STRIDE


def _text(value) -> str | None:
    return value if isinstance(value, str) and value else None


def _number(value) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def listing_header(row: dict) -> str:
    """Tiêu đề + dòng meta `loại · phường, quận, tỉnh · giá X tỷ · N m² · K phòng ngủ` (lặp ở mọi chunk)."""
    place = ", ".join(p for p in (_text(row.get(k)) for k in ("ward", "district", "province")) if p)
    parts = [_text(row.get("property_type")), place]
    price, area, beds = (_number(row.get(k)) for k in ("price_vnd", "area_m2", "bedrooms"))
    if price is not None:
        parts.append(f"giá {price / 1e9:.2f} tỷ")
    if area is not None:
        parts.append(f"{area:g} m²")
    if beds is not None:
        parts.append(f"{beds:g} phòng ngủ")
    return f"{_text(row.get('title')) or ''}\n" + " · ".join(p for p in parts if p)


def chunk_rows(rows: list[dict], tokenizer, special, settings) -> tuple[pd.DataFrame, list[list[int]], int]:
    """Chunk một lô tin. Trả (bảng chunk, input_ids từng chunk, số tin bị cắt ở `max_chunks_per_listing`)."""
    records, sequences, truncated = [], [], 0
    for rec, seqs, trunc in iter_chunk_batches(rows, tokenizer, special, settings):
        records += rec
        sequences += seqs
        truncated += trunc
    return chunk_table(records), sequences, truncated


def chunk_table(records: list[tuple]) -> pd.DataFrame:
    table = pd.DataFrame.from_records(records, columns=["point_id", "listing_id", "chunk_index", "text", "n_tokens"])
    return table.astype({"point_id": "int64", "listing_id": "int64", "chunk_index": "int16", "n_tokens": "int16"})


def iter_chunk_batches(rows: list[dict], tokenizer, special, settings):
    """Từng lô `TOKENIZE_BATCH` tin: (bản ghi chunk, input_ids, số tin bị cắt). Chỉ một lô nằm dưới dạng list Python."""
    for start in range(0, len(rows), TOKENIZE_BATCH):
        records, sequences, truncated = [], [], 0
        batch = rows[start:start + TOKENIZE_BATCH]
        headers = [listing_header(r) for r in batch]
        bodies = [_text(r.get("description")) or "" for r in batch]
        header_ids = tokenizer(headers, add_special_tokens=False)["input_ids"]
        body_enc = tokenizer(bodies, add_special_tokens=False, return_offsets_mapping=True)
        for row, header, body, h_ids, b_ids, offsets in zip(batch, headers, bodies, header_ids,
                                                            body_enc["input_ids"], body_enc["offset_mapping"], strict=True):
            h_ids = list(h_ids[:settings.header_max_tokens])
            budget = settings.chunk_size - len(h_ids) - len(special.newline)
            starts = chunk_starts(len(b_ids), budget, settings.chunk_overlap)
            if len(starts) > settings.max_chunks_per_listing:
                truncated += 1
                starts = starts[:settings.max_chunks_per_listing]
            head_ids = [special.bos, *special.passage_prefix, *h_ids, *special.newline]
            lid = int(row["listing_id"])
            for index, s in enumerate(starts):
                window = offsets[s:s + budget]
                text = header + ("\n" + body[window[0][0]:window[-1][1]] if window else "")
                ids = head_ids + list(b_ids[s:s + budget]) + [special.eos]
                sequences.append(ids)
                records.append((point_id(lid, index), lid, index, text, len(ids) - 2 - len(special.passage_prefix)))
        yield records, sequences, truncated


def n_parts(n_listings: int, part_size: int) -> int:
    return math.ceil(n_listings / part_size)


def read_part(clean_path: Path, part: int, part_size: int, columns=None) -> pa.Table:
    """Dòng [part·part_size, (part+1)·part_size) của bảng sạch; chỉ đọc các row group giao với khoảng đó."""
    pf = pq.ParquetFile(clean_path)
    lo, hi = part * part_size, (part + 1) * part_size
    tables, first = [], 0
    for g in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if first < hi and first + n > lo:
            t = pf.read_row_group(g, columns=list(columns) if columns else None)
            tables.append(t.slice(max(lo - first, 0), min(hi, first + n) - max(lo, first)))
        first += n
    if not tables:
        raise IndexError(f"part {part} nằm ngoài bảng sạch ({pf.metadata.num_rows} dòng)")
    return pa.concat_tables(tables)


def write_parquet_atomic(table: pa.Table | pd.DataFrame, path: Path) -> None:
    if isinstance(table, pd.DataFrame):
        table = pa.Table.from_pandas(table, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def chunk_part(settings, layout, part: int) -> dict:
    """Việc của một tiến trình tokenize: đọc part, ghi `chunks/` + `tok/`. Trả số chunk, số tin bị cắt."""
    from vre.index import tokenize

    tok = tokenize.load_fast_tokenizer()  # không kéo torch vào worker (commit)
    rows = read_part(settings.clean_path, part, settings.part_size, PART_COLUMNS).to_pylist()
    records, flats, lengths, truncated = [], [], [], 0
    for rec, seqs, trunc in iter_chunk_batches(rows, tok, tokenize.special_ids(tok), settings):
        records += rec
        truncated += trunc
        lengths.append(np.fromiter((len(q) for q in seqs), dtype=np.int64, count=len(seqs)))
        flats.append(np.fromiter((t for q in seqs for t in q), dtype=np.int32, count=int(lengths[-1].sum())))
    table = chunk_table(records)
    tokenize.save_tokens(layout.tok(part), np.concatenate(flats), np.concatenate(lengths))
    write_parquet_atomic(table, layout.chunks(part))  # ghi sau cùng: chunks có mặt ⇒ tok đã có
    return {"part": part, "n_chunks": len(table), "truncated": truncated,
            "n_tokens_max": int(np.max(table["n_tokens"])) if len(table) else 0}
