"""Chunk theo cửa sổ token (data-contract §2.3, §4.2): công thức số chunk, trần 64, point_id, tokenize mỗi tin đúng 1 lần."""

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vre.index import chunk
from vre.index.tokenize import SpecialIds

SPECIAL = SpecialIds(bos=0, eos=2, pad=1, passage_prefix=(9, 10), query_prefix=(8,), newline=())
CFG = SimpleNamespace(chunk_size=20, chunk_overlap=5, header_max_tokens=6, max_chunks_per_listing=3)


class WordTokenizer:
    """Tokenizer giả: mỗi từ một token (id = độ dài từ + 100); ghi lại mọi văn bản được tokenize."""

    def __init__(self):
        self.calls, self.texts = 0, []

    def __call__(self, texts, add_special_tokens=False, return_offsets_mapping=False):
        self.calls += 1
        self.texts.extend(texts)
        ids, offsets = [], []
        for text in texts:
            spans, pos = [], 0
            for word in text.split():
                start = text.index(word, pos)
                spans.append((start, start + len(word)))
                pos = start + len(word)
            ids.append([100 + e - s for s, e in spans])
            offsets.append(spans)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def _row(listing_id: int, n_words: int, **extra) -> dict:
    return {"listing_id": listing_id, "title": f"Tin {listing_id}", "property_type": "Nhà", "province": "Hà Nội",
            "district": "Cầu Giấy", "ward": None, "price_vnd": 3.2e9, "area_m2": 65.0, "bedrooms": 2,
            "description": " ".join(f"w{i}" for i in range(n_words)), **extra}


def test_chunk_count_matches_formula():
    assert len(chunk.chunk_starts(2000, 400, 100)) == chunk.n_chunks_formula(2000, 400, 100) == 7
    for length in (0, 1, 299, 300, 301, 551, 1000):
        assert len(chunk.chunk_starts(length, 300, 50)) == chunk.n_chunks_formula(length, 300, 50)


def test_point_id_is_reversible():
    lid = 2**53 - 1
    for index in (0, 1, 63):
        pid = chunk.point_id(lid, index)
        assert chunk.listing_of(pid) == (lid, index)
        assert pid < 2**59
    assert chunk.point_id(lid, 0) == lid * 64


def test_header_skips_missing_fields():
    header = chunk.listing_header({"title": "Bán nhà", "property_type": "Nhà", "province": "Hà Nội",
                                   "price_vnd": float("nan"), "area_m2": None, "bedrooms": None})
    assert header == "Bán nhà\nNhà · Hà Nội"
    full = chunk.listing_header(_row(1, 0))
    assert "giá 3.20 tỷ" in full and "65 m²" in full and "2 phòng ngủ" in full


def test_chunks_are_built_from_one_tokenization_per_listing():
    tok = WordTokenizer()
    rows = [_row(1, 0), _row(2, 30), _row(3, 500)]
    table, sequences, truncated = chunk.chunk_rows(rows, tok, SPECIAL, CFG)
    headers = [chunk.listing_header(r) for r in rows]
    assert tok.calls == 2  # một lần cho mọi header, một lần cho mọi mô tả
    assert sorted(tok.texts) == sorted(headers + [r["description"] for r in rows])
    per_listing = table.groupby("listing_id").size().to_dict()
    header_len = len(headers[1].split())
    budget = CFG.chunk_size - min(header_len, CFG.header_max_tokens)
    assert per_listing[1] == 1  # không có mô tả: một chunk chỉ có header
    assert per_listing[2] == chunk.n_chunks_formula(30, budget, CFG.chunk_overlap)
    assert per_listing[3] == CFG.max_chunks_per_listing and truncated == 1
    assert table["point_id"].tolist() == [chunk.point_id(lid, i) for lid, i in zip(table["listing_id"], table["chunk_index"], strict=True)]
    for (_, rec), ids in zip(table.iterrows(), sequences, strict=True):
        assert rec["text"].startswith(headers[rec["listing_id"] - 1])
        assert ids[0] == SPECIAL.bos and ids[-1] == SPECIAL.eos and tuple(ids[1:3]) == SPECIAL.passage_prefix
        assert len(ids) - 2 - len(SPECIAL.passage_prefix) == rec["n_tokens"] <= CFG.chunk_size


def test_windows_overlap_and_text_matches_ids():
    tok = WordTokenizer()
    row = _row(7, 40)
    table, sequences, _ = chunk.chunk_rows([row], tok, SPECIAL, CFG)
    header_ids = tok(chunk.listing_header(row).split("\n"))["input_ids"]
    n_header = min(sum(len(x) for x in header_ids), CFG.header_max_tokens)
    budget = CFG.chunk_size - n_header
    bodies = [t.split("\n", 2)[-1].split() for t in table["text"]]
    assert bodies[0][:1] == ["w0"] and len(bodies[0]) == budget
    assert bodies[1][0] == f"w{budget - CFG.chunk_overlap}"  # cửa sổ sau chồng lấn đúng `overlap` token
    for words, ids in zip(bodies, sequences, strict=True):
        assert len(ids) == 1 + 2 + n_header + len(words) + 1


def test_read_part_spans_row_groups(tmp_path):
    path = tmp_path / "listings.parquet"
    pq.write_table(pa.table({"listing_id": list(range(30))}), path, row_group_size=7)
    assert chunk.read_part(path, 1, 10, ["listing_id"]).column("listing_id").to_pylist() == list(range(10, 20))
    assert chunk.read_part(path, 2, 13, ["listing_id"]).column("listing_id").to_pylist() == list(range(26, 30))
    with pytest.raises(IndexError):
        chunk.read_part(path, 3, 10)
    assert chunk.n_parts(30, 10) == 3 and chunk.n_parts(31, 10) == 4
