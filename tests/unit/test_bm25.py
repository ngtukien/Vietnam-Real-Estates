"""BM25 sparse (data-contract §4, §5): token id, tf gộp, trọng số theo avgdl toàn corpus, part BM25 trên đĩa."""

import zlib
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vre.index import bm25


def test_token_id_is_crc32_31bit():
    assert bm25.token_id("sổ_hồng") == zlib.crc32("sổ_hồng".encode()) & 0x7FFFFFFF


def test_doc_terms_aggregates_tf_sorted():
    ids, tfs, doc_len = bm25.doc_terms(["sổ", "hồng", "sổ"])
    assert ids == sorted(ids) and doc_len == 3
    assert dict(zip(ids, tfs, strict=True)) == {bm25.token_id("sổ"): 2, bm25.token_id("hồng"): 1}


def test_weights_follow_bm25_formula():
    k1, b = bm25.K1, bm25.B
    (w,) = bm25.weights([2], doc_len=6, avgdl=3.0)
    assert w == pytest.approx(2 * (k1 + 1) / (2 + k1 * (1 - b + b * 6 / 3.0)))
    short, long = bm25.weights([1], 2, 4.0)[0], bm25.weights([1], 8, 4.0)[0]
    assert short > long  # tài liệu dài hơn avgdl bị phạt


def test_document_and_query_vectors():
    doc = bm25.document_vector([5, 9], [1, 3], doc_len=4, avgdl=4.0)
    assert doc.indices == [5, 9] and doc.values[1] > doc.values[0]
    query = bm25.query_vector("Sổ hồng, sổ hồng!")
    assert query.values == [1.0] * len(query.indices) and query.indices == sorted(set(query.indices))


def test_tokenize_vi_lowercases_and_drops_punctuation():
    tokens = bm25.tokenize_vi("Nhà Mặt Tiền , giá 5 tỷ !")
    assert tokens and all(t == t.lower() for t in tokens)
    assert "," not in tokens and "!" not in tokens and "5" in tokens


def _layout(tmp_path):
    return SimpleNamespace(chunks=lambda p: tmp_path / "chunks" / f"part_{p:04d}.parquet",
                           bm25=lambda p: tmp_path / "bm25" / f"part_{p:04d}.parquet")


def test_bm25_part_writes_parquet_and_corpus_avgdl(tmp_path):
    layout = _layout(tmp_path)
    texts = {0: ["nhà đẹp quận bảy", "đất nền"], 1: ["căn hộ hai phòng ngủ gần chợ"]}
    for part, rows in texts.items():
        layout.chunks(part).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"point_id": list(range(part * 10, part * 10 + len(rows))), "text": rows}),
                       layout.chunks(part))
    outs = [bm25.bm25_part(layout, part) for part in texts]
    table = pq.read_table(layout.bm25(0))
    assert table.schema.equals(bm25.BM25_SCHEMA)
    assert table.column("point_id").to_pylist() == [0, 1]
    assert not list(tmp_path.glob("bm25/*.tmp"))
    total = sum(o["doc_len_sum"] for o in outs)
    assert bm25.corpus_avgdl(layout, [0, 1]) == pytest.approx(total / 3)
    with pytest.raises(FileNotFoundError):
        bm25.corpus_avgdl(layout, [2])
