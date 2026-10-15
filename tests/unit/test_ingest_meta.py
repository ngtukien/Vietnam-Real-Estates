"""Dấu logic của ingest: artifact trung gian/bảng sạch chỉ được dùng lại khi cùng logic sinh ra chúng."""

import json

from vre.data import ingest


def test_meta_roundtrip_and_mismatch(tmp_path):
    assert not ingest.meta_matches(tmp_path)
    ingest.write_meta(tmp_path)
    assert ingest.meta_matches(tmp_path)
    meta = tmp_path / ingest.META_NAME
    meta.write_text(json.dumps({"logic": "khac"}), encoding="utf-8")
    assert not ingest.meta_matches(tmp_path)


def test_logic_hash_covers_existing_files_and_is_stable():
    assert ingest.logic_hash() == ingest.logic_hash()
    assert len(ingest.logic_hash()) == 16
