"""Tín hiệu phối hợp: phiên Phase 3/4 (p3, p4) gửi được P3_RESULTS_READY; người gửi lạ vẫn bị từ chối."""

import json

import pytest

from vre.common import write_handoff
from vre.config import load_settings


def test_p3_results_ready(tmp_path):
    settings = load_settings(data_root=tmp_path)
    path = write_handoff(settings, "P3_RESULTS_READY", "p3", {"eval": "x"})
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "P3_RESULTS_READY.json" and doc["from"] == "p3" and doc["seq"] == 1 and doc["payload"] == {"eval": "x"}
    assert json.loads(write_handoff(settings, "P3_RESULTS_READY", "p3", {}).read_text(encoding="utf-8"))["seq"] == 2


def test_unknown_sender_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_handoff(load_settings(data_root=tmp_path), "NOTE", "p9", {})
