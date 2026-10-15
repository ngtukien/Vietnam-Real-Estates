"""results/embedding_benchmark.json (data-contract §7.4): bản sao thẳng .cache/scratch/bench_final.jsonl."""

import json
from pathlib import Path

import pytest

from vre.config import load_settings
from vre.data.ingest import BENCH_SOURCE, copy_embedding_benchmark

FIELDS = {"model", "params_m", "dim", "corpus", "queries", "docs_per_s", "hours_2_8m_chunks", "peak_vram_gb",
          "recall_at_1", "recall_at_10", "mrr_at_10", "token_budget", "sweep", "gpu_util_avg", "vram_used_max_mb",
          "power_avg_w", "sm_clock_avg_mhz"}
REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED = REPO_ROOT / "results" / "embedding_benchmark.json"
SOURCE = Path(r"D:\Vietnam-Real-Estates") / BENCH_SOURCE


def _check(items):
    assert isinstance(items, list) and len(items) == 4
    for item in items:
        assert FIELDS <= set(item), item.get("model")
    best = max(items, key=lambda x: x["mrr_at_10"])
    assert best["model"] == "intfloat/multilingual-e5-large"


@pytest.mark.skipif(not COMMITTED.is_file(), reason="chưa chạy ingest")
def test_committed_file_shape():
    _check(json.loads(COMMITTED.read_text(encoding="utf-8")))


@pytest.mark.skipif(not SOURCE.is_file(), reason="file số đo nguồn chỉ có trên máy dự án")
def test_copy_is_verbatim(tmp_path):
    root = tmp_path
    (root / BENCH_SOURCE).parent.mkdir(parents=True)
    (root / BENCH_SOURCE).write_text(SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    out = copy_embedding_benchmark(load_settings(data_root=root))
    items = json.loads(out.read_text(encoding="utf-8"))
    source_lines = [json.loads(x) for x in SOURCE.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert items == source_lines
    _check(items)
