"""Chỉ mục chính sau lượt chạy dài (phase-02 §Verify). Tự skip tới khi `results/night_report.json` có `status=completed`."""

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.integration
SAMPLE_LISTINGS = 1000


@pytest.fixture(scope="module")
def index():
    from vre.config import load_settings
    from vre.index import qdrant_store
    from vre.pipeline.state import layout_for

    settings = load_settings()
    layout = layout_for(settings, dry_run=False)
    if not layout.report_path.exists() or json.loads(layout.report_path.read_text(encoding="utf-8"))["status"] != "completed":
        pytest.skip("lượt chạy dài chưa completed")
    state = json.loads(layout.state_path.read_text(encoding="utf-8"))
    return settings, layout, state, qdrant_store.client(settings, timeout=60)


def test_alias_points_and_dimension(index):
    from vre.config import EMBED_MODEL_SLUG, EMBED_MODELS
    from vre.index import qdrant_store

    settings, _, state, qc = index
    assert qdrant_store.alias_target(qc, settings.collection_alias) == settings.collection
    info = qc.get_collection(settings.collection_alias)
    assert info.points_count == sum(p["n_chunks"] for p in state["parts"].values())
    assert info.config.params.vectors["dense"].size == EMBED_MODELS[EMBED_MODEL_SLUG][1] == 1024


def test_random_listings_have_chunk_zero(index):
    from vre.index.chunk import point_id

    settings, _, _, qc = index
    ids = pq.read_table(settings.clean_path, columns=["listing_id"]).column("listing_id").to_numpy()
    sample = np.random.default_rng(42).choice(ids, size=min(SAMPLE_LISTINGS, len(ids)), replace=False)
    found = qc.retrieve(settings.collection_alias, ids=[point_id(int(i), 0) for i in sample], with_payload=["listing_id"])
    assert sorted(p.payload["listing_id"] for p in found) == sorted(int(i) for i in sample)


def test_dense_bm25_and_province_filter(index):
    from qdrant_client import models

    from vre.index import bm25
    from vre.index.embed import get_encoder

    settings, _, _, qc = index
    query = "căn hộ 2 phòng ngủ quận 7 dưới 4 tỷ"
    dense = get_encoder(settings).encode_texts([query], kind="query")[0].astype(np.float32).tolist()
    assert len(qc.query_points(settings.collection_alias, query=dense, using="dense", limit=10).points) == 10
    assert qc.query_points(settings.collection_alias, query=bm25.query_vector(query), using="bm25", limit=10).points
    flt = models.Filter(must=[models.FieldCondition(key="province", match=models.MatchValue(value="Hồ Chí Minh"))])
    hits = qc.query_points(settings.collection_alias, query=dense, using="dense", limit=10, query_filter=flt,
                           with_payload=["province"]).points
    assert hits and all(h.payload["province"] == "Hồ Chí Minh" for h in hits)
