"""Bất biến Knowledge Graph sau import (data-contract §6.3, §7.3). Tự skip nếu Neo4j không chạy hoặc chưa có graph_report."""

import json
import time

import pytest

from vre.config import load_settings

pytestmark = pytest.mark.integration

MARKET_QUERY = ("MATCH (d:District {name: $district})<-[:STAT_OF]-"
                "(s:MarketStat {property_type: 'ALL', month: 'ALL', deal_type: 'sale'}) RETURN s.median_price_m2_mil AS m")
FULL_GRAPH_MIN_LISTINGS = 100_000  # dưới ngưỡng này là bản thử 1.000 tin, không chắc có Quận 7


@pytest.fixture(scope="module")
def graph():
    from vre.graph import client

    settings = load_settings()
    try:
        client.use(settings.neo4j_uri)
        client.run_cypher("RETURN 1")
    except Exception as exc:  # Neo4j tắt/không kết nối được ⇒ skip, không fail
        pytest.skip(f"Neo4j không chạy: {exc.__class__.__name__}")
    for name in ("graph_report.json", "graph_report_trial.json"):
        path = settings.results_dir / name
        if path.exists():
            return client, json.loads(path.read_text(encoding="utf-8"))
    pytest.skip("chưa có results/graph_report.json")


def test_report_invariants_hold(graph):
    _, report = graph
    assert {"schema_version", "generated_at", "git_sha", "import_seconds", "nodes", "relationships",
            "invariants", "communities"} <= set(report)
    assert report["invariants"] == {"listing_count_ok": True, "price_sum_ok": True}
    assert {"n_wards", "n_clusters", "modularity", "k", "seed"} <= set(report["communities"])


def test_listing_count_matches_report(graph):
    client, report = graph
    n = int(client.run_cypher("MATCH (l:Listing) RETURN count(l) AS n").iloc[0]["n"])
    assert n == report["nodes"]["Listing"]


def test_market_stat_lookup_is_fast(graph):
    client, report = graph
    client.run_cypher(MARKET_QUERY, {"district": "Quận 7"})  # làm ấm plan cache
    t0 = time.perf_counter()
    df = client.run_cypher(MARKET_QUERY, {"district": "Quận 7"})
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"MarketStat Quận 7: {len(df)} dòng, {elapsed_ms:.1f} ms")
    assert elapsed_ms < 100
    if report["nodes"]["Listing"] >= FULL_GRAPH_MIN_LISTINGS:
        assert len(df) == 1 and df.iloc[0]["m"] > 0
    stat_rows = client.run_cypher("MATCH (s:MarketStat) WHERE s.n < 5 RETURN count(s) AS c").iloc[0]["c"]
    assert stat_rows == 0
