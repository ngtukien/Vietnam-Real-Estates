"""Nạp Knowledge Graph bằng `neo4j-admin database import full` (data-contract §6, §7.3; phase-01 Bước 6).

stop neo4j → `docker compose run --rm --no-deps neo4j neo4j-admin database import full ...` → up --wait →
`graph/post_import.cypher` → đợi index ONLINE → kiểm bất biến → `results/graph_report.json`.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from vre.common import CODE_ROOT, listings_fingerprint, result_header, write_json_atomic
from vre.config import Settings
from vre.graph import client
from vre.index import graph_export

POST_IMPORT = CODE_ROOT / "graph" / "post_import.cypher"
CONTAINER_IMPORT_DIR = "/import"
PRICE_SUM_RTOL = 1e-6
INDEX_TIMEOUT_S = 3600
INDEX_POLL_S = 5


def compose(settings: Settings, *args: str) -> subprocess.CompletedProcess[str]:
    """`docker compose -f <compose_file> ...`; lỗi ⇒ raise kèm phần cuối stdout/stderr."""
    cmd = ["docker", "compose", "-f", str(settings.compose_file), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-40:])
        raise RuntimeError(f"Lệnh thất bại ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc


def admin_import_args(info: dict[str, Any], threads: int, heap: str) -> list[str]:
    """Đối số `docker compose run` cho neo4j-admin (cú pháp đã kiểm bằng `--help` trong neo4j:5.26-community)."""
    nodes = [f"--nodes={CONTAINER_IMPORT_DIR}/{name}" for name in info["nodes"].values()]
    rels = [f"--relationships={CONTAINER_IMPORT_DIR}/{name}" for name in info["relationships"]]
    return ["run", "--rm", "--no-deps", "-e", f"HEAP_SIZE={heap}", "neo4j",
            "neo4j-admin", "database", "import", "full", "neo4j", "--overwrite-destination=true",
            *nodes, *rels, "--skip-bad-relationships=false", "--bad-tolerance=0", f"--threads={int(threads)}",
            f"--max-off-heap-memory={heap}", "--report-file=/logs/import.report"]


def cypher_statements(text: str) -> list[str]:
    """Tách script Cypher theo `;`, bỏ dòng chú thích `//`."""
    body = "\n".join(line for line in text.splitlines() if not line.strip().startswith("//"))
    return [s.strip() for s in body.split(";") if s.strip()]


def wait_indexes_online(timeout_s: int = INDEX_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        pending = client.run_cypher("SHOW INDEXES YIELD name, state WHERE state <> 'ONLINE' RETURN name, state")
        if pending.empty:
            return
        failed = pending[pending["state"] == "FAILED"]
        if not failed.empty:
            raise RuntimeError(f"Index FAILED: {', '.join(failed['name'])}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Index chưa ONLINE sau {timeout_s}s: {pending.to_dict('records')}")
        time.sleep(INDEX_POLL_S)


def scalar_query(query: str) -> Any:
    df = client.run_cypher(query)
    return None if df.empty else df.iloc[0, 0]


def graph_counts() -> tuple[dict[str, int], dict[str, int]]:
    labels = list(graph_export.NODES)
    types = sorted({rel[1] for rel in graph_export.RELS})
    nodes = {lb: int(scalar_query(f"MATCH (n:`{lb}`) RETURN count(n)")) for lb in labels}
    rels = {t: int(scalar_query(f"MATCH ()-[r:`{t}`]->() RETURN count(r)")) for t in types}
    return nodes, rels


def check_invariants(expected_n: int, expected_sum: float) -> dict[str, bool]:
    df = client.run_cypher("MATCH (l:Listing) RETURN count(l) AS n, sum(l.price_vnd) AS s")
    n, total = int(df.iloc[0]["n"]), float(df.iloc[0]["s"] or 0.0)
    return {"listing_count_ok": n == expected_n,
            "price_sum_ok": math.isclose(total, expected_sum, rel_tol=PRICE_SUM_RTOL)}


def report_path(settings: Settings, limit: int | None) -> Path:
    """Bản thử (`limit`) ghi file riêng để không đè/giả mạo báo cáo bản full."""
    return settings.results_dir / ("graph_report.json" if limit is None else "graph_report_trial.json")


def already_done(settings: Settings) -> dict[str, Any] | None:
    path = report_path(settings, None)
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if not all(report.get("invariants", {}).values()):
        return None
    try:
        client.use(settings.neo4j_uri)
        in_graph = int(scalar_query("MATCH (l:Listing) RETURN count(l)") or 0)
    except Exception as exc:  # Neo4j không truy vấn được ⇒ coi như chưa xong, nạp lại
        print(f"Không kiểm được Neo4j ({exc.__class__.__name__}); nạp lại.")
        return None
    return report if in_graph == listings_fingerprint(settings.clean_path)["n_listings"] else None


def run(settings: Settings, limit: int | None = None, threads: int = 12, heap: str = "4G") -> dict[str, Any]:
    """Xuất CSV → neo4j-admin import → constraint/index → bất biến → graph_report. `limit`: thử với N tin đầu."""
    if limit is None and (done := already_done(settings)) is not None:
        print("đã xong")
        return done
    t0 = time.monotonic()
    info = graph_export.export(settings, limit=limit)
    print(f"Xuất CSV: {info['n_listings']} tin, {time.monotonic() - t0:.0f}s")
    client.use(settings.neo4j_uri)  # đóng driver cũ trước khi dừng server
    compose(settings, "stop", "neo4j")
    t1 = time.monotonic()
    try:
        out = compose(settings, *admin_import_args(info, threads, heap))
        print("\n".join(out.stdout.strip().splitlines()[-8:]))
    finally:
        import_seconds = time.monotonic() - t1
        compose(settings, "up", "-d", "--wait", "neo4j")  # luôn bật lại Neo4j, kể cả khi import lỗi
    client.use(settings.neo4j_uri)
    for stmt in cypher_statements(POST_IMPORT.read_text(encoding="utf-8")):
        client.run_cypher(stmt, write=True)
    wait_indexes_online()
    nodes, rels = graph_counts()
    report = {**result_header(), "import_seconds": round(import_seconds, 1), "nodes": nodes, "relationships": rels,
              "invariants": check_invariants(info["n_listings"], info["price_vnd_sum"]),
              "communities": info["communities"]}
    write_json_atomic(report_path(settings, limit), report)
    print(f"graph_report: invariants={report['invariants']} ({time.monotonic() - t0:.0f}s)")
    return report
