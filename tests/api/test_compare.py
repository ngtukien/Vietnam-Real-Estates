"""`POST /api/compare`: 4 hệ thống song song, mỗi hệ thống đúng thứ tự §4, `compare_done` cuối cùng."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from api_helpers import ScriptedBackend, assert_order, make_app, parse_sse, request

Q = "căn hộ 2pn q7 dưới 4 tỷ"


def run(app: Any, **body: Any) -> tuple[Any, list[tuple[str, dict[str, Any]]]]:
    resp = request(app, "POST", "/api/compare", json={"question": Q, **body})
    assert resp.status_code == 200, resp.text
    return resp, parse_sse(resp.text)


def per_system(events: list[tuple[str, dict[str, Any]]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for name, payload in events[:-1]:
        out[payload["system"]].append(name)
    return out


def test_all_four_systems(app: Any) -> None:
    resp, events = run(app)
    assert events[-1][0] == "compare_done" and "system" not in events[-1][1]
    groups = per_system(events)
    assert set(groups) == {"llm_only", "basic_rag", "graph", "hybrid"}
    for names in groups.values():
        assert_order(names)
    final = events[-1][1]
    assert final["request_id"] == resp.headers["x-request-id"]
    assert sorted(final["systems_ok"]) == sorted(groups) and final["systems_failed"] == []
    assert app.state.slots._sem._value == app.state.slots.capacity


def test_subset_and_failure() -> None:
    _, events = run(make_app(), systems=["basic_rag", "graph"], question="x [fake:error]")
    assert set(per_system(events)) == {"basic_rag", "graph"}
    assert sorted(events[-1][1]["systems_failed"]) == ["basic_rag", "graph"]


def test_duplicate_systems_rejected(app: Any) -> None:
    resp = request(app, "POST", "/api/compare", json={"question": Q, "systems": ["graph", "graph"]})
    assert resp.status_code == 400 and resp.json()["error"]["code"] == "invalid_request"


def test_partially_unavailable_streams_error_for_that_system() -> None:
    status = {"llm_only": (True, None), "basic_rag": (True, None), "graph": (False, "Neo4j chưa chạy"), "hybrid": (True, None)}
    _, events = run(make_app(ScriptedBackend(status=status)))
    groups = per_system(events)
    assert groups["graph"] == ["meta", "error", "done"]
    err = next(p for n, p in events if n == "error" and p["system"] == "graph")
    assert err["code"] == "system_unavailable" and "Neo4j" in err["message"]
    assert events[-1][1]["systems_failed"] == ["graph"]


def test_all_unavailable_is_503() -> None:
    status: dict[str, tuple[bool, str | None]] = {s: (False, "down") for s in ("llm_only", "basic_rag", "graph", "hybrid")}
    resp = request(make_app(ScriptedBackend(status=status)), "POST", "/api/compare", json={"question": Q})
    assert resp.status_code == 503 and resp.json()["error"]["code"] == "system_unavailable"
