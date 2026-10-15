"""Đồ thị con cho giao diện (api-contract §3 `/api/graph/subgraph`): `id = "<Label>:<key>"`, chỉ truy vấn chỉ đọc."""

from __future__ import annotations

from typing import Any

from api.errors import ApiError
from api.schemas import ErrorCode, NodeLabel, RelType
from api.services.neo import cypher

KEY_PROP: dict[str, str] = {"Listing": "listing_id", "PropertyType": "name", "Cluster": "cluster_id"}
INT_KEYS = frozenset({"Listing", "Cluster"})
LABELS = frozenset(n.value for n in NodeLabel)
RELS = frozenset(r.value for r in RelType)
NAME_MAX = 80

NEIGHBORS = """
MATCH (n:{label} {{{prop}: $key}})-[r]-(m)
RETURN type(r) AS type, startNode(r) = n AS outgoing, labels(m) AS labels, properties(m) AS props
LIMIT $lim
"""
NODE = "MATCH (n:{label} {{{prop}: $key}}) RETURN properties(n) AS props LIMIT 1"
LOCATION_CHAIN = """
MATCH (l:Listing {listing_id: $key})-[:IN_WARD|IN_DISTRICT|IN_PROVINCE]->(a)
MATCH p = (a)-[:IN_DISTRICT|IN_PROVINCE|IN_CLUSTER*1..2]->(b)
UNWIND relationships(p) AS r
RETURN DISTINCT type(r) AS type, labels(startNode(r)) AS sl, properties(startNode(r)) AS sp,
       labels(endNode(r)) AS el, properties(endNode(r)) AS ep
LIMIT 20
"""


def _label(labels: list[str]) -> str | None:
    return next((lb for lb in labels if lb in LABELS), None)


def node_id(label: str, props: dict[str, Any]) -> str:
    return f"{label}:{props.get(KEY_PROP.get(label, 'key'))}"


def to_node(label: str, props: dict[str, Any]) -> dict[str, Any]:
    if label == "Listing":
        name = str(props.get("title") or props.get("listing_id"))[:NAME_MAX]
    else:
        name = str(props.get("name") or props.get("label") or props.get(KEY_PROP.get(label, "key")))
    flat = {k: v for k, v in props.items() if v is None or isinstance(v, (str, int, float, bool))}
    return {"id": node_id(label, props), "label": label, "name": name, "props": flat}


def parse_key(label: str, key: str) -> str | int:
    if label not in INT_KEYS:
        return key
    try:
        return int(key)
    except ValueError:
        raise ApiError(ErrorCode.INVALID_REQUEST, f"key của {label} phải là số nguyên",
                       {"fields": [{"loc": ["query", "key"], "msg": "phải là số nguyên"}]}) from None


def subgraph(label: str, key: str | int, limit: int) -> dict[str, Any] | None:
    """None ⇒ node không tồn tại (404)."""
    prop = KEY_PROP.get(label, "key")
    found = cypher(NODE.format(label=label, prop=prop), {"key": key})
    if not found:
        return None
    center = to_node(label, found[0]["props"])
    nodes: dict[str, dict[str, Any]] = {center["id"]: center}
    edges: dict[tuple[str, str, str], dict[str, str]] = {}

    def add_edge(rel: str, source: str, target: str) -> None:
        if rel in RELS:
            edges[(source, target, rel)] = {"source": source, "target": target, "type": rel}

    rows = cypher(NEIGHBORS.format(label=label, prop=prop), {"key": key, "lim": limit + 1})
    truncated = len(rows) > limit
    for r in rows[:limit]:
        other_label = _label(r["labels"])
        if other_label is None:
            continue
        other = to_node(other_label, r["props"])
        nodes.setdefault(other["id"], other)
        src, dst = (center["id"], other["id"]) if r["outgoing"] else (other["id"], center["id"])
        add_edge(r["type"], src, dst)

    if label == "Listing":  # thêm chuỗi Phường → Quận → Tỉnh và Cụm để đồ thị con có ngữ cảnh địa lý
        for r in cypher(LOCATION_CHAIN, {"key": key}):
            sl, el = _label(r["sl"]), _label(r["el"])
            if sl is None or el is None:
                continue
            s, e = to_node(sl, r["sp"]), to_node(el, r["ep"])
            nodes.setdefault(s["id"], s)
            nodes.setdefault(e["id"], e)
            add_edge(r["type"], s["id"], e["id"])
    return {"nodes": list(nodes.values()), "edges": list(edges.values()), "truncated": truncated}
