"""`GET /api/graph/subgraph?listing_id=` hoặc `?node_label=&key=&limit=` (đúng MỘT trong hai dạng)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from api.errors import ApiError
from api.routers.common import backend, errors
from api.schemas import ErrorCode, NodeLabel, Subgraph
from api.services.backend import Backend
from api.services.graph_view import parse_key

router = APIRouter(prefix="/api/graph", tags=["graph"])
MAX_SAFE_INT = 2**53 - 1


def _bad(msg: str) -> ApiError:
    return ApiError(ErrorCode.INVALID_REQUEST, msg, {"fields": [{"loc": ["query"], "msg": msg}]})


@router.get("/subgraph", response_model=Subgraph, responses=errors(400, 401, 404, 429, 503))
def subgraph(listing_id: int | None = Query(None, ge=1, le=MAX_SAFE_INT), node_label: NodeLabel | None = None,
             key: str | None = Query(None, min_length=1, max_length=300), limit: int = Query(40, ge=1, le=100),
             b: Backend = Depends(backend)) -> dict[str, Any]:
    by_node = node_label is not None or key is not None
    if (listing_id is not None) == by_node:
        raise _bad("Cần đúng một trong hai dạng: listing_id, hoặc node_label + key")
    parsed: str | int
    if listing_id is not None:
        label, parsed = NodeLabel.LISTING.value, listing_id
    elif node_label is None or key is None:
        raise _bad("node_label và key phải đi cùng nhau")
    else:
        label, parsed = node_label.value, parse_key(node_label.value, key)
    out = b.subgraph(label, parsed, limit)
    if out is None:
        raise ApiError(ErrorCode.NOT_FOUND, f"Không có node {label}:{parsed}")
    return out
