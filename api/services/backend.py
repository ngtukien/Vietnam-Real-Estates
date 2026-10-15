"""Giao diện backend mà router gọi. `RealBackend` = DuckDB/Neo4j/Qdrant + facade lõi (api-contract §7);
`FakeBackend` (`api.services.fake`) = fixtures (§6). Router không import `vre.*` trực tiếp."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any, Protocol

from api.deps import ApiSettings
from api.errors import ApiError
from api.schemas import ErrorCode, SystemId

log = logging.getLogger("vre.api")

SYSTEM_NAMES: dict[str, str] = {
    "llm_only": "LLM thuần (không truy xuất)",
    "basic_rag": "Basic RAG",
    "graph": "Graph (Text2Cypher)",
    "hybrid": "Hybrid (tự động)",
}
STATUS_TTL_S = 30.0

Status = dict[str, tuple[bool, str | None]]


class Backend(Protocol):
    def health(self) -> dict[str, Any]: ...
    def info(self) -> dict[str, Any]: ...
    def filter_options(self) -> dict[str, Any]: ...
    def listing(self, listing_id: int) -> dict[str, Any] | None: ...
    def valuate(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def stats_overview(self) -> dict[str, Any]: ...
    def market(self, province: str, district: str | None, property_type: str | None, deal_type: str) -> dict[str, Any] | None: ...
    def geo_areas(self, level: str, province: str | None) -> dict[str, Any]: ...
    def subgraph(self, label: str, key: str | int, limit: int) -> dict[str, Any] | None: ...
    def benchmark(self) -> dict[str, Any] | None: ...
    def system_status(self) -> Status: ...
    def stream_answer(self, system: str, question: str, filters: dict[str, Any] | None, top_k: int, use_rerank: bool,
                      stream_tokens: bool) -> Iterator[tuple[str, dict[str, Any]]]: ...


def _read_json(path: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


class RealBackend:
    def __init__(self, settings: ApiSettings):
        self.settings = settings
        self._status: tuple[float, Status] | None = None

    # ---- facade lõi QA (Lane A1). Import lười: nhánh API không có `vre.qa`.
    def system_status(self) -> Status:
        now = time.monotonic()
        if self._status and now - self._status[0] < STATUS_TTL_S:
            return self._status[1]
        try:
            from vre.qa import system_status
        except ImportError as error:
            log.warning("Chưa có facade vre.qa (%s) — mọi hệ thống QA unavailable", error)
            reason = "Lõi QA (vre.qa) chưa được cài trên server"
            status: Status = {s.value: (False, reason) for s in SystemId}
        else:
            status = {str(k): (bool(v[0]), v[1]) for k, v in system_status().items()}
        self._status = (now, status)
        return status

    def stream_answer(self, system: str, question: str, filters: dict[str, Any] | None, top_k: int, use_rerank: bool,
                      stream_tokens: bool) -> Iterator[tuple[str, dict[str, Any]]]:
        from vre.qa import stream_answer

        return iter(stream_answer(system, question, filters=filters, top_k=top_k, use_rerank=use_rerank,
                                  stream_tokens=stream_tokens))

    def valuate(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            from vre.valuation.predict import valuate
        except ImportError as error:
            raise ApiError(ErrorCode.SYSTEM_UNAVAILABLE, f"Module định giá không nạp được ({type(error).__name__})") from error
        try:
            result: dict[str, Any] = valuate(payload)
        except ValueError as error:
            raise ApiError(ErrorCode.INVALID_REQUEST, str(error)) from error
        except (RuntimeError, FileNotFoundError) as error:  # predict.ModelUnavailableError kế thừa FileNotFoundError
            raise ApiError(ErrorCode.SYSTEM_UNAVAILABLE, str(error)) from error
        return result

    # ---- meta
    def health(self) -> dict[str, Any]:
        import httpx

        from api import API_VERSION
        from vre.config import load_settings

        vs = load_settings()
        try:
            qdrant = httpx.get(f"{vs.qdrant_url.rstrip('/')}/readyz", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            qdrant = False
        try:
            from api.services.neo import cypher

            neo4j = bool(cypher("RETURN 1 AS ok"))
        except ApiError:
            neo4j = False
        key = {"groq": vs.groq_api_key, "openai": vs.openai_api_key, "anthropic": vs.anthropic_api_key}.get(vs.llm_provider, "")
        return {"status": "ok" if qdrant and neo4j else "degraded", "qdrant": qdrant, "neo4j": neo4j,
                "llm_configured": bool(key), "version": API_VERSION}

    def info(self) -> dict[str, Any]:
        from api.services import stats
        from api.services.neo import cypher
        from vre.config import load_settings

        vs = load_settings()
        manifest = _read_json(vs.index_dir / f"manifest_{vs.collection}.json")
        if manifest is None:
            raise ApiError(ErrorCode.SYSTEM_UNAVAILABLE, f"Chưa có manifest chỉ mục {vs.collection}")
        status = self.system_status()
        try:
            graph_ok = int(cypher("MATCH (l:Listing) RETURN count(l) AS n")[0]["n"]) == int(manifest["n_listings"])
        except ApiError:
            graph_ok = False
        return {
            "systems": [{"id": s.value, "name": SYSTEM_NAMES[s.value], "available": status[s.value][0],
                         "unavailable_reason": status[s.value][1]} for s in SystemId],
            "index": {"collection": manifest["collection"], "model_slug": manifest["model_slug"],
                      "n_listings": int(manifest["n_listings"]), "n_points": int(manifest["n_points"])},
            "graph_ok": graph_ok,
            "data_period": stats.data_period(self.settings),
            "limits": {"max_question_chars": 0, "rate_per_min": 0},  # router điền từ ApiSettings
        }

    # ---- dữ liệu
    def filter_options(self) -> dict[str, Any]:
        from api.services import stats

        return stats.filter_options(self.settings)

    def listing(self, listing_id: int) -> dict[str, Any] | None:
        from api.services import listings

        return listings.get_detail(self.settings, listing_id)

    def stats_overview(self) -> dict[str, Any]:
        from api.services import stats

        return stats.overview(self.settings)

    def market(self, province: str, district: str | None, property_type: str | None, deal_type: str) -> dict[str, Any] | None:
        from api.services import stats

        return stats.market(province, district, property_type, deal_type)

    def geo_areas(self, level: str, province: str | None) -> dict[str, Any]:
        from api.services import geo

        return geo.areas(level, province)

    def subgraph(self, label: str, key: str | int, limit: int) -> dict[str, Any] | None:
        from api.services import graph_view

        return graph_view.subgraph(label, key, limit)

    def benchmark(self) -> dict[str, Any] | None:
        from vre.config import load_settings

        results = load_settings().results_dir
        summary = _read_json(results / "eval_summary.json")
        if summary is None:
            return None
        return {**summary, "ablation": _read_json(results / "ablation.json"),
                "embedding_benchmark": _read_json(results / "embedding_benchmark.json")}


def make_backend(settings: ApiSettings) -> Backend:
    if settings.fake_backends:
        from api.services.fake import FakeBackend

        return FakeBackend()
    return RealBackend(settings)
