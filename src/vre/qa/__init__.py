"""Facade QA cho API (api-contract §7): `stream_answer`, `answer`, `system_status`.

API (Lane B) chỉ gọi ba hàm này, không import `vre.rag`/`vre.graph`/`vre.hybrid` trực tiếp.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any, Literal, get_args

from vre.common import utc_now
from vre.config import get_settings
from vre.qa.pipeline import PREPARE, Run, generate_events
from vre.rag.filters import QueryFilters

SystemId = Literal["llm_only", "basic_rag", "graph", "hybrid"]
SYSTEMS: tuple[str, ...] = get_args(SystemId)
MAX_TOP_K = 20
STATUS_TTL_S = 30.0
log = logging.getLogger(__name__)


def _validate(system: str, question: str, filters: dict | None, top_k: int) -> None:
    if system not in SYSTEMS:
        raise ValueError(f"system phải là một trong {SYSTEMS}, nhận {system!r}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question rỗng")
    if not 1 <= int(top_k) <= MAX_TOP_K:
        raise ValueError(f"top_k phải trong 1..{MAX_TOP_K}")
    if filters is not None:
        QueryFilters.model_validate(filters)  # ValidationError là ValueError


def error_payload(error: BaseException) -> dict[str, str]:
    """Lỗi → `{code, message}` (ErrorCode api-contract §1), không stack trace."""
    name = type(error).__name__
    module = type(error).__module__ or ""
    if isinstance(error, TimeoutError):
        return {"code": "timeout", "message": "Hết thời gian xử lý câu hỏi"}
    if module.startswith("openai") or module.startswith("anthropic"):
        return {"code": "upstream_llm_error", "message": f"LLM lỗi ({name})"}
    if module.startswith(("neo4j", "qdrant_client", "grpc", "httpx", "httpcore")) or isinstance(error, ConnectionError):
        return {"code": "system_unavailable", "message": f"Dịch vụ dữ liệu không trả lời ({name})"}
    return {"code": "internal_error", "message": "Lỗi khi xử lý"}


def _usage(run: Run, started: float) -> dict[str, Any]:
    s = get_settings()
    tokens_in = sum(c.input_tokens for c in run.calls)
    tokens_out = sum(c.output_tokens for c in run.calls)
    usd = (tokens_in * (s.llm_price_input or 0.0) + tokens_out * (s.llm_price_output or 0.0)) / 1e6
    return {"latency_ms": round((time.perf_counter() - started) * 1000, 1), "input_tokens": tokens_in,
            "output_tokens": tokens_out, "usd": round(usd, 6), "cached": bool(run.calls) and all(c.cached for c in run.calls)}


def _events(run: Run, started: float) -> Iterator[tuple[str, dict]]:
    yield "meta", {"question": run.question, "started_at": utc_now()}
    try:
        yield from generate_events(run, PREPARE[run.system](run))
        yield "usage", _usage(run, started)
        yield "done", {"status": "ok"}
    except Exception as error:  # sau sự kiện đầu tiên không raise nữa (api-contract §7)
        log.warning("QA %s lỗi: %s", run.system, type(error).__name__, exc_info=True)
        payload = error_payload(error)
        run.error = run.error or f"{payload['code']}: {type(error).__name__}"
        yield "error", payload
        yield "done", {"status": "error"}


def stream_answer(system: SystemId, question: str, filters: dict | None = None, top_k: int = 5,
                  use_rerank: bool = False, stream_tokens: bool = True) -> Iterator[tuple[str, dict]]:
    """(event_name, payload) đúng thứ tự §4, không gồm `request_id`/`system`. Input sai ⇒ ValueError TRƯỚC sự kiện đầu."""
    _validate(system, question, filters, top_k)
    run = Run(system, question.strip(), filters, int(top_k), use_rerank, stream_tokens)
    return _events(run, time.perf_counter())


def answer(system: SystemId, question: str, filters: dict | None = None, top_k: int = 5,
           use_rerank: bool = False) -> dict:
    """Bản không stream cho eval: các khoá của `results/eval/raw/answers.jsonl` (data-contract §7.7) trừ `id`."""
    _validate(system, question, filters, top_k)
    run = Run(system, question.strip(), filters, int(top_k), use_rerank, stream_tokens=False)
    usage: dict[str, Any] = {}
    for name, payload in _events(run, time.perf_counter()):
        if name == "usage":
            usage = payload
    if not usage:
        usage = _usage(run, time.perf_counter())
    return {"system": system, "answer": run.answer, "route": run.route, "sources": list(dict.fromkeys(run.sources)),
            "contexts": run.contexts, "cypher": run.cypher, "error": run.error, "fallback": run.fallback,
            "unsupported_citations": run.unsupported, "latency_ms": usage.get("latency_ms"),
            "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
            "cached": usage.get("cached", False)}


_status_lock = threading.Lock()
_status_cache: tuple[float, dict] | None = None


def _llm_configured() -> tuple[bool, str | None]:
    s = get_settings()
    key = {"groq": s.groq_api_key, "openai": s.openai_api_key, "anthropic": s.anthropic_api_key,
           "gemini": s.gemini_api_key}.get(s.llm_provider)
    return (True, None) if key else (False, f"thiếu khoá API cho LLM_PROVIDER={s.llm_provider}")


def _manifest() -> dict | None:
    import json

    path = get_settings().index_dir / f"manifest_{get_settings().collection}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _qdrant_ok() -> tuple[bool, str | None]:
    from vre.rag import qdrant_search

    try:
        points = qdrant_search.client().get_collection(qdrant_search.collection()).points_count or 0
    except Exception as error:
        return False, f"Qdrant không trả lời ({type(error).__name__})"
    return (True, None) if points > 0 else (False, "collection Qdrant rỗng")


def _neo4j_ok() -> tuple[bool, str | None]:
    from vre.graph.client import run_cypher

    manifest = _manifest()
    try:
        n = int(run_cypher("MATCH (l:Listing) RETURN count(l) AS n", timeout=5)["n"].iloc[0])
    except Exception as error:
        return False, f"Neo4j không trả lời ({type(error).__name__})"
    if manifest is None:
        return False, "thiếu manifest chỉ mục"
    if n != int(manifest["n_listings"]):
        return False, f"Neo4j có {n} Listing, manifest {manifest['n_listings']}"
    return True, None


def system_status() -> dict[str, tuple[bool, str | None]]:
    """(available, unavailable_reason) cho /api/info; cache 30s để /api/info không dò dịch vụ mỗi lần."""
    global _status_cache
    with _status_lock:
        if _status_cache is not None and time.monotonic() - _status_cache[0] < STATUS_TTL_S:
            return dict(_status_cache[1])
        llm, qdrant, neo4j = _llm_configured(), _qdrant_ok(), _neo4j_ok()

        def both(*parts: tuple[bool, str | None]) -> tuple[bool, str | None]:
            bad = [reason for ok, reason in parts if not ok]
            return (not bad, "; ".join(r for r in bad if r) or None)

        status = {"llm_only": llm, "basic_rag": both(llm, qdrant), "graph": both(llm, neo4j),
                  "hybrid": both(llm, qdrant)}  # hybrid lui về Basic RAG khi Neo4j lỗi
        _status_cache = (time.monotonic(), status)
        return dict(status)


__all__ = ["SYSTEMS", "SystemId", "answer", "error_payload", "stream_answer", "system_status"]
