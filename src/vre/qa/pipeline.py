"""Bốn hệ thống QA (llm_only, basic_rag, graph, hybrid) dưới dạng chuỗi sự kiện api-contract §4.

Mỗi hệ thống chuẩn bị ngữ cảnh (`Prepared`: sự kiện trước câu trả lời theo đúng thứ tự, prompt, nguồn), rồi
`generate_events` stream câu trả lời. Trạng thái của một lượt nằm trong `Run` (không có biến toàn cục ⇒ an toàn đa luồng).
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from vre.graph.text2cypher import GRAPH_ANSWER_SYSTEM, cypher_result, graph_context, row_listing_ids, text2cypher
from vre.hybrid import valuation as hv
from vre.hybrid.router import GRAPH_CONTEXT_SYSTEM, router, safe_text2cypher
from vre.llm.client import LLMResult, stream_llm
from vre.qa.cards import cards, order_cards
from vre.rag import qdrant_search
from vre.rag.citations import check_citations, normalize_citations
from vre.rag.filters import FILTER_DROPPED, QueryFilters
from vre.rag.generate import ANSWER_SYSTEM, LLM_ONLY_SYSTEM, build_prompt, format_context, strip_context_echo, wrap_documents
from vre.rag.retrieve import retrieve, search_hybrid

Event = tuple[str, dict[str, Any]]
log = logging.getLogger(__name__)


@dataclass
class Run:
    system: str
    question: str
    manual: dict[str, Any] | None = None
    top_k: int = 5
    use_rerank: bool = False
    stream_tokens: bool = True
    calls: list[LLMResult] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)
    route: str | None = None
    cypher: str | None = None
    error: str | None = None
    fallback: bool = False
    answer: str = ""
    cited: list[int] = field(default_factory=list)
    unsupported: list[int] = field(default_factory=list)


@dataclass
class Prepared:
    events: list[Event]
    prompt: str
    system_prompt: str
    cards: list[dict[str, Any]]


def _in_thread(pool: ThreadPoolExecutor, fn: Any, *args: Any) -> Any:
    """Chạy trong luồng phụ với BẢN SAO ngữ cảnh (giữ `llm.cache.fresh()` của luồng gọi — B1)."""
    return pool.submit(contextvars.copy_context().run, fn, *args)


def _filters_event(found: dict[str, Any]) -> Event:
    f: QueryFilters = found["filters"] or QueryFilters()
    return "filters", {"filters": f.model_dump(), "source": found["source"], "note": found["note"]}


def _rag_part(run: Run, found: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    hits = found["hits"]
    run.calls += found["calls"]
    run.sources += [int(x) for x in hits["listing_id"]] if not hits.empty else []
    return format_context(hits), cards(hits)


def _graph_cards(ids: list[int], limit: int) -> list[dict[str, Any]]:
    try:
        return cards(qdrant_search.cards_by_ids(ids[:limit]))
    except Exception as error:  # thẻ tin là phụ: câu trả lời đồ thị vẫn đúng khi Qdrant không trả lời
        log.warning("không lấy được thẻ tin từ Qdrant: %s", type(error).__name__)
        return []


def prepare_llm_only(run: Run) -> Prepared:
    return Prepared([], f"CÂU HỎI: {run.question}", LLM_ONLY_SYSTEM, [])


def prepare_basic_rag(run: Run) -> Prepared:
    found = retrieve(run.question, run.top_k, True, run.use_rerank, run.manual)
    contexts, rag_cards = _rag_part(run, found)
    run.contexts += contexts
    return Prepared([_filters_event(found)], build_prompt(run.question, contexts, found["note"]), ANSWER_SYSTEM, rag_cards)


def _apply_graph(run: Run, t2c: dict[str, Any]) -> tuple[Event, str]:
    run.calls += t2c["calls"]
    run.cypher, run.error = t2c["cypher"] or None, t2c["error"]
    ids = row_listing_ids(t2c["rows"])
    run.sources += ids
    context = graph_context(t2c)
    run.contexts.append(context)
    return ("cypher", cypher_result(t2c)), context


def prepare_graph(run: Run) -> Prepared:
    t2c = text2cypher(run.question)
    event, context = _apply_graph(run, t2c)
    prompt = f"{wrap_documents([context])}\n\nCÂU HỎI: {run.question}"
    return Prepared([event], prompt, GRAPH_ANSWER_SYSTEM, _graph_cards(run.sources, run.top_k * 2))


def _prepare_valuation(run: Run, found: dict[str, Any]) -> Prepared | None:
    """None ⇒ không định giá được (thiếu thông tin / chưa có model): người gọi lui về Basic RAG."""
    from vre.valuation.predict import valuate

    payload, call = hv.extract_input(run.question)
    run.calls.append(call)
    try:
        result = valuate(payload)
    except (ValueError, RuntimeError):
        return None
    comp_filters = hv.comparable_filters(payload)
    hits, note = search_hybrid(run.question, run.top_k, comp_filters), None
    if hits.empty:
        hits, note = search_hybrid(run.question, run.top_k), FILTER_DROPPED
    run.calls += found["calls"]  # truy xuất song song đã tốn token dù bị bỏ
    run.sources += [int(x) for x in hits["listing_id"]] if not hits.empty else []
    contexts = [hv.valuation_context(payload, result), *format_context(hits)]
    run.contexts += contexts
    events: list[Event] = [("filters", {"filters": comp_filters.model_dump(), "source": "llm", "note": note}),
                           ("valuation", result)]
    return Prepared(events, build_prompt(run.question, contexts, note), ANSWER_SYSTEM + hv.VALUATION_SYSTEM_NOTE,
                    cards(hits))


def prepare_hybrid(run: Run) -> Prepared:
    """router ∥ truy xuất RAG; graph/hybrid ⇒ Text2Cypher, lỗi/rỗng ⇒ lui về Basic RAG; valuation ⇒ định giá."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        rag_job = _in_thread(pool, retrieve, run.question, run.top_k, True, run.use_rerank, run.manual)
        routed = router(run.question)
        run.calls.append(routed["call"])
        run.route = routed["route"]
        events: list[Event] = [("route", {"route": run.route, "reason": routed["reason"]})]
        t2c = safe_text2cypher(run.question) if run.route in ("graph", "hybrid") else None
        found = rag_job.result()
    if run.route == "valuation":
        prepared = _prepare_valuation(run, found)
        if prepared is not None:
            prepared.events[:0] = events
            return prepared
        run.fallback = True
    graph_ok = t2c is not None and not t2c["error"] and not t2c["rows"].empty
    run.fallback = run.fallback or (run.route in ("graph", "hybrid") and not graph_ok)
    use_rag = run.route != "graph" or not graph_ok
    if use_rag:
        events.append(_filters_event(found))
    else:
        run.calls += found["calls"]
    contexts: list[str] = []
    all_cards: list[dict[str, Any]] = []
    if t2c is not None:
        event, context = _apply_graph(run, t2c)
        events.append(event)
        if graph_ok:
            contexts.append(context)
            all_cards += _graph_cards(row_listing_ids(t2c["rows"]), run.top_k * 2)
        else:
            run.contexts.remove(context)
    note = None
    if use_rag:
        rag_contexts, rag_cards = _rag_part(run, found)
        contexts += rag_contexts
        run.contexts += rag_contexts
        all_cards += rag_cards
        note = found["note"]
    run.sources = list(dict.fromkeys(run.sources))
    system_prompt = GRAPH_CONTEXT_SYSTEM if graph_ok else ANSWER_SYSTEM
    return Prepared(events, build_prompt(run.question, contexts, note), system_prompt, all_cards)


PREPARE = {"llm_only": prepare_llm_only, "basic_rag": prepare_basic_rag, "graph": prepare_graph,
           "hybrid": prepare_hybrid}


def generate_events(run: Run, prepared: Prepared) -> Iterator[Event]:
    """Sự kiện chuẩn bị → `sources` → `token`* → `answer` (văn bản đã chuẩn hoá [Tin#id])."""
    yield from prepared.events
    limit = run.top_k * 2
    source_cards = order_cards(prepared.cards, limit)
    yield "sources", {"sources": source_cards}
    stream = stream_llm(prepared.prompt, system=prepared.system_prompt)
    for piece in stream:
        if run.stream_tokens:
            yield "token", {"text": piece}
    if stream.result is None:
        raise RuntimeError("LLM stream kết thúc không có kết quả")
    run.calls.append(stream.result)
    run.answer = normalize_citations(strip_context_echo(stream.result.text))
    run.cited, run.unsupported = check_citations(run.answer, run.sources)
    yield "answer", {"answer": run.answer, "cited": run.cited, "unsupported_citations": run.unsupported,
                     "fallback": run.fallback, "route": run.route}
