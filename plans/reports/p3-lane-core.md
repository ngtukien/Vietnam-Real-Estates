# Phase 3 — Lane A1 (core QA on the new index) — report

**Status:** DONE_WITH_CONCERNS · branch `feat/p3-core` (worktree `D:\Vietnam-Real-Estates-wt\core`), pushed, working tree clean.
`team-tools-unavailable: applied-from-prompt`.

Commits (on top of `519253c`):
- `76a4d86` wip(rag): QueryFilters per contract (one condition list → Qdrant/SQL/pandas), alias search, 16-digit citations, KG schema few-shots, stream_llm, context-local fresh()
- `b6362e6` feat(qa): vre.qa facade (SSE event order, answers.jsonl keys, system_status), KG text2cypher + valuation route, valuate contract, CLI ask/eval gt, bug tests B1/B3/B4/B5/B7
- `f176a2a` perf(rag): CPU fp16 query encoder with sliced safetensors load (ask peak commit 5.9 → 2.96 GB), [Cypher] citations for graph context, Cypher latency log, event validation against api/schemas.py

No PR opened (the lead integrates lanes).

## A1 checklist

| # | Item | Status | Where |
|---|---|---|---|
| 1 | `QueryFilters` per api-contract §2 (ward, deal_type, project, street, month_from/to, `min/max_area_m2`). One condition list compiled to Qdrant `Filter` (months as `MatchAny` on `published_month`), SQL WHERE and a pandas mask. LLM extraction keeps the money strings. Manual filters override per field (`source` llm/manual/merged). | ✓ | `src/vre/rag/filters.py` |
| 2 | Retrieval reads alias `vre_chunks`. e5-large query: prefix `query: `, mean pooling, L2. `threshold` removed (B5). | ✓ | `rag/retrieve.py`, `rag/qdrant_search.py` (new), `rag/query_encoder.py` (new) |
| 3 | Citations regex `\[Tin#(\d{1,16})\]`. 【】［］ are normalized. Sources are checked by `listing_id`. | ✓ | `rag/citations.py` |
| 4 | `graph/schema.py`: schema prompt plus 10 few-shots (MarketStat aggregation, Project/Street, Month trend, Amenity, Cluster "similar areas", multi-hop project → district → province, `WHERE m.n >= 20`). `text2cypher` uses it; B7 fixed. | ✓ | `graph/schema.py` (new), `graph/text2cypher.py` |
| 5 | `Route` gains `valuation`. LLM extracts `ValuationInput` → `valuate()` → comparables (same district/type, area ±20%) → answer with forecast, interval and `[Tin#id]`. If extraction or the model fails, it falls back to RAG with `fallback=true`. | ✓ | `hybrid/router.py`, `hybrid/valuation.py` (new), `qa/pipeline.py` |
| 6 | `stream_llm()` (OpenAI SDK `stream=True`, `include_usage`, Groq `x_groq` fallback). A cached answer comes back as one chunk. B1 fixed with a `ContextVar`. | ✓ | `llm/client.py`, `llm/cache.py` |
| 7 | `vre.qa` facade: `stream_answer`, `answer`, `system_status`, with events in the §4 order. | ✓ | `qa/__init__.py`, `qa/pipeline.py`, `qa/cards.py` (all new) |
| — | CLI `ask --system <id> QUESTION` and `eval gt` (lazy `vre.eval.ground_truth.run`) | ✓ | `scripts/vre_pipeline.py` |
| — | `valuate(payload)` per §7: prefers `full`, falls back to `tabular`; ValueError → 400, RuntimeError (`ModelUnavailableError` now subclasses RuntimeError) → 503 | ✓ (see deviation D3) | `valuation/predict.py` |

## Bug proofs

| Bug | Test |
|---|---|
| B1 | `tests/unit/test_llm_cache.py::test_fresh_is_context_local` (2 threads) + `test_fresh_restores_after_exception`; `qa.pipeline._in_thread` copies the context into worker threads |
| B3 | `tests/unit/test_graph_schema.py::test_no_duplicate_scalar_helpers` (no `_value`/`_clean`/`scalar` defined in rag/graph/hybrid/qa/llm; all of them use `vre.common.scalar`). One more `_clean` remains in `src/vre/data/eda.py`, which this lane does not own. |
| B4 | `tests/unit/test_filters_equivalence.py` (200 fake rows × 11 filter cases; the Qdrant side runs `QdrantClient(":memory:")`, the SQL side DuckDB, plus the pandas mask: all three return the same set) |
| B5 | `test_graph_schema.py::test_no_cosine_threshold_in_rag` (grep `threshold` in `vre/rag` = 0) |
| B7 | `test_graph_schema.py::test_cypher_system_logs_fallback` (the warning includes the exception type; the failure is not cached) |

## Verify (real outputs)

- `pytest -q tests/unit` → **224 passed in 40.32s** (baseline 180)
- `ruff check src tests scripts` → **All checks passed!**
- `mypy src/vre` → **47 errors in 11 files** (main baseline: 68 in 19). **0 errors in my modules**: rag, graph, hybrid, llm, qa, common, config, valuation/predict. Before this run, config, common, llm/client, rag/filters, rag/retrieve, rag/generate, graph/text2cypher and hybrid/router all had errors.
- Contract check: `tests/unit/test_qa.py::test_events_validate_against_api_schemas` validates every event emitted by all 4 systems (graph, basic_rag, valuation and fallback routes) against Lane B's `EVENT_MODELS` in `api/schemas.py`. It loads that file from the sibling worktree, or from `api/` after merge, and skips if neither exists. **Passes.**

Manual runs (one at a time, free commit ≥ 6 GB checked first; 11 real Groq calls in total):

1. `ask --system hybrid "Quận nào ở Hà Nội có giá/m² căn hộ trung vị cao nhất?"`
   - route `graph`, `fallback: false`, `[vre.graph.text2cypher] Cypher chạy 30 ms, 1 dòng (lần 1)`; on the cached rerun: 37 ms.
   - Cypher: `MATCH (m:MarketStat {scope: 'district', property_type: 'Căn hộ chung cư', month: 'ALL', deal_type: 'sale'})-[:STAT_OF]->(d:District)-[:IN_PROVINCE]->(:Province {key: 'Hà Nội'}) WHERE m.n >= 20 RETURN d.name AS district, round(m.median_price_m2_mil, 2) AS median_price_m2_mil, m.n AS n ORDER BY median_price_m2_mil DESC LIMIT 1` → rows `[{"district": "Tây Hồ", "median_price_m2_mil": 107.14, "n": 6379}]`.
   - Answer: "Quận Tây Hồ … khoảng **107,14 triệu đồng/m²** (số mẫu ≥ 20) [Cypher]." `unsupported_citations: []`.
   - The first run cited a made-up `[Tin#1]`, because the answer prompt forced `[Tin#ID]` on aggregate numbers. Fixed by giving `GRAPH_CONTEXT_SYSTEM` a `[Cypher]` source rule.
   - Peak commit **2.96 GB**.
2. `ask --system basic_rag "Căn hộ 2pn q7 dưới 4 tỷ có hồ bơi"`
   - filters `{district: "Quận 7", bedrooms: 2, max_price_mil: 4000.0}`, source `llm`, note null.
   - 5 sources, all Quận 7 with 2 bedrooms. The answer cites `[Tin#3134563763903940]`, `[Tin#653246868779482]`, `[Tin#2106690535929527]` (full 16-digit ids), `unsupported_citations: []`.
   - Peak commit **2.94 GB**.
3. `ask --system hybrid "Căn hộ 70m2 2 phòng ngủ ở Quận 7, Hồ Chí Minh chào 4,5 tỷ có đắt không?"`
   - route `valuation`; filters area 56–84 m².
   - valuation event: `variant: full`, 23.41 tr/m², 1.64 tỷ, interval 1.43–1.94 tỷ, verdict `above_market`; the answer cites `[Định giá]` and `[Tin#5180195771937688]`.
   - Peak commit **6.51 GB**; see concern C2.
- `eval gt` → `ModuleNotFoundError: No module named 'vre.eval.ground_truth'`. Expected: the command is wired lazily, and Lane A2's module is not on this branch.

## Memory (lead request)

Measured with psutil `peak_pagefile` (peak private commit):

| What | Peak private commit |
|---|---|
| Old path (GPU `vre.index.embed.Encoder`) | 5.9–6.5 GB |
| CUDA context alone | 1.85 GB |
| A 1 GB CUDA tensor | +1.0 GB |
| `from_pretrained` fp32 staging + safetensors copy-on-write mmap | +3.5 GB |
| `import numpy` with 32-thread OpenBLAS | 0.79 GB |

On Windows (WDDM), CUDA allocations count toward private commit, so the GPU query encoder could not get under 3 GB.

Fix: `vre/rag/query_encoder.py` runs on CPU in fp16. It builds the model on `meta`, then reads the safetensors file with `readinto` in ≤ 64 MB slices and casts each slice to fp16. Results:
- Parity with `vre.index.embed` on GPU: cosine 1.00001 / 0.99999 / 1.0.
- Query embedding takes ~0.2 s; the model load is ~10 s, once per process.
- The CLI sets `OPENBLAS_NUM_THREADS=4`, saving 0.67 GB.

After the fix: basic_rag `ask` 2.94 GB, hybrid graph `ask` 2.96 GB (target ≤ 3 GB met). The reranker only loads when `use_rerank=True`, and the QA path never reads `listings.parquet` (cards come from the Qdrant payload, or from Qdrant `retrieve` on `listing_id*64` for Cypher rows).

## Deviations from api-contract §4/§7 and decisions taken

- **D1 `sources` order:** sources are sent before the answer, so the "cited first" order cannot be known yet. Cards come in score order with `cited=false`; the `answer` event carries `cited`.
- **D2 `route` in `answer` / answers.jsonl:** `null` for llm_only, basic_rag and graph; only `hybrid` sets it.
- **D3 `valuate` legacy form:** `valuate(payload)` prefers full (the contract form). The legacy call `valuate(input, settings)` keeps its old `tabular` default, because `tests/unit/test_valuation_features.py` is outside my ownership and asserts `tabular`. Recommend the lead update that test and drop the legacy branch.
- **D4 hybrid `filters` event:** emitted only when RAG context is used (route hybrid/basic_rag, or graph fallback). On a valuation route it carries the comparable filters with source `llm`. Pre-answer events are computed first and then emitted, so the §4 order holds even when a graph route falls back.
- **D5 `system_status`:** graph needs Neo4j `count(Listing)` = manifest `n_listings`. Hybrid needs only LLM + Qdrant, since it falls back to RAG. Results are cached 30 s.
- **D6 error codes after `meta`:** openai/anthropic → `upstream_llm_error`; neo4j, qdrant, grpc, httpx and ConnectionError → `system_unavailable`; TimeoutError → `timeout`; anything else → `internal_error` with a generic message. Logs carry the type only, never the question text.
- **D7 Claude provider:** `stream_llm` returns a single chunk (no real streaming yet).
- **D8 month filter:** Qdrant can only match keywords, so a month range becomes the list of months inside `DATA_PERIOD = ("2025-06","2026-03")`. A range entirely outside that period matches nothing in all three compilers.
- `ask_rag`, `ask_graph` and `ask_hybrid` were removed because `vre.qa` replaces them (reference check: no callers left). `ask_llm_only` stays, since a test uses it.

## Concerns (lead decision needed)

- **C1 — `full` valuation is unreliable for hypothetical inputs.** A hypothetical property has no listing embedding, so the PCA features are NaN. For a 70 m² Quận 7 apartment: `full` gives 23.4 tr/m², `tabular` 53.0, baseline ward median 96.1. The contract's "prefer full" therefore gives a much lower price, and the demo answer called a 4.5 tỷ listing "đắt" off that number. Recommend changing api-contract §7 to prefer `tabular` when the input has no text/embedding, or giving `full` an embedding of a synthesized description. I kept the contract as written.
- **C2 — valuation route peak is 6.51 GB.** Parsing the 264 MB `xgb_full_sale.json` peaks at about +2 GB once per process (tabular, 105 MB: about +1.3 GB). Fix outside my ownership: `valuation/train.py` should also save `.ubj`, which `search_cache` already uses. Until then, a cold valuation ask needs ≥ 7 GB free commit.
- **C3 —** the API process (Lane B) should set `OPENBLAS_NUM_THREADS=4` before importing numpy, or the lead should add it to `vre_env.ps1` / Dockerfile; that is 0.67 GB of commit.
- **C4 —** `query_encoder.py` holds its own low-memory loader and pooling (~25 lines) next to `vre.index.embed.Encoder`, which I don't own. Its docstring says why. Later, the loader could move into `vre.index.embed` as `load_fp16_low_memory` so both encoders share it.
- **C5 —** the scratchpad was shared with another lane: its `run.ps1` overwrote mine. I switched to `scratchpad/core/`.

## Not done

- Nothing from A1 or B1/B3/B4/B5/B7 is missing within my ownership.
- The extra `_clean` in `data/eda.py` (B3 scope, not owned) is still there.
- A2–A5 are not part of this run.
