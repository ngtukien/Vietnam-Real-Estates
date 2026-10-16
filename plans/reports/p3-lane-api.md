# Phase 3, Lane B (API): báo cáo

**Status:** DONE_WITH_CONCERNS (concerns are listed under "Cần lead quyết"; none of them block Lane C)

Branch `feat/p3-api` is pushed. HEAD is `2c88ce1`, the same as `origin/feat/p3-api`. Working tree is clean. No PR was opened: under phase-03-lanes §1/§5 the lead merges at integration.

| Commit | Contents |
|---|---|
| `8adfd1e` | `api/schemas.py` + `api/openapi.json` (pushed first so Lane C could start) |
| `b13cd35` | app, middleware, SSE, routers, services, fake mode + fixtures, `tests/api/**` |
| `66b337e` | `Dockerfile.api`, `Dockerfile.api.dockerignore`, `docker-compose.yaml` profile `app`, `requirements/base.txt` |
| `2c88ce1` | `api/__init__.py` puts the repo's own `src/` on `sys.path`, so `python -m api.export_openapi` and `uvicorn api.main:app` run without PYTHONPATH |

## Đã làm

- **Schemas (§1, §2, §3, §4, §8)**
  - All 55 schema names are in `components.schemas`, including the SSE payload models and the enums not attached to any route.
  - `separate_input_output_schemas=False`, so no `-Input`/`-Output` names appear.
  - The 422 response and `HTTPValidationError` are removed from the docs, because 422 is mapped to 400.
  - Response fields typed `T | null` are always `required`.
  - Request models use `extra="forbid"`.
- **Middleware (§0, B8)**
  - Order: request-id, then CORS, then body-size (8 KiB, both declared Content-Length and streamed bodies), then rate limit by IP, then auth (`X-API-Key` header only), then media type (415), then the handler.
  - `X-Request-Id` (8 hex) and security headers are on every response, including 500s.
  - The 500 envelope hides details. Logs never contain the question text, only its length.
  - The limiter sweeps idle keys every 60s.
  - Without `VRE_API_TOKEN`, only loopback clients are accepted.
- **SSE (`api/sse.py`)**
  - The sync facade runs in a thread. Every payload gets `request_id`/`system` added and is validated against its `Event*` model.
  - If the facade fails to emit `meta`, the API synthesizes it. If it fails to emit `done`, the API adds `error` then `done`. A payload that breaks the contract becomes `internal_error`.
  - `VRE_CHAT_TIMEOUT_S` produces `error{timeout}` then `done`.
  - The question slot is released only after the facade thread has actually stopped. This holds on timeout and on client disconnect too.
  - Heartbeat `: ping` every 15s, `sep="\n"`.
  - `/api/compare` runs the systems in parallel, interleaves their events, and ends with `compare_done`.
- **Facade calls.** All lazy imports, using the exact §7 signatures:
  - `vre.qa.stream_answer` and `vre.qa.system_status` (cached for 30s).
  - `vre.valuation.predict.valuate`: `ValueError` → 400; `RuntimeError` or `FileNotFoundError` → 503.
  - `vre.qa` is not on this branch, so in real mode `/api/info` reports every system as unavailable with a reason, and `/api/chat` returns 503 `system_unavailable`.
- **Real-mode services**
  - Listing detail, overview and filter options read DuckDB (`v_listings`, 3 GB / 4 threads, a cursor per query).
  - The overview counts `n_raw_rows` and `n_duplicates_removed` from `ingest_report.json`.
  - Market stats come from MarketStat. Geo uses Province/District/Ward nodes plus `data/geo/centroids.json`; if that file is missing, lat/lon are null and coverage is 0.
  - The subgraph includes 1-hop neighbours, plus the Ward→District→Province chain and Cluster for a listing.
  - `/api/benchmark` merges `eval_summary.json`, `ablation.json` and `embedding_benchmark.json`.
- **Fake mode (§6)**
  - 19 fixtures, built from real data: 4 real listings in Quận 7, the overview and filters from DuckDB, a real `valuate()` result, and MarketStat/Province/Ward numbers copied from Neo4j. `benchmark.json` is fake and says so in `_fixture`.
  - Hooks for Lane C E2E:
    - A question containing `[fake:error]` streams meta → error → done.
    - With `system=hybrid`, a question containing `định giá` takes the valuation route.
    - `VRE_FAKE_DELAY_MS` adds a delay between events.
    - Every card in the sources opens via `/api/listings/{id}`.
- **Tests.** 9 files under `tests/api`:
  - Required list: `test_contract.py`, `test_fixtures_valid.py`, `test_guards.py`, `test_sse_order.py`, `test_compare.py`, `test_valuate.py`, `test_limiter_cleanup.py`.
  - Extra: `test_rest_fake.py`.
  - Helpers: `api_helpers.py`.
  - `test_contract` parses `api-contract.md` directly (enums §1, TS fields §2/§3, the SSE table §4, names §8) and checks that the committed `openapi.json` is current.
- **Docker**
  - `Dockerfile.api`: python:3.13-slim, CPU torch, base + transformers 5.14 / tokenizers 0.22.2 / xgboost, non-root user, healthcheck.
  - `Dockerfile.api.dockerignore` limits the build context to code only.
  - Compose profile `app`:
    - `api`: mounts `.cache/huggingface`, results, models, data/clean, data/geo and index, all read-only. Depends on qdrant/neo4j `service_healthy`. Healthcheck with `start_period: 120s`. Bound to `127.0.0.1:8000`.
    - `web`: built from `Dockerfile.web`, bound to `127.0.0.1:8080`.
  - Not built in this run, per your instruction. `docker compose --profile app config --quiet` returns rc=0.

## Lệnh verify + kết quả thật

All commands were run from `D:\Vietnam-Real-Estates-wt\api`.

| Command | Result |
|---|---|
| `python -m pytest -q tests/api` | `102 passed in 3.35s` |
| `python -m pytest -q tests/unit` | `180 passed in 39.67s` |
| `python -m ruff check api tests` | `All checks passed!` |
| `python -m mypy api` | 0 errors in `api/`. It prints `Found 13 errors in 5 files`, all in `src/vre/{config,common,valuation/features,valuation/search_cache,valuation/train}.py`. I did not touch those files; mypy follows imports into them. |
| `python -m api.export_openapi --check` | rc=0 |
| Fake curl (uvicorn `--timeout-keep-alive 5`, `VRE_FAKE_BACKENDS=1`, hybrid question) | 200 `text/event-stream`, `cache-control: no-store`, `x-accel-buffering: no`, `x-request-id`. Events in order: meta → route → filters → cypher → sources → token×6 → answer → usage → done. The server was stopped afterwards. |

Real-mode sanity was done in-process, without fake mode:

| Request | Result |
|---|---|
| `/api/health` | 200, qdrant and neo4j both true |
| `market?province=Hồ Chí Minh&district=Quận 7` | 200, 381 ms |
| `geo/areas?level=district` | 200, 78 ms. Lane C's centroids are already in use. |
| `subgraph?listing_id=531409273736` | 200, 358 ms |
| `subgraph` by Province, limit 5 | 200, 140 ms |
| `subgraph?listing_id=7` | 404 |
| `/api/benchmark` | 404: eval has not run yet |
| `/api/chat` | 503: `vre.qa` is missing |

The DuckDB endpoints (listing detail, overview, filter options) were exercised by building the fixtures from real data. I did not rerun them through the server, because free commit was only 4.9 GB.

## Chưa làm

- **The Docker image was not built**, per your instruction (free commit 4.9 GB).
- **`/api/info` has not been run end-to-end in real mode.** It reads `data_period` from DuckDB, and I skipped it because commit memory was low.
- **The lifespan only checks the manifest and does not load the encoder.** §7 has no warm-up function, so the facade loads the encoder on the first question.
- **The `vre.api` INFO logs (request, length, ms) do not show under uvicorn** because no logging config is installed. WARNING and above do show.

## Cần lead quyết / ghi chú cho lane khác

1. **Rate limit uses two buckets.**
   - `/api/chat` and `/api/compare` use `VRE_CHAT_RATE_PER_MIN` (20).
   - The other `/api/*` endpoints (not health) use a new key `VRE_API_RATE_PER_MIN` (300). The web page loads many GETs at once, so sharing 20/minute would make it block itself.
   - A compare request counts as one hit and holds `min(len(systems), VRE_CHAT_MAX_CONCURRENT)` slots.
2. **Compare when some systems are down.** If only some requested systems are unavailable, the stream still opens and each unavailable system gets meta → `error{system_unavailable}` → done. The request returns 503 only if none of the systems can run.
3. **New key `VRE_API_ALLOW_REMOTE_WITHOUT_TOKEN`.** Compose sets it to `1` because a client inside the Docker network is never loopback, and the port is published only on 127.0.0.1.
   - `Dockerfile.api` runs uvicorn with `--proxy-headers --forwarded-allow-ips *`, so the rate limit sees the real IP behind nginx.
   - **Lane C:** nginx must listen on 8080 in the container, proxy `/api/` to `http://api:8000` with `proxy_buffering off`, and set `X-Forwarded-For`.
4. **CORS in compose is `http://127.0.0.1:8080,http://localhost:8080`.** Outside Docker the default is `:5173`, as in the contract.
5. **For A1:**
   - `predict.valuate()` currently defaults to `variant="tabular"`. §7 says to prefer `full`. The API only calls `valuate(payload)`, so this has to change in predict.
   - `ModelUnavailableError` subclasses `FileNotFoundError`, not `RuntimeError`. The API catches both.
   - `stream_answer` must emit `meta` itself, and every payload must match the `Event*` models exactly. If it doesn't, the API turns it into `internal_error` and logs `payload sai hợp đồng`.
6. **The `/docs` page is off and the API's CSP is `default-src 'none'`.** The schema is at `/api/openapi.json`, which also needs auth when a token is set.

**Status:** DONE_WITH_CONCERNS. **Summary:** the API is done and green, and Lane C can use the fake mode and `openapi.json` now. **Concerns:** the Docker image is not built, and A1 still has to fix predict to prefer `full`.
