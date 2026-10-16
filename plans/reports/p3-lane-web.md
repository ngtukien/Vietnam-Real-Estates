# Phase 3 — Lane C (Web) report

**Status:** DONE_WITH_CONCERNS
**Branch:** `feat/p3-web` (worktree `D:\Vietnam-Real-Estates-wt\web`). Pushed, HEAD `e531c16`. No PR opened; the lead integrates.

## What was done

| Item | Files |
|---|---|
| Vite 8 + React 19 + TS strict + Tailwind v4 + shadcn-style UI (radix-ui), TanStack Query, react-router 7, react-markdown/remark-gfm, recharts 3, maplibre-gl 6, react-force-graph-2d | `web/package.json`, `vite.config.ts`, `tsconfig.json`, `eslint.config.js`, `src/components/ui/*` |
| Hand-written SSE parser (fetch + ReadableStream, CRLF/split-chunk/UTF-8 safe, rejects unknown events or bad JSON) | `src/api/sse.ts` |
| API client: `X-API-Key` from sessionStorage `vre_api_key`, never read from the URL; parses the error envelope (code, request_id, Retry-After) | `src/api/client.ts` |
| `types.ts` **generated** from Lane B's `api/openapi.json` (openapi-typescript `--default-non-nullable false`, so request fields with defaults stay optional) | `src/api/types.ts`, script `gen:types` |
| Chat reducer that follows the §4 event order: tokens build up and are then replaced by `answer`; `cited` is updated; `error`→`done`; an early stream close becomes `protocol_error`; compare events from different systems can interleave | `src/features/chat/reducer.ts` |
| 6 pages: **Chat** (stream, markdown + tables, `[Tin#id]` link scrolls to and focuses the source card, manual filter panel, Cypher panel with a **visible error**, KG graph per listing, route/fallback badges, usage, `aria-live="polite"`, history `vre_history_v1` capped at 50, `?q=` only prefills the input) · **Compare** (4 columns; a failing column does not break the others; `compare_done`) · **Map** (no external tiles; province/district polygons + circles sized by `n_listings`, colour by median price; coverage; table alternative) · **Valuation** (form, result, SHAP chart, verdict) · **Market** (bar chart by area, monthly line, data tables) · **Benchmark** (9 categories × 4 systems with CI error bars, pairwise, retrieval, cost, ablation, embedding_benchmark) | `src/pages/*` |
| Dark mode (`vre_theme`), settings dialog for the API key, skip link, footer attribution (data CC BY-NC 4.0 + geoBoundaries) | `src/components/Layout.tsx` |
| `Dockerfile.web` (node:24-alpine → nginx:alpine, build context = repo root, port 8080); `web/nginx.conf`: `/api/` → `http://api:8000`, `proxy_buffering off`, `proxy_read_timeout 180s`, `X-Forwarded-For` + `X-Real-IP`, CSP `'self'`, SPA fallback | `Dockerfile.web`, `web/nginx.conf` |
| `scripts/build_geo.py` | see the geo section below |

## Verify (actual output, 2026-10-15)

- `npm ci` → OK (warning: install scripts for `better-sqlite3` and `msgpackr-extract`, pulled in by mapshaper, not approved; not needed)
- `npm run lint` → `eslint . --max-warnings 0`, 0 problems
- `npm run typecheck` → `tsc --noEmit`, 0 errors (types generated from openapi.json)
- `npm run test` → `Test Files 3 passed (3) · Tests 22 passed (22)` (sse.ts parser, money/number formatting, chat reducer + linkCitations)
- `npm run build` → `✓ built`; chunks: maplibre 1.03 MB (280 kB gz), charts 471 kB, index 566 kB, separate maplibre worker
- `npm run e2e` → **`11 passed (19.7s)`**. Ran against **Lane B's real fake API** (`D:\Vietnam-Real-Estates-wt\api`, `uvicorn api.main:app`, `VRE_FAKE_BACKENDS=1`, started and stopped by Playwright `webServer`), plus Vite dev with proxy. 1 worker, chromium only. axe (wcag2a/aa, 21a/aa) gives **0 serious/critical** on chat, compare, valuation, map, market, benchmark, and in dark mode. After the run, nothing was listening on ports 8000 or 5173.
  - The "Cypher error shown" case uses `page.route` to inject SSE, because Lane B's fixtures have no Cypher error. The other cases use the fake API (`[fake:error]` → error, `định giá` → valuation route).
- `ruff check scripts/build_geo.py` → All checks passed

## Geo (`scripts/build_geo.py`)

- Outputs exist: `web/public/geo/districts.json` (1.13 MB, 708 features, `key`/`name`/`province`), `web/public/geo/provinces.json` (392 kB, 63 provinces, **dissolved from the districts**), `D:\Vietnam-Real-Estates\data\geo\centroids.json` (709 items). Raw files are in `data/geo/raw/` (gitignored).
- **coverage: province 1.0, district 0.9969**. The 2 unmatched are `Bình Phước|Phú Riềng` (missing from ADM2 2020) and `Hòa Bình|Văn Giang` (bad source data).
- Method: every URL is HEAD-checked (200) at fixed commit `wmgeolab/geoBoundaries@9469f09`; ADM2 is simplified with `mapshaper -simplify 5% keep-shapes`. Each district's province is found by **name first** (data district names), using ADM1 point-in-polygon only to break ties. Raw ADM1 (2008, Wikipedia) is off by about 0.1°, so pure geometry wrongly put Từ Sơn, Văn Giang and Khoái Châu in Hà Nội. Merged districts are aliased: Q2+Q9→Thủ Đức, Long Điền+Đất Đỏ→Long Đất, Tân Thành→Phú Mỹ, Quảng Uyên+Phục Hòa→Quảng Hòa, Tĩnh Gia→Nghi Sơn.

## Concerns / decisions for the lead

1. **The license is not CC BY 4.0 as the contract says.** The geoBoundaries API reports ADM1 = Public Domain (vintage 2008), ADM2 = OCHA ROAP **CC BY 3.0 IGO** (vintage 2020). `centroids.json.source` and the footer use the real licenses. The lead should fix data-contract §8, phase-03 step C and the README (Lane D).
2. `provinces.json` is dissolved from ADM2 instead of using raw ADM1, which was too coarse to display. This differs from the spec, which reads as if ADM1 would be simplified directly.
3. `npm` uses Node 24 + **TypeScript ~5.9**, because typescript-eslint and openapi-typescript do not yet accept TS 6/7.
4. The maplibre chunk is 1 MB. It is lazy-loaded only on `/map` and acceptable for a demo.
5. The map e2e only proves the canvas renders, there are no worker errors and no external requests. It does not assert pixel colours of the polygons.
6. `types.ts` must be regenerated (`npm --prefix web run gen:types`) whenever Lane B changes `api/openapi.json`. The script path `../api/openapi.json` is correct once merged into main.
7. CI (Lane D): set `VRE_PY` and `VRE_API_DIR` (defaults: `python`, repo root) or start the fake API first. `playwright.config.ts` reuses the server at `:8000/api/health`. Install with `npx playwright install --with-deps chromium`.

## Not done

- Storybook (not in the project).
- `e2e:real` (lead, step E).
- Docker image build (`docker compose --profile app build web`) was not run, to save RAM. It needs a run in step E.
