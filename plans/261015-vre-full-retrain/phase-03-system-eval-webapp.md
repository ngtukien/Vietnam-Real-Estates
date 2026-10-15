# Phase 3 — Hệ thống + đánh giá + webapp + tài liệu (ban ngày)

> **Phạm vi khi cook file này: CHỈ phase này.** Dừng sau bước cuối của file (E — Tích hợp). Không làm việc của phase khác, không đọc `plan.md`
> để làm tiếp. Chỉ bắt đầu khi cổng ở mục "Điều kiện tiên quyết" đạt.

Plan tổng: `plans/261015-vre-full-retrain/plan.md` · Thiết kế: `plans/reports/brainstorm-261015-0007-full-retrain-redesign.md` §2.1, §2.3–§2.6, §9
· Hợp đồng (đọc trước, KHÔNG chép lại vào brief): `plans/261015-vre-full-retrain/api-contract.md` (API ↔ web, facade Python §7),
`plans/261015-vre-full-retrain/data-contract.md` (Qdrant §5, Neo4j §6, results §7).

## Mục tiêu

RAG/Text2Cypher/Router chạy trên chỉ mục mới (Qdrant alias `vre_chunks`, KG Neo4j mới) + tool `valuate()`; benchmark ~100 câu, 9 nhóm,
đáp án chuẩn bằng SQL DuckDB trên toàn bộ bảng sạch; eval runner + ablation + error analysis + bootstrap CI → `results/*.json`;
FastAPI (SSE) + React/Vite/TS/Tailwind/shadcn webapp + Dockerfile; 5 notebook mới chỉ ĐỌC `results/`; README; CI
(ruff, mypy, pytest + coverage, Playwright); sửa các bug cũ liệt kê ở report §1.2. RAG/GraphRAG là chủ đề chính; định giá chỉ là module phụ.

## Điều kiện tiên quyết

- Cổng hai phiên (coordination §10): `D:\Vietnam-Real-Estates\results\handoff\` có đủ **`P1_DONE.json`, `INDEX_DONE.json` và `P2_DONE.json`**; `main`
  chứa cả `feat/p1-foundation` và `feat/p2-index`; worktree `D:\Vietnam-Real-Estates-p2` đã gỡ. Thiếu file nào ⇒ dừng, không bắt đầu.
- Phase 1 xong (package `src/vre`, `vre.config`/`vre.checks`, bảng sạch, KG Neo4j, model định giá tabular, `embedding_benchmark.json`).
- Phase 2 xong: `results/night_report.json.status = completed`, alias `vre_chunks` → `vre_chunks_e5_large` green, `test_index_complete.py` xanh,
  `valuation_metrics.variants.full` có số (chạy sau `P1_DONE`).
- Từ phase này chỉ còn một checkout (`D:\Vietnam-Real-Estates`); quyền sửa file dùng chung thuộc lead Lane A.
- `.env` có `GROQ_API_KEY` (đây là phase duy nhất dùng nó); `JUDGE_MODEL ≠ LLM_MODEL`.
- Embedding duy nhất: `intfloat/multilingual-e5-large` (cache sẵn ở `HF_HOME`; không tải model embedding nào khác).
- Mọi lệnh chạy sau `. D:\Vietnam-Real-Estates\scripts\vre_env.ps1` (cache/TMP/npm/Playwright về `D:\Vietnam-Real-Estates\.cache\`).
- `docker compose -f D:\Vietnam-Real-Estates\docker-compose.yaml up -d --wait qdrant neo4j` (project `vre`).

## Prior art tái dùng (phạm vi đã đọc: `src/vre/**` sau Phase 1 = các module di chuyển từ `src/*.py` cũ; `app/` cũ đã xoá ở Phase 1 — xem git history `b4a7663:app/server.py`)

| Nguồn | Dùng cho |
|---|---|
| `vre/rag/retrieve.py` (từ `src/rag.py`): `search_hybrid` 1 lượt `query_batch_points`, `_by_listing`, `rrf`, `rerank`, `retrieve` + bỏ lọc khi 0 kết quả (`FILTER_DROPPED`) | giữ thuật toán, đổi sang alias + payload mới |
| `vre/rag/generate.py`: `ANSWER_SYSTEM`, `LLM_ONLY_SYSTEM`, `wrap_documents` chống prompt injection, `build_prompt`, `ask_llm_only`, `ask_rag` | giữ; thêm stream |
| `vre/rag/citations.py`: `normalize_citations`, `check_citations` | sửa regex cho id 16 chữ số |
| `vre/graph/text2cypher.py` + `guard.py` + `client.py`: chặn regex + giao dịch chỉ đọc + timeout + `ensure_limit` + retry gửi lỗi lại cho LLM | giữ cơ chế; viết lại schema prompt + few-shot cho KG mới |
| `vre/hybrid/router.py`: router Pydantic, RAG chạy song song với router/Text2Cypher, fallback RAG, `GRAPH_CONTEXT_SYSTEM` | thêm route `valuation` |
| `vre/llm/{client,cache}.py`: Groq/OpenAI/Anthropic, structured output đóng schema, fallback model 429/503, cache đĩa ghi nguyên tử | thêm stream, sửa `fresh()` |
| `vre/eval/{metrics,judge,runner}.py` (từ `src/eval.py`): recall/RR/MRR, judge 0–2, REFUSAL_SYSTEM, RAGAS faithfulness/relevancy/context precision, `error_cases` | mở rộng: nDCG, dup_cluster, bootstrap, nguyên nhân mới |
| `b4a7663:app/server.py`: `RateLimiter`, `Busy` + semaphore, `SECURITY_HEADERS`, `listing_card`, không log nội dung câu hỏi, chặn mở ra ngoài loopback khi thiếu token | port sang FastAPI (sửa bug thứ tự auth/limit, dọn limiter) |
| `vre/valuation/predict.py` (Phase 1) | tool `valuate()` + `/api/valuate` |

## Bug cũ phải sửa trong phase này (report §1.2 #4, #7, #8, #9, #11)

| # | Bug | Sửa | Test chứng minh |
|---|---|---|---|
| B1 | `llm.fresh()` đổi biến global, không thread-safe | `contextvars.ContextVar` | `test_llm_cache.py::test_fresh_is_context_local` (2 luồng) |
| B2 | Judge = generator khi provider openai/anthropic (`_DEFAULTS[...][2] = None`) | eval từ chối chạy khi `JUDGE_MODEL == LLM_MODEL` trừ khi `VRE_ALLOW_SAME_JUDGE=1` (ghi vào `eval_summary.llm`) | `test_judge_config.py` |
| B3 | Trùng `_value` (graph) / `_clean` (qdrant_store) | 1 hàm `vre.common.scalar` | grep 0 định nghĩa trùng |
| B4 | Trùng `rag.matches` vs `qdrant_store.to_filter` | 1 danh sách điều kiện trong `vre/rag/filters.py` biên dịch ra Qdrant `Filter`, SQL WHERE (DuckDB) và mask pandas | `test_filters_equivalence.py` (cùng 200 dòng giả: 3 cách cho cùng tập) |
| B5 | Ngưỡng cosine 0,80 vô tác dụng (P2-04) | bỏ tham số `threshold` khỏi `search_dense` | grep `threshold` trong `vre/rag` = 0 |
| B6 | P4-02 ghi đè `questions.csv` | GT chỉ ghi `results/benchmark_gt.json`; xoá `benchmark/questions.csv` | `test_questions_file_untouched` (hash trước/sau `eval gt`) |
| B7 | `cypher_system()` nuốt mọi exception im lặng | log warning kèm loại lỗi, vẫn dùng schema mặc định | `test_cypher_system_logs_fallback` |
| B8 | Web cũ: token trong URL, `?q=` tự submit, limiter sau auth (sai key không bị chặn), dict limiter phình, không timeout socket, ẩn lỗi Cypher, a11y | viết lại theo api-contract §0/§4/§5; limiter dọn khoá rảnh mỗi 60s; uvicorn `timeout_keep_alive=5` + timeout câu hỏi | `tests/api/test_guards.py`, Playwright a11y (axe) |
| B9 | CI tự liệt kê package | CI cài `requirements/dev.txt` | workflow diff |
| B10 | Test phụ thuộc `.env` local | `VRE_LOAD_DOTENV=0` trong conftest (Phase 1) + CI không có `.env` | CI xanh không có `.env` |
| B11 | Notebook 04 không output, `[ĐIỀN SAU]`, P2-11 ghi "trả lời đúng" sai GT | notebook mới chỉ đọc results | `test_notebooks.py` (không `[ĐIỀN SAU]`, chạy được) |

## Lane và file ownership (fan-out song song an toàn — không lane nào sửa file của lane khác)

| Lane | Agent gợi ý | Sở hữu (tạo/sửa) | Phụ thuộc |
|---|---|---|---|
| **A — Lõi QA + eval** (lead) | `t1k-fullstack-developer` | `src/vre/rag/**`, `src/vre/graph/**` (trừ `communities.py`), `src/vre/hybrid/**`, `src/vre/llm/**`, `src/vre/qa/**` (mới), `src/vre/eval/**`, `src/vre/common.py`, `src/vre/valuation/predict.py`, `benchmark/questions.jsonl` (mới), xoá `benchmark/questions.csv`, `scripts/vre_pipeline.py` (lệnh `eval`, `ablation`, `report`), `tests/unit/test_{rag,graph,hybrid,llm,eval,qa,filters}*.py` | — |
| **B — API** | `t1k-web-core-developer` | `api/**` (mới), `tests/api/**` (mới), `Dockerfile.api`, `docker-compose.yaml` (thêm profile `app`), `requirements/base.txt` (thêm `fastapi`, `uvicorn[standard]`, `sse-starlette`, `httpx`) | api-contract §7 (code thật của A, hoặc stub có cùng chữ ký trong lúc chờ) |
| **C — Web** | `t1k-web-ui-developer` | `web/**` (mới), `Dockerfile.web`, `web/nginx.conf`, `scripts/build_geo.py` (mới) | api-contract (dùng `VRE_FAKE_BACKENDS=1` của B để dev) |
| **D — Notebook, README, CI** | `t1k-docs-manager` + `t1k-fullstack-developer` | `notebooks/01_eda.ipynb` … `05_valuation.ipynb` (mới), `README.md`, `.github/workflows/ci.yml` (mới) + xoá `.github/workflows/tests.yml`, `tests/test_notebooks.py`, `requirements/dev.txt` (thêm `nbclient`) | results/*.json từ A (notebook 02/04 chờ eval xong) |

File dùng chung chỉ lead (Lane A) sửa: `src/vre/config.py`, `pyproject.toml`, `.env.example`, `.gitignore`. Lane B sửa `requirements/base.txt`
một lần duy nhất ở đầu lane. Mỗi lane làm trong `git worktree` riêng do lead tạo trước khi fan-out (rule parallel-teammate-git-index-race);
commit dạng pathspec.

## Các bước

### A1 — Lõi QA trên chỉ mục mới (agent ~10h)
1. `vre/rag/filters.py`: `QueryFilters` theo api-contract §2 (thêm `ward`, `deal_type`, `project`, `street`, `month_from/to`; đổi `min_area/max_area` → `min_area_m2/max_area_m2`);
   biên dịch 1 nguồn → Qdrant `Filter` (tháng: `MatchAny` trên `published_month`), SQL, mask (B4). LLM trích filter giữ nguyên chuỗi tiền như cũ.
   Bộ lọc chỉnh tay (API) ghi đè từng trường LLM trích (`source="merged"`).
2. `vre/rag/retrieve.py`: đọc alias `vre_chunks`; query encoder e5-large theo `vre.config.EMBED_MODELS[EMBED_MODEL_SLUG]` (data-contract §9.1; luôn tiền tố `query: `, mean pooling, L2); bỏ `threshold` (B5).
3. `vre/rag/citations.py`: regex `\[Tin#(\d{1,16})\]`, chuẩn hoá 【】［］ như cũ; nguồn hợp lệ so theo `listing_id`.
4. `vre/graph/schema.py` (mới): schema prompt + 8–10 few-shot cho KG mới: tổng hợp qua `MarketStat` (KHÔNG quét Listing), Project/Street, Month,
   Amenity, Cluster ("khu vực tương tự" = phường cùng `Cluster`), đường đa chặng dự án → quận → so sánh. Bao gồm ví dụ có `WHERE n >= 20`
   (sửa lỗi Cypher A2 slide). `text2cypher.py` dùng schema này; B7.
5. `vre/hybrid/router.py`: `Route` thêm `valuation`; route valuation ⇒ LLM trích `ValuationInput` (structured output) → `valuate()` →
   RAG lấy tin tương tự (cùng quận/loại/diện tích ±20%) → trả lời có số dự báo + khoảng + `[Tin#id]` so sánh.
6. `vre/llm/client.py`: `stream_llm()` (OpenAI SDK `stream=True`, usage cuối stream); bản cache trả cả câu trả lời như 1 mảnh; B1.
7. `vre/qa/__init__.py`: facade api-contract §7 (`stream_answer`, `answer`, `system_status`) — phát sự kiện đúng thứ tự §4.

Verify: `& $VRE_PY -m pytest -q tests/unit -k "rag or graph or hybrid or llm or qa or filters"`; thủ công (mạng, 4 lời gọi):
`& $VRE_PY scripts\vre_pipeline.py ask --system hybrid "Quận nào ở Hà Nội có giá/m² căn hộ trung vị cao nhất?"` in route `graph`,
Cypher dùng `MarketStat`, latency Cypher < 1s.

### A2 — Benchmark ~100 câu + GT SQL (agent ~6h; máy ~5 phút)
`benchmark/questions.jsonl` theo data-contract §7.7. Phân bổ: lookup 12, constraint 12, aggregate 12, multihop 12, temporal 10,
comparison 10, project_street 10, valuation 10, unanswerable 12 (= 100). ≥ 15 câu gắn tag `no_diacritics` hoặc `abbreviation` (vd "q7", "hxh", "2pn", "sg").
Quy tắc soạn: khám phá dữ liệu bằng DuckDB trước (thực thể có thật, đủ tin); lookup GT ≤ 5 `dup_cluster_id`; constraint GT ≥ 1 tin;
aggregate/comparison yêu cầu `n ≥ 20` mỗi nhóm; multihop thật sự nhiều chặng (dự án → quận → so với trung vị tỉnh); temporal dùng `month`;
amenity dùng khoá từ điển (không substring tự do); unanswerable: ngoài phạm vi thời gian (trước 06/2025, sau 03/2026), thông tin không có
trong dữ liệu (lãi suất, quy hoạch), tỉnh sau sáp nhập hỏi theo tên mới không có trong dữ liệu cũ...; valuation: mô tả 1 BĐS giả định,
GT = `area_m2 × median(price_m2_mil)` của tin cùng quận × loại × diện tích ±20% trong 2026-02…03 (`value_tolerance=0.15`).
`vre/eval/questions.py` (đọc + validate pydantic), `vre/eval/ground_truth.py` (chạy SQL trên `v_listings`, ghi `results/benchmark_gt.json`,
kiểm: `ids` không rỗng, `value` đúng 1 dòng, `refusal` có `gt_sql=null`). Lệnh: `vre_pipeline.py eval gt`.
Người dùng duyệt nhanh 10 câu ngẫu nhiên (in câu + SQL + đáp án) trước khi chạy eval — agent dùng `AskUserQuestion` (đạt / sửa câu nào).
Verify: `& $VRE_PY scripts\vre_pipeline.py eval gt` exit 0; `& $VRE_PY -m pytest -q tests/unit/test_questions.py` (schema, phân bổ 9 nhóm, id duy nhất,
mọi SQL chỉ `SELECT`/`WITH`, đọc `v_listings`).

### A3 — Eval runner, metric, bootstrap, error analysis (agent ~8h; máy 45–90 phút)
1. `vre/eval/runner.py`: 100 câu × 4 hệ thống qua `vre.qa.answer` (ThreadPool 4, cache LLM, retry backoff theo 429 của Groq), ghi
   `results/eval/raw/answers.jsonl` từng dòng ngay khi xong (resume: bỏ (id, system) đã có).
2. `vre/eval/metrics.py`: thêm `ndcg_at_k`; khớp theo `dup_cluster_id` (data-contract §2.2); Recall@5/10 (capped), MRR cho `gt_kind=ids`.
3. `vre/eval/judge.py`: điểm 0–2 (giữ thang cũ, tolerance từ câu hỏi), refusal; RAGAS faithfulness/answer_relevancy/context_precision cho
   `basic_rag`, `graph`, `hybrid` trên câu không phải unanswerable; `answer_relevancy` dùng encoder query e5-large.
4. `vre/eval/bootstrap.py` (mới): 10.000 mẫu lại theo câu hỏi, ghép cặp, seed 42 → CI 95% + p-value cho các cặp `hybrid–basic_rag`,
   `basic_rag–llm_only`, `hybrid–graph`, theo tổng và theo nhóm aggregate/multihop (report §9).
5. `vre/eval/errors.py` (mới): gán nguyên nhân theo enum data-contract §7.7 (router sai so `route_gold`, Cypher lỗi, Cypher chạy nhưng sai,
   filter sai = filter LLM loại mất tin GT, retrieval miss = GT không nằm trong top-k, bịa = `unsupported_citations` không rỗng hoặc judge 0
   khi ngữ cảnh chứa đáp án, không từ chối…), ghi `results/error_analysis.json` (≥ 2 ví dụ mỗi nguyên nhân có xuất hiện).
6. `vre/eval/summary.py`: ghi `results/eval_summary.json` (§7.7) — gồm router confusion, cost (USD theo `LLM_PRICE_*`), latency p50/p95.
Preview-first (rule preview-first-batch): `vre_pipeline.py eval run --limit 5` trước, in kết quả + thời gian/câu + số lời gọi, rồi `AskUserQuestion`
để chạy đủ. Lệnh: `eval run`, `eval score`, `eval report` (hoặc `eval all`).
Verify: `results/eval_summary.json` có 4 hệ thống × 9 nhóm, mọi `ci95` có 2 số; `& $VRE_PY -m pytest -q tests/unit/test_eval_*.py`
(bootstrap trên dữ liệu giả có chênh lệch biết trước ⇒ CI chứa giá trị thật; CI hẹp lại khi n tăng).

### A4 — Ablation (agent ~4h; máy ~1,5–2h GPU/Qdrant)
Chỉ dùng e5-large — **không chạy model embedding nào khác**. So sánh giữa các model embedding lấy từ `results/embedding_benchmark.json`
(số đo 2026-10-15, data-contract §7.4), trình bày cạnh bảng ablation ở notebook 02 / slide.
Trên chỉ mục đầy đủ (chỉ retrieval, filter LLM lấy từ cache của A3, không sinh câu trả lời): dense vs BM25 vs hybrid; pre-filter có/không;
rerank có/không (`RERANK_MODEL` cũ). Trên **subcorpus** = 200.000 tin ngẫu nhiên (seed 42) ∪ mọi tin GT của câu `gt_kind=ids`: `chunk_size`
256/400/512 với e5-large (chunk 400 tái dùng `.npy` của Phase 2; embed lại 256/512 cho subcorpus, ~35–45 phút mỗi cỡ @~99 chunk/s).
Collection `vre_ablation_e5_large_c{size}`, xoá sau khi đo. Ghi `results/ablation.json` (`embedding` luôn `"e5_large"`). Lệnh: `vre_pipeline.py ablation`.
Verify: mọi run có `recall_at_5`, `n_questions`; `test_ablation_corpus.py` (subcorpus chứa 100% tin GT).

### A5 — Examples + README results block (agent ~1h)
`vre_pipeline.py report examples` → `results/examples.json` (data-contract §7.8). `vre_pipeline.py report readme` thay khối giữa
`<!-- results:start -->` và `<!-- results:end -->` trong README bằng bảng sinh từ `results/*.json` (không gõ tay số).

### B — API FastAPI (agent ~10h)
Cấu trúc: `api/main.py` (app factory, middleware theo thứ tự api-contract §0, lifespan nạp encoder + kiểm manifest), `api/schemas.py`
(Pydantic, đúng tên/kiểu/enum hợp đồng), `api/deps.py` (settings, `RateLimiter` có dọn khoá, auth header, semaphore + chờ 60s),
`api/sse.py` (sse-starlette, heartbeat 15s), `api/routers/{meta,chat,compare,valuate,stats,geo,graph,listings}.py`,
`api/services/{stats,geo,listings,graph_view}.py` (DuckDB đọc `listings.parquet` theo `listing_id` — file đã sort nên row-group pruning;
MarketStat qua Neo4j; centroid từ `data/geo/centroids.json` do Lane C sinh), `api/services/fake.py` + `api/fixtures/*.json` (§6),
`api/openapi.json` (sinh bằng `& $VRE_PY -m api.export_openapi`). Chạy: `& $VRE_PY -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --timeout-keep-alive 5`.
Không bind ngoài loopback khi thiếu `VRE_API_TOKEN` (giữ chính sách cũ). `Dockerfile.api`: `python:3.13-slim`, torch CPU + `requirements/base.txt`
+ transformers/tokenizers==0.22.2 + xgboost; mount `./.cache/huggingface:/hf:ro` (`HF_HOME=/hf`, `HF_HUB_OFFLINE=1`), `./results:/app/results:ro`,
`./models:/app/models:ro`, `./data/clean:/app/data/clean:ro`, `./index:/app/index:ro`. Compose profile `app`: `api` (depends_on qdrant/neo4j
`service_healthy`, healthcheck `/api/health`, `start_period: 120s`, port `127.0.0.1:8000`), `web` (port `127.0.0.1:8080`).
Tests (`tests/api/`, httpx `ASGITransport`, `VRE_FAKE_BACKENDS=1`): `test_contract.py` (OpenAPI ↔ bảng hợp đồng), `test_fixtures_valid.py`,
`test_guards.py` (415/413/400/401/429 thứ tự đúng: sai key vẫn bị rate-limit; `Retry-After`), `test_sse_order.py` (thứ tự sự kiện §4,
`error`→`done`), `test_compare.py` (4 hệ thống, `compare_done`), `test_valuate.py`, `test_limiter_cleanup.py`.
Verify: `& $VRE_PY -m pytest -q tests/api`; `curl -N -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" -d '{"question":"căn hộ 2pn q7 dưới 4 tỷ","system":"hybrid"}'` stream đủ sự kiện.

### C — Web React (agent ~16h)
Stack: Vite + React 19 + TypeScript strict + Tailwind v4 + shadcn/ui + TanStack Query + react-router + react-markdown/remark-gfm +
recharts + maplibre-gl + react-force-graph-2d; SSE POST bằng parser tự viết `web/src/api/sse.ts` (fetch + ReadableStream, có vitest).
`web/src/api/types.ts` sinh bằng `npx openapi-typescript ../api/openapi.json -o src/api/types.ts` (commit). Trang:
1. **Chat** — stream token, markdown + bảng, thẻ nguồn (click `[Tin#id]` cuộn tới thẻ), panel bộ lọc chỉnh tay (tỉnh/quận/loại/giá/diện tích/tháng),
   panel Cypher (câu lệnh + bảng rows + **lỗi hiển thị rõ**), đồ thị đường đi (`/api/graph/subgraph?listing_id=`), nhãn route/fallback,
   usage (latency/token/USD). `aria-live="polite"` cho vùng stream.
2. **So sánh 4 hệ** — 4 cột `/api/compare`, cùng câu hỏi, cột lỗi không làm hỏng cột khác.
3. **Bản đồ** — MapLibre KHÔNG dùng tile ngoài: lớp polygon tỉnh/quận từ GeoJSON đã đơn giản hoá + chấm tròn theo `n_listings`, màu theo
   `median_price_m2_mil`; `coverage` hiển thị.
4. **Định giá** — form `ValuationInput` + kết quả + biểu đồ SHAP + verdict khi có giá chào.
5. **Dashboard thị trường** — giá/m² theo quận (bar), xu hướng tháng (line) từ `/api/stats/market`.
6. **Benchmark** — bảng 9 nhóm × 4 hệ có CI (error bar), ablation, đọc `/api/benchmark`.
Lịch sử chat `localStorage` (khoá `vre_history_v1`, tối đa 50), dark mode, nhập API key ở hộp thoại cài đặt (sessionStorage, KHÔNG đọc từ URL;
`?q=` chỉ điền sẵn ô nhập, không tự gửi — B8).
`scripts/build_geo.py`: tải ranh giới hành chính geoBoundaries gbOpen VNM ADM1/ADM2 (CC BY 4.0, địa giới trước 2025 — kiểm vintage và kiểm URL
bằng `curl -sI` trước khi tải; lưu `data/geo/raw/`, gitignored), đơn giản hoá (`npx mapshaper -simplify 5% keep-shapes`), khớp tên với
`normalize_place` (bỏ dấu + tiền tố), xuất `web/public/geo/{provinces,districts}.json` (ghi công nguồn trong README + footer web) và
`data/geo/centroids.json` cho API; in `coverage`.
`Dockerfile.web`: `node:24-alpine` build → `nginx:alpine`; `web/nginx.conf` proxy `/api` → `api:8000` với `proxy_buffering off`,
`proxy_read_timeout 180s` (SSE).
Tests: vitest (`sse.ts`, format tiền, reducer chat); Playwright (`web/e2e/*.spec.ts`, chạy với API `VRE_FAKE_BACKENDS=1`): chat stream +
nguồn + Cypher lỗi hiển thị; so sánh 4 cột; định giá; bản đồ render; benchmark; điều hướng bàn phím; `@axe-core/playwright` 0 vi phạm
mức serious/critical.
Verify: `npm --prefix web ci`; `npm --prefix web run lint`; `npm --prefix web run typecheck`; `npm --prefix web run test`; `npm --prefix web run build`;
`npm --prefix web run e2e` (Playwright browsers ở `PLAYWRIGHT_BROWSERS_PATH` trên D).

### D — Notebook, README, CI (agent ~10h)
1. Xoá notebook cũ đã làm ở Phase 1; tạo `notebooks/01_eda.ipynb`, `02_rag.ipynb`, `03_knowledge_graph.ipynb`, `04_graphrag_eval.ipynb`,
   `05_valuation.ipynb`. Chỉ import `json`, `pathlib`, `pandas`, `matplotlib`, `IPython.display`; chỉ đọc `results/*.json`, `results/eda/*.png`.
   Mọi con số trong markdown được in từ JSON (không gõ tay). Lưu notebook ĐÃ chạy (có output).
   01: ingest_report + eda_summary + 8 hình; 02: embedding_benchmark (vì sao chốt e5-large: chất lượng + tốc độ 4 model đo 2026-10-15), cấu hình embedding cố định
   (`night_report.model_slug`, `embed.token_budget`), night_report (thời gian từng bước, `peaks`, counters), ablation, examples.rag; 03: graph_report (node/cạnh, communities),
   schema KG (Mermaid trong markdown), examples.graph; 04: eval_summary (9 nhóm × 4 hệ
   + CI), pairwise, router confusion, cost, error_analysis; 05: valuation_metrics (tabular vs full vs baseline, theo loại/tỉnh, SHAP top).
2. `tests/test_notebooks.py`: import được phép, không `[ĐIỀN SAU]`, không số điện thoại/khoá trong output, chạy được bằng `nbclient`
   (marker `notebooks`, < 2 phút).
3. `README.md` viết lại (tiếng Việt): mục tiêu, kiến trúc (Mermaid: Parquet → DuckDB → embedding → Qdrant + Neo4j → Router → LLM → API → Web),
   dữ liệu + license CC-BY-NC-4.0 + ghi công geoBoundaries, yêu cầu ổ D + `vre_env.ps1` + dữ liệu Docker ở `docker-data/`, lệnh pipeline,
   lượt chạy dài (~9h, resume bằng cùng lệnh), chạy API/web/Docker,
   khối kết quả sinh tự động (A5), hạn chế.
4. `.github/workflows/ci.yml` (xoá `tests.yml`): KHÔNG `paths:` (job bắt buộc — rule ci-cd-trigger-design #3); `concurrency` cancel-in-progress
   theo `${{ github.workflow }}-${{ github.ref }}`; jobs: `lint` (ruff + mypy `src/vre api`, `timeout-minutes: 5`), `unit`
   (`pip install -r requirements/dev.txt` + `pip install -e . --no-deps`, `pytest -m "not integration and not gpu" --cov=vre --cov=api --cov-fail-under=70`,
   10 phút), `notebooks` (10 phút), `web` (Node 24, `npm ci`, lint, typecheck, vitest, build, `npx playwright install --with-deps chromium`,
   khởi động API fake rồi `npm run e2e`, 20 phút). Bỏ qua commit của `t1k-ci-sync[bot]`.
Verify: push nhánh ⇒ cả 4 job xanh (`gh run list --branch <branch> --limit 1`).

### E — Tích hợp (lead, ~3h)
1. `docker compose --profile app up -d --build --wait` ⇒ web `http://127.0.0.1:8080` hỏi được thật (không fake) cả 4 hệ.
2. Kiểm tra chéo hợp đồng: Playwright 1 kịch bản chạy với backend THẬT (`e2e:real`, chạy tay, không trong CI) — chat hybrid + compare + valuate.
3. `& $VRE_PY -m pytest -q` (toàn bộ, cả integration + gpu) = 0 fail; ruff/mypy 0 lỗi; `npm --prefix web run e2e` xanh.
4. Commit theo pathspec từng lane, PR, CI xanh.

## Tiêu chí hoàn thành

- [ ] `results/eval_summary.json`, `ablation.json`, `error_analysis.json`, `examples.json`, `benchmark_gt.json` hợp lệ §7.7–7.8 và được commit.
- [ ] Benchmark 100 câu, 9 nhóm đúng phân bổ, mọi GT có SQL; 10 câu ngẫu nhiên đã được người dùng duyệt.
- [ ] Kết luận Hybrid vs Basic RAG vs LLM-only ở nhóm aggregate/multihop có CI + p-value; nếu không có ý nghĩa thống kê thì ghi trung thực.
- [ ] `valuation_metrics.variants.full` (từ Phase 2) có số và MAPE so với baseline phường được báo cáo.
- [ ] So sánh embedding ở notebook/README chỉ đọc `embedding_benchmark.json`; `ablation.json` chỉ có `embedding = "e5_large"`.
- [ ] API: mọi endpoint api-contract có test; thứ tự SSE đúng; guard đúng thứ tự.
- [ ] Web: 6 trang chạy; Playwright + axe xanh; không token trong URL.
- [ ] 5 notebook chạy được, chỉ đọc results, không `[ĐIỀN SAU]`.
- [ ] README không có số gõ tay ngoài khối sinh tự động.
- [ ] CI 4 job xanh, coverage ≥ 70%.
- [ ] Bug B1–B11 có test chứng minh.

## Rủi ro

| Rủi ro | L | I | Điểm | Giảm thiểu |
|---|---|---|---|---|
| Rate limit Groq khi ~2.800 lời gọi eval | 4 | 3 | 12 | Cache LLM, ThreadPool 4, backoff, resume theo dòng `answers.jsonl`, smoke 5 câu trước |
| GT benchmark sai (SQL không khớp ý câu hỏi) | 3 | 4 | 12 | SQL lưu kèm, test cấu trúc, người dùng duyệt 10 câu, error analysis xem lại câu mà mọi hệ đều sai |
| Ba lane lệch hợp đồng | 3 | 3 | 9 | api-contract là SSOT, `types.ts` sinh từ OpenAPI, `test_contract.py`, fake backend dùng fixture đã validate |
| LLM chép sai id 16 chữ số trong `[Tin#id]` | 3 | 3 | 9 | Đo `bad_citation_rate`; regex chuẩn hoá; prompt nhắc chép nguyên id; báo cáo trung thực |
| Hybrid không thắng có ý nghĩa | 3 | 2 | 6 | Báo cáo trung thực (report §9 cho phép) + error analysis chỉ nguyên nhân |
| SSE bị buffer qua nginx/Docker | 2 | 3 | 6 | `proxy_buffering off`, `X-Accel-Buffering: no`, test e2e real |
| GeoJSON không khớp tên quận | 3 | 2 | 6 | Chuẩn hoá tên, `coverage` hiển thị, bản đồ vẫn dùng được khi thiếu điểm |
| Phase vượt thời gian | 4 | 3 | 12 | Fan-out 4 lane trong worktree; thứ tự ưu tiên A1→A2→A3 (đường găng chủ đề chính) trước web nâng cao |
| Dockerfile.api nặng (torch) | 3 | 2 | 6 | torch CPU wheel, HF cache mount read-only thay vì copy model vào image |

## Timeline

| Hạng mục | Agent | Máy | Phụ thuộc |
|---|---|---|---|
| A1 lõi QA | 10h | — | cổng `P1_DONE` + `INDEX_DONE` + `P2_DONE` |
| A2 benchmark | 6h | 5 phút | A1 (chỉ cần schema) |
| A3 eval | 8h | 45–90 phút (mạng) | A1, A2 |
| A4 ablation | 4h | 1,5–2 giờ | A1, A2 |
| A5 examples/README block | 1h | 5 phút | A3 |
| B API | 10h | — | api-contract §7 (stub được) |
| C web | 16h | — | api-contract + fake của B |
| D notebook/README/CI | 10h | 5 phút | A3–A5 cho nội dung notebook 02/04 |
| E tích hợp | 3h | 15 phút | tất cả |
| **Tổng** | **~68 giờ agent (~2,5–3 ngày với 4 lane song song)** | **~3–4 giờ máy** | Đường găng: A1 → A2 → A3 → A5 → D(notebook 04) → E |
