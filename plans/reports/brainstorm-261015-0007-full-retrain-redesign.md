# Brainstorm: Làm lại toàn bộ dự án Vietnam Real Estates (RAG → GraphRAG) trên đủ 3,5 triệu tin

Ngày: 2026-10-15 · Trạng thái: CHỜ DUYỆT · Triển khai ở phiên sau (phiên này chỉ khảo sát + lập kế hoạch)

## 0. Quyết định đã chốt với người dùng

| Câu hỏi | Quyết định |
|---|---|
| Phạm vi "train" | Giữ RAG/GraphRAG làm chủ đề chính (không bỏ). Index đủ 3,5M dòng **+ thêm** model ML định giá nhà như module phụ |
| Embedding | Model chạy được trong ngân sách ~7 giờ (xem §3) |
| Webapp | Viết lại: FastAPI (SSE streaming) + React/Vite/TS + Tailwind/shadcn |
| Ràng buộc outline môn học | Bỏ, tự do tái cấu trúc notebook/slide |
| Tài nguyên | Toàn bộ máy (tắt Unity khi chạy) · **chỉ dùng ổ D**, không ghi gì vào C |
| Venv | Được tự do sửa `.venv` trong dự án |

## 1. Hiện trạng (tổng hợp 4 audit + profile dữ liệu)

### 1.1 Dự án làm gì
Trợ lý hỏi đáp thị trường BĐS VN, so sánh 3 (thực ra 4) hệ thống:
- **Basic RAG**: LLM trích filter → Qdrant (E5-small dense 384d + BM25 pyvi sparse, RRF k=60, pre-filter payload) → LLM trả lời có trích `[Tin#ID]`.
- **Graph (Text2Cypher)**: Neo4j, schema `Listing-OF_TYPE->PropertyType`, `Listing-IN_WARD->Ward-IN_DISTRICT->District-IN_PROVINCE->Province`; LLM sinh Cypher, guard read-only.
- **Hybrid**: LLM router chọn rag/graph/hybrid, fallback RAG.
- **LLM-only** baseline (có trong code, không có trong UI).
- LLM: Groq `gpt-oss-120b`, judge `gpt-oss-20b`.

### 1.2 Vấn đề lớn (vì sao "sơ sài")
1. **Chỉ dùng 20.000 dòng ĐẦU stream** (`config.py:57 N_ROWS`), không phải mẫu ngẫu nhiên → lệch HCM/HN.
2. **Đánh giá chưa từng chạy**: notebook 04 không có output, còn 6 chỗ `[ĐIỀN SAU]`; không có `results/`. Bảng kết quả trên slide (11/24, 14/24, 18/24…) không có nguồn tái lập trong repo.
3. **Benchmark 28 câu**: "multihop" thực chất là group-by đơn; gold set bằng substring (`thang máy`, `hồ bơi`); lookup chỉ 1 id, không chịu tin trùng; không có câu thời gian / so sánh / dự án / đường; GT tính trên 20k nên vô nghĩa khi lên 3,5M; P4-02 ghi đè `questions.csv`.
4. **EDA 1 cell 2 biểu đồ**; ngưỡng 0.80 ở P2-04 vô tác dụng (cosine E5 dồn ~0.9); P2-11 ghi "trả lời đúng" nhưng sai so với GT; rerank không chứng minh được.
5. **KG nông**: chỉ cây hành chính + loại; Project/Street/hướng chỉ là property; amenity extraction không nạp vào graph; community tính xong không dùng; bảng P3-12 stale.
6. **`listing_id` = số thứ tự dòng** trong stream → không ổn định, benchmark hardcode id.
7. **Webapp**: stdlib `http.server` + vanilla JS; không streaming, không render markdown, ẩn lỗi Cypher, không so sánh mode, không bộ lọc/bản đồ; ~20 bug (token trong URL, `?q=` tự submit, rate-limit không chặn sai key, dict limiter phình, không timeout socket, a11y…).
8. **Test/CI**: test đơn vị khá tốt nhưng phụ thuộc `.env` local; CI tự liệt kê package thay vì `requirements.txt`; không lint/coverage/E2E.
9. **Infra**: không có Dockerfile app; Neo4j 2G heap quá nhỏ cho 3,5M; healthcheck `wget` chưa kiểm chứng.
10. **Venv hỏng**: `transformers 5.14` cần `tokenizers<=0.23.0` nhưng cài 0.23.1 → **ĐÃ SỬA trong phiên này** (hạ 0.22.2, cài đủ requirements, `pip check` sạch).
11. **Code**: `graph.py` 522 dòng (trộn load/vis/community/Text2Cypher), `rag.py` 346, `eval.py` 276; trùng lặp (`_value`/`_clean`, `matches` vs `to_filter`); `llm.fresh()` đổi global không thread-safe; judge = generator khi provider openai/anthropic.
12. **`scripts/index_full.py`** đã có pipeline full streaming + resume, nhưng: đọc qua HF stream (resume phải tải lại), chunk đơn luồng tokenize 3 lần/tin, embed → upsert Qdrant (wait=True) → 9 khối MERGE Neo4j chạy tuần tự, `batch_size=64`, avgdl BM25 chỉ từ batch đầu.

### 1.3 Profile dữ liệu thật (3.500.744 dòng, 10 shard, 1,6GB) — ĐO THẬT
- Đã tải về: `D:\Vietnam-Real-Estates\data\raw\vietnam-real-estates\shard_0000..0009.parquet` (gitignored).
- Thời gian 06/2025 → 03/2026; 63 tỉnh (theo địa giới CŨ, trước sáp nhập 7/2025); HCM 1,36M, HN 1,14M, ĐN 186k.
- Loại: Nhà 1,58M · Đất 833k · Căn hộ 763k · Biệt thự/Liền kề 270k · Shophouse 60k.
- **`price` là chuỗi** → phải parse; null price 6,2%.
- Thiếu: ward 14%, street 39%, project 71%, bedroom 46%, floor 77%, road_width 81%, house_depth 98%.
- Area p1/p50/p99 = 25/80/2200 m² → có outlier cần lọc.
- Description: trung vị 466 ký tự, p99 1884; tổng 1,9 tỷ ký tự. Token XLM-R/tin: p50 179, p90 309, p99 553 → **~1,04 chunk/tin** với chunk 400.
- **~815k tin trùng gần đúng** (name + 300 ký tự đầu description) → còn ~2,7M tin duy nhất.

### 1.4 Phần cứng — ĐO THẬT
Ryzen 9 7945HX 16C/32T · RAM 39,7GB · RTX 4060 Laptop 8GB · D trống 72GB, C trống 24GB.
⚠ Pagefile tối đa 8GB (theo phiên Claude khác) → máy từng crash khi hết commit; ⚠ **disk image Docker Desktop đang ở C**.

## 2. Kiến trúc mới đề xuất

```
data/raw (parquet)  ──►  [1] ingest + clean + dedup (DuckDB, đa luồng, theo shard)
                          └─► data/clean/listings.parquet  (listing_id ổn định = hash nội dung)
                                 │
     ┌───────────────────────────┼─────────────────────────────┬───────────────────────┐
     ▼                           ▼                             ▼                       ▼
[2] EDA đầy đủ (DuckDB)   [3] chunk + embed (GPU)        [5] KG export CSV       [6] model định giá
  → results/eda/*.png       + BM25 pyvi (CPU pool)         → neo4j-admin import     XGBoost CUDA
                            → embeddings fp16 .npy          (nhanh hơn MERGE)       + SHAP
                            [4] Qdrant upload song song
                                 │                             │                       │
                                 └──────────── src/vre (RAG · Text2Cypher · Router · Valuation) ─┘
                                                     │
                                [7] Eval: benchmark ~100 câu, GT bằng SQL DuckDB trên full data
                                                     │
                                api/ FastAPI (SSE) ──► web/ React (chat · so sánh 4 hệ · bản đồ · định giá · dashboard)
```

### 2.1 Cấu trúc code mới
```
src/vre/
  config.py              pydantic-settings, mọi tham số env-configurable (chunk, top_k, rrf_k, batch…)
  data/   ingest.py clean.py dedup.py eda.py
  index/  chunk.py embed.py bm25.py qdrant_store.py graph_export.py
  rag/    filters.py retrieve.py generate.py citations.py
  graph/  schema.py text2cypher.py guard.py communities.py
  hybrid/ router.py
  valuation/ features.py train.py predict.py explain.py
  eval/   questions.py ground_truth.py metrics.py judge.py runner.py
  llm/    client.py cache.py
scripts/  vre_pipeline.py  (Typer CLI: ingest | eda | embed | qdrant | graph | price | eval | all --resume)
api/      FastAPI app (routers: chat, compare, valuate, stats, graph)
web/      React + Vite + TS + Tailwind + shadcn/ui + TanStack Query + MapLibre
notebooks/ 01_eda · 02_rag · 03_knowledge_graph · 04_graphrag_eval · 05_valuation (chỉ ĐỌC kết quả cache, không tự tính lại nặng)
tests/    unit (mock) · integration (skipUnless Qdrant/Neo4j) · web E2E (Playwright)
```
Mỗi file ≤200 dòng; tách `graph.py` 522 dòng thành 4 module; xoá trùng lặp đã liệt kê.

### 2.2 Dữ liệu
- Đọc **parquet local** bằng DuckDB/Polars (bỏ HF streaming → resume không phải tải lại).
- `listing_id` = hash ổn định (blake2b của title+description chuẩn hoá) → benchmark không còn phụ thuộc thứ tự dòng.
- Dedup 2 tầng: trùng chính xác + gần trùng (MinHash/SimHash trên description) → giữ `dup_cluster_id` để lookup chấp nhận tin repost.
- Parse `price` chuỗi; tách **bán vs cho thuê**; lọc outlier theo IQR / phân vị theo (loại × tỉnh).
- Chuẩn hoá địa danh (đã có `normalize_place`), thêm cột `admin_2025` (map tỉnh cũ → tỉnh mới sau sáp nhập) làm điểm cộng nội dung.

### 2.3 Knowledge Graph sâu hơn
- Node mới: `Project`, `Street`, `Month` (thời gian), `Amenity` (trích bằng regex/từ điển có biến thể: "thang máy|TM", "hẻm xe hơi|HXH|hẻm ô tô"… rẻ hơn LLM cho 2,7M tin).
- Cạnh mới: `IN_PROJECT`, `ON_STREET`, `POSTED_IN`, `HAS_AMENITY`.
- **Thuộc tính tổng hợp tính sẵn** trên District/Ward/Project (median giá/m² theo loại, số tin, theo tháng) → Cypher tổng hợp chạy ms thay vì quét 2,7M node (tránh timeout 10s).
- Community Leiden trên kNN ward → node `Cluster` và **dùng thật** cho câu hỏi "khu vực tương tự".
- Load bằng `neo4j-admin database import` từ CSV (thay vì MERGE từng lô).

### 2.4 Model định giá (module phụ, không thay RAG)
- XGBoost `device=cuda` (chạy GPU trên Windows ổn), target `log(price/m²)`, tách theo bán/thuê.
- Feature: loại, tỉnh/quận/phường (target encoding có K-fold), diện tích, phòng, mặt tiền, đường, hướng, tháng, amenity flags, **+ PCA 32 chiều của chính embedding bge-m3** (tái dùng, không tốn thêm).
- Split theo thời gian (train 06/2025–01/2026, test 02–03/2026) để tránh rò rỉ.
- Báo cáo: MAE/MAPE/R² theo loại & tỉnh, baseline median-theo-phường, SHAP.
- Tích hợp: tool `valuate()` cho router ("căn này giá hợp lý không?") → câu hỏi kiểu hybrid RAG + ML.

### 2.5 Đánh giá
- **~100 câu** chia 9 nhóm: lookup, constraint, aggregate, true multi-hop (dự án→quận→so sánh), temporal, comparison, project/street, valuation, unanswerable; có câu không dấu/viết tắt.
- GT tính bằng **SQL DuckDB trên full data**, lưu kèm SQL để kiểm chứng; gold set amenity dùng từ điển biến thể; lookup chấp nhận cả `dup_cluster`.
- Metric: Recall@5/10, MRR, nDCG; điểm 0–2 có judge khác model sinh; RAGAS faithfulness/relevancy/context precision; latency/token/USD; **bootstrap CI** cho chênh lệch giữa hệ thống.
- Ablation: e5-small (cũ) vs bge-m3; dense vs BM25 vs hybrid; có/không pre-filter; có/không rerank; chunk 256/400/512.
- Error analysis có phân loại nguyên nhân (router sai, Cypher sai, filter sai, retrieval miss, LLM bịa).

### 2.6 Webapp mới
- FastAPI: `/api/chat` (SSE stream), `/api/compare` (chạy song song 4 hệ), `/api/valuate`, `/api/stats`, `/api/graph/subgraph`; auth token qua header (bỏ token trong URL), rate-limit trước auth, timeout, dọn limiter, Pydantic schema.
- React: chat có markdown + bảng; thẻ nguồn; **chế độ so sánh 4 cột**; bộ lọc tỉnh/quận/loại/giá/diện tích chỉnh tay; **bản đồ** (centroid quận/phường từ GeoJSON hành chính, vì dataset không có toạ độ); **trang định giá** (form + SHAP); **dashboard thị trường** (giá/m² theo quận, xu hướng theo tháng); hiển thị Cypher + bảng kết quả + đồ thị đường đi (react-force-graph); trang **kết quả benchmark**; lịch sử chat localStorage; dark mode; a11y.
- Docker: Dockerfile api + web, compose có app, Neo4j/Qdrant sizing cho full, `start_period`, log volume, named volume.

## 3. Chọn embedding cho ngân sách ~7 giờ — SỐ ĐO THẬT trên RTX 4060 (4.096 tin thật, fp16)

| Model | Tham số | tin/giây | ~2,8M chunk | Ghi chú |
|---|---|---|---|---|
| multilingual-e5-small (hiện tại) | 118M | 890 | 0,9h | baseline cũ |
| multilingual-e5-base | 278M | 350 | 2,2h | |
| **BAAI/bge-m3** | 568M | 74 (chưa tối ưu) | 10,5h chưa tối ưu → **~7–8h sau tối ưu (ước tính)** | mạnh nhất cho tiếng Việt, 1024d, 8192 ctx |
| multilingual-e5-large | 560M | 64 | 12h | chậm hơn bge-m3, không lợi gì |
| gte-multilingual-base | 305M | lỗi (remote code không tương thích transformers 5) | – | loại |

**Đề xuất: `BAAI/bge-m3`** (hoặc bản fine-tune tiếng Việt `AITeamVN/Vietnamese_Embedding`, cùng kiến trúc, cùng tốc độ — chốt bằng mini-eval 500 câu ở phiên triển khai), chạy trên **~2,7M tin sau dedup**, với tối ưu:
1. Tokenize ở worker CPU riêng (bench 74/s gồm cả tokenize trên luồng chính), batch 128, sắp theo độ dài, SDPA, pinned memory.
2. Dedup trước khi embed (bỏ ~815k tin trùng → tiết kiệm ~3h).
3. Checkpoint mỗi 50k → chia được 2 đêm nếu cần.
**Cổng an toàn:** đo 50k tin đầu; nếu dự báo > 9h thì chuyển `e5-base` (~2,2h) và ghi rõ trong báo cáo. Đo tối ưu chưa hoàn tất trong phiên này (dừng giữa chừng để nhường RAM cho phiên Unity) → 7–8h là **ước tính**, cần xác nhận.

**Cập nhật 2026-10-15** (user chốt, thay đề xuất và cổng an toàn ở trên):
- Embedding cuối cùng: **`intfloat/multilingual-e5-large`** — duy nhất, không còn chọn/lùi model; các model khác đã xoá khỏi cache, không tải lại.
  Số đo lại trên máy này (20k tin ngẫu nhiên đã dedup, 1.000 truy vấn title→description, fp16, SDPA): e5-large 99 tin/s bền vững (104/s ở
  token budget 32768, tốt nhất trong quét 16k/32k/64k/128k), GPU util 91%, Recall@1 0,656, Recall@10 0,816, MRR@10 0,711 — chất lượng cao
  nhất trong 4 model đo; bge-m3 0,681 @120/s, e5-base 0,643 @430/s, e5-small 0,649 @802/s. Số thô: `.cache/scratch/bench_final.jsonl`
  (script `.cache/scratch/bench_final.py`), được sao chép vào `results/embedding_benchmark.json`. Bảng 4.096 tin ở trên là số đo cũ, đã lỗi thời.
- Spec: `e5_large`, 1024d, mean pooling theo attention mask, tiền tố `passage: `/`query: `, `max_length=512`, `embed_token_budget=32768`, L2, fp16;
  collection `vre_chunks_e5_large`.
- **Không giới hạn thời gian**: bỏ ngân sách ~7 giờ, bỏ cổng chọn model theo dự báo giờ. Lượt chạy dài ~9h (embed ~7,7–7,9h cho ~2,75M chunk
  còn lại + HNSW/verify + định giá full ~1h luôn chạy), resume bằng cùng lệnh. Bước 50k trở thành hiệu chỉnh/dry-run (quét budget, GPU util ≥ 90%, ETA).
- **Không cấu hình máy**: máy dành riêng cho dự án; bỏ mọi việc chuẩn bị máy ở §4–§5 và mọi kiểm tra/watchdog bộ nhớ hệ thống. Dữ liệu Qdrant/Neo4j
  bind-mount vào `D:\Vietnam-Real-Estates\docker-data\{qdrant,neo4j}`. Chi tiết: `plans/261015-vre-full-retrain/plan.md` mục "Quyết định bổ sung".

## 4. Ước tính thời gian chạy (máy đã tắt Unity)

| Bước | Tài nguyên | Ước tính | Căn cứ |
|---|---|---|---|
| 1. Ingest + clean + dedup | CPU 16 luồng, ~6GB RAM | 5–15 phút | đọc 10 shard mất 18s (đo) |
| 2. EDA đầy đủ + biểu đồ | CPU | 5–10 phút | DuckDB |
| 3a. Chunk + tokenize | CPU pool 6 tiến trình | chạy chồng với 3b | tokenize 20k mất 3s (đo) |
| 3b. **Embedding bge-m3** | GPU (~3–4GB VRAM) | **7–8h** (cổng: >9h → e5-base 2,2h) | §3 |
| 3c. BM25 pyvi | CPU pool 8 tiến trình | 20–60 phút, chồng với 3b | ước tính, chưa đo |
| 4. Qdrant upload + build HNSW | Docker, 6 luồng | 30–60 + 30–60 phút | ước tính |
| 5. Neo4j import (neo4j-admin) + aggregate | Docker, 8 luồng | 15–30 phút | ước tính; MERGE cũ sẽ ~1–2h |
| 6. Model định giá XGBoost CUDA + tuning | GPU (sau bước 3b) | 45–90 phút | ước tính |
| 7. Benchmark ~100 câu × 4 hệ + judge + RAGAS | mạng (Groq) | 45–90 phút | phụ thuộc rate limit Groq |
| **Tổng** | | **~10–13h đồng hồ** | |

Chia theo phase (§6):
- **Ban ngày Phase 1** (bước 1, 2, 5, 6-tabular + đo 50k): ~1,5–2,5h máy chạy.
- **Đêm Phase 2** (bước 3a–3c, 4, 6-đầy đủ): mục tiêu **≤ 7h**. Model embedding được chọn bằng số đo 50k để vừa khung này.
- **Ban ngày Phase 3** (bước 7): 45–90 phút máy chạy, cộng thời gian làm code/webapp.

## 5. Phân bổ tài nguyên khi chạy

| Tài nguyên | Phân bổ |
|---|---|
| GPU 8GB | Chỉ embedding (fp16, ~3–4GB); sau đó XGBoost CUDA; khi serve: bge-m3 encode query (~1,2GB) |
| CPU 32 luồng | 1 tiến trình điều phối GPU + 4 worker tokenize · 8 worker pyvi · Qdrant 6 · Neo4j 8 · chừa 4–6 cho OS |
| RAM 40GB | `.wslconfig memory=18GB` cho Docker: Qdrant ~5GB (vector int8 trong RAM, gốc on_disk), Neo4j heap 4G + pagecache 6G · pipeline Python ≤10GB (xử lý theo shard) · OS ~8GB |
| Ổ D (trống 72GB) | data clean ~2GB · embeddings fp16 ~5,5GB · Qdrant ~15GB · Neo4j ~6GB · model HF 8,5GB (đã có) → ~37GB |
| Ổ C | **không ghi gì**: `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, `TMP` trỏ về `D:\Vietnam-Real-Estates\.cache\` |

**Việc người dùng cần làm trước phiên triển khai:**
1. Docker Desktop → Settings → Resources → Advanced → **Disk image location** chuyển sang ổ D.
2. Tăng pagefile (đặt trên D, ví dụ 16–32GB) để tránh crash do hết commit.
3. Tắt Unity / lên lịch tránh build Unity của phiên khác.
4. Có `GROQ_API_KEY` trong `.env`.

## 6. Lộ trình triển khai — 4 phase (theo yêu cầu: ít phase, 1 phase treo máy qua đêm)

### Phase 1 — Nền tảng + chuẩn bị đêm chạy (ban ngày, người dùng có mặt)
- Tái cấu trúc `src/vre` + config env + CLI `scripts/vre_pipeline.py`.
- Ingest/clean/dedup DuckDB → `data/clean/listings.parquet`; EDA đầy đủ → `results/eda/`.
- KG mới: export CSV + `neo4j-admin import` + thuộc tính tổng hợp + community (15–30 phút, chạy luôn ban ngày).
- Model định giá **bản tabular** (XGBoost CUDA, chưa có feature embedding) (~30–45 phút).
- Pipeline đêm: chunk → embed → BM25 → Qdrant, có checkpoint/resume, log, retry, watchdog RAM/VRAM.
- **Đo thật 50k tin** với bge-m3 tối ưu → chọn model cho đêm theo quy tắc: tổng dự báo đêm ≤ 6,5h → bge-m3; ngược lại → e5-base. Ghi quyết định vào `results/run_plan.json`.
- Preflight: Docker disk ở D, pagefile, `.wslconfig`, ổ D đủ chỗ, HF/TORCH/PIP/TMP trỏ D, không có tiến trình Unity, Qdrant/Neo4j healthy.
- Unit test xanh. Kết thúc phase: đưa người dùng **1 lệnh** để khởi động đêm chạy.

### Phase 2 — Chạy treo máy qua đêm (~7h, người dùng ngủ, toàn bộ tài nguyên)
- 1 lệnh: `vre_pipeline.py night --resume` chạy không cần người:
  1. Chunk + tokenize (CPU pool) ‖ embed GPU (model đã chọn ở Phase 1) ‖ BM25 pyvi (CPU pool) — chạy chồng nhau.
  2. Upload Qdrant song song (HNSW tắt khi nạp) → bật HNSW + build.
  3. Nếu còn thời gian: feature PCA từ embedding + train lại model định giá bản đầy đủ.
  4. Ghi `results/night_report.json` (thời gian từng bước, số điểm, lỗi, tài nguyên đỉnh).
- Tự xử lý lỗi: retry có backoff, checkpoint mỗi 50k; nếu crash, sáng chạy lại cùng lệnh sẽ tiếp tục, không làm lại từ đầu.
- Không gọi LLM/mạng trong đêm (tránh rate limit làm treo); eval để Phase 3.

### Phase 3 — Hệ thống + đánh giá + webapp + tài liệu (ban ngày)
- RAG/Text2Cypher/Router dùng index mới + tool `valuate()`.
- Benchmark ~100 câu, GT SQL, eval runner, ablation, error analysis → `results/*.json`.
- FastAPI + React webapp + Docker; 5 notebook mới (đọc `results/`); README; CI (ruff, mypy, pytest, coverage, Playwright).

### Phase 4 — Slide (phiên riêng) — xem §7
Mọi số liệu lấy từ `results/*.json` của Phase 3.

## 7. Ghi chú cho slide (làm ở phiên riêng)

Hiện trạng: `slide/…có-lý-thuyết.pptx` 39 slide (16:9, Calibri, palette navy/teal/cam nhất quán, mọi slide có speaker notes) + `hoc-phan-4-graphrag.html` (tài liệu ôn cho Người 4, slide 30–38). Cả 2 **chưa được commit**.

### Phải sửa (số liệu stale — toàn bộ gắn với 20k)
- Số đếm dữ liệu: 20.000 / 19.925 / 73 trùng / 48 giá sửa / 21.030 chunk / 4,8% / 133s / 32,3MB.
- Tỷ lệ thiếu: 50,8% / 75,2% / 14,5% → thay bằng số full (46% / 71% / 14%).
- EDA slide 7 (histogram, tỉnh 6888/6805…, "HN+HCM 69%").
- Mọi ví dụ `Tin#ID` (1, 1198, 14515, 8940, 12955, 15405, 3138, 18456, 1205) — id sẽ đổi sang hash.
- Graph: 22.116 node / 41.978 cạnh + mọi số con; Leiden 460/13/Q=0,77/245,8/37,0.
- Toàn bộ kết quả slide 15, 33, 35, 36, 37, 38 (chưa có nguồn tái lập!) + latency/token.
- Số nằm cả trong **tiêu đề** (slide 6, 11, 16, 19, 28, 29, 33, 35, 38), **speaker notes**, **chart nhúng** (7 chart) và **mảng JS trong file HTML**.
- Sửa mâu thuẫn đã biết: 24→~100 câu, 3→4 hệ; Thủ Đức 1.229 vs 1.367; Cypher A2 thiếu `WHERE n>=20`; note slide 37 ngược chart; slide 15 chart trùng bảng slide 35; "RTX 3050" → RTX 4060.

### Thêm mới
- 1 sơ đồ kiến trúc end-to-end (Parquet → DuckDB → bge-m3 → Qdrant + Neo4j → Router → LLM → Web).
- Slide "Scale lên 3,5M": thời gian từng bước, tài nguyên, kỹ thuật (dedup, checkpoint, neo4j-admin, int8).
- EDA thật (3–4 slide): phân bố giá/m² theo loại & vùng, xu hướng theo tháng, dữ liệu thiếu, outlier, trùng lặp 23%.
- Ablation embedding (e5-small 20k vs bge-m3 3,5M) + bảng kết quả 9 nhóm câu có CI.
- Error analysis (ví dụ thất bại thật theo nguyên nhân).
- Module định giá ML (kết quả + SHAP) như phần mở rộng, không thay chủ đề.
- Demo webapp (ảnh chụp/clip: so sánh 4 hệ, bản đồ, định giá).
- Hạn chế & hướng phát triển; kết luận/takeaways trước Q&A.

### Redesign
- Giảm wall-of-text ở slide lý thuyết 5, 8, 12, 17, 24, 30, 34 (30–40 shape, chữ 12–15pt): mỗi slide 1 ý, đẩy chữ xuống notes.
- Caption tối thiểu 14pt (hiện 10–11pt không đọc được trên máy chiếu).
- Thay sơ đồ vẽ tay bằng shape bằng sơ đồ sạch (Mermaid/Excalidraw xuất SVG) cùng palette.
- Chart xuất từ notebook (cùng theme), không nhập tay số vào chart nhúng.
- Code Cypher: chỉ 1 khối/slide, highlight phần quan trọng.
- Quyết định file HTML: tái sinh từ cùng nguồn số liệu (JSON kết quả) hoặc bỏ, để không lệch số lần nữa.
- **Số liệu trên slide phải lấy từ `results/*.json`** (sinh ra bởi pipeline) → 1 nguồn sự thật.

## 8. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| bge-m3 thực tế > 9h | Cổng 50k → e5-base; checkpoint chia nhiều đêm |
| Hết RAM/commit → crash (từng xảy ra) | Xử lý theo shard, giới hạn WSL 18GB, tăng pagefile trên D, không chạy cùng build Unity |
| Ổ C đầy do Docker | Chuyển disk image Docker sang D trước |
| Cypher tổng hợp timeout trên 2,7M node | Thuộc tính tổng hợp tính sẵn + index; few-shot dùng chúng |
| Rate limit Groq khi eval ~1.600 lời gọi | Cache LLM (đã có), retry backoff, chạy theo lô |
| Địa giới 2025 (63→34 tỉnh) gây nhầm câu hỏi | Giữ địa giới cũ của dataset + cột map sang địa giới mới |
| License dataset CC-BY-NC-4.0 | Ghi nguồn, chỉ dùng phi thương mại |

## 9. Tiêu chí thành công
- Pipeline `vre_pipeline.py all --resume` chạy hết 3,5M từ parquet local, tái lập được, không ghi vào C.
- Notebook 04 và mọi số trên slide sinh từ `results/*.json`; không còn `[ĐIỀN SAU]`.
- Benchmark ~100 câu có GT SQL kiểm chứng; báo cáo có CI và ablation.
- Hybrid > Basic RAG > LLM-only có ý nghĩa thống kê ở nhóm aggregate/multi-hop (hoặc báo cáo trung thực nếu không).
- Model định giá vượt baseline median-theo-phường (MAPE thấp hơn rõ).
- Webapp: stream, so sánh 4 hệ, bản đồ, định giá, dashboard; Playwright E2E xanh; CI xanh.

## 10. Câu hỏi còn mở
Không còn — các quyết định ở §0 đã chốt; lựa chọn embedding cuối (bge-m3 vs bản fine-tune tiếng Việt vs fallback e5-base) được quyết định bằng số đo ở cổng 50k của phase P2 theo quy tắc đã ghi ở §3.
