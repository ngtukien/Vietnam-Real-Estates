# Phase 2 — Pipeline index + lượt chạy dài e5-large (phiên `p2`, chạy song song với Phase 1)

> **Phạm vi khi cook file này: CHỈ phase này.** Dừng sau bước cuối của file (Bước 7). Không làm việc của phase khác, không đọc `plan.md`
> để làm tiếp. Phase 1 và Phase 2 chạy song song trong hai phiên (`p1` = Phase 1, `p2` = phiên này), phối hợp theo
> [`coordination.md`](coordination.md) — đọc file đó TRƯỚC khi làm bất cứ gì.

Đọc trước khi code (đường dẫn tuyệt đối, đọc được từ worktree p2):
`D:\Vietnam-Real-Estates\plans\261015-vre-full-retrain\coordination.md` (tên phiên, tín hiệu, ownership, git, tài nguyên) ·
`D:\Vietnam-Real-Estates\plans\261015-vre-full-retrain\data-contract.md` (§2.3, §4, §5, §7.5, §7.6; §9 = khoá config, kiểm tra fail-fast, CLI ↔ hàm mà code của p2 phải khớp).
(Tên file giữ `phase-02-overnight-run.md` để link không gãy; tên lệnh/artefact giữ `night`, `night.ps1`, `night_report.json` — nghĩa là "lượt chạy dài",
không gắn với ban đêm.)

## Mục tiêu

1. **Viết** code chỉ mục và lượt chạy dài: `src/vre/index/{chunk,tokenize,embed,bm25,qdrant_store,qdrant_upload}.py`,
   `src/vre/pipeline/{state,stages,night,pca,report}.py`, `scripts/night.ps1` + test của chúng — theo đúng bề mặt data-contract §9.
2. **Chạy** dry-run thật trên 2 part (bắt lỗi code trước lượt dài — kiểm tính đúng, không phải đo).
3. **Chạy** lượt dài, không người, không gọi mạng/LLM, **không giới hạn thời gian**: chunk + tokenize → embed GPU bằng `intfloat/multilingual-e5-large`
   → BM25 pyvi → upload Qdrant `vre_chunks_e5_large` (HNSW xây song song khi nạp) → chờ `green` → alias `vre_chunks` → kiểm chứng → PCA 32 chiều.
   Ghi `results/night_report.json`. Bị ngắt ở đâu ⇒ chạy lại đúng cùng lệnh tiếp tục từ checkpoint.
4. **Chạy** định giá bản `full` (GPU) bằng code định giá của p1 sau khi nhánh p1 đã merge, rồi merge nhánh p2.

Embedding cố định, không đo lại, không chọn model (data-contract §9.1): `intfloat/multilingual-e5-large`, 1024 chiều, mean pooling theo attention mask,
tiền tố `passage: ` (tài liệu) / `query: ` (truy vấn), `max_length=512`, fp16, L2-normalize, `embed_token_budget=32768` (số đo 2026-10-15:
~99 tin/s bền vững, 104/s ở budget này). Không có bước quét budget, không có cổng GPU util, không có ETA.

Ngoài phạm vi: Neo4j, ingest, EDA, định giá tabular (p1); benchmark/LLM (Phase 3). Không sửa file dùng chung (coordination §5) — cần đổi ⇒ `CHANGE_REQUEST`.

## Máy

Toàn bộ máy rảnh và dành riêng cho dự án: Ryzen 9 7945HX 32 luồng, ~35GB RAM dùng được, RTX 4060 Laptop 8GB. **Không có bước cấu hình máy nào**
— người dùng không chỉnh hay dọn dẹp gì. Vì p1 chạy cùng lúc, máy được chia theo coordination §8: **GPU chỉ p2 dùng**; p2 dùng
`tokenize_workers=6`, `bm25_workers=10`, Qdrant `max_indexing_threads=12`; p1 dùng CPU còn lại (DuckDB/XGBoost 12 luồng) và Neo4j.

## Điều kiện tiên quyết

- Bước 0–2 (viết code): chỉ cần đọc được hợp đồng. Bước 3 (test): đã nhận `SKELETON_READY`. Bước 4–5 (chạy): đã nhận `CLEAN_DATA_READY`.
  Bước 6: đã nhận `P1_DONE`.
- Kiểm tra fail-fast nằm ở đầu lệnh `night` (data-contract §9.2–§9.3): `e5_large_cached` (`HF_HUB_OFFLINE=1`; không bao giờ tải model khác),
  `gpu`, `qdrant_healthy`, `d_free_space` (≥ 45 GB), `env_paths_on_d`, `clean_data_ready` (fingerprint khớp `CLEAN_DATA_READY.json`).
- Mọi lệnh chạy sau `. D:\Vietnam-Real-Estates\scripts\vre_env.ps1` (bản trong worktree p2 sau rebase cũng trỏ về `D:\Vietnam-Real-Estates\.cache\`).

## Prior art tái dùng (phạm vi tìm: `b4a7663:src/`, `b4a7663:scripts/`, `.cache/scratch/`, `b4a7663:tests/`)

| Nguồn (đọc bằng `git show b4a7663:<path>` — p1 sẽ xoá bản cũ khỏi cây làm việc) | Tái dùng ở đâu (p2) |
|---|---|
| `src/data.py` — `chunk_starts`, `n_chunks_formula`, `listing_header`, `chunk_listing` | `vre/index/chunk.py` |
| `src/qdrant_store.py` — BM25 sparse (`K1`, `B`, `token_id`, `bm25_document`, `bm25_query`), int8 quantization + `rescore`, `to_filter`, `search_many` | `vre/index/qdrant_store.py`, `vre/index/bm25.py` |
| `src/vector.py` — fp16 trên GPU, lock một luồng | `vre/index/embed.py` (thay bằng encoder tự viết theo token-budget) |
| `scripts/index_full.py` — state nguyên tử, tạo Pool TRƯỚC khi nạp CUDA, `spawn` trên Windows, digest blake2b, manifest | `vre/pipeline/state.py`, `vre/pipeline/night.py` (ghi file qua `vre.common.write_json_atomic` của p1) |
| `.cache/scratch/bench_final.py` — e5-large fp16 + SDPA + pre-tokenize, sort theo độ dài, lô theo ngân sách token | `vre/index/embed.py` (chỉ phần encoder; KHÔNG mang phần đo `nvidia-smi`/quét budget) |
| `tests/test_presentation_notebooks.py` — test chunk, BM25 | `tests/unit/test_chunk.py`, `test_bm25.py` |

## File ownership

Theo coordination §5. p2 tạo/sửa DUY NHẤT: `src/vre/index/{chunk,tokenize,embed,bm25,qdrant_store,qdrant_upload}.py`,
`src/vre/pipeline/{state,stages,night,pca,report}.py`, `scripts/night.ps1`, `tests/unit/test_{chunk,bm25,qdrant_payload,state,pca}.py`,
`tests/integration/{test_qdrant_roundtrip,test_index_complete}.py`, `tests/gpu/test_embed_parity.py`. Không tạo `__init__.py` (p1 tạo rỗng sẵn).
Sinh ra (gitignored): `index/**`, `data/features/`, `models/valuation/*_full*`, `logs/night-*.log`, `docker-data/qdrant/`.
Sinh ra (commit): `results/night_report.json`, `results/valuation_metrics.json` (thêm `variants.full`, sau `P1_DONE`).
Mỗi file Python ≤ 200 dòng; mọi tham số đọc từ `vre.config` (data-contract §9.1).

## Các bước (theo thứ tự; tín hiệu gửi/chờ ghi đậm)

### Bước 0 — Chuẩn bị phiên + worktree (10 phút)
1. Phiên mang tên `p2` (coordination §2); `ListAgents` để thấy `p1`.
2. Có `D:\Vietnam-Real-Estates\results\handoff\SKELETON_READY.json`? Có ⇒ `git -C D:\Vietnam-Real-Estates worktree add -b feat/p2-index D:\Vietnam-Real-Estates-p2 <sha>`;
   chưa ⇒ tạo từ `main` (coordination §6). Mọi việc của p2 làm trong `D:\Vietnam-Real-Estates-p2`.
3. Đọc hợp đồng + prior art. Không chờ p1 — sang Bước 1 ngay.

### Bước 1 — Viết code chỉ mục (agent ~3,5h; chưa chạy)
Theo data-contract §2.3, §4, §5.
- `tokenize.py` + `chunk.py`: tokenizer fast của e5-large (XLM-R), **tokenize mỗi tin đúng 1 lần** (header 1 lần, body 1 lần với
  offsets); `input_ids = [bos] + prefix_ids + header_ids[:header_max_tokens] + nl_ids + body_ids[window] + [eos]`; text lưu =
  header + "\n" + body[span]. Cắt ở `max_chunks_per_listing`. Token ghi tạm `index/tok/part_NNNN.npz` (mảng phẳng int32 + offsets), xoá sau khi part
  đã embed (tránh pickle dữ liệu lớn qua queue).
- `embed.py`: `AutoModel.from_pretrained(..., dtype=float16, attn_implementation="sdpa")`, `inference_mode`, sort theo độ dài trong part,
  lô theo ngân sách token `settings.embed_token_budget` (32768, cố định), tối đa `embed_max_batch` câu, pinned memory + `non_blocking`; mean pooling theo
  attention mask; tiền tố `passage: ` / `query: `; L2-normalize; xuất fp16 theo thứ tự gốc. `torch.cuda.OutOfMemoryError` ⇒ `empty_cache`, giảm ngân sách ½,
  thử lại (đếm `oom_retries`); dưới `embed_min_token_budget` (4096) mà vẫn OOM ⇒ raise.
- `bm25.py`: pool `spawn` `bm25_workers` tiến trình, pyvi, ra `token_ids/tfs/doc_len` (data-contract §4); trọng số theo §5 với `avgdl` từ state.
- `qdrant_store.py`: tạo collection đúng §5 (HNSW bật khi nạp: `indexing_threshold=qdrant_indexing_threshold`, `max_indexing_threads=qdrant_max_indexing_threads`;
  kiểm tên khoá cấu hình đúng cho v1.15), alias, payload index, `wait_green(timeout)`, `count_part(lo, hi)`, `smoke_search(settings, query, collection, limit)`
  (đúng chữ ký §9.3; dense + BM25, không LLM).
- `qdrant_upload.py`: dựng point (payload từ lát `listings.parquet` của part, bỏ khoá null), `upload_points(..., parallel=qdrant_upload_parallel,
  batch_size=qdrant_upload_batch, max_retries=5, wait=False)` qua gRPC, bọc `tenacity` (6 lần, backoff mũ 2–120s); Qdrant không trả lời
  > 2 phút ⇒ một lần `docker compose -f D:\Vietnam-Real-Estates\docker-compose.yaml up -d --wait qdrant`, rồi tiếp tục retry.

### Bước 2 — Viết code lượt chạy dài (agent ~3h; chưa chạy)
Theo data-contract §4.1, §7.5.
- `state.py`: §4.1, ghi nguyên tử (`vre.common.write_json_atomic`), kiểm fingerprint; state dry-run tách riêng dưới `index/dryrun/`.
- `stages.py` + `night.py` — `run(settings, resume, dry_run) -> int`: một dòng giữ máy thức trong tiến trình
  `ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED)` (không đổi thiết lập hệ thống), hoàn tác trong `finally`;
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. Điều phối: producer tokenize (tối đa `tokenize_prefetch_parts`=3 part đi trước GPU), BM25 pool song song,
  GPU loop ở tiến trình chính, luồng upload. `bm25_avgdl` tính một lần khi mọi part đã `bm25` và trước upload đầu tiên (§4.1). Sau khi mọi part
  `uploaded`: `wait_green` (≤ `qdrant_green_timeout_s`) → alias `vre_chunks` → kiểm `count = Σ n_chunks` → `smoke_search` 3 truy vấn cố định → PCA →
  manifest `index/manifest_{collection}.json`. Report luôn ghi (`try/finally`). Ổ D < `min_free_disk_gb_part` trước một part ⇒ checkpoint, mã 20.
  Lỗi không phục hồi sau retry ⇒ `status=failed`, mã 20. `dry_run=True` ⇒ collection `dryrun_collection`, `dryrun_parts`=2 part đầu, MỌI artifact (chunks, tok, emb, bm25,
  `night_state.json`, `night_report.json`) dưới `index/dryrun/` thay vì `index/` và `results/`, bỏ PCA, không đặt alias.
- `pca.py`: vector tin = trung bình vector các chunk của tin (đọc `.npy` theo part, fp32), `IncrementalPCA` `pca_dim` chiều fit trên `pca_fit_sample` tin
  thuộc tháng train, transform toàn bộ → `data/features/emb_pca32_e5_large.parquet` (§4).
- `report.py`: ghi `night_report.json` đúng §7.5 (`peaks` từ psutil/torch; `embed.chunks_per_s_avg` chỉ là dòng thông tin); `status(settings, dry_run)`
  trả `status`, số part embedded/uploaded/tổng, `points_in_qdrant`, trạng thái HNSW, đỉnh RAM/VRAM, số lỗi (không ETA).
- `scripts/night.ps1`: dot-source `D:\Vietnam-Real-Estates\scripts\vre_env.ps1`; `docker compose -f D:\Vietnam-Real-Estates\docker-compose.yaml up -d --wait qdrant`;
  vòng tối đa `night_max_attempts` lần (đọc bằng `vre_pipeline.py config-get`): chạy `& $VRE_PY "$PSScriptRoot\vre_pipeline.py" night --resume`,
  ghi `D:\Vietnam-Real-Estates\logs\night-<yyMMdd-HHmm>.log` (Tee-Object); mã 0 ⇒ thoát 0; mã 20 hoặc crash ⇒ chờ `night_retry_wait_s` (120s) rồi chạy lại
  cùng lệnh; hết số lần ⇒ thoát 20.

### Bước 3 — Chờ `SKELETON_READY`, rebase, viết + chạy test (agent ~2h; máy ~10 phút)
1. **Chờ `SKELETON_READY`** (coordination §4: kiểm file; chưa có ⇒ lệnh nền thăm dò ≤ 5 phút). Nhận ⇒ `git -C D:\Vietnam-Real-Estates-p2 rebase <sha>`.
   Thiếu khoá config/thư viện ⇒ **gửi `CHANGE_REQUEST`**, làm việc khác trong lúc chờ `CHANGE_APPLIED`, rồi rebase lên sha đó.
2. Viết test: unit (không mạng/GPU/Docker) `test_chunk.py` (công thức số chunk, cap 64, `point_id` đảo ngược, tokenize 1 lần — đếm lời gọi tokenizer giả),
   `test_bm25.py`, `test_qdrant_payload.py` (payload đúng §5, bỏ null), `test_state.py` (resume, ghi nguyên tử, fingerprint lệch ⇒ lỗi, state dry-run tách
   riêng), `test_pca.py` (shape, trung bình chunk theo tin, fit chỉ trên tháng train); integration (`@pytest.mark.integration`, tự skip nếu Qdrant không chạy)
   `test_qdrant_roundtrip.py` (collection tạm, 100 point, dense+bm25+filter), `test_index_complete.py` (tự skip tới khi `night_report.status=completed`; nội
   dung ở §Verify); GPU (`@pytest.mark.gpu`) `tests/gpu/test_embed_parity.py` (encoder tự viết vs `SentenceTransformer.encode` trên 256 câu: cosine ≥ 0,999,
   cả tiền tố `passage: ` và `query: `).
3. Chạy: `& $VRE_PY -m pytest -q tests/unit/test_chunk.py tests/unit/test_bm25.py tests/unit/test_qdrant_payload.py tests/unit/test_state.py tests/unit/test_pca.py`
   = 0 fail; `& $VRE_PY -m pytest -q -m gpu` = 0 fail; `& $VRE_PY -m pytest -q tests/integration/test_qdrant_roundtrip.py` = 0 fail;
   `& $VRE_PY -m ruff check src/vre/index src/vre/pipeline tests` = 0 lỗi. Commit pathspec.

### Bước 4 — Chờ `CLEAN_DATA_READY`, chạy dry-run 2 part (máy ≤ 15 phút)
1. **Chờ `CLEAN_DATA_READY`** (file + tin). Đọc `fingerprint` từ file.
2. Chạy: `& $VRE_PY scripts\vre_pipeline.py night --dry-run` ⇒ `index/dryrun/night_report.json` có `status=completed`, Qdrant `vre_dryrun` count = Σ n_chunks của 2 part,
   `smoke-search --collection vre_dryrun "căn hộ 2 phòng ngủ quận 7"` ≥ 5 hit. Report có thể ghi thông lượng như một dòng thông tin — không phải ngưỡng.
3. Kiểm resume: chạy lại `night --dry-run --no-resume` với state dry-run mới, Ctrl+C sau khi part 0000 `uploaded`, chạy lại `night --dry-run` ⇒ log in
   "skip part 0000" và hoàn tất.
4. Xoá collection `vre_dryrun` và cả thư mục `index/dryrun/`. Lỗi ở đây ⇒ sửa code, chạy lại test Bước 3, chạy lại dry-run.

### Bước 5 — Chạy lượt dài (~8h máy; người ~5 phút) → **gửi `INDEX_STARTED`**, sau đó **`INDEX_DONE`** hoặc **`INDEX_FAILED`**
1. Khởi động bằng lệnh nền của phiên (Bash/PowerShell `run_in_background: true`) — người dùng không phải gõ gì; nếu muốn tự chạy tay thì đúng lệnh:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File D:\Vietnam-Real-Estates-p2\scripts\night.ps1
   ```
2. **Gửi `INDEX_STARTED`** `{"collection":"vre_chunks_e5_large","n_parts":<n>,"log":"<đường dẫn log>","attempt":1}`.
3. Kiểm part đầu (~10–15 phút sau khi bắt đầu): `& $VRE_PY scripts\vre_pipeline.py status` cho thấy part 0000 `uploaded` và count khớp. Không khớp ⇒ dừng
   tiến trình, sửa, quay lại Bước 4.
4. Chờ lệnh nền kết thúc (tiến trình thoát đánh thức phiên; không thăm dò bằng tin nhắn).
5. Thoát 0 ⇒ chạy §Verify ⇒ đạt thì commit pathspec `results/night_report.json` và **gửi `INDEX_DONE`** `{"collection","alias":"vre_chunks","n_points","pca_path"}`.
   Thoát 20 sau `night_max_attempts` lần ⇒ **gửi `INDEX_FAILED`** `{"error","step","log","resumable":true}`; đọc `errors[]` + log, sửa code (chạy lại test Bước 3),
   chạy lại ĐÚNG lệnh ở mục 1 (chỉ phần còn lại được làm) và **gửi lại `INDEX_STARTED`** với `attempt` tăng.

Các bước bên trong lệnh `night` (thứ tự + thời gian dự kiến; mọi bước chạy chồng nhau trừ khi ghi "tuần tự"; ~2,8M chunk @ ~99–104 chunk/s,
toàn bộ part embed trong lượt này):

| # | Bước | Tài nguyên | Thời gian dự kiến | Điều kiện xong |
|---|---|---|---|---|
| 0 | Kiểm tra fail-fast + nạp state + kiểm fingerprint + nạp model | CPU, GPU | 2–4 phút | mọi kiểm tra đạt |
| 1 | Chunk + tokenize (`tokenize_workers=6`, tối đa 3 part đi trước GPU) | CPU | chồng với 3 | mọi part `chunked` |
| 2 | BM25 pyvi (`bm25_workers=10`) → `bm25_avgdl` | CPU | 15–45 phút, chồng với 3 | mọi part `bm25`, `bm25_avgdl` ghi state |
| 3 | Embed (GPU, fp16, `embed_token_budget=32768`) | GPU | **~7,7–7,9h** | mọi part `embedded` |
| 4 | Upload Qdrant gRPC (`qdrant_upload_parallel=8`), `wait=False`, kiểm đếm theo part | Docker CPU | chồng với 3; đuôi ~5 phút | mọi part `uploaded` |
| 5 | Chờ HNSW (xây song song khi nạp) tới `green` (tuần tự) | Docker CPU | 10–20 phút đuôi | `status=green`, `indexed_vectors ≥ 0,99·points` |
| 6 | Alias `vre_chunks` → `vre_chunks_e5_large`; `count = Σ n_chunks`; `smoke_search` 3 truy vấn; manifest | — | 2–5 phút | count khớp, mỗi truy vấn ≥ 5 hit |
| 7 | PCA 32 chiều (CPU) | CPU | ~10–20 phút | `pca_done=true`, file §4 có đủ `n_listings` dòng |
| 8 | Ghi `results/night_report.json` (luôn, cả khi lỗi) | — | < 1 phút | file hợp lệ §7.5 |
| | **Tổng** | | **~8,3h** | |

Xử lý lỗi (tự động, không cần người):

| Sự cố | Phát hiện | Hành động | Kết quả trong report |
|---|---|---|---|
| CUDA OOM | `torch.cuda.OutOfMemoryError` | `empty_cache`, ngân sách token ½, làm lại lô; < 4096 token vẫn OOM ⇒ dừng mã 20 | `counters.oom_retries`, `counters.batch_size_final` |
| Qdrant lỗi/timeout | exception client | tenacity 6 lần, backoff mũ 2–120s; > 2 phút không phản hồi ⇒ `docker compose -f ... up -d --wait qdrant` một lần; vẫn lỗi ⇒ mã 20 | `counters.upload_retries`, `errors[]` |
| Part upload thiếu điểm | `count(listing_id ∈ [lo,hi]) ≠ n_chunks` | upsert lại part (point_id tất định ⇒ không nhân bản) | `errors[]` nếu sau 2 lần vẫn lệch |
| File part hỏng (crash giữa lúc ghi) | ghi `.tmp` + `os.replace`; khi resume kiểm `npy.shape[0] = n_chunks` | làm lại đúng part đó | — |
| Ổ D trống < 5 GB | kiểm trước mỗi part | checkpoint, mã 20 (không xoá gì tự động) | `errors[]` |
| Mất điện / khởi động lại / crash tiến trình | lần chạy sau thấy `runs[-1].ended_at` rỗng | tiếp tục từ state; không làm lại part đã đánh dấu | `resumed_runs` |
| Fingerprint bảng sạch ≠ state (p1 ingest lại) | lúc khởi động | dừng mã 20 với thông báo (không trộn chỉ mục) ⇒ `INDEX_FAILED`, `NOTE` cho p1 | `errors[]` |

Không bước nào gọi LLM hay mạng ngoài loopback Docker. Mã thoát: 0 = completed, 20 = lỗi cần người.

### Bước 6 — Chờ `P1_DONE`, rebase lên `main`, chạy định giá `full` (máy ~45–75 phút)
1. **Chờ `P1_DONE`** (thường đã có từ trước khi lượt dài xong). `git -C D:\Vietnam-Real-Estates-p2 fetch` rồi `git rebase main` (nhánh `main` đã chứa p1).
2. Chạy toàn bộ test trong worktree p2: `& $VRE_PY -m pytest -q` (cả `integration`, `gpu`) = 0 fail.
3. Chạy: `& $VRE_PY scripts\vre_pipeline.py price --variant full` (code của p1, GPU `xgb_device_full=cuda`, đọc PCA từ Bước 5) ⇒ `valuation_metrics.json.variants.full` có số.
4. Commit pathspec `results/valuation_metrics.json`.

### Bước 7 — Merge + bàn giao (agent ~0,5h) → **gửi `P2_DONE`**
1. Push `feat/p2-index`, mở PR "Phase 2 index + full run" (KHÔNG commit `index/`, `data/`, `models/`, `docker-data/`).
2. `AskUserQuestion` xin duyệt merge vào `main`; được duyệt ⇒ merge, cập nhật checkout chính, chạy `& $VRE_PY -m pytest -q` toàn bộ trên `main` = 0 fail.
3. **Gửi `P2_DONE`** `{"branch":"feat/p2-index","sha":"<sha main sau merge>","valuation_full_mape":<số>}`.
4. `git worktree remove D:\Vietnam-Real-Estates-p2`; `git branch -D feat/p2-index`. Dừng phiên — không làm Phase 3.

## Verify (bằng lệnh, không dựa vào "có vẻ ổn")

```powershell
& $VRE_PY -c "import json;r=json.load(open(r'D:\Vietnam-Real-Estates\results\night_report.json',encoding='utf-8'));print(r['status'],r['counters']['parts_uploaded'],r['counters']['parts_total'],r['qdrant'])"
curl -s http://127.0.0.1:6333/collections/vre_chunks | & $VRE_PY -c "import sys,json;d=json.load(sys.stdin)['result'];print(d['status'],d['points_count'],d['indexed_vectors_count'])"
& $VRE_PY scripts\vre_pipeline.py smoke-search "căn hộ 2 phòng ngủ quận 7 dưới 4 tỷ"
& $VRE_PY -m pytest -q -m integration tests/integration/test_index_complete.py
& $VRE_PY -c "import json;v=json.load(open(r'D:\Vietnam-Real-Estates\results\valuation_metrics.json',encoding='utf-8'))['variants'];print(v['full']['mape'],v['baseline_ward_median']['mape'])"
```
`test_index_complete.py`: alias tồn tại; `points_count = Σ n_chunks` từ state; mọi `listing_id` của 1.000 tin ngẫu nhiên có point chunk 0;
dense search + BM25 + filter `province` trả kết quả; số chiều vector = `EMBED_MODELS["e5_large"]` dim = 1024.

Đọc `status` khi đang chạy / sau khi xong:

| Thấy | Ý nghĩa | Làm gì |
|---|---|---|
| `status=completed`, HNSW `green`, alias đặt, `pca_done` | Xong chỉ mục | `INDEX_DONE`, sang Bước 6 |
| Đang chạy (`runs[-1].ended_at` rỗng, log còn ghi) | Chưa xong | Đợi lệnh nền |
| `status=failed` | Lỗi không tự phục hồi sau 3 lần | `INDEX_FAILED`; đọc `errors[]` + log; sửa (test), chạy lại ĐÚNG lệnh |

## Tiêu chí hoàn thành

- [ ] `INDEX_STARTED`, `INDEX_DONE`, `P2_DONE` có trong `results/handoff/` và đã gửi tin cho `p1`.
- [ ] Unit + integration + gpu test của p2 xanh (gồm `test_embed_parity` cosine ≥ 0,999); ruff 0 lỗi; không file > 200 dòng.
- [ ] Dry-run 2 part completed, resume không làm lại part đã xong, đã dọn `vre_dryrun`.
- [ ] `night_report.status = completed` (qua nhiều lần chạy lại được phép; `resumed_runs` ghi số lần); report đúng §7.5.
- [ ] Qdrant alias `vre_chunks` → `vre_chunks_e5_large`; `status=green`; `points_count = Σ n_chunks`; `test_index_complete.py` xanh; `smoke-search` ≥ 5 tin.
- [ ] `valuation_metrics.variants.full` có số (MAE/MAPE/R²) và được so với `baseline_ward_median`.
- [ ] Không có file mới trên C do pipeline (kiểm `%LOCALAPPDATA%\Temp`, `%APPDATA%\NVIDIA\ComputeCache` không có mtime trong lượt chạy); dữ liệu Qdrant trong `D:\Vietnam-Real-Estates\docker-data\`.
- [ ] Nhánh p2 đã merge vào `main` sau p1; toàn bộ test xanh trên `main`.

## Rủi ro

| Rủi ro | L | I | Điểm | Giảm thiểu |
|---|---|---|---|---|
| Lỗi code chỉ lộ ở quy mô lớn | 2 | 3 | 6 | Dry-run 2 part + kiểm part đầu của lượt thật; checkpoint nên sửa xong chỉ chạy phần còn lại |
| Encoder tự viết lệch sentence-transformers | 2 | 4 | 8 | `test_embed_parity` cosine ≥ 0,999 trước dry-run |
| Embed chậm hơn số 2026-10-15 (nhiệt GPU laptop, CPU chia với p1) | 3 | 2 | 6 | Không có giới hạn thời gian ⇒ chỉ kéo dài lịch; checkpoint mỗi part |
| Qdrant chậm/lỗi trên bind mount `docker-data\qdrant` | 2 | 4 | 8 | Round-trip của p1 + dry-run trên đúng bind mount; retry + 1 lần `compose up`; upsert tất định |
| Tiến trình bị ngắt giữa chừng | 2 | 3 | 6 | Checkpoint nguyên tử sau mỗi part mỗi stage; `night.ps1` tự chạy lại; chạy lại cùng lệnh để tiếp tục |
| CUDA OOM ở lô dài | 2 | 2 | 4 | Ngân sách token ½ rồi thử lại; budget 32768 đỉnh ~2GB VRAM ở benchmark |
| p1 ghi lại bảng sạch giữa lượt chạy | 1 | 5 | 5 | Bất biến coordination §8 + kiểm fingerprint ⇒ dừng mã 20, không trộn chỉ mục |
| Chạy nhầm code của checkout p1 (editable install) | 2 | 4 | 8 | Không cài editable; `vre_pipeline.py` tự thêm `src` của chính worktree vào `sys.path` (coordination §7) |
| HNSW build lâu hơn dự kiến | 2 | 2 | 4 | Chờ tối đa 3h; chưa green ⇒ `failed` có thông tin, chạy lại lệnh chỉ chờ tiếp |
| Ổ D đầy | 1 | 4 | 4 | Kiểm ≥ 45GB đầu lệnh, ≥ 5GB mỗi part; ước tính dùng ~37GB |

## Timeline (chỉ các bước của phase này)

| Bước | Agent | Máy | Phụ thuộc | Tín hiệu |
|---|---|---|---|---|
| 0 chuẩn bị + worktree | 0,2h | — | — | — |
| 1 code chỉ mục | 3,5h | — | 0 | — |
| 2 code lượt dài | 3h | — | 1 | — |
| 3 rebase + test | 2h | 10 phút | 2 + chờ `SKELETON_READY` | gửi `CHANGE_REQUEST` nếu cần |
| 4 dry-run 2 part | 0,3h | ≤ 15 phút | 3 + chờ `CLEAN_DATA_READY` | — |
| 5 lượt dài | 0,3h | ~8,3h | 4 | gửi `INDEX_STARTED` → `INDEX_DONE`/`INDEX_FAILED` |
| 6 định giá `full` | 0,3h | 45–75 phút | 5 + chờ `P1_DONE` | — |
| 7 merge + bàn giao | 0,5h | 10 phút | 6 | gửi `P2_DONE` |
| **Tổng** | **~10 giờ agent** | **~9,5 giờ máy** | Đường găng của p2: 1 → 2 → 3 → 4 → 5 → 6 → 7 | |
