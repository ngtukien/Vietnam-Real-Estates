# Plan: Làm lại Vietnam Real Estates RAG/GraphRAG trên đủ 3,5M tin

Ngày lập: 2026-10-15 · Cập nhật: 2026-10-15 (chốt e5-large; bỏ đo hiệu năng; Phase 1 và 2 chạy song song hai phiên) · Trạng thái: sẵn sàng cho `/t1k:cook`
Phase 1 + Phase 2 cùng lúc
Thiết kế đã duyệt (SSOT quyết định): `plans/reports/brainstorm-261015-0007-full-retrain-redesign.md` (§3 có ghi chú cập nhật 2026-10-15)
Hợp đồng dùng chung: [`data-contract.md`](data-contract.md) (parquet, chunk, Qdrant, Neo4j, `results/*.json`, §9 bề mặt code dùng chung) ·
[`api-contract.md`](api-contract.md) (FastAPI ↔ React, SSE, facade Python) · [`coordination.md`](coordination.md) (hai phiên song song p1/p2:
tín hiệu, file handoff, ownership, git, chia tài nguyên).

## Tóm tắt

Thay hệ thống demo chạy trên 20.000 dòng đầu stream bằng pipeline tái lập được trên toàn bộ 3.500.744 dòng (≈2,7M tin sau dedup), đọc
parquet local, chỉ ghi ổ D. RAG/GraphRAG vẫn là chủ đề chính (Basic RAG · Text2Cypher · Hybrid router · LLM-only baseline); thêm model
định giá XGBoost như module phụ và tool `valuate()`. Mọi con số cho notebook/README/slide sinh từ `results/*.json`.

Ràng buộc chốt 2026-10-15:
- **Embedding duy nhất, cố định, không đo lại: `intfloat/multilingual-e5-large`** (`e5_large`, 1024 chiều, mean pooling, tiền tố `passage: `/`query: `,
  `max_length=512`, `embed_token_budget=32768`, L2, fp16). Số đo đã có trên máy này (2026-10-15): 99 tin/s bền vững (104/s ở budget 32768),
  Recall@1 0,656, Recall@10 0,816, MRR@10 0,711 — chất lượng cao nhất trong 4 model đã đo (`.cache/scratch/bench_final.jsonl`, sao chép thẳng thành
  `results/embedding_benchmark.json`). Không có bước quét budget, cổng GPU util, ETA hay benchmark lại. Không giới hạn thời gian: lượt chạy dài
  chạy tới khi xong (~8,3h chỉ mục + ~1h định giá full), resume bằng cùng lệnh.
- **Toàn bộ máy rảnh và dành riêng cho dự án** (Ryzen 9 7945HX 32 luồng, ~35GB RAM dùng được, RTX 4060 Laptop 8GB). Không có bước người dùng
  chỉnh Docker/hệ điều hành hay đóng ứng dụng. "Dữ liệu chỉ ở ổ D" đạt bằng `envguard` (cache/TMP) + bind mount Qdrant/Neo4j vào
  `D:\Vietnam-Real-Estates\docker-data\{qdrant,neo4j}` (gitignored). Khi Phase 1 và 2 chạy cùng lúc, máy được chia theo coordination §8 (GPU chỉ p2).
- **Phase 1 và Phase 2 chạy song song** trong hai phiên Claude Code (`p1`, `p2`), giao tiếp bằng `SendMessage` + file handoff
  `results/handoff/<SIGNAL>.json` (coordination.md).

## Phases

| # | Phase | Phiên / khi chạy | Phạm vi / file chính | Effort agent | Máy chạy |
|---|---|---|---|---|---|
| 1 | [Foundation](phase-01-foundation.md) | phiên `p1`, song song với Phase 2 | khung `src/vre` + `vre/config.py` + `vre/checks.py` + `scripts/vre_pipeline.py` + `pyproject.toml` + `requirements/*` + `docker-compose.yaml` (bind mount `docker-data/`) + `.gitignore`; `vre/data/**` (ingest/clean/dedup/EDA), `vre/index/graph_*`, `vre/graph/**`, `graph/post_import.cypher`, `vre/valuation/**` (tabular trên CPU), di chuyển module cũ, test của các module này | **L** (~17 giờ) | ~1,5–2 giờ |
| 2 | [Pipeline index + lượt chạy dài e5-large](phase-02-overnight-run.md) | phiên `p2`, song song với Phase 1 | viết `vre/index/{chunk,tokenize,embed,bm25,qdrant_store,qdrant_upload}.py`, `vre/pipeline/{state,stages,night,pca,report}.py`, `scripts/night.ps1` + test; dry-run 2 part; chạy lượt dài → `index/**`, Qdrant `vre_chunks_e5_large` + alias, PCA, `results/night_report.json`; định giá `full` (GPU) | **M** (~10 giờ) | ~9,5 giờ (embed ~7,7–7,9h + đuôi HNSW/verify/PCA ~0,5h + dry-run ~0,25h + định giá full ~1h) |
| 3 | [System + eval + webapp](phase-03-system-eval-webapp.md) | ban ngày, sau Phase 1 + 2 | `src/vre/{rag,graph,hybrid,llm,qa,eval}/**`, `benchmark/questions.jsonl`, `api/**`, `web/**`, Dockerfiles, `notebooks/01..05`, `README.md`, `.github/workflows/ci.yml` | **L** (~68 giờ, 2,5–3 ngày với 4 lane) | 3–4 giờ |
| 4 | [Slides](phase-04-slides.md) | phiên riêng | `slide/*.pptx`, `slide/*.html`, `scripts/slide_numbers.py`, `slide/numbers_map.yaml`, `slide/assets/**` | **M** (~13 giờ) | < 10 phút |

## Thứ tự phụ thuộc

```
            ┌── p1: khung ──SKELETON_READY──┐
Phase 1 ────┤   ingest ──CLEAN_DATA_READY───┼──┐   EDA · KG · định giá tabular (CPU) · test ── merge ──P1_DONE──┐
            │                               │  │                                                              │
Phase 2 ────┴── p2: viết code index ── rebase+test ── dry-run ──INDEX_STARTED── lượt dài ~8,3h ──INDEX_DONE──┤
                                                                                                             ▼
                                                    p2: rebase main → định giá full (GPU) → merge ──P2_DONE──► Phase 3 ──► Phase 4
```
- Phase 1 ∥ Phase 2: hai phiên, thứ tự nội bộ do tín hiệu quyết định (coordination §3). p2 viết code ngay từ đầu; test sau `SKELETON_READY`;
  chạy dry-run + lượt dài chỉ sau `CLEAN_DATA_READY`; định giá `full` chỉ sau `P1_DONE` (cần code định giá của p1 đã merge).
- Phase 3 bắt đầu khi có đủ `P1_DONE`, `INDEX_DONE`, `P2_DONE` trong `results/handoff/` (định giá `full` đã xong, `main` chứa cả hai nhánh). Phase 4 sau Phase 3.
- Trong Phase 3: fan-out 4 lane (lõi QA+eval · API · web · notebook/README/CI) trong `git worktree` riêng, nối qua `api-contract.md`.
- Đường găng tổng: p1 khung → p1 ingest → p2 dry-run → p2 lượt dài (embed ~7,8h) → p2 định giá full (sau `P1_DONE`) → P3 lõi QA → benchmark → eval → notebook 04 → tích hợp → P4.
- Sở hữu file xuyên phase: file dùng chung (`scripts/vre_pipeline.py`, `src/vre/config.py`, `src/vre/checks.py`, `pyproject.toml`, `requirements/*`,
  `docker-compose.yaml`, `.gitignore`) chỉ p1 sửa trong Phase 1–2 (p2 gửi `CHANGE_REQUEST`); sau `P1_DONE` chuyển cho p2; Phase 3 nhận sau khi
  cả hai đã merge. Bảng đầy đủ: coordination §5.

## Feasibility

- Reuse: code hiện có (`src/*.py`, `scripts/index_full.py`, `app/server.py`, `.cache/scratch/bench_final.py`, `profile.py`) cung cấp BM25 sparse,
  int8 + rescore, Text2Cypher guard 2 lớp, router + fallback, judge/RAGAS, checkpoint nguyên tử, rate limiter — được di chuyển/tái dùng (chi tiết ở
  mục "Prior art" từng phase). Số đo embedding đã có sẵn (`.cache/scratch/bench_final.jsonl`) — không benchmark lại. MỚI: DuckDB ingest + SimHash,
  neo4j-admin import + MarketStat, XGBoost, kiểm tra fail-fast mức dự án, CLI Typer, giao thức handoff hai phiên, benchmark SQL, bootstrap, FastAPI SSE, React app.
- Độ phức tạp: complex (dữ liệu lớn trên một máy 8GB VRAM, lượt chạy dài không người song song với phiên khác, 4 hệ thống + webapp).

## Tiêu chí thành công (report §9, đo được)

| Tiêu chí | Lệnh / bằng chứng |
|---|---|
| Pipeline `vre_pipeline.py all --resume` chạy hết 3,5M từ parquet local, tái lập, không ghi C | `results/{ingest_report,graph_report,embedding_benchmark,night_report}.json` đủ; chạy lại `all --resume` in "đã xong" mọi bước; kiểm Temp/NVIDIA cache trên C không có file mới; dữ liệu Qdrant/Neo4j trong `docker-data/` |
| Hai phiên phối hợp không mất tín hiệu, không đè file | `results/handoff/` có đủ `SKELETON_READY`…`P2_DONE`; `git log` hai nhánh không có commit chạm file của bên kia (coordination §5) |
| Notebook 04 + mọi số trên slide sinh từ `results/*.json`; không còn `[ĐIỀN SAU]` | `pytest tests/test_notebooks.py`; `scripts/slide_numbers.py check` = 0 unmapped |
| Benchmark ~100 câu có GT SQL; báo cáo có CI và ablation | `benchmark/questions.jsonl` 100 dòng/9 nhóm; `eval_summary.json.by_category[*][*].ci95`; `ablation.json.runs` |
| Hybrid > Basic RAG > LLM-only có ý nghĩa thống kê ở aggregate/multi-hop, HOẶC báo cáo trung thực nếu không | `eval_summary.json.pairwise` (diff, ci95, p_value) |
| Model định giá vượt baseline median-theo-phường | `valuation_metrics.json`: `variants.full.mape` (và `tabular`) < `variants.baseline_ward_median.mape` |
| Webapp: stream, so sánh 4 hệ, bản đồ, định giá, dashboard; Playwright E2E xanh; CI xanh | `npm --prefix web run e2e`; `gh run list --branch <branch> --limit 1` = success (4 job) |

## Risk Assessment

| Risk | Likelihood (1-5) | Impact (1-5) | Score | Mitigation |
|------|-----------------|--------------|-------|------------|
| Tín hiệu giữa hai phiên bị lỡ (tin bị giữ chờ duyệt, phiên đóng) ⇒ một bên chờ mãi | 3 | 3 | 9 | Mỗi tín hiệu ghi file `results/handoff/<SIGNAL>.json` TRƯỚC khi gửi tin; bên chờ thăm dò file ≤ 5 phút bằng lệnh nền (coordination §4) |
| Xung đột file / đè việc của nhau giữa p1 và p2 | 2 | 4 | 8 | Ownership map không giao nhau (coordination §5); file dùng chung chỉ p1 sửa, p2 gửi `CHANGE_REQUEST`; mỗi phiên một worktree + nhánh, commit pathspec |
| Tranh tài nguyên khi hai phiên chạy cùng lúc (GPU, CPU, RAM) | 2 | 3 | 6 | GPU chỉ p2; XGBoost tabular `device=cpu`; chia luồng CPU và RAM DuckDB/Neo4j cố định (coordination §8); ingest của p1 xong trước khi lượt dài bắt đầu |
| Hai worktree chạy hai bộ container / đọc nhầm dữ liệu | 2 | 4 | 8 | Compose `name: vre`, luôn `-f D:\Vietnam-Real-Estates\docker-compose.yaml`; mọi đường dẫn dữ liệu tuyệt đối dưới `VRE_DATA_ROOT`; không cài editable (coordination §7) |
| p1 ghi lại bảng sạch khi p2 đang chạy dài | 1 | 5 | 5 | Bất biến coordination §8; fingerprint lệch ⇒ lượt chạy dừng mã 20, không trộn chỉ mục |
| Qdrant/Neo4j chậm hoặc lỗi trên bind mount `docker-data/` | 3 | 4 | 12 | Phase 1 Bước 1 round-trip (ghi → restart → đọc) trước `SKELETON_READY`; Phase 2 dry-run trên đúng bind mount; fail ⇒ dừng, `AskUserQuestion` |
| Lỗi code chỉ lộ ở quy mô lớn | 2 | 3 | 6 | Dry-run 2 part + test parity encoder trước lượt dài; kiểm part đầu; checkpoint mỗi part nên sửa xong chỉ chạy phần còn lại |
| Lượt chạy dài bị ngắt (mất điện, khởi động lại, crash) | 2 | 3 | 6 | Giữ máy thức trong tiến trình; checkpoint nguyên tử mỗi part mỗi stage; `night.ps1` tự chạy lại 3 lần; chạy lại cùng lệnh để tiếp tục |
| Embed chậm hơn số đo 2026-10-15 (nhiệt GPU laptop) | 3 | 2 | 6 | Không có giới hạn thời gian ⇒ chỉ kéo dài lịch |
| Thiếu RAM khi ingest / chạy dài | 2 | 4 | 8 | Xử lý theo shard/part 50k; DuckDB 12GB + temp trên D; tối đa 3 part đi trước GPU; CUDA OOM ⇒ ½ batch và thử lại |
| Ghi nhầm vào ổ C (cache HF/torch/NVIDIA/pip/npm, TMP, dữ liệu Docker) | 2 | 4 | 8 | `vre_env.ps1` + `envguard` raise + kiểm `env_paths_on_d`; Qdrant/Neo4j bind mount trên D; kiểm Temp ở tiêu chí hoàn thành |
| Rate limit Groq khi eval (~2.800 lời gọi) | 4 | 3 | 12 | Cache LLM, 4 luồng, backoff, resume theo dòng, smoke 5 câu trước |
| GT benchmark sai ⇒ kết luận sai | 3 | 4 | 12 | GT bằng SQL lưu kèm, test cấu trúc, người dùng duyệt 10 câu, error analysis |
| Cypher tổng hợp timeout trên 2,7M node | 2 | 3 | 6 | MarketStat + thuộc tính tổng hợp tính sẵn, few-shot dùng chúng, timeout 10s giữ nguyên |
| Lệch hợp đồng giữa lane API/web/lõi | 3 | 3 | 9 | `api-contract.md`/`data-contract.md` là SSOT; `types.ts` sinh từ OpenAPI; test hợp đồng + fixture validate |
| LLM chép sai id 16 chữ số trong `[Tin#id]` | 3 | 3 | 9 | Đo `bad_citation_rate`; regex chuẩn hoá; báo cáo trung thực |
| Địa giới 2025 (63→34) gây nhầm | 2 | 2 | 4 | Giữ địa giới cũ của dataset + cột `province_2025`; câu unanswerable kiểm hành vi |
| License CC-BY-NC-4.0 (dataset), CC BY 4.0 (geoBoundaries) | 1 | 3 | 3 | Ghi nguồn README + footer web; chỉ phi thương mại |
| Ước lượng effort vượt (phase lớn) | 4 | 3 | 12 | Phase 1 ∥ Phase 2; fan-out lane có file ownership riêng; ưu tiên đường găng chủ đề RAG/GraphRAG trước tính năng web nâng cao |

Không còn rủi ro ≥ 15. Rủi ro hạ tầng cao nhất (bind mount, 12) được kiểm bằng lệnh ở Phase 1 Bước 1 trước khi p2 đầu tư vào lượt chạy dài.

## Rollback

- Phase 1: nhánh `feat/p1-foundation`; code cũ nằm trong git history (`b4a7663`). Dữ liệu sinh ra (`data/clean`, `models`, `docker-data/neo4j`) xoá được,
  tái tạo bằng CLI. Volume Docker cũ không bị xoá.
- Phase 2: nhánh `feat/p2-index` trong worktree `D:\Vietnam-Real-Estates-p2`; chỉ tạo collection mới `vre_chunks_e5_large` + alias; xoá collection/alias
  là quay lại; checkpoint cho phép làm lại từng part. Merge vào `main` theo thứ tự p1 → p2 (coordination §6) nên revert được từng nhánh.
- Phase 3: mỗi lane một worktree/nhánh; revert theo PR. `results/*.json` sinh lại bằng `vre_pipeline.py eval all`.
- Phase 4: sao lưu deck trong `slide/_backup/` trước khi sửa.

## Timeline (xuyên phase — file phase chỉ có timeline của riêng nó)

| Phase | Effort | Notes |
|-------|--------|-------|
| Phase 1 (p1) | L — ~17 giờ agent · ~1,5–2 giờ máy | Chạy song song Phase 2; gửi `SKELETON_READY` sau ~2h, `CLEAN_DATA_READY` sau ~5,5h |
| Phase 2 (p2) | M — ~10 giờ agent · ~9,5 giờ máy | Song song Phase 1; lượt dài bắt đầu sau `CLEAN_DATA_READY` + dry-run; định giá `full` sau `P1_DONE` |
| Phase 3 | L — ~68 giờ agent · 3–4 giờ máy | Blocked by `P1_DONE` + `INDEX_DONE` + `P2_DONE`; 4 lane song song |
| Phase 4 | M — ~13 giờ agent · ~1 giờ người | Blocked by Phase 3 results + quyết định file HTML |
| **Total** | **~108 giờ agent + 1 lượt chạy dài ~8,3 giờ** (≈ 5–6 ngày lịch) | Critical path: p1 khung → p1 ingest → p2 dry-run → p2 lượt dài → p2 định giá full → P3 lõi QA → benchmark → eval → notebook 04 → P4 |

Lịch song song Phase 1 + 2: wall-clock ≈ max(việc của p1 ≈ 19h, chờ khung + ingest ≈ 5,5h + dry-run ~0,3h + lượt dài ~8,3h ≈ 14h)
+ đuôi của p2 sau `P1_DONE` (rebase, test, định giá full, merge ≈ 1,5h) ≈ **~20 giờ**, thay vì tổng tuần tự ≈ 38 giờ (19h p1 + 19,5h p2).
Phần lớn thời gian chạy dài của p2 nằm gọn trong lúc p1 làm EDA/KG/định giá/test.

## Quyết định bổ sung

Quyết định được chốt sau ngày lập plan ghi tại đây (ngày + nội dung + người chốt). Phase 4 bước 0 đọc mục này.
Dòng có "(thay bằng …)" là lịch sử, không còn hiệu lực.

| Ngày | Quyết định | Người chốt |
|---|---|---|
| 2026-10-15 | File `slide/hoc-phan-4-graphrag.html` được **giữ và tái sinh** từ `results/*.json` (mọi số + mảng JS sinh tự động, `slide_numbers.py html`) | user |
| 2026-10-15 | ~~Chốt embedding đề xuất ban đầu ở brainstorm §3 (budget 16384, cổng lùi model theo dự báo giờ)~~ **(thay bằng e5-large — xem dòng "Chốt e5-large" bên dưới)** | user |
| 2026-10-15 | ~~Loại bản fine-tune tiếng Việt nêu ở brainstorm §3; ứng viên chỉ gồm các model đã benchmark~~ **(thay bằng e5-large: chỉ còn một model)** | user |
| 2026-10-15 | **Commit** `results/*.json` + ảnh `results/eda/*.png` (bỏ khỏi `.gitignore`) và **commit slide** (pptx + html) lần đầu ở Phase 4 | user |
| 2026-10-15 | **Không theo dõi trên Plane** (dự án môn học cá nhân, workspace không có project phù hợp). Chạy cook với `--no-plane` | user |
| 2026-10-15 | ~~Lượt chạy kết thúc `partial` ⇒ chạy lại ngay cùng lệnh~~ **(thay bằng: không còn giới hạn thời gian; chưa xong/lỗi thì chạy lại cùng lệnh để resume)** | user |
| 2026-10-15 | **Chốt e5-large, không giới hạn thời gian.** Embedding duy nhất `intfloat/multilingual-e5-large` (`e5_large`, 1024d, mean pooling, `passage: `/`query: `, `max_length=512`, `embed_token_budget=32768`, L2, fp16); collection `vre_chunks_e5_large`, emb `index/emb/e5_large/`. Đo 2026-10-15 (20k tin, 1.000 truy vấn): 99/s bền vững (104/s @32768), R@1 0,656, R@10 0,816, MRR@10 0,711 — tốt nhất 4 model đo (`.cache/scratch/bench_final.jsonl` → `results/embedding_benchmark.json`). Model khác đã xoá khỏi cache, không tải lại. Bỏ hạn giờ, bỏ chọn model theo dự báo; định giá full luôn chạy sau chỉ mục; resume bằng cùng lệnh. ~~Bước "cổng 50k" thành hiệu chỉnh/dry-run (quét budget, GPU util ≥ 90%, ETA)~~ **(thay bằng dòng "Bỏ đo hiệu năng" bên dưới)** | user |
| 2026-10-15 | **Không cấu hình máy** (thiết lập đĩa Docker, bộ nhớ ảo, cấu hình WSL, đóng ứng dụng khác, cập nhật hệ điều hành, checklist nguồn/nắp máy…): toàn bộ máy rảnh và dành riêng cho dự án; kiểm tra chỉ ở mức dự án; bỏ watchdog RAM. **Dữ liệu Docker bind-mount vào D** (`D:\Vietnam-Real-Estates\docker-data\{qdrant,neo4j}`, gitignored) thay cho named volume. Giữ `envguard`, keep-awake `SetThreadExecutionState` trong tiến trình, CUDA OOM ⇒ ½ batch | user |
| 2026-10-15 | **Bỏ đo hiệu năng; e5-large cố định token_budget 32768.** Bỏ bước hiệu chỉnh (quét `embed_token_budget`, cổng GPU util ≥ 90%, ETA/dự báo giờ), `results/run_plan.json`, module theo dõi GPU và mọi bước đo lại/benchmark; cấu hình model đọc từ `vre.config`. `results/embedding_benchmark.json` là bản sao thẳng số đo đã có. GIỮ: dry-run 2 part trước lượt dài (bắt lỗi code; thông lượng chỉ là dòng thông tin), test parity encoder, mọi kiểm tra fail-fast (gộp vào đầu lệnh qua `vre.checks`), mọi bước tính năng/đầu ra và test | user |
| 2026-10-15 | **Phase 1 và 2 chạy song song 2 phiên** (`p1`, `p2`), giao tiếp `SendMessage` + file handoff `results/handoff/<SIGNAL>.json` (thăm dò ≤ 5 phút); mỗi phiên một worktree/nhánh, ownership không giao nhau, file dùng chung chỉ p1 sửa; GPU chỉ p2, XGBoost tabular trên CPU; merge p1 → `main` rồi p2 → `main`. SSOT: `coordination.md` | user |
