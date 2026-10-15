# Vietnam Real Estates — từ RAG đến GraphRAG

Trợ lý hỏi đáp thị trường bất động sản Việt Nam trên toàn bộ dataset tin đăng
[tinixai/vietnam-real-estates](https://huggingface.co/datasets/tinixai/vietnam-real-estates).
Dự án so sánh bốn cách trả lời trên cùng dữ liệu, cùng bộ câu hỏi và cùng mô hình ngôn ngữ:

| Hệ thống | Cách tìm bằng chứng | Mạnh ở |
| --- | --- | --- |
| **LLM only** | Không truy xuất; cùng mô hình, không có ngữ cảnh | Baseline: đo truy xuất thêm được gì |
| **Basic RAG** | LLM trích bộ lọc → dense (`multilingual-e5-large`) + BM25 trên Qdrant, gộp RRF, lọc trước theo payload → LLM trả lời có `[Tin#id]` | Tìm theo mô tả tự do: "gần trường", "có thang máy" |
| **Graph** | Text2Cypher: LLM sinh Cypher chỉ đọc trên đồ thị tri thức Neo4j → LLM diễn giải kết quả | Câu tổng hợp, đếm, xếp hạng, đa chặng |
| **Hybrid** | Router chọn Basic RAG, Graph, định giá hoặc gộp ngữ cảnh; đồ thị lỗi hoặc rỗng thì lui về Basic RAG | Câu vừa có điều kiện cấu trúc vừa có mô tả |

RAG/GraphRAG là chủ đề chính. Module định giá (XGBoost dự đoán giá/m²) là phần phụ, dùng làm tool `valuate()` cho router.

## Mục lục

- [Kiến trúc](#kiến-trúc)
- [Dữ liệu và giấy phép](#dữ-liệu-và-giấy-phép)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)
- [Pipeline](#pipeline)
- [Đánh giá](#đánh-giá)
- [API, web và Docker](#api-web-và-docker)
- [Notebook](#notebook)
- [Kết quả](#kết-quả)
- [Kiểm thử và CI](#kiểm-thử-và-ci)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Hạn chế](#hạn-chế)

## Kiến trúc

```mermaid
flowchart LR
  raw[Parquet thô] --> duck[DuckDB: làm sạch, khử trùng, bảng sạch]
  duck --> chunk[Chunk + BM25]
  chunk --> emb[Embedding multilingual-e5-large]
  emb --> qd[(Qdrant: dense + sparse)]
  duck --> kg[(Neo4j: đồ thị tri thức)]
  duck --> xgb[XGBoost định giá]
  q[Câu hỏi] --> router{Router}
  router -->|basic_rag| qd
  router -->|graph| kg
  router -->|hybrid| qd & kg
  router -->|valuation| xgb
  qd & kg & xgb --> llm[LLM sinh câu trả lời có trích dẫn]
  llm --> api[FastAPI SSE]
  api --> web[Web React]
```

- **Bảng sạch** (`data/clean/listings.parquet`): DuckDB đọc các shard thô, chuẩn hoá địa danh, sửa giá nhập theo nghìn đồng, ẩn số điện thoại
  và email, khử trùng chính xác + trùng tiền tố, gom tin gần trùng bằng SimHash (giữ lại, đánh dấu `dup_cluster_id`).
- **Chỉ mục vector**: collection Qdrant `vre_chunks_e5_large` (alias `vre_chunks`), mỗi điểm có vector dense và vector sparse BM25
  (tách từ `pyvi`), payload index cho lọc trước. Lý do chọn `multilingual-e5-large` ở notebook `02_rag` (đọc `results/embedding_benchmark.json`).
- **Đồ thị tri thức**: `Listing` gắn với `Ward`/`District`/`Province`, `Street`, `Project`, `PropertyType`, `Month`, `Amenity`; `MarketStat`
  là thống kê giá tính sẵn theo vùng × loại × tháng; `Cluster` là cộng đồng Leiden của các phường. Text2Cypher chạy trong giao dịch chỉ đọc,
  có chặn regex, `LIMIT` và timeout.
- **LLM**: Groq (API tương thích OpenAI), structured output cho bộ lọc, Cypher, router và chấm điểm; cache đĩa theo nội dung prompt.
  Mô hình chấm (`JUDGE_MODEL`) phải khác mô hình sinh (`LLM_MODEL`).

## Dữ liệu và giấy phép

- Dataset: [tinixai/vietnam-real-estates](https://huggingface.co/datasets/tinixai/vietnam-real-estates), giấy phép
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (chỉ dùng phi thương mại, ghi công tác giả dataset).
  Kết quả và mô hình sinh từ dữ liệu này thừa hưởng điều kiện phi thương mại.
- Ranh giới hành chính cho bản đồ: [geoBoundaries](https://www.geoboundaries.org/) gbOpen VNM ADM1/ADM2,
  giấy phép [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (địa giới trước sắp xếp năm `2025`). Sinh lại bằng
  `python scripts/build_geo.py`; nguồn được ghi công ở chân trang web.
- Giá trong dữ liệu là giá chào bán do người đăng tự nhập, không phải giá giao dịch.

## Yêu cầu môi trường

- Windows, mọi thứ trên ổ **D** (dữ liệu, cache Hugging Face/pip/torch, TMP, npm, Playwright). Trước mọi lệnh, nạp biến môi trường:

  ```powershell
  . D:\Vietnam-Real-Estates\scripts\vre_env.ps1   # đặt VRE_DATA_ROOT, HF_HOME, TMP... về D:\Vietnam-Real-Estates\.cache và tạo $VRE_PY
  ```

- Python `3.13` trong `.venv`; GPU CUDA cho bước embedding và biến thể định giá `full`.
- Docker Desktop: Qdrant và Neo4j chạy bằng `docker-compose.yaml` (project `vre`). Dữ liệu dịch vụ nằm trong named volume
  (đĩa WSL của Docker Desktop đặt trên ổ D); log Neo4j bind mount ra `docker-data/`.

```powershell
& $VRE_PY -m pip install -r requirements/ml.txt        # torch CUDA: cài wheel từ index PyTorch trước (xem đầu file)
& $VRE_PY -m pip install -r requirements/dev.txt       # pytest, ruff, mypy, nbclient, ipykernel
Copy-Item .env.example .env                            # điền GROQ_API_KEY, NEO4J_PASSWORD
docker compose -f D:\Vietnam-Real-Estates\docker-compose.yaml up -d --wait qdrant neo4j
```

Test không đọc `.env` (`VRE_LOAD_DOTENV=0` trong `tests/conftest.py`), nên chạy được trên máy không có khoá.

## Pipeline

Mọi bước là một lệnh của `scripts/vre_pipeline.py`, idempotent (artifact đã có và fingerprint khớp thì in "đã xong" rồi thoát), và
kiểm điều kiện tiên quyết trước khi chạy (thiếu shard, thiếu model trong cache, dịch vụ chưa sẵn sàng, ổ D đầy… thì dừng ngay).

| Lệnh | Làm gì | Ghi ra |
| --- | --- | --- |
| `& $VRE_PY scripts/vre_pipeline.py ingest` | Nạp, làm sạch, khử trùng | `data/clean/`, `results/ingest_report.json`, `results/embedding_benchmark.json` |
| `& $VRE_PY scripts/vre_pipeline.py eda` | Thống kê mô tả + hình | `results/eda_summary.json`, `results/eda/*.png` |
| `& $VRE_PY scripts/vre_pipeline.py graph` | Nạp đồ thị Neo4j bằng `neo4j-admin import` | `results/graph_report.json` |
| `& $VRE_PY scripts/vre_pipeline.py price --variant tabular` | Định giá trên đặc trưng cấu trúc (CPU) | `models/valuation/`, `results/valuation_metrics.json` |
| `& $VRE_PY scripts/vre_pipeline.py night --resume` | Chunk → BM25 → embed → upload Qdrant → HNSW → kiểm tra → PCA | `index/`, `results/night_report.json` |
| `& $VRE_PY scripts/vre_pipeline.py price --variant full` | Định giá thêm vector mô tả nén PCA (GPU) | `results/valuation_metrics.json` |
| `& $VRE_PY scripts/vre_pipeline.py all --resume` | Toàn bộ chuỗi trên theo thứ tự | |
| `& $VRE_PY scripts/vre_pipeline.py status` | Tiến độ lượt chạy dài | |

**Lượt chạy dài** (`night`): embed và upload toàn bộ chunk mất nhiều giờ (thời gian thật của lần chạy gần nhất ở
`results/night_report.json` → `wall_hours`, hiển thị trong notebook `02_rag`). Bị ngắt (mất điện, tắt máy, lỗi bộ nhớ) thì chạy lại
**đúng lệnh cũ**: pipeline đọc trạng thái từng phần và tiếp tục từ phần chưa xong. Chạy thử nhỏ trước bằng `--dry-run`.

## Đánh giá

- `benchmark/questions.jsonl`: bộ câu hỏi chia theo nhóm `lookup`, `constraint`, `aggregate`, `multihop`, `temporal`, `comparison`,
  `project_street`, `valuation`, `unanswerable`; mỗi câu có tuyến đúng cho router. Đáp án chuẩn tính bằng SQL DuckDB trên toàn bộ bảng
  sạch (lưu kèm câu SQL), không gõ tay, ghi vào `results/benchmark_gt.json`; file câu hỏi không bao giờ bị code ghi đè.
- `& $VRE_PY scripts/vre_pipeline.py eval` chạy bốn hệ thống trên mọi câu, chấm bằng LLM giám khảo khác mô hình sinh, tính recall/MRR/nDCG,
  các chỉ số kiểu RAGAS, chi phí và độ trễ, khoảng tin cậy bootstrap ghép cặp theo câu hỏi, rồi phân tích nguyên nhân lỗi
  → `results/eval_summary.json`, `results/error_analysis.json`.
- `& $VRE_PY scripts/vre_pipeline.py ablation` so sánh cấu hình truy hồi (dense/BM25/hybrid, lọc trước, rerank, kích thước chunk) trên
  subcorpus chứa mọi tin đáp án → `results/ablation.json`. So sánh giữa các model embedding chỉ lấy từ `results/embedding_benchmark.json`.
- `& $VRE_PY scripts/vre_pipeline.py report examples` → `results/examples.json`; `& $VRE_PY scripts/vre_pipeline.py report readme` điền
  khối [Kết quả](#kết-quả) bên dưới.

## API, web và Docker

API FastAPI (`api/`), hợp đồng ở `plans/261015-vre-full-retrain/api-contract.md`, OpenAPI ở `api/openapi.json`:

```powershell
& $VRE_PY -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --timeout-keep-alive 5
$env:VRE_FAKE_BACKENDS = '1'   # chạy API với dữ liệu giả (không cần Qdrant/Neo4j/LLM) để phát triển web và chạy e2e
```

- `POST /api/chat` và `POST /api/compare` trả luồng SSE (token, nguồn, Cypher, route, usage). Có giới hạn kích thước body, độ dài câu hỏi,
  số câu đồng thời và tần suất theo IP; rate limit chạy trước kiểm khoá.
- Đặt `VRE_API_TOKEN` thì mọi `/api/*` (trừ `/api/health`) cần header `X-API-Key`; không nhận khoá qua URL. Không có token thì API chỉ
  bind loopback.

Web React + Vite + TypeScript (`web/`): Chat, So sánh bốn hệ, Bản đồ, Định giá, Dashboard thị trường, Benchmark.

```powershell
npm --prefix web ci
npm --prefix web run build
npm --prefix web run e2e        # Playwright + axe, cần API chạy với VRE_FAKE_BACKENDS=1
```

Docker (profile `app`): `docker compose --profile app up -d --build --wait` dựng `api` và `web`; mở `http://127.0.0.1:8080`.
Container API mount read-only cache Hugging Face, `results/`, `models/`, `data/clean/` và `index/` từ máy.

## Notebook

Năm notebook trong `notebooks/` chỉ **đọc** `results/*.json` và `results/eda/*.png` (không tính lại, không gọi mạng); mọi con số được in
từ JSON. File kết quả chưa có thì notebook hiện ô "chưa có" kèm lệnh sinh. Thư mục kết quả lấy từ `VRE_DATA_ROOT`, mặc định là gốc repo.

| Notebook | Nội dung |
| --- | --- |
| `01_eda.ipynb` | Nạp/làm sạch (`ingest_report`), thống kê mô tả và các hình EDA (`eda_summary`) |
| `02_rag.ipynb` | So sánh embedding và lý do chọn `multilingual-e5-large`, cấu hình cố định, lượt dựng chỉ mục (`night_report`), ablation, ví dụ truy hồi |
| `03_knowledge_graph.ipynb` | Lược đồ đồ thị (Mermaid), quy mô node/cạnh, cộng đồng Leiden (`graph_report`), ví dụ Text2Cypher |
| `04_graphrag_eval.ipynb` | Độ chính xác theo nhóm × hệ thống kèm CI, so sánh cặp, router, chi phí, phân tích lỗi |
| `05_valuation.ipynb` | `tabular` vs `full` vs baseline trung vị phường, theo loại và tỉnh, SHAP (`valuation_metrics`) |

Sinh lại nguồn và chạy lại (lưu kèm output): `& $VRE_PY notebooks/build_notebooks.py` (hoặc chỉ một số: `... build_notebooks.py 02 04`).

## Kết quả

Khối dưới đây do `scripts/vre_pipeline.py report readme` sinh từ `results/*.json`; không sửa tay.

<!-- results:start -->
<!-- results:end -->

## Kiểm thử và CI

```powershell
& $VRE_PY -m pytest -q -m "not integration and not gpu and not notebooks"   # unit, không mạng/GPU/Docker
& $VRE_PY -m pytest -q -m notebooks tests/test_notebooks.py                   # chạy cả năm notebook bằng nbclient
& $VRE_PY -m pytest -q -m integration                                         # cần Qdrant + Neo4j đang chạy
& $VRE_PY -m ruff check src tests scripts
```

CI (`.github/workflows/ci.yml`) chạy trên mỗi push và pull request, không có `.env`: `lint` (ruff + mypy), `unit` (pytest + coverage),
`notebooks` (chạy notebook trên `results/` đã commit), `web` (lint, typecheck, vitest, build, Playwright với API giả).

## Cấu trúc thư mục

```text
src/vre/            Package chính: config, checks, data (ingest, eda), index (chunk, BM25, Qdrant, đồ thị), pipeline (lượt chạy dài),
                    rag, graph (Text2Cypher), hybrid (router), llm, eval, valuation, qa (facade cho API)
scripts/            vre_pipeline.py (CLI), vre_env.ps1 (biến môi trường ổ D), night.ps1, build_geo.py
benchmark/          questions.jsonl — bộ câu hỏi đánh giá
api/                FastAPI + SSE, chế độ dữ liệu giả, OpenAPI
web/                React + Vite + TypeScript + Tailwind
notebooks/          Năm notebook trình bày + build_notebooks.py
results/            JSON kết quả nhỏ (được commit); cache, eval thô và parquet bị bỏ qua
graph/              Cypher sau import (constraint, index)
tests/              unit, integration, gpu, api, notebook
plans/              Kế hoạch và hợp đồng dữ liệu/API
data/ index/ models/ docker-data/   Sinh ra khi chạy pipeline, không đưa vào Git
```

## Hạn chế

- Giá chào bán do người dùng tự nhập: có nhiễu, có tin đăng lại có sửa chữ mà SimHash không gom được; không phản ánh giá giao dịch.
- Dữ liệu tập trung mạnh ở Hà Nội và Hồ Chí Minh (tỷ trọng in ở notebook `01_eda`); kết luận cho tỉnh ít tin kém tin cậy hơn.
- Địa danh theo địa giới trước sắp xếp năm `2025`; bản đồ và đồ thị chưa phản ánh đơn vị hành chính mới.
- Số tin thuê quá ít để huấn luyện mô hình định giá thuê (lý do ghi ở `results/valuation_metrics.json` → `rent_model`).
- Bộ lọc, Cypher, router và câu trả lời phụ thuộc LLM; cần khoá Groq khi chạy lần đầu, lần sau đọc cache. LLM có thể chép sai id trong
  `[Tin#id]` (đo bằng `bad_citation_rate`).
- Benchmark là bộ câu hỏi có hạn: so sánh tương đối giữa bốn hệ thống, kèm khoảng tin cậy; không có ý nghĩa thống kê thì ghi đúng như vậy.
