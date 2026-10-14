# Phase 1 — Nền tảng: khung package, bảng sạch, EDA, Knowledge Graph, định giá tabular (phiên `p1`, chạy song song với Phase 2)

> **Phạm vi khi cook file này: CHỈ phase này.** Dừng sau bước cuối của file (Bước 9). Không làm việc của phase khác, không đọc `plan.md`
> để làm tiếp. Phase 1 và Phase 2 chạy song song trong hai phiên (`p1` = phiên này, `p2` = Phase 2), phối hợp theo
> [`coordination.md`](coordination.md) — đọc file đó TRƯỚC khi làm bất cứ gì.

Đọc trước khi code (đường dẫn tuyệt đối, đọc được từ mọi worktree):
`D:\Vietnam-Real-Estates\plans\261015-vre-full-retrain\coordination.md` (tên phiên, tín hiệu, ownership, git, tài nguyên) ·
`D:\Vietnam-Real-Estates\plans\261015-vre-full-retrain\data-contract.md` (schema dữ liệu; §9 = khoá config, kiểm tra, CLI mà p2 dựa vào) ·
thiết kế đã duyệt `D:\Vietnam-Real-Estates\plans\reports\brainstorm-261015-0007-full-retrain-redesign.md`.

## Mục tiêu

Cuối phase này có: package `src/vre` + config + kiểm tra fail-fast + CLI Typer `scripts/vre_pipeline.py` (đúng bề mặt data-contract §9, để p2 viết
code song song); `docker-compose.yaml` chạy Qdrant + Neo4j trên bind mount ổ D; bảng sạch ~2,7M tin với `listing_id` ổn định và `dup_cluster_id`;
`results/embedding_benchmark.json` (bản sao thẳng số đo đã có); EDA đầy đủ trong `results/eda/`; Knowledge Graph mới đã nạp vào Neo4j bằng
`neo4j-admin`; model định giá tabular XGBoost **trên CPU** (`tree_method="hist"`, `device="cpu"`) — không bao giờ tranh GPU với p2; code nhánh
`full` của định giá sẵn sàng để p2 chạy sau lượt chạy dài; module cũ đã di chuyển sang `vre.*`; unit/integration test của các module p1 xanh;
nhánh `feat/p1-foundation` đã merge vào `main`.

Phiên này **không** viết code chỉ mục/embedding/lượt chạy dài (việc của p2, Phase 2) và **không dùng GPU**.
Ngoài phạm vi: RAG/Text2Cypher/Router chức năng mới, benchmark, webapp, notebook mới, README, CI đầy đủ (Phase 3); slide (Phase 4). Không gọi LLM.

## Máy

Toàn bộ máy rảnh và dành riêng cho dự án: Ryzen 9 7945HX 32 luồng, ~35GB RAM dùng được, RTX 4060 Laptop 8GB. **Không có bước cấu hình máy
nào** — người dùng không đổi thiết lập Docker/hệ điều hành, không đóng ứng dụng; agent cũng không đụng tới. Vì hai phiên chạy cùng lúc, máy được
chia theo coordination §8: p1 không dùng GPU; DuckDB 12 luồng / 12GB, XGBoost 12 luồng; Neo4j heap 4G + pagecache 4G.

Embedding của dự án đã chốt và không đo lại (data-contract §9.1, §7.4): `intfloat/multilingual-e5-large`, 1024 chiều, mean pooling, tiền tố
`passage: `/`query: `, `max_length=512`, fp16, `embed_token_budget=32768`. Phase 1 chỉ ghi các giá trị này vào `vre.config` và sao chép số đo
2026-10-15 sang `results/embedding_benchmark.json`.

## Điều kiện tiên quyết (mức dự án — đều thành kiểm tra fail-fast trong lệnh, data-contract §9.2)

1. 10 shard `D:\Vietnam-Real-Estates\data\raw\vietnam-real-estates\shard_0000..0009.parquet` có mặt (`raw_shards_present`, đầu lệnh `ingest`).
2. `intfloat/multilingual-e5-large` có trong `D:\Vietnam-Real-Estates\.cache\huggingface\hub` (`e5_large_cached`; p1 chỉ dùng tokenizer của nó để đếm token EDA).
3. Dịch vụ Docker healthy sau Bước 1 (`qdrant_healthy`, `neo4j_healthy`).
4. Ổ D trống ≥ 45 GB (`d_free_space`).
5. `.env` có `GROQ_API_KEY` (`groq_key_present`, chỉ kiểm tồn tại; Phase 3 mới dùng).
6. Mọi biến cache/TMP trỏ vào D (`env_paths_on_d`).

Môi trường đã có (đo 2026-10-15): `.venv` Python 3.13.5; torch 2.13.0+cu126, transformers 5.14.1, **tokenizers 0.22.2 (đã hạ để
khớp transformers 5.14 — phải ghim)**, sentence-transformers 5.7.0, xgboost 3.4.1, typer 0.27.1, qdrant-client 1.15.1,
neo4j 6.2.0, pyvi 0.1.1, psutil 7.2.2, tenacity 9.1.4, pyarrow 25.0.1, pandas 3.0.5, numpy 2.5.1, pydantic 2.14.1, PyYAML 6.0.3.
CHƯA cài (`pip show` trong `.venv`, 2026-10-15): `duckdb`, `pydantic-settings`, `ruff`, `mypy`, `pytest-cov`, `fastapi`.
Node 24.13 / npm 11.19 có sẵn. `.venv` dùng chung với p2: chỉ p1 cài package; KHÔNG `pip install -e .` (coordination §7).

## Biến môi trường ổ D (mọi lệnh)

Mọi lệnh chạy trong PowerShell đã dot-source `scripts\vre_env.ps1` (tạo ở Bước 1):

```powershell
. D:\Vietnam-Real-Estates\scripts\vre_env.ps1
```
File này đặt (tạo thư mục nếu chưa có): `VRE_DATA_ROOT=D:\Vietnam-Real-Estates`, `HF_HOME=D:\Vietnam-Real-Estates\.cache\huggingface`,
`TORCH_HOME=...\.cache\torch`, `PIP_CACHE_DIR=...\.cache\pip`, `TMP`/`TEMP=...\.cache\tmp`, `CUDA_CACHE_PATH=...\.cache\nv`,
`MPLCONFIGDIR=...\.cache\matplotlib`, `XDG_CACHE_HOME=...\.cache\xdg`, `JUPYTER_DATA_DIR/JUPYTER_RUNTIME_DIR/IPYTHONDIR=...\.cache\jupyter\*`,
`npm_config_cache=...\.cache\npm`, `PLAYWRIGHT_BROWSERS_PATH=...\.cache\ms-playwright`, `PYTHONUTF8=1`, `TOKENIZERS_PARALLELISM=false`,
và `$VRE_PY = D:\Vietnam-Real-Estates\.venv\Scripts\python.exe`. Mọi đường `.cache` trỏ về checkout chính (dùng chung với worktree p2).
`vre.envguard.apply()` (gọi ĐẦU TIÊN trong `scripts/vre_pipeline.py`, trước mọi import torch/HF) đặt các mặc định tương tự bằng
`os.environ.setdefault` và **raise** nếu bất kỳ biến nào ở trên trỏ vào ổ khác `D:` (errors over silent fallbacks).
DuckDB: `temp_directory = D:\Vietnam-Real-Estates\.cache\duckdb_tmp`.

## Prior art tái dùng (đã đọc, phạm vi tìm: `src/`, `scripts/`, `graph/`, `tests/`, `app/`, `.cache/scratch/`, `docker-compose.yaml`)

| Nguồn | Tái dùng ở đâu (p1) |
|---|---|
| `src/config.py` — `COL` map cột thô, mọi tham số env (LLM, Qdrant, Neo4j, chat) | `src/vre/config.py` (pydantic-settings, prefix `VRE_`, giữ tên env LLM cũ `GROQ_API_KEY`, `LLM_MODEL`, `JUDGE_MODEL`…) |
| `src/data.py` — `normalize_text`, `mask_pii`, `normalize_place`, `_PROVINCE_ALIASES`, quy tắc sửa giá nghìn đồng | `vre/data/places.py`, `vre/data/clean.py` (phần chunk `chunk_starts`, `n_chunks_formula`, `listing_header`, `chunk_listing` do p2 tái dùng — p2 đọc từ `b4a7663:src/data.py`) |
| `src/graph.py` — `run_cypher` (giao dịch chỉ đọc, timeout, max_rows), `FORBIDDEN`/`is_read_only`/`ensure_limit`, `text2cypher`, `ask_graph`, `ward_profiles`/`ward_communities`/`community_table`/`modularity` | `vre/graph/client.py`, `guard.py`, `text2cypher.py`, `communities.py` (di chuyển nguyên trạng; Phase 3 sửa chức năng) |
| `src/rag.py`, `src/hybrid.py`, `src/llm.py`, `src/eval.py` | di chuyển cơ học sang `vre/rag/*`, `vre/hybrid/router.py`, `vre/llm/{client,cache}.py`, `vre/eval/{metrics,judge,runner}.py` (Phase 3 sửa bug + chức năng) |
| `scripts/index_full.py::save_state` — ghi nguyên tử (`tmp.replace`) | `vre.common.write_json_atomic` (p2 dùng cho state/report) |
| `.cache/scratch/bench_final.jsonl` — số đo 4 model 2026-10-15 | sao chép thẳng thành `results/embedding_benchmark.json` (một dòng cuối lệnh `ingest`, data-contract §7.4) |
| `.cache/scratch/profile.py` — khoá trùng gần `name\|description[:300]` (~815k trùng) | tầng dedup 2 (data-contract §2.2) |
| `tests/test_presentation_notebooks.py` — test `normalize_place`, `mask_pii`, `clean`, money/filters, guard, RRF, metrics | chuyển sang `tests/unit/*` của p1 (test chunk/BM25 cũ chuyển cho p2: p2 đọc từ `b4a7663`) |
| `tests/test_chatbot.py` — test guard HTTP của server cũ | bỏ cùng `app/` (API mới viết lại ở Phase 3) |

Không có sẵn (tìm `grep -ri "duckdb\|simhash\|minhash\|neo4j-admin\|xgboost\|typer" src scripts tests` → 0 kết quả): DuckDB ingest, SimHash,
neo4j-admin import, model định giá, CLI Typer, kiểm tra fail-fast mức dự án — đều là MỚI.

## File ownership

SSOT sở hữu là coordination §5 (p1 sở hữu mọi file dùng chung; p2 sở hữu `vre/index/{chunk,tokenize,embed,bm25,qdrant_*}.py`, `vre/pipeline/*`,
`scripts/night.ps1` và test của chúng — p1 KHÔNG tạo/sửa các file đó). Chi tiết của p1:

**Tạo mới**
- `pyproject.toml` (setuptools, package `vre` trong `src/`; `[tool.pytest.ini_options] pythonpath = ["src"]`, markers `integration`, `gpu`, `slow`; cấu hình `ruff`, `mypy`)
- `requirements/base.txt`, `requirements/ml.txt`, `requirements/dev.txt`
- `scripts/vre_env.ps1`, `scripts/vre_pipeline.py`
- `src/vre/__init__.py`, `config.py`, `envguard.py`, `common.py`, `checks.py`
- `src/vre/{data,index,graph,rag,hybrid,llm,eval,valuation,pipeline}/__init__.py` (rỗng — để p2 chỉ thêm module, không chạm `__init__`)
- `src/vre/data/`: `places.py`, `clean.py`, `ingest.py`, `dedup.py`, `simhash.py`, `amenities.py`, `amenities.yaml`, `admin_2025.csv`, `views.sql`, `duck.py`, `eda.py`, `eda_plots.py`
- `src/vre/index/`: `graph_export.py`, `graph_import.py`
- `src/vre/graph/`: `client.py`, `guard.py`, `schema.py`, `text2cypher.py`, `communities.py`
- `src/vre/rag/`: `filters.py`, `retrieve.py`, `generate.py`, `citations.py`
- `src/vre/hybrid/router.py`; `src/vre/llm/`: `client.py`, `cache.py`; `src/vre/eval/`: `metrics.py`, `judge.py`, `runner.py`
- `src/vre/valuation/`: `features.py`, `train.py`, `predict.py`, `explain.py`
- `graph/post_import.cypher`
- `tests/conftest.py`, `tests/unit/test_*.py` của p1 (danh sách ở Bước 8), `tests/integration/test_neo4j_invariants.py`

**Sửa**: `requirements.txt` (chỉ còn `-r requirements/ml.txt` và `-r requirements/dev.txt`), `docker-compose.yaml`, `.env.example`, `.gitignore`,
`.github/workflows/tests.yml` (tối thiểu để không gãy; Phase 3 làm CI đầy đủ).

**Xoá** (sau pre-delete reference check `grep -rn "from src\|import src" --include=*.py --include=*.ipynb .`):
`src/__init__.py`, `src/common.py`, `src/config.py`, `src/data.py`, `src/eval.py`, `src/graph.py`, `src/hybrid.py`, `src/llm.py`,
`src/qdrant_store.py`, `src/rag.py`, `src/vector.py`, `scripts/index_full.py`, `graph/load.cypher`, `app/server.py`,
`app/static/{app.js,index.html,style.css}`, `notebooks/00_setup.ipynb` … `04_graphrag_eval.ipynb`, `tests/test_chatbot.py`,
`tests/test_presentation_notebooks.py`. (Lịch sử git `b4a7663` giữ bản cũ — p2 đọc prior art chỉ mục từ đó. `benchmark/questions.csv` GIỮ — Phase 3 thay.)

**Sinh ra (gitignored)**: `data/interim/`, `data/clean/`, `data/graph_import/`, `models/valuation/*_tabular*`, `logs/`, `.cache/duckdb_tmp/`,
`docker-data/neo4j/`, `results/handoff/` (cùng p2).
**Sinh ra (commit)**: `results/ingest_report.json`, `results/eda_summary.json`, `results/eda/*.png`, `results/graph_report.json`,
`results/valuation_metrics.json` (bản `tabular`), `results/embedding_benchmark.json`.

Mỗi file Python ≤ 200 dòng (code-conventions). Mọi hằng số có tên trong `config.py` hoặc đầu module.

## Các bước (theo thứ tự; tín hiệu gửi/chờ ghi đậm)

### Bước 0 — Chuẩn bị phiên (5 phút)
1. Phiên mang tên `p1` (coordination §2); `ListAgents` để thấy `p2` (nếu đã mở — chưa mở cũng không chờ).
2. `git status` sạch ngoài `plans/`, `slide/`, `.gitignore`; `git switch -c feat/p1-foundation` trong `D:\Vietnam-Real-Estates`.
3. Tạo `D:\Vietnam-Real-Estates\results\handoff\` (rỗng).

### Bước 1 — Viết code khung package + compose, chạy dịch vụ (agent ~2h; máy ~5 phút) → **gửi `SKELETON_READY`**
Mục tiêu: p2 có thể rebase và viết code theo đúng bề mặt data-contract §9 càng sớm càng tốt — bước này chỉ làm phần dùng chung.
1. `scripts/vre_env.ps1` như mục "Biến môi trường ổ D".
2. `requirements/base.txt`: `pandas>=2.2`, `pyarrow>=15`, `numpy>=1.26`, `duckdb>=1.1`, `pydantic>=2,<3`, `pydantic-settings>=2.4`,
   `pyyaml>=6`, `qdrant-client>=1.15,<1.16` (khớp server v1.15.5), `neo4j>=5,<7`, `pyvi>=0.1,<0.2`, `python-dotenv>=1,<2`,
   `openai>=1.60,<4`, `anthropic>=1.11,<2`, `tenacity>=9`, `psutil>=7`, `typer>=0.12`, `networkx>=3.2,<4`, `python-igraph>=0.11`,
   `leidenalg>=0.10`, `scikit-learn>=1.4`, `matplotlib>=3.8`.
   `requirements/ml.txt`: `-r base.txt`, `torch` (comment: cài wheel cu126 từ index PyTorch), `transformers>=5.14,<5.15`,
   **`tokenizers==0.22.2  # transformers 5.14 yêu cầu tokenizers<=0.23.0; 0.23.1 làm hỏng venv (đã sửa 2026-10-15)`**,
   `sentence-transformers>=5,<6`, `xgboost>=3.0`.
   `requirements/dev.txt`: `-r base.txt`, `pytest>=9`, `pytest-cov>=6`, `ruff>=0.6`, `mypy>=1.11`, `nbformat>=5.10,<6`.
   (Phase 3 thêm `fastapi`, `uvicorn`, `sse-starlette`, `httpx` vào base.)
3. `pyproject.toml`; cài: `& $VRE_PY -m pip install -r requirements\ml.txt -r requirements\dev.txt` (không `-e .`);
   `& $VRE_PY -m pip check` phải sạch; `& $VRE_PY -m pip show tokenizers` = 0.22.2.
4. `src/vre/config.py`: `Settings(BaseSettings)` đúng TỪNG khoá + mặc định ở data-contract §9.1 (không thêm, không đổi tên; cần khoá mới ⇒ sửa
   §9.1 trước và gửi `NOTE` cho p2). Đọc `.env` CHỈ khi `VRE_LOAD_DOTENV != "0"` (test đặt 0 — sửa bug "test phụ thuộc `.env` local").
   Mọi đường dẫn dữ liệu là property tuyệt đối từ `data_root`. Bảng model một mục `EMBED_MODELS`, `EMBED_MODEL_SLUG = "e5_large"`; không có logic
   chọn model, không có model dự phòng. Giữ mọi tham số LLM/RAG/chat cũ qua `validation_alias`.
5. `src/vre/envguard.py`, `src/vre/common.py` (`write_json_atomic`, `git_sha`, `write_handoff`, `scalar`), `src/vre/checks.py` — đủ 9 kiểm tra
   ở data-contract §9.2, mỗi kiểm tra là hàm thuần trên "facts" + một collector (ai-driven-design: script phát facts), `require(settings, *names)`.
6. `src/vre/*/__init__.py` cho mọi subpackage (rỗng).
7. `scripts/vre_pipeline.py` (Typer, ≤ 200 dòng, chỉ nối lệnh → hàm): đủ mọi lệnh ở data-contract §9.3 (`ingest`, `eda`, `graph`,
   `price --variant tabular|full`, `night [--resume/--no-resume] [--dry-run]`, `status [--dry-run]`, `smoke-search QUERY`, `signal`, `config-get`,
   `all --resume`; Phase 3 thêm `eval ...`, `ablation`, `report ...`). Dòng đầu: thêm `<repo>/src` vào `sys.path`, rồi `envguard.apply()`.
   Mỗi lệnh gọi `checks.require(...)` với đúng tập kiểm tra ở bảng §9.3 rồi import LƯỜI hàm đích bên trong thân lệnh (lệnh của p2 chưa có code
   vẫn hiện trong `--help`; gọi trước khi p2 merge ⇒ `ImportError` rõ ràng). Mọi lệnh idempotent: artifact đã có + fingerprint khớp ⇒ in "đã xong", thoát 0.
8. `docker-compose.yaml` mới — dữ liệu chỉ ở ổ D **mà không đụng thiết lập Docker**, một bộ container duy nhất cho cả hai worktree (coordination §7):
   - top-level `name: vre`; bind mount tuyệt đối `${VRE_DATA_ROOT:-D:/Vietnam-Real-Estates}/docker-data/...` (gitignored; rule docker-volume-discipline
     chấp nhận bind mount tường minh) thay cho named volume.
   - `qdrant`: `qdrant/qdrant:v1.15.5`, ports `127.0.0.1:6333:6333`, `127.0.0.1:6334:6334`, bind `.../docker-data/qdrant:/qdrant/storage`,
     không giới hạn `cpus`; luồng tối ưu hoá đặt ở collection (data-contract §5, p2 làm); healthcheck `bash -c ':> /dev/tcp/127.0.0.1/6333'`, `start_period: 30s`.
   - `neo4j`: `neo4j:5.26-community`, `NEO4J_AUTH`, heap `4G`, pagecache `4G`, `NEO4J_server_memory_transaction_total_max=2G` (chia máy với p2,
     coordination §8), ports `127.0.0.1:7474:7474`, `127.0.0.1:7687:7687`, bind `.../docker-data/neo4j/data:/data`, `.../docker-data/neo4j/logs:/logs`,
     `.../data/graph_import:/import:ro`, healthcheck `cypher-shell -u neo4j -p "$${NEO4J_PASSWORD}" "RETURN 1"`, `start_period: 90s`.
   - Volume cũ `neo4j_data`/`neo4j_full_data`/`qdrant_data` (dữ liệu mẫu 20k): KHÔNG xoá, không prune; báo tên cho người dùng ở Bước 9.
9. `.gitignore`: bỏ `results/`; thêm `results/cache/`, `results/eval/raw/`, `results/**/*.parquet`, `results/handoff/`, `models/`, `docker-data/`,
   `data/` (giữ), `index/` (giữ). `.env.example` thêm `VRE_DATA_ROOT`.
10. `tests/conftest.py`: `VRE_LOAD_DOTENV=0`, thư mục tạm dưới `.cache/tmp/pytest`, fixture DataFrame nhỏ.

Verify (chạy):
```powershell
& $VRE_PY -c "import sys; sys.path.insert(0, 'src'); import vre.config, vre.checks, vre.common"
& $VRE_PY scripts\vre_pipeline.py --help        # liệt kê đủ lệnh §9.3
docker compose -f D:\Vietnam-Real-Estates\docker-compose.yaml config -q
docker compose -f D:\Vietnam-Real-Estates\docker-compose.yaml up -d --wait qdrant neo4j
curl -s http://127.0.0.1:6333/readyz
```
`Get-ChildItem docker-data\qdrant, docker-data\neo4j\data` có file mới; round-trip bind mount: tạo collection tạm 1.000 point →
`docker compose -f ... restart qdrant` → count vẫn 1.000 → xoá (chứng minh Qdrant chạy đúng trên bind mount trước khi p2 đầu tư vào lượt chạy dài).

Hoàn tất bước: commit pathspec mọi file của bước này + `plans/261015-vre-full-retrain/` (coordination §6); đọc lại sha bằng `git rev-parse HEAD`;
**gửi `SKELETON_READY`** `{"branch":"feat/p1-foundation","sha":"<sha>"}` (file trước bằng `vre_pipeline.py signal`, rồi `SendMessage` tới `p2`).

### Việc thường trực từ Bước 1 tới Bước 9 — xử lý `CHANGE_REQUEST` của p2
Nhận `CHANGE_REQUEST` (tin hoặc file `results/handoff/CHANGE_REQUEST-<id>.json`): áp dụng vào file dùng chung trong nhánh p1 (sửa data-contract §9
trước nếu là khoá/lệnh), chạy lại test liên quan, commit pathspec, **gửi `CHANGE_APPLIED`** `{"id","sha"}`. Không đổi thiết kế ngoài yêu cầu;
yêu cầu mâu thuẫn plan ⇒ trả `NOTE` giải thích, không tự quyết thay p2.

### Bước 2 — Viết code ingest + clean + dedup (agent ~3h)
Theo data-contract §2–§3. Hai giai đoạn, xử lý theo shard để bộ nhớ tiến trình có giới hạn:
1. **Theo shard** (`ingest.py`, `ProcessPoolExecutor(max_workers=settings.ingest_workers)`, context `spawn`): đọc 1 shard bằng pyarrow
   (chỉ cột cần), tính `listing_id`, `prefix_key` (uint64), `simhash` (uint64), chuẩn hoá NFC/địa danh (cache `dict` theo giá trị
   duy nhất), mask PII, parse giá + sửa nghìn đồng, `deal_type`, `amenities`, `n_nonnull`; ghi `data/interim/shard_NNNN.parquet`;
   giải phóng bảng trước khi nhận shard kế. Bỏ dòng thiếu cả title+description / loại lạ (đếm).
2. **Toàn cục** (`dedup.py` + `duck.py`, DuckDB `threads=settings.duckdb_threads` (12), `memory_limit` = `duckdb_memory_gb` (12GB), `temp_directory` trên D,
   `preserve_insertion_order=false`): kiểm va chạm `listing_id` (GROUP BY listing_id HAVING count(DISTINCT key_exact_hash2) > 1 — hash phụ thứ hai
   để so khoá); `row_number() OVER (PARTITION BY prefix_key ORDER BY published_at DESC NULLS LAST, n_nonnull DESC, listing_id)`;
   phân vị outlier theo nhóm; ghi `dup_map.parquet`. `duck.py` nạp `views.sql` và thay `{clean_path}` bằng đường dẫn tuyệt đối (data-contract §3.1).
3. **SimHash** (`simhash.py`, numpy): 4 band × 16 bit, nhóm theo `(province, property_type)`, Hamming bằng `np.bitwise_count`,
   union-find mảng numpy, bỏ bucket > 500. Gán `dup_cluster_id`.
4. Ghi `data/clean/listings.parquet` (sort `listing_id`, row group 100k, zstd) + `results/ingest_report.json`.
5. Một dòng cuối `ingest.run`: sao chép thẳng `.cache/scratch/bench_final.jsonl` → `results/embedding_benchmark.json` (mảng JSON, data-contract §7.4;
   không lọc trường, không đo gì).
6. `src/vre/data/admin_2025.csv` (63 dòng) + `views.sql` + `amenities.yaml` (data-contract §3.2).

### Bước 3 — Chạy ingest (máy 10–15 phút) → **gửi `CLEAN_DATA_READY`**
```powershell
& $VRE_PY scripts\vre_pipeline.py ingest
& $VRE_PY -c "import duckdb;c=duckdb.connect();print(c.sql(\"select count(*), count(distinct listing_id), sum(n_raw_rows), sum(price_vnd) from 'D:/Vietnam-Real-Estates/data/clean/listings.parquet'\").fetchall())"
```
Đạt: `count = count distinct`; `sum(n_raw_rows) + dropped = 3500744`; count trong [2,55M; 2,85M] (đo trước ~2,7M — lệch hơn thì
dừng và giải thích); `listing_id_collisions = 0`; peak RAM tiến trình < 24 GB (in trong log); `embedding_benchmark.json` có 4 phần tử;
chạy lại lệnh lần 2 in "đã xong".
Hoàn tất bước: commit pathspec code Bước 2 + `results/ingest_report.json` + `results/embedding_benchmark.json`; **gửi `CLEAN_DATA_READY`**
`{"path":"D:\\Vietnam-Real-Estates\\data\\clean\\listings.parquet","n_listings":<count>,"fingerprint":{"n_listings":<count>,"price_vnd_sum":<sum>}}`.
**Từ đây KHÔNG ghi lại `listings.parquet`** (coordination §8: cần ingest lại ⇒ `NOTE` cho p2 trước, đợi p2 dừng lượt chạy).

### Bước 4 — Di chuyển module cũ + xoá file cũ (agent ~1,5h)
Di chuyển cơ học module cũ sang `src/vre/{rag,graph,hybrid,llm,eval}` (đổi import `src.` → `vre.`), tách `graph.py` 522 dòng
thành `client.py` (driver, `use`, `run_cypher`), `guard.py` (`FORBIDDEN`, `is_read_only`, `ensure_limit`), `text2cypher.py`
(`SCHEMA*`, `CYPHER_*`, `cypher_system`, `text2cypher`, `ask_graph`, `rows_context`), `communities.py` (`modularity`, `ward_profiles`,
`ward_communities`, `community_table`). KHÔNG mang theo code chỉ phục vụ notebook/nạp MERGE cũ: `load_blocks`, `graph_rows`,
`run_block`, `reset_graph`, `expected_counts`, `graph_matches*`, `mini_graph`, `draw_graph`, `neighborhood`, `district_html`,
`schema_graph`, `QUERIES`, `SQL_EQUIV`, `extract_amenities` (LLM) — mỗi tên phải grep = 0 tham chiếu còn lại trước khi xoá.
`src/llm.py` → `llm/client.py` (provider, `_ask_openai`, `_ask_anthropic`, `_close_objects`, `_schema`) + `llm/cache.py`
(khoá cache, đọc/ghi nguyên tử, `fresh`). Cache LLM chuyển sang `results/cache/llm/` (giữ đường dẫn cũ để tận dụng cache có sẵn).
Không sửa logic ở bước này (Phase 3 sửa); chỉ đổi import + vị trí. Sau đó xoá các file ở mục "Xoá".
Verify: `& $VRE_PY -c "import sys; sys.path.insert(0,'src'); import vre.rag.retrieve, vre.graph.text2cypher, vre.hybrid.router"`;
`grep -rn "from src\|import src" --include=*.py --include=*.ipynb .` = 0. Commit pathspec.

### Bước 5 — Viết + chạy EDA (agent ~1,5h; máy 5–10 phút)
`eda.py` (truy vấn DuckDB trên `v_listings` → dict) + `eda_plots.py` (matplotlib backend Agg, palette navy/teal/cam như slide,
caption ≥ 14pt khi xuất 1600×900). Ghi đúng 8 file hình + `results/eda_summary.json` (data-contract §7.2); `tokens_per_listing` đếm bằng
tokenizer (chỉ tokenizer, không nạp model) của e5-large trên mẫu 20k tin.
Verify: `& $VRE_PY scripts\vre_pipeline.py eda`; `Get-ChildItem results\eda\*.png | Measure-Object` = 8; JSON có `n_listings` = số dòng bảng sạch.
Commit pathspec code + `results/eda_summary.json` + `results/eda/*.png`.

### Bước 6 — Viết + chạy Knowledge Graph (agent ~3h; máy 20–40 phút)
1. `graph/post_import.cypher`: constraint + index theo data-contract §6.3, mỗi lệnh `IF NOT EXISTS`, khối `// name:` như format cũ.
2. `graph_export.py`: DuckDB `COPY ... TO '<data_root>/data/graph_import/<file>.csv' (HEADER, DELIMITER ',', QUOTE '"')` cho mọi node/quan hệ ở
   data-contract §6.1–6.2 (header neo4j-admin: `:ID(<Label>)`, thuộc tính có kiểu `:long`/`:double`, `:LABEL`, `:START_ID(..)`, `:END_ID(..)`, `:TYPE`).
   Thuộc tính tổng hợp + `MarketStat` tính bằng `median()`/`quantile_cont()`. Communities: `communities.ward_profiles` +
   `ward_communities` (k=5, seed 42) chạy trên DataFrame phường tổng hợp từ DuckDB (không nạp 2,7M dòng vào pandas).
   Chuỗi trong CSV: escape dấu `"`; bỏ ký tự điều khiển.
3. `graph_import.py`: `docker compose -f <compose_file> stop neo4j` → `docker compose -f <compose_file> run --rm --no-deps neo4j neo4j-admin database import full neo4j --overwrite-destination=true --nodes=... --relationships=... --skip-bad-relationships=false --threads=12`
   (ghi vào bind mount `docker-data/neo4j/data`; kiểm cú pháp bằng `neo4j-admin database import full --help` trong image trước; thử import với
   1.000 tin trước bản full) → `docker compose -f <compose_file> up -d --wait neo4j` → chạy `graph/post_import.cypher` → đợi index ONLINE
   (`SHOW INDEXES YIELD state`) → kiểm bất biến (count + sum giá) → `results/graph_report.json`. Neo4j chỉ p1 dùng — dừng/khởi động lại không ảnh hưởng p2.

Verify: `& $VRE_PY scripts\vre_pipeline.py graph`; trong cypher-shell:
`MATCH (l:Listing) RETURN count(l)` = n_listings; `MATCH (d:District {name:'Quận 7'})<-[:STAT_OF]-(s:MarketStat {property_type:'ALL', month:'ALL', deal_type:'sale'}) RETURN s.median_price_m2_mil` trả 1 dòng < 100 ms;
`graph_report.invariants` đều `true`. Commit pathspec code + `results/graph_report.json`.

### Bước 7 — Viết + chạy định giá tabular trên CPU (agent ~2,5h; máy 30–60 phút)
`features.py`: từ `v_listings` (`deal_type='sale'`, `price_m2_mil` not null, không outlier diện tích); target `log(price_m2_mil)`;
split theo tháng train 2025-06…2026-01 / test 2026-02…2026-03 (tháng 2026-01 làm validation cho early stopping);
target encoding K-fold (5 fold, chỉ trên train; test dùng mapping toàn train; smoothing m=20) cho `province`, `district` key, `ward` key,
`project` key, `street` key; `property_type`, `house_direction` dạng categorical native của XGBoost; số: diện tích, phòng, tầng,
mặt tiền, đường, chỉ số tháng; 1 cột 0/1 cho mỗi amenity. Bản `full` = bản `tabular` + join `pca_00..pca_31` theo `listing_id` từ
`data/features/emb_pca32_e5_large.parquet` (data-contract §4; p2 sinh file này).
`train.py` — `run(settings, variant)`: `xgboost` `tree_method="hist"`, `device = settings.xgb_device_tabular` (`cpu`) cho `tabular`,
`settings.xgb_device_full` (`cuda`) cho `full`, `nthread = settings.xgb_nthread` (12); tìm ngẫu nhiên 8 cấu hình (seed 42) với early stopping 100 vòng;
baseline median-theo-phường (lùi quận → tỉnh → loại). `explain.py`: SHAP bằng `booster.predict(DMatrix, pred_contribs=True)` trên 20k mẫu test
(không thêm thư viện `shap`). `predict.py`: hàm `valuate(input)` dùng cho Phase 3 (trả đúng `ValuationResult` của api-contract §3).
Thuê: chỉ huấn luyện nếu `n_rent ≥ rent_min_rows` (đo trước: ~56/350k ở shard 0 ⇒ gần như chắc chắn ghi `rent_model.trained=false` kèm lý do).
Ghi `models/valuation/xgb_tabular_sale.json`, `encoders_tabular.json`, `results/valuation_metrics.json` (`variants.full = null`; p2 điền sau).
Chạy: `& $VRE_PY scripts\vre_pipeline.py price --variant tabular` (KHÔNG chạy `--variant full` — p2 chạy sau `P1_DONE`; nhánh `full` được kiểm bằng
unit test với PCA giả).
Verify: JSON có `variants.tabular.mape < variants.baseline_ward_median.mape` (nếu không: báo trung thực trong JSON, không chỉnh để "đẹp" — rule
green-that-proves-nothing); log xác nhận XGBoost chạy `device=cpu`. Commit pathspec code + `results/valuation_metrics.json`.

### Bước 8 — Viết + chạy unit/integration test của p1 (agent ~2,5h)
Unit (không mạng, không GPU, không Docker): `test_config.py` (env override; envguard raise khi đường dẫn ở C; mọi khoá §9.1 có mặt đúng mặc định),
`test_checks.py` (facts giả: mỗi kiểm tra §9.2 có ca fail và ca pass — rule green-that-proves-nothing), `test_places.py`, `test_clean.py`
(parse giá chuỗi, sửa nghìn đồng, deal_type, cờ outlier), `test_ids.py` (listing_id tất định, ≤ 2⁵³−1, trùng chính xác cùng id, phát hiện va chạm
giả lập), `test_dedup.py` (prefix + đại diện), `test_simhash.py`, `test_amenities.py`, `test_admin_2025.py`, `test_embedding_benchmark.py`
(data-contract §7.4: 4 phần tử, khớp file nguồn), `test_graph_export.py` (header CSV, khoá node), `test_valuation_features.py` (split thời gian,
target encoding không rò rỉ fold, join PCA giả cho `full`), `test_guard.py`, `test_rag_core.py` (chuyển từ test cũ: money/filters, RRF, citations,
wrap_documents), `test_metrics.py`.
Integration (`@pytest.mark.integration`, tự skip nếu Neo4j không chạy): `test_neo4j_invariants.py` (đọc graph_report + truy vấn MarketStat).
Không chạy test `-m gpu` (GPU thuộc p2).
Verify: `& $VRE_PY -m pytest -q tests/unit` (chỉ file của p1) = 0 fail; `& $VRE_PY -m pytest -q tests/integration/test_neo4j_invariants.py` = 0 fail;
`& $VRE_PY -m ruff check src tests scripts` = 0 lỗi; không file nào > 200 dòng
(`Get-ChildItem src -Recurse -Filter *.py | % { (Get-Content $_).Count }`). Commit pathspec.

### Bước 9 — Bàn giao + merge (agent ~0,5h) → **gửi `P1_DONE`**
1. `.github/workflows/tests.yml` tối thiểu: `pip install -r requirements/dev.txt`, `pytest -m "not integration and not gpu"` (pythonpath từ pyproject),
   `timeout-minutes: 10`.
2. Kiểm không ghi C: `Get-ChildItem $env:LOCALAPPDATA\Temp -Recurse -ErrorAction SilentlyContinue | ? LastWriteTime -gt <giờ bắt đầu phase>` không có
   file do pipeline tạo (ghi kết quả vào mô tả PR).
3. Commit pathspec phần còn lại, push nhánh, mở PR "Phase 1 foundation" (KHÔNG commit `data/`, `index/`, `models/`, `docker-data/`). Báo người dùng
   tên các volume Docker cũ (không xoá).
4. `AskUserQuestion` xin duyệt merge PR vào `main`; được duyệt ⇒ merge, `git switch main && git pull --ff-only`, chạy
   `& $VRE_PY -m pytest -q -m "not gpu"` trên `main` = 0 fail (test `gpu`/chỉ mục thuộc p2, chạy ở Phase 2).
5. **Gửi `P1_DONE`** `{"branch":"feat/p1-foundation","sha":"<sha main sau merge>"}`. Từ đây quyền sửa file dùng chung chuyển cho p2 (coordination §5).
   Xoá nhánh đã merge (`git branch -D feat/p1-foundation`). Dừng phiên — không làm tiếp Phase 2/3.

## Tiêu chí hoàn thành (tất cả phải có bằng chứng)

- [ ] `SKELETON_READY`, `CLEAN_DATA_READY`, `P1_DONE` có trong `results/handoff/` và đã gửi tin cho `p2`.
- [ ] `pip check` sạch; `tokenizers==0.22.2` ghim trong `requirements/ml.txt` có comment lý do; không bản cài editable trong `.venv`.
- [ ] `config.py`, `checks.py`, `vre_pipeline.py` khớp đúng data-contract §9 (test `test_config.py`, `--help`).
- [ ] Qdrant/Neo4j ghi dữ liệu vào `docker-data/` trên D (round-trip bind mount ở Bước 1 đạt); compose có `name: vre`.
- [ ] `data/clean/listings.parquet` đúng schema data-contract §3; `ingest_report.listing_id_collisions = 0`; `embedding_benchmark.json` hợp lệ §7.4.
- [ ] 8 hình + `eda_summary.json`; `graph_report.invariants` đều true; Cypher MarketStat < 100 ms.
- [ ] `valuation_metrics.json.variants.tabular` có đủ MAE/MAPE/R² + so baseline; huấn luyện trên CPU.
- [ ] Unit + integration test của p1 xanh; ruff 0 lỗi; không file nào > 200 dòng.
- [ ] Không có file mới trên C do pipeline.
- [ ] Nhánh p1 đã merge vào `main`, test `-m "not gpu"` xanh trên `main`.

## Rủi ro

| Rủi ro | L | I | Điểm | Giảm thiểu |
|---|---|---|---|---|
| Thiếu RAM khi ingest (8 worker + DuckDB) | 2 | 4 | 8 | Xử lý theo shard, giải phóng bảng trước shard kế; DuckDB 12GB + temp trên D; ingest chạy TRƯỚC lượt chạy dài của p2; peak RAM < 24GB là tiêu chí |
| Qdrant/Neo4j chậm hoặc lỗi trên bind mount `docker-data/` | 3 | 4 | 12 | Round-trip test ở Bước 1 trước khi gửi `SKELETON_READY`; fail ⇒ dừng, `AskUserQuestion` (không tự đổi thiết lập Docker), `NOTE` cho p2 |
| Bề mặt config/CLI lệch với code p2 | 2 | 4 | 8 | data-contract §9 là SSOT; `test_config.py` khẳng định đủ khoá; đổi chỉ qua `CHANGE_REQUEST`/`NOTE` |
| Ghi lại bảng sạch khi p2 đang chạy dài | 1 | 5 | 5 | Bất biến coordination §8; fingerprint lệch ⇒ lượt chạy của p2 dừng mã 20 thay vì trộn chỉ mục |
| Sai cú pháp/headers `neo4j-admin import` (community 5.26) | 3 | 3 | 9 | Đọc `--help` trong image trước; thử import với 1.000 tin trước bản full |
| Dedup lệch xa số đo (~2,7M) | 2 | 3 | 6 | Khoá prefix giống hệt `profile.py`; khoảng chấp nhận [2,55M; 2,85M], ngoài khoảng ⇒ dừng điều tra |
| SimHash bucket khổng lồ (tin mẫu template) → chậm | 3 | 2 | 6 | Giới hạn 500/bucket + đếm bucket bỏ qua |
| XGBoost CPU chậm hơn GPU | 3 | 1 | 3 | Không có hạn giờ; 12 luồng; chỉ ảnh hưởng lịch của p1 |
| Ghi nhầm vào C (cache NVIDIA, temp Python, pip) | 3 | 3 | 9 | `vre_env.ps1` + `envguard` raise + kiểm `env_paths_on_d` + kiểm Temp ở Bước 9 |
| Map 63→34 tỉnh sai | 2 | 2 | 4 | Test 63 dòng / 34 giá trị; chỉ là cột phụ, không ảnh hưởng RAG |

## Timeline (chỉ các bước của phase này)

| Bước | Agent | Máy | Phụ thuộc | Tín hiệu |
|---|---|---|---|---|
| 0 chuẩn bị phiên | 0,1h | — | — | — |
| 1 khung + compose + dịch vụ | 2h | 5 phút | 0 | gửi `SKELETON_READY` |
| 2 code ingest | 3h | — | 1 | — |
| 3 chạy ingest | 0,2h | 10–15 phút | 2 | gửi `CLEAN_DATA_READY` |
| 4 di chuyển module cũ | 1,5h | — | 1 | — |
| 5 EDA | 1,5h | 5–10 phút | 3 | — |
| 6 KG | 3h | 20–40 phút | 1, 3 | — |
| 7 định giá tabular (CPU) | 2,5h | 30–60 phút | 3 | — |
| 8 test | 2,5h | 5 phút | 2–7 | — |
| 9 bàn giao + merge | 0,5h | 5 phút | 8 | gửi `P1_DONE` |
| **Tổng** | **~17 giờ agent** | **~1,5–2 giờ máy** | Đường găng của p1: 1 → 2 → 3 (mở khoá lượt chạy dài của p2) → 6 → 8 → 9 | |

Bước 4–7 độc lập với nhau sau Bước 3 (file riêng) — có thể fan-out sub-agent trong phiên p1, mỗi lane một `git worktree` riêng của p1, commit pathspec;
file dùng chung vẫn chỉ phiên chính p1 sửa.
