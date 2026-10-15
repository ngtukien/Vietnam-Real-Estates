# Phối hợp hai phiên song song — Phase 1 (`p1`) và Phase 2 (`p2`)

SSOT cho mọi thứ hai phiên Claude Code phải thống nhất khi Phase 1 và Phase 2 chạy CÙNG LÚC. File phase chỉ trỏ tới đây,
không chép lại. Đổi bất kỳ mục nào ở đây ⇒ báo phiên kia trước (tin `NOTE`, §3), rồi mới làm theo nội dung mới.

Đọc kèm: [`data-contract.md`](data-contract.md) (§9 = bề mặt code dùng chung: khoá `vre.config`, lệnh CLI, chữ ký hàm; §7.10 = schema file handoff).
Mọi đường dẫn plan trong tài liệu này là tuyệt đối (`D:\Vietnam-Real-Estates\plans\261015-vre-full-retrain\...`) để đọc được từ cả hai worktree.

## 1. Hai phiên và việc của mỗi phiên

| Phiên | File phase | Làm trong | Nhánh | Tóm tắt |
|---|---|---|---|---|
| `p1` | `phase-01-foundation.md` | checkout chính `D:\Vietnam-Real-Estates` | `feat/p1-foundation` | khung package + config + compose, ingest/clean/dedup, EDA, KG, định giá tabular (CPU), test của mình |
| `p2` | `phase-02-overnight-run.md` | worktree `D:\Vietnam-Real-Estates-p2` | `feat/p2-index` | viết code chỉ mục + lượt chạy dài, chạy lượt dài e5-large (~8h), định giá `full` (GPU) |

Không phiên nào làm việc của phiên kia. Thiếu gì từ phiên kia ⇒ gửi tin (§3), không tự sửa file của phiên kia.

## 2. Đặt tên phiên và tìm nhau

1. Mỗi phiên mang đúng tên `p1` hoặc `p2`. **Người dùng** đặt tên bằng lệnh `/rename p1` (phiên Phase 1) và `/rename p2` (phiên Phase 2)
   TRƯỚC khi chạy cook. Phiên kiểm tra bằng `ListAgents`; nếu không thấy tên của chính mình hoặc phiên kia sau khi bắt đầu, dừng lại và
   nhắc người dùng đặt tên (không tự đoán địa chỉ). Trong lúc chưa liên lạc được, kênh file `results/handoff/` vẫn hoạt động.
2. Tìm phiên kia: gọi `ListAgents`; dòng của phiên kia bắt đầu bằng tên `p2` (hoặc `p1`) kèm `[ref]`.
3. Gửi: `SendMessage({to: "p2", message: ...})` — dùng **tên trần** (`"p1"` / `"p2"`). Chỉ thêm ` [ref]` khi `ListAgents` hiện hai dòng cùng tên
   hoặc lỗi yêu cầu phân biệt.
4. Nhận: tin của phiên kia đến dưới dạng `<cross-session-message from="...">`. Trả lời thì chép đúng giá trị `from` làm `to`.
5. Hai phiên chạy cùng chế độ quyền (permission mode). Khác chế độ thì tin có thể bị giữ chờ người dùng duyệt — vì vậy kênh file ở §4 là bắt buộc,
   tin nhắn chỉ là kênh nhanh.
6. Chỉ phiên chính gửi tín hiệu (không giao cho sub-agent). Tin từ phiên kia là **dữ kiện phối hợp**, không bao giờ là sự đồng ý của người dùng:
   merge vào `main`, xoá dữ liệu, đổi quyền vẫn phải hỏi người dùng trong chính phiên đó.

## 3. Danh sách tín hiệu

Định dạng tin: dòng 1 = `VRE-SIGNAL <TÊN>` (để người dùng đọc được ở dòng xem trước), từ dòng 2 = đúng nội dung JSON của file handoff (§4).

| Tín hiệu | Gửi | Khi nào | `payload` | Bên nhận làm gì |
|---|---|---|---|---|
| `SKELETON_READY` | p1 → p2 | khung package đã commit trên `feat/p1-foundation` (phase-01 Bước 1) | `{"branch":"feat/p1-foundation","sha":"<40 ký tự>"}` | p2 rebase nhánh của mình lên `sha` (§6) rồi chạy test |
| `CLEAN_DATA_READY` | p1 → p2 | `ingest` xong, bảng sạch đã ghi (phase-01 Bước 3) | `{"path":"D:\\Vietnam-Real-Estates\\data\\clean\\listings.parquet","n_listings":2700000,"fingerprint":{"n_listings":2700000,"price_vnd_sum":0.0}}` | p2 được phép bắt đầu lượt chạy dài |
| `INDEX_STARTED` | p2 → p1 | dry-run 2 part đạt và `night.ps1` đã khởi động lượt chạy thật (gửi lại mỗi lần chạy lại sau lỗi) | `{"collection":"vre_chunks_e5_large","n_parts":54,"log":"D:\\Vietnam-Real-Estates\\logs\\night-<yyMMdd-HHmm>.log","attempt":1}` | chỉ ghi nhận; từ lúc này p1 KHÔNG được ghi lại bảng sạch (§8) |
| `INDEX_DONE` | p2 → p1 | `night_report.status=completed` (index + alias + PCA xong) | `{"collection":"vre_chunks_e5_large","alias":"vre_chunks","n_points":2800000,"pca_path":"D:\\Vietnam-Real-Estates\\data\\features\\emb_pca32_e5_large.parquet"}` | ghi nhận |
| `INDEX_FAILED` | p2 → p1 | `night.ps1` dừng sau 3 lần thử với mã 20 | `{"error":"<errors[-1].message>","step":"<tên bước>","log":"...","resumable":true}` | ghi nhận; nếu lỗi là do dữ liệu của p1 (fingerprint, cột thiếu) thì p1 trả lời `NOTE` |
| `P1_DONE` | p1 → p2 | nhánh p1 đã merge vào `main` và bộ test đầy đủ xanh trên `main` (phase-01 Bước 9) | `{"branch":"feat/p1-foundation","sha":"<sha của main sau merge>"}` | p2 rebase lên `main`, chạy định giá `full` |
| `P2_DONE` | p2 → p1 | định giá `full` xong, nhánh p2 đã merge vào `main`, test đầy đủ xanh (phase-02 Bước 7) | `{"branch":"feat/p2-index","sha":"<sha của main sau merge>","valuation_full_mape":0.0}` | — (phiên p1 thường đã kết thúc; file là bằng chứng cho Phase 3) |
| `CHANGE_REQUEST` | p2 → p1 | p2 cần đổi file dùng chung (thêm khoá config, thêm dependency, sửa compose…) | `{"id":"CR-01","file":"src/vre/config.py","change":"thêm khoá ...","reason":"..."}` | p1 áp dụng, commit, trả `CHANGE_APPLIED` |
| `CHANGE_APPLIED` | p1 → p2 | đã commit thay đổi theo `CHANGE_REQUEST` | `{"id":"CR-01","sha":"<40 ký tự>"}` | p2 `git rebase <sha>` (hoặc cherry-pick đúng commit đó) |
| `NOTE` | hai chiều | thông tin không đổi trạng thái (cảnh báo, đổi kế hoạch, hỏi nhanh) | `{"text":"..."}` | đọc, trả lời nếu cần |

`P2_DONE` và `CHANGE_REQUEST`/`CHANGE_APPLIED`/`NOTE` là các tín hiệu bổ sung cần cho quy trình (định giá `full` cần code của p1 đã merge; file
dùng chung chỉ p1 sửa).

## 4. Kênh bền — file handoff (không bao giờ kẹt vì lỡ tin)

Thư mục: `D:\Vietnam-Real-Estates\results\handoff\` (tuyệt đối, cùng một chỗ cho cả hai worktree; gitignored). Schema file: data-contract §7.10.

Thứ tự khi GỬI (luôn đủ cả hai, file trước):
1. Ghi file nguyên tử bằng lệnh của dự án (khi khung package đã có):
   `& $VRE_PY D:\Vietnam-Real-Estates\scripts\vre_pipeline.py signal <TÊN> --from p1 --payload '<json>'`
   (ghi `<TÊN>.json.tmp` rồi `os.replace` ⇒ `<TÊN>.json`). Tín hiệu có `id` (`CHANGE_REQUEST`, `CHANGE_APPLIED`) ghi `<TÊN>-<id>.json`;
   `NOTE` ghi `NOTE-<from>-<yyMMddHHmmss>.json`. Gửi lại cùng tín hiệu (vd `INDEX_STARTED` lần 2) ghi đè file, tăng `seq`.
   Trước khi có lệnh `signal` (chỉ p2 lúc mới mở phiên, và chỉ cho `NOTE`): ghi tay `.tmp` rồi đổi tên.
2. `SendMessage` tới phiên kia (§2).

Khi CHỜ một tín hiệu:
- Kiểm file trước (có thể đã đến khi phiên mình đang tắt). Có ⇒ coi như đã nhận.
- Chưa có ⇒ chạy lệnh nền (Bash `run_in_background: true`), thăm dò **tối đa mỗi 5 phút** (`handoff_poll_s=300`):
  `until [ -f /d/Vietnam-Real-Estates/results/handoff/CLEAN_DATA_READY.json ]; do sleep 300; done; cat /d/Vietnam-Real-Estates/results/handoff/CLEAN_DATA_READY.json`.
  Lệnh nền thoát khi file xuất hiện và đánh thức phiên. Tin `SendMessage` đến trước thì dùng luôn — nhưng đọc payload từ FILE (file là SSOT).
- Không dùng `sleep` tiền cảnh, không gửi tin hỏi "xong chưa?".
- Nhận trùng (tin + file) là bình thường: xử lý một lần theo `seq`.

## 5. Sở hữu file — không giao nhau

| Sở hữu | File / thư mục |
|---|---|
| **p1 (dùng chung — chỉ p1 sửa)** | `pyproject.toml`, `src/vre/config.py`, `requirements/*`, `requirements.txt`, `docker-compose.yaml`, `.gitignore`, `.env.example`, `scripts/vre_pipeline.py`, `scripts/vre_env.ps1`, `src/vre/__init__.py`, `src/vre/common.py`, `src/vre/envguard.py`, `src/vre/checks.py` (kiểm tra fail-fast dùng chung), mọi `src/vre/*/__init__.py`, `tests/conftest.py` |
| **p1 (riêng)** | `src/vre/data/**`, `src/vre/index/graph_export.py`, `src/vre/index/graph_import.py`, `src/vre/graph/**`, `src/vre/rag/**`, `src/vre/hybrid/**`, `src/vre/llm/**`, `src/vre/eval/**`, `src/vre/valuation/**`, `graph/post_import.cypher`, `.github/workflows/tests.yml`, mọi file cũ bị xoá (`src/*.py`, `app/**`, `notebooks/0*.ipynb`, `scripts/index_full.py`, `graph/load.cypher`, `tests/test_*.py` cũ), `tests/unit/test_{config,checks,places,clean,ids,dedup,simhash,amenities,admin_2025,embedding_benchmark,graph_export,valuation_features,guard,rag_core,metrics}.py`, `tests/integration/test_neo4j_invariants.py`, `plans/261015-vre-full-retrain/**` (chỉ commit, không sửa nội dung) |
| **p2 (riêng)** | `src/vre/index/{chunk,tokenize,embed,bm25,qdrant_store,qdrant_upload}.py`, `src/vre/pipeline/{state,stages,night,pca,report}.py`, `scripts/night.ps1`, `tests/unit/test_{chunk,bm25,qdrant_payload,state,pca}.py`, `tests/integration/{test_qdrant_roundtrip,test_index_complete}.py`, `tests/gpu/test_embed_parity.py` |

Artifact dữ liệu (không commit, theo người ghi): p1 ghi `data/interim/`, `data/clean/`, `data/graph_import/`, `models/valuation/*_tabular*`,
`results/{ingest_report,eda_summary,graph_report,embedding_benchmark}.json`, `results/eda/`, `results/valuation_metrics.json` (bản `tabular`),
`docker-data/neo4j/`; p2 ghi `index/**`, `data/features/`, `models/valuation/*_full*`, `results/night_report.json`, `results/valuation_metrics.json`
(thêm `full`, chỉ sau `P1_DONE` ⇒ không ghi chồng thời gian), Qdrant (`docker-data/qdrant/`). `results/handoff/` cả hai ghi, mỗi tín hiệu một người gửi.

Cần đổi file dùng chung ⇒ p2 gửi `CHANGE_REQUEST`; p1 sửa trong nhánh p1, commit, trả `CHANGE_APPLIED`. Sau `P1_DONE` phiên p1 kết thúc ⇒
quyền sửa file dùng chung chuyển sang p2 (ghi rõ trong commit message).

## 6. Git

- p1: checkout chính `D:\Vietnam-Real-Estates`, `git switch -c feat/p1-foundation` từ `main`. Commit đầu tiên của p1 (trong Bước 1) gồm cả
  `plans/261015-vre-full-retrain/**` để phiên kia và Phase 3 đọc được plan từ git.
- p2: tạo worktree riêng NGAY khi mở phiên, từ `main`:
  `git -C D:\Vietnam-Real-Estates worktree add -b feat/p2-index D:\Vietnam-Real-Estates-p2 main`
  (nếu `SKELETON_READY.json` đã có thì tạo thẳng từ `sha` trong file). p2 soạn code trong worktree này; khi nhận `SKELETON_READY`:
  `git -C D:\Vietnam-Real-Estates-p2 rebase <sha>`. File hai bên không giao nhau ⇒ rebase không xung đột; xung đột ⇒ dừng, gửi `NOTE`, không tự sửa file của p1.
- Commit **chỉ dạng pathspec**: `git commit -m "<msg>" -- <đường dẫn...>` (file mới: `git add <đường dẫn cụ thể>` trước). Cấm `git add .`/`-A`, `commit -a`.
- Không phiên nào chạy `checkout`/`switch`/`reset`/`branch -f` trong worktree của phiên kia, không commit vào nhánh của phiên kia.
- Đọc code của phiên kia: `git -C <worktree kia> show <sha>:<path>` hoặc đọc file trong worktree kia — không sửa.
- Merge cuối: (1) p1: push `feat/p1-foundation`, mở PR, người dùng duyệt (`AskUserQuestion` trong phiên p1), merge vào `main`, chạy toàn bộ test trên `main`
  ⇒ `P1_DONE`. (2) p2: `git rebase main` (sau `P1_DONE`), chạy toàn bộ test, push, PR, người dùng duyệt, merge ⇒ `P2_DONE`.
  Sau mỗi merge: `pytest -q` toàn bộ (cả `integration`, `gpu`) = 0 fail trên `main`.
- Dọn: sau `P2_DONE`, p2 `git worktree remove D:\Vietnam-Real-Estates-p2` và xoá nhánh đã merge (`git branch -D`).

## 7. Dữ liệu và môi trường dùng chung

- **Code tách, dữ liệu chung.** Mọi đường dẫn dữ liệu là TUYỆT ĐỐI dưới `VRE_DATA_ROOT` (mặc định `D:\Vietnam-Real-Estates`):
  `data/`, `index/`, `results/`, `models/`, `logs/`, `docker-data/`, `.cache/`. Code không bao giờ dùng đường dẫn dữ liệu tương đối theo thư mục
  hiện hành (data-contract §9.1). Vì vậy p2 chạy code từ `D:\Vietnam-Real-Estates-p2` nhưng đọc/ghi dữ liệu ở `D:\Vietnam-Real-Estates\...`.
- **Docker một bộ duy nhất.** Mọi lệnh compose dùng file của checkout chính và tên project cố định:
  `docker compose -f D:\Vietnam-Real-Estates\docker-compose.yaml ...` (file có `name: vre`, bind mount tuyệt đối). Không chạy compose từ worktree p2
  (sẽ tạo container thứ hai trùng cổng). Qdrant và Neo4j do p1 khởi động ở Bước 1; p2 chỉ `up -d --wait qdrant` khi cần khởi động lại.
- **`.venv` dùng chung** `D:\Vietnam-Real-Estates\.venv`. KHÔNG `pip install -e .` trong Phase 1–2 (bản cài editable trỏ cố định vào một worktree).
  `scripts/vre_pipeline.py` tự thêm `<repo của chính nó>/src` vào `sys.path`; pytest dùng `pythonpath = ["src"]` trong `pyproject.toml`.
  Chỉ p1 cài package; p2 cần thư viện mới ⇒ `CHANGE_REQUEST`.
- `scripts/vre_env.ps1` dùng chung (đặt cache/TMP về `D:\Vietnam-Real-Estates\.cache\`, `$VRE_PY`). Mỗi phiên dot-source bản trong worktree của mình.

## 8. Chia tài nguyên khi cả hai đang chạy

Toàn bộ máy (Ryzen 9 7945HX 32 luồng, ~35GB RAM dùng được, RTX 4060 Laptop 8GB) rảnh và dành riêng cho dự án; không có bước cấu hình máy.
Bảng dưới chỉ chia máy đó giữa hai phiên để không tranh nhau.


| Tài nguyên | p1 | p2 |
|---|---|---|
| GPU (RTX 4060 8GB) | **không dùng** (XGBoost tabular `device="cpu"`; không chạy test `-m gpu`) | toàn quyền: embed, test parity, định giá `full` |
| CPU (32 luồng) | DuckDB `duckdb_threads=12`; XGBoost `xgb_nthread=12`; ingest `ingest_workers=8` (chỉ lúc ingest, trước khi p2 chạy dài) | tokenize `tokenize_workers=2`; BM25 `bm25_workers=4` (CR-04); Qdrant indexing `qdrant_max_indexing_threads=12` |
| RAM | DuckDB `duckdb_memory_gb=12`; Neo4j heap 4G + pagecache 4G | tối đa 3 part đi trước GPU |
| Dịch vụ | Neo4j (chỉ p1 dùng; import dừng/khởi động lại Neo4j) | Qdrant (chỉ p2 dùng) |

Bất biến:
- Sau `CLEAN_DATA_READY`, p1 **không ghi lại** `data/clean/listings.parquet`. Cần ingest lại ⇒ gửi `NOTE` cho p2 TRƯỚC, đợi p2 dừng lượt chạy, rồi mới ghi;
  sau đó gửi lại `CLEAN_DATA_READY` (seq mới) — lượt chạy của p2 sẽ thấy fingerprint lệch và phải làm lại chỉ mục.
- p1 không đụng Qdrant/`index/`; p2 không đụng Neo4j/`data/graph_import/`.

## 9. Sự cố phối hợp

| Sự cố | Phát hiện | Xử lý |
|---|---|---|
| Lỡ tin nhắn | không có tin | lệnh nền thấy file trong ≤ 5 phút (§4) |
| Phiên bị đóng/khởi động lại | — | khi mở lại: đọc `results/handoff/*.json` + `git log` nhánh mình + `night_state.json` để biết đang ở đâu, tiếp tục bước kế |
| Rebase xung đột | `git rebase` dừng | `git rebase --abort`, gửi `NOTE` kèm danh sách file; vi phạm §5 thì bên sở hữu sửa |
| p2 cần đổi file dùng chung gấp | — | vẫn `CHANGE_REQUEST`; trong lúc chờ chỉ làm việc khác, không sửa file đó |
| Phiên kia im lặng lâu | không có file mới | không gửi tin thúc; chờ file. Quá lâu bất thường thì báo người dùng trong phiên mình |

## 10. Cổng sang Phase 3

Phase 3 chỉ bắt đầu khi `results/handoff/` có đủ `P1_DONE.json`, `INDEX_DONE.json` và `P2_DONE.json` (định giá `full` đã xong và đã merge),
và `main` chứa cả hai nhánh.
