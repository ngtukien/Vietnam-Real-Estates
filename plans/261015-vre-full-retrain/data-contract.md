# Data Contract — VRE full retrain (dùng chung Phase 1–3)

Phase 1 (`p1`) và Phase 2 (`p2`) chạy song song trong hai phiên; quy tắc phối hợp ở [`coordination.md`](coordination.md).

SSOT cho mọi định dạng dữ liệu trên đĩa, trong Qdrant, trong Neo4j và trong `results/*.json`.
Phase nào tạo/đọc artifact nào ghi ở cột "Writer / Reader". Đổi bất kỳ trường nào ở đây ⇒ sửa file này
TRƯỚC, rồi sửa code ở mọi phase dùng nó. Không phase nào được tự chế lại schema trong brief của mình.

Quy ước chung:
- Tên trường `snake_case`. Tiền: VND (`*_vnd`, DOUBLE). Giá/m²: triệu đồng/m² (`*_m2_mil`). Diện tích m² (`*_m2`).
- Tháng: chuỗi `YYYY-MM` (giờ UTC của `published_at`).
- `null` = không có dữ liệu (KHÔNG dùng 0, "", "None", NaN cho "không có").
- Mọi file JSON trong `results/` có `schema_version` (int, bắt đầu 1) và `generated_at` (ISO-8601 UTC) và `git_sha` (7 ký tự, `"dirty"` nếu working tree bẩn).
- Ghi file luôn nguyên tử: ghi `*.tmp` rồi `os.replace` (học từ `scripts/index_full.py::save_state`).
- Không lưu trường dẫn xuất (rule `code-conventions.md` → No Derived Fields). Ngoại lệ có chủ đích được đánh dấu **[materialized]** kèm lý do.

---

## 1. Nguồn thô (read-only)

`data/raw/vietnam-real-estates/shard_0000..0009.parquet` — 10 shard, mỗi shard 1 row group (~350k dòng),
tổng **3.500.744** dòng, license CC-BY-NC-4.0 (chỉ dùng phi thương mại, ghi nguồn `tinixai/vietnam-real-estates`).

| Cột thô | Kiểu thô | Ghi chú đo thật |
|---|---|---|
| `name` | string | tiêu đề |
| `description` | string | trung vị 466 ký tự, p99 1884 |
| `property_type_name` | string | 5 giá trị: `Nhà`, `Đất`, `Căn hộ chung cư`, `Biệt thự/Nhà liền kề`, `Shophouse` |
| `province_name`, `district_name`, `ward_name`, `street_name`, `project_name` | string | địa giới CŨ (63 tỉnh); thiếu ward 14%, street 39%, project 71% |
| `price` | **string** | chuỗi chỉ gồm chữ số, đơn vị VND (`"7450000000"`); null 6,2%; có giá 8 chữ số (nhập theo nghìn đồng hoặc giá thuê) |
| `area` | double | m²; p1/p50/p99 = 25/80/2200 |
| `floor_count`, `frontage_width`, `house_depth`, `road_width`, `bedroom_count`, `bathroom_count` | double | thiếu nhiều (bedroom 46%, floor 77%, road_width 81%, depth 98%) |
| `house_direction`, `balcony_direction` | string | |
| `published_at` | string | `2025-06-01T05:12:56.941000` (06/2025 → 03/2026) |

`(source_shard, source_row)` = (số shard 0–9, chỉ số dòng 0-based trong shard) là khoá vật lý của 1 dòng thô.

---

## 2. Định danh

### 2.1 `listing_id` (BIGINT, 53-bit, ổn định)

```
norm(s)     = lower(NFC(collapse_whitespace(s or ""))).strip()
key_exact   = norm(name) + "\x1f" + norm(description)
listing_id  = int.from_bytes(blake2b(key_exact.encode("utf-8"), digest_size=8).digest(), "big") & (2**53 - 1)
```
- Tính TRƯỚC khi mask PII (mask chỉ áp vào cột hiển thị), để id không đổi khi đổi regex mask.
- 53-bit để an toàn với JavaScript `Number` (≤ 9.007.199.254.740.991) → API trả `listing_id` là số JSON.
- Hai dòng thô trùng chính xác (`key_exact` bằng nhau) có cùng `listing_id` → tầng dedup 1 gộp tự nhiên.
- **Va chạm**: nếu hai `key_exact` KHÁC nhau cho cùng `listing_id` → ingest dừng với lỗi liệt kê cặp (fail-loud,
  không salt ngầm). Xác suất ≈ 4·10⁻⁴ cho 2,7M khoá.
- Trích dẫn trong câu trả lời: `[Tin#<listing_id>]` (số thập phân). Regex chuẩn: `\[Tin#(\d{1,16})\]`.

### 2.2 Dedup và `dup_cluster_id`

| Tầng | Khoá | Hành động |
|---|---|---|
| 1. Trùng chính xác | `listing_id` | gộp, giữ 1 đại diện |
| 2. Trùng gần (prefix) | `prefix_key = blake2b(norm(name + "\|" + description)[:300])` (đúng khoá đã đo ~815k trùng trong `.cache/scratch/profile.py`) | gộp, giữ 1 đại diện → corpus ~2,7M |
| 3. Cụm repost (SimHash) | SimHash 64-bit trên 3-shingle từ của `norm(description)`; ứng viên qua 4 band × 16 bit trong cùng `(province, property_type)`; Hamming ≤ 3; bucket > 500 phần tử bỏ qua và đếm vào `ingest_report.simhash_skipped_buckets` | KHÔNG xoá; chỉ hợp nhất cụm (union-find) |

- Đại diện của một nhóm tầng 1/2: `published_at` mới nhất → nhiều trường khác null nhất → `listing_id` nhỏ nhất.
- `dup_cluster_id` = `listing_id` nhỏ nhất trong cụm tầng 3 (singleton ⇒ chính nó).
- Benchmark lookup: một tin truy xuất được tính ĐÚNG nếu `dup_cluster_id` của nó bằng `dup_cluster_id` của tin đáp án.

### 2.3 Chunk và point id

- `chunk_index` ∈ [0, 63]; tin dài hơn bị cắt ở 64 chunk (đếm vào `night_report.counters.chunks_truncated_listings`).
- `point_id = listing_id * 64 + chunk_index` (< 2⁵⁹, hợp lệ với Qdrant unsigned int, đảo ngược được:
  `listing_id = point_id // 64`). Point của thẻ tin = `listing_id * 64` (chunk 0).

---

## 3. Bảng sạch — `data/clean/listings.parquet`

Writer: Phase 1 `vre.data` (`vre_pipeline.py ingest`). Reader: Phase 1 EDA/graph/price, Phase 2 dry-run + lượt chạy dài (`night`, chỉ sau tín hiệu `CLEAN_DATA_READY`), Phase 3 eval/api.
Đường dẫn tuyệt đối `settings.clean_path` (§9.1) — cả hai worktree đọc cùng một file.
Một file, sắp xếp theo `listing_id` tăng dần, row group 100.000 dòng, nén zstd. Thứ tự theo hash ⇒ mọi đoạn liên tiếp
là mẫu ngẫu nhiên (dùng để chia part; dry-run 2 part đầu là mẫu đại diện).

| Cột | Kiểu (DuckDB) | Null | Mô tả |
|---|---|---|---|
| `listing_id` | BIGINT | không | §2.1, UNIQUE |
| `dup_cluster_id` | BIGINT | không | §2.2 |
| `n_raw_rows` | INTEGER | không | số dòng thô đã gộp vào đại diện này (tầng 1+2), ≥ 1 |
| `source_shard` | SMALLINT | không | shard của dòng đại diện |
| `source_row` | INTEGER | không | dòng của đại diện trong shard |
| `title` | VARCHAR | có | `name` đã chuẩn hoá NFC + mask PII (`[phone_number]`, `[email]`) |
| `description` | VARCHAR | có | như trên |
| `property_type` | VARCHAR | không | 1 trong 5 giá trị §1 (dòng có loại lạ bị loại, đếm trong ingest_report) |
| `deal_type` | VARCHAR | không | `sale` \| `rent`. `rent` nếu `title` khớp `^\s*(cần\s+)?cho\s+thuê` hoặc (`description` 200 ký tự đầu khớp `cho thuê` VÀ giá < 1e8). Ngược lại `sale` |
| `province` | VARCHAR | có | `normalize_place(kind="province")` |
| `province_2025` | VARCHAR | có | tỉnh sau sáp nhập 07/2025, tra `src/vre/data/admin_2025.csv` |
| `district`, `ward` | VARCHAR | có | `normalize_place` |
| `street`, `project` | VARCHAR | có | NFC + collapse whitespace |
| `price_vnd` | DOUBLE | có | `TRY_CAST(price AS DOUBLE)`; > 0; sửa nghìn-đồng như `src/data.py::clean` (bán, < 1e8, ×1000 nếu giá/m² sau sửa ∈ [5, 2000] tr) ; không sửa được ⇒ null |
| `price_fixed_thousand` | BOOLEAN | không | true nếu đã nhân 1000 |
| `area_m2` | DOUBLE | có | > 0 |
| `floors` | SMALLINT | có | |
| `frontage_m`, `depth_m`, `road_width_m` | DOUBLE | có | |
| `bedrooms`, `bathrooms` | SMALLINT | có | |
| `house_direction`, `balcony_direction` | VARCHAR | có | |
| `published_at` | TIMESTAMP | có | |
| `is_price_outlier` | BOOLEAN | không | giá/m² ngoài [p1, p99] của nhóm `(deal_type, property_type, province)` có ≥ 200 tin; nhóm nhỏ hơn dùng ngưỡng cố định [0,05; 2000] tr/m². **[materialized]**: phụ thuộc phân vị toàn cục, không tính lại rẻ ở query |
| `is_area_outlier` | BOOLEAN | không | area < 10 hoặc > 10.000 hoặc ngoài [p0.5, p99.5] của `(property_type, province)` khi nhóm có ≥ 200 tin (nhóm nhỏ hơn chỉ dùng ngưỡng cố định, tránh loại min/max của mọi nhóm nhỏ) **[materialized]** cùng lý do |
| `amenities` | VARCHAR[] | không (có thể rỗng) | khoá amenity (§3.2) khớp trong title+description **[materialized]**: regex trên 1,9 tỷ ký tự, đắt |

### 3.1 View chuẩn `v_listings` (DuckDB, `src/vre/data/views.sql`)

Mọi SQL đọc dữ liệu (EDA, GT benchmark, export graph, valuation) dùng view này, KHÔNG tự tính lại:

```sql
CREATE OR REPLACE VIEW v_listings AS
SELECT *,
  price_vnd / 1e9                                              AS price_bn,
  CASE WHEN NOT is_price_outlier AND NOT is_area_outlier AND price_vnd IS NOT NULL AND area_m2 IS NOT NULL
       THEN price_vnd / area_m2 / 1e6 END                      AS price_m2_mil,
  strftime(published_at, '%Y-%m')                              AS month
FROM read_parquet('{clean_path}');
```
`{clean_path}` được `vre/data/duck.py` thay bằng `settings.clean_path` tuyệt đối (không dùng đường dẫn tương đối — hai worktree, một bộ dữ liệu).

### 3.2 Từ điển amenity — `src/vre/data/amenities.yaml`

```yaml
# key: {name: tên hiển thị, patterns: [regex không dấu-nhạy, áp trên lower(NFC(text))]}
thang_may:   {name: "Thang máy",      patterns: ['thang\s*máy', '\btm\b']}
ho_boi:      {name: "Hồ bơi",         patterns: ['hồ\s*bơi', 'bể\s*bơi']}
hem_xe_hoi:  {name: "Hẻm xe hơi",     patterns: ['hẻm\s*(xe\s*hơi|ô\s*tô|oto)', '\bhxh\b']}
so_hong:     {name: "Sổ hồng/sổ đỏ",  patterns: ['sổ\s*(hồng|đỏ)', '\bshr\b']}
mat_tien:    {name: "Mặt tiền",       patterns: ['mặt\s*tiền', '\bmt\b']}
gan_truong:  {name: "Gần trường",     patterns: ['gần\s*trường']}
view_song:   {name: "View sông",      patterns: ['view\s*sông', 'view\s*hồ']}
gara:        {name: "Gara/ô tô vào nhà", patterns: ['gara', 'ô\s*tô\s*(vào|đỗ)\s*(trong\s*)?nhà']}
noi_that:    {name: "Đầy đủ nội thất", patterns: ['full\s*nội\s*thất', 'đầy\s*đủ\s*nội\s*thất', '\bfull\s*nt\b']}
an_ninh:     {name: "An ninh 24/7",   patterns: ['bảo\s*vệ\s*24', 'an\s*ninh\s*24']}
```
Phase 1 có thể THÊM khoá (≥ 10, ≤ 30), không đổi tên khoá đã có sau khi Phase 3 viết benchmark. Test
`tests/unit/test_amenities.py` khớp mỗi pattern với ≥ 2 câu mẫu dương và 1 câu âm.

### 3.3 `data/clean/dup_map.parquet`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `source_shard`, `source_row` | SMALLINT, INTEGER | khoá dòng thô |
| `row_listing_id` | BIGINT | `listing_id` của chính dòng thô |
| `kept_listing_id` | BIGINT | đại diện trong bảng sạch; null nếu dòng bị loại |
| `reason` | VARCHAR | `kept` \| `exact_dup` \| `prefix_dup` \| `dropped_no_text` \| `dropped_bad_type` |

Bất biến (test tích hợp Phase 1): `count(*) = 3.500.744`; `count(*) FILTER (reason='kept') = count(listings)`.

### 3.4 Map địa giới 2025 — `src/vre/data/admin_2025.csv`

`province_old,province_2025` — 63 dòng, `province_old` đúng chuỗi sau `normalize_place`. Test: mọi tỉnh xuất hiện trong
bảng sạch có trong file; số giá trị `province_2025` khác nhau = 34.

---

## 4. Artifact chỉ mục (Phase 2 ghi, Phase 3 đọc)

Mọi đường dẫn dưới đây tương đối với `VRE_DATA_ROOT` (mặc định `D:\Vietnam-Real-Estates`); code luôn ghép thành đường dẫn tuyệt đối.

Part = 50.000 tin liên tiếp của `listings.parquet` (part `k` = dòng `[k·50000, (k+1)·50000)`), `NNNN` = 4 chữ số.
~54 part cho ~2,7M tin. `model_slug` = `e5_large` (giá trị duy nhất; model `intfloat/multilingual-e5-large`, đã chốt 2026-10-15).

| Đường dẫn | Định dạng | Nội dung |
|---|---|---|
| `index/chunks/part_NNNN.parquet` | parquet | `point_id` BIGINT, `listing_id` BIGINT, `chunk_index` SMALLINT, `text` VARCHAR, `n_tokens` SMALLINT |
| `index/bm25/part_NNNN.parquet` | parquet | `point_id` BIGINT, `token_ids` UINTEGER[], `tfs` USMALLINT[], `doc_len` INTEGER (số token pyvi) |
| `index/emb/{model_slug}/part_NNNN.npy` | npy float16 `[n_chunks, dim]`, L2-normalized | hàng `i` ↔ hàng `i` của `chunks/part_NNNN.parquet` |
| `index/tok/part_NNNN.npz` | npz (int32 phẳng + offsets) | token tạm của part, xoá sau khi part đã `embedded` |
| `index/state/night_state.json` | JSON | §4.1 |
| `index/dryrun/**` | như trên | mọi artifact của dry-run 2 part (chunks, tok, emb, bm25, `night_state.json`, `night_report.json`; collection `vre_dryrun`), xoá cả thư mục sau khi dry-run đạt |
| `data/features/emb_pca32_e5_large.parquet` | parquet | `listing_id` BIGINT, `pca_00` … `pca_31` FLOAT (`pca_dim=32`); vector tin = trung bình vector các chunk; `IncrementalPCA` fit trên `pca_fit_sample` tin tháng train. Writer: Phase 2 `vre.pipeline.pca`; Reader: `vre.valuation` bản `full` |

### 4.1 `index/state/night_state.json` (checkpoint, ghi nguyên tử sau MỖI part của MỖI stage)

```json
{"schema_version":1,"model_slug":"e5_large","collection":"vre_chunks_e5_large","n_parts":54,
 "listings_fingerprint":{"n_listings":0,"price_vnd_sum":0.0},
 "parts":{"0000":{"chunked":true,"bm25":true,"embedded":true,"uploaded":true,"n_chunks":52013}},
 "bm25_avgdl":null,
 "hnsw_enabled":false,"hnsw_green":false,"alias_set":false,"pca_done":false,
 "runs":[{"started_at":"...","ended_at":"...","exit":"ok|error"}]}
```
Resume = đọc file này; part có cờ `true` không làm lại. Một part chỉ được đánh `embedded` sau khi `.npy` đã `os.replace`
xong; `uploaded` sau khi Qdrant `count(filter listing_id ∈ part)` = `n_chunks` của part. Fingerprint lệch với
bảng sạch hiện tại ⇒ dừng với lỗi (không trộn chỉ mục của hai lần ingest). `bm25_avgdl` = trung bình `doc_len` trên MỌI part, tính một lần khi
mọi part đã `bm25` và TRƯỚC lần upload đầu tiên (BM25 chạy trước GPU nhiều part; upload bám theo embed nên tổng thời gian không đổi).
Định giá `full` không nằm trong state này (chạy bằng lệnh `price --variant full` riêng, §7.6).

### 4.2 Quy tắc chunk

Quy tắc chunk (giữ từ `src/data.py::chunk_listing`, strategy `fixed`): header = title + dòng meta
(`property_type · ward, district, province · giá X tỷ · N m² · K phòng ngủ`) lặp ở MỌI chunk; body = cửa sổ token
của description theo tokenizer XLM-R của e5-large, `chunk_size=400`, `chunk_overlap=100` (env-configurable).
Tiền tố model: `"passage: "` cho tài liệu/chunk, `"query: "` cho truy vấn (bắt buộc với e5-large). Token id BM25 = `crc32(token) & 0x7FFFFFFF`
(giữ `qdrant_store.token_id`), token = `pyvi.ViTokenizer.tokenize(lower(text))`, bỏ token không có ký tự chữ/số.

---

## 5. Qdrant (server `qdrant/qdrant:v1.15.5`, client `qdrant-client 1.15.x`)

| Collection | Dùng cho | Writer |
|---|---|---|
| `vre_chunks_e5_large` | chỉ mục chính (toàn bộ ~2,8M chunk) | Phase 2 |
| alias `vre_chunks` → `vre_chunks_e5_large` | API/RAG luôn đọc alias | Phase 2 (tạo alias khi xong) |
| `vre_dryrun` | dry-run 2 part trước lượt chạy dài (bắt lỗi code, không phải đo), xoá ngay sau khi đạt | Phase 2 |
| `vre_ablation_e5_large_c{chunk_size}` | ablation chunk size trên subcorpus ~200k | Phase 3 |

Dữ liệu Qdrant nằm ở named volume `vre_qdrant_data` → `/qdrant/storage` (ext4 trong ổ đĩa WSL của Docker Desktop ở `D:\DockerDesktopWSL`; đổi 2026-10-15 vì store mmap trên bind mount NTFS vừa chậm vừa mất nhất quán). Ở Phase 1–2 chỉ p2 dùng Qdrant.

Cấu hình collection chính:
- vector `dense`: `size` = 1024; `distance=Cosine`; `on_disk=true`.
- `hnsw_config`: `m=16`, `ef_construct=100`, `on_disk=false`. HNSW xây SONG SONG khi nạp (`optimizers_config.indexing_threshold=20000`, `max_indexing_threads=12`) vì CPU rảnh trong lúc GPU embed
  (quyết định 2026-10-15) ⇒ không còn đuôi build HNSW 45–60 phút sau upload.
- `quantization_config`: scalar int8, `always_ram=true`. Truy vấn: `rescore=true`, `oversampling=VRE_QDRANT_OVERSAMPLING` (mặc định 2.0).
- sparse vector `bm25`: `modifier=IDF`. Trọng số tài liệu: `tf·(k1+1) / (tf + k1·(1 − b + b·doc_len/avgdl))`,
  `k1=1.2`, `b=0.75`, `avgdl` = `night_state.bm25_avgdl` (§4.1, trên toàn corpus, cố định cho cả lần nạp). Truy vấn: trọng số 1.
- `on_disk_payload=true`.

Payload mỗi point (null ⇒ BỎ khoá, không ghi `null`). **[materialized]** từ bảng sạch: cần cho pre-filter + thẻ tin.

| Khoá | Kiểu | Payload index |
|---|---|---|
| `listing_id` | int | integer |
| `dup_cluster_id` | int | integer |
| `chunk_index` | int | — |
| `text` | string (chunk) | — |
| `title` | string | — |
| `description_excerpt` | string (≤ 400 ký tự đầu description) | — |
| `property_type` | string | keyword |
| `deal_type` | string | keyword |
| `province`, `district`, `ward` | string | keyword |
| `street`, `project` | string | keyword |
| `price_vnd` | float | float |
| `area_m2` | float | float |
| `price_m2_mil` | float (null nếu outlier) | float |
| `bedrooms`, `bathrooms` | int | float (bedrooms) |
| `published_month` | string `YYYY-MM` | keyword |
| `amenities` | string[] | keyword |

Manifest chỉ mục: `index/manifest_{collection}.json` = `{schema_version, collection, model_slug, dim, n_listings, n_points, price_vnd_sum, bm25_avgdl, created_at}` (API dùng để kiểm tra Neo4j khớp, thay `read_manifest` cũ).

---

## 6. Neo4j (image `neo4j:5.26-community`, DB `neo4j`, nạp bằng `neo4j-admin database import full`)

Dữ liệu ở named volume `vre_neo4j_data` → `/data` (ext4 trên D, xem §5; log: bind mount `docker-data\neo4j\logs` → `/logs`); `neo4j-admin`
chạy qua `docker compose run` nên ghi vào cùng volume. Heap 4G, pagecache 4G (chia máy với lượt chạy dài của p2, coordination §8).

### 6.1 Node

| Label | Khoá (unique) | Thuộc tính |
|---|---|---|
| `Listing` | `listing_id` (long) | `title`, `property_type`, `deal_type`, `price_vnd`, `area_m2`, `price_m2_mil`*, `bedrooms`, `bathrooms`, `floors`, `frontage_m`, `road_width_m`, `direction`, `published_at` (string ISO), `month`, `dup_cluster_id` |
| `PropertyType` | `name` | — |
| `Province` | `key` (= name) | `name`, `name_2025`, `n_listings`, `median_price_m2_mil`* |
| `District` | `key` = `province\|district` | `name`, `n_listings`, `median_price_m2_mil`* |
| `Ward` | `key` = `province\|district\|ward` | `name`, `n_listings`, `median_price_m2_mil`* |
| `Street` | `key` = `province\|district\|street` | `name`, `n_listings` |
| `Project` | `key` = `province\|lower(project)` | `name` (cách viết phổ biến nhất), `n_listings`, `median_price_m2_mil`* |
| `Month` | `key` = `YYYY-MM` | `year`, `month` |
| `Amenity` | `key` (khoá §3.2) | `name` |
| `Cluster` | `cluster_id` (long) | `n_wards`, `median_price_m2_mil`*, `label` (chuỗi mô tả: tỉnh chính + khoảng giá) |
| `MarketStat` | `key` = `scope\|area_key\|property_type\|month\|deal_type` | `scope` (`province`\|`district`\|`ward`\|`project`), `area_key`, `property_type` (tên loại hoặc `ALL`), `month` (`YYYY-MM` hoặc `ALL`), `deal_type`, `n`, `median_price_m2_mil`, `p25_price_m2_mil`, `p75_price_m2_mil`, `median_price_vnd`, `median_area_m2` |

`*` **[materialized]** (tổng hợp tính sẵn để Cypher tổng hợp chạy ms thay vì quét 2,7M node, tránh timeout 10s).
Tính bằng DuckDB `median()` trên `v_listings` (chỉ `deal_type='sale'`, `price_m2_mil IS NOT NULL`), lưu cùng lúc với import.
`MarketStat` chỉ tạo khi `n ≥ 5`.

### 6.2 Quan hệ

| Kiểu | Từ → Đến | Ghi chú |
|---|---|---|
| `OF_TYPE` | Listing → PropertyType | |
| `IN_WARD` | Listing → Ward | khi có ward |
| `IN_DISTRICT` | Listing → District | CHỈ khi thiếu ward (giữ quy ước cũ) ; Ward → District; Street → District; Project → District (quận phổ biến nhất của dự án) |
| `IN_PROVINCE` | Listing → Province | chỉ khi thiếu cả district; District → Province |
| `ON_STREET` | Listing → Street | |
| `IN_PROJECT` | Listing → Project | |
| `POSTED_IN` | Listing → Month | |
| `HAS_AMENITY` | Listing → Amenity | |
| `IN_CLUSTER` | Ward → Cluster | Leiden trên kNN phường (k=5, seed 42; giữ `ward_profiles`/`ward_communities`), phường ≥ 8 tin |
| `STAT_OF` | MarketStat → Province \| District \| Ward \| Project | |

### 6.3 Constraint/index (sau import, `graph/post_import.cypher`)

Unique: `Listing.listing_id`, `PropertyType.name`, `Province.key`, `District.key`, `Ward.key`, `Street.key`,
`Project.key`, `Month.key`, `Amenity.key`, `Cluster.cluster_id`, `MarketStat.key`.
Range index: `District.name`, `Ward.name`, `Street.name`, `Project.name`, `Listing.price_vnd`, `Listing.area_m2`,
`Listing.price_m2_mil`, `Listing.bedrooms`, `Listing.month`, `MarketStat(scope, property_type, month, deal_type)`.
Full-text `project_name_ft` trên `Project.name`, `street_name_ft` trên `Street.name` (CHỈ code API/resolver dùng;
Cypher do LLM sinh bị cấm `CALL`).

CSV import (`data/graph_import/`): node header dùng ID space, vd `:ID(Listing),listing_id:long,title,...,:LABEL`;
quan hệ `:START_ID(Listing),:END_ID(Ward),:TYPE`. ID của node chuỗi = khoá. Bất biến kiểm tra sau import
(`results/graph_report.json`): `count(Listing) = n_listings` của bảng sạch; `sum(Listing.price_vnd)` khớp `rtol=1e-6`.

---

## 7. `results/` (JSON nhỏ — được COMMIT; dữ liệu lớn bị ignore)

`.gitignore`: bỏ dòng `results/`, thay bằng `results/cache/`, `results/eval/raw/`, `results/**/*.parquet`.

### 7.1 `results/ingest_report.json` (Phase 1)
```json
{"schema_version":1,"generated_at":"...","git_sha":"...",
 "raw_rows":3500744,"shards":10,"seconds":0.0,
 "dropped":{"no_text":0,"bad_type":0},
 "dedup":{"exact_dup":0,"prefix_dup":0,"kept":0,"simhash_clusters_multi":0,"simhash_listings_in_multi":0,"simhash_skipped_buckets":0},
 "price":{"null_or_unparsable":0,"fixed_thousand":0,"outliers":0},"area_outliers":0,
 "deal_type":{"sale":0,"rent":0},"missing_pct":{"ward":0.0,"street":0.0,"project":0.0,"bedrooms":0.0},
 "listing_id_collisions":0}
```

### 7.2 `results/eda_summary.json` + `results/eda/*.png` (Phase 1)
`{schema_version, generated_at, git_sha, n_listings, period:{min_month,max_month}, by_property_type:[{property_type,n,median_price_m2_mil}], top_provinces:[{province,n,share}], hn_hcm_share, monthly:[{month,n,median_price_m2_mil}], missing_pct:{<col>:pct}, description_chars:{p50,p90,p99}, tokens_per_listing:{p50,p90,p99}, figures:[{file,title,caption}]}`.
Tên file hình cố định: `price_m2_by_type.png`, `price_m2_by_region.png`, `monthly_trend.png`, `missing_values.png`, `outliers.png`, `duplicates.png`, `top_provinces.png`, `top_districts_hcm_hn.png`.

### 7.3 `results/graph_report.json` (Phase 1)
`{schema_version, generated_at, git_sha, import_seconds, nodes:{<Label>:n}, relationships:{<TYPE>:n}, invariants:{listing_count_ok:bool, price_sum_ok:bool}, communities:{n_wards, n_clusters, modularity, k, seed}}`.

### 7.4 `results/embedding_benchmark.json` (Phase 1 lệnh `ingest` sao chép; Phase 3/4 đọc)
Ngoại lệ có chủ đích của quy ước chung: file là **mảng JSON** ở cấp cao nhất (không có `schema_version`/`generated_at`/`git_sha`) — bản sao thẳng
số đo 2026-10-15 từ `.cache/scratch/bench_final.jsonl` (mỗi dòng JSONL → một phần tử, giữ thứ tự, không lọc trường), làm bằng MỘT dòng code cuối lệnh `ingest`.
Không chạy lại benchmark, không đo lại gì, không tải model nào. `embed_token_budget=32768` lấy từ số đo này và cố định trong `vre.config` (§9.1). Đây là nguồn DUY NHẤT cho so sánh embedding ở notebook/slide/ablation.
```json
[{"model":"intfloat/multilingual-e5-large","params_m":560,"dim":1024,"corpus":20000,"queries":1000,
  "docs_per_s":98.8,"hours_2_8m_chunks":7.87,"peak_vram_gb":2.0,
  "recall_at_1":0.656,"recall_at_10":0.816,"mrr_at_10":0.711,
  "token_budget":32768,"sweep":{"16384":{"docs_per_s":102.0,"peak_vram_gb":1.55},"32768":{"docs_per_s":104.3,"peak_vram_gb":1.97}},
  "gpu_util_avg":91.4,"vram_used_max_mb":3449.0,"power_avg_w":35.1,"sm_clock_avg_mhz":1131}]
```
| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `model` | string | tên HF của model đã đo (đúng chuỗi trong file nguồn) |
| `params_m`, `dim` | int | số tham số (triệu), số chiều vector |
| `corpus`, `queries` | int | 20.000 tin ngẫu nhiên đã dedup; 1.000 truy vấn title → description |
| `docs_per_s` | float | thông lượng bền vững, fp16, SDPA |
| `hours_2_8m_chunks` | float | thời gian ước tính cho ~2,8M chunk |
| `peak_vram_gb` | float | VRAM đỉnh do torch cấp phát |
| `recall_at_1`, `recall_at_10`, `mrr_at_10` | float | chất lượng truy hồi title → description |
| `token_budget` | int | budget tốt nhất của model |
| `sweep` | object `{"<budget>": {docs_per_s, peak_vram_gb}}` | quét 16384/32768/65536/131072 |
| `gpu_util_avg`, `vram_used_max_mb`, `power_avg_w`, `sm_clock_avg_mhz` | float | lấy mẫu `nvidia-smi` lúc benchmark 2026-10-15 (chỉ để trình bày) |
| các trường khác có trong file nguồn (vd `min_free_*`) | — | giữ nguyên vì là bản sao thẳng; không đọc |

Test `tests/unit/test_embedding_benchmark.py`: đúng 4 phần tử; đủ các trường ở bảng trên; khớp từng dòng của file nguồn; phần tử
`intfloat/multilingual-e5-large` có `mrr_at_10` lớn nhất.

### 7.5 `results/night_report.json` (Phase 2)
```json
{"schema_version":1,"generated_at":"...","git_sha":"...",
 "status":"completed|failed","model_slug":"e5_large","collection":"vre_chunks_e5_large",
 "started_at":"...","finished_at":"...","wall_hours":0.0,"resumed_runs":0,
 "steps":[{"name":"checks|chunk|bm25|embed|upload|hnsw|verify|pca|report",
           "status":"done|failed","started_at":"...","finished_at":"...","seconds":0.0,
           "items_done":0,"items_total":0,"note":null}],
 "counters":{"parts_total":0,"parts_embedded":0,"parts_uploaded":0,"chunks":0,"points_in_qdrant":0,
             "chunks_truncated_listings":0,"oom_retries":0,"upload_retries":0,"batch_size_final":0},
 "embed":{"token_budget":32768,"chunks_per_s_avg":0.0},
 "bm25":{"avgdl":0.0},
 "peaks":{"ram_used_gb":0.0,"vram_peak_gb":0.0},
 "qdrant":{"status":"green","indexed_vectors":0,"points":0},
 "errors":[{"at":"...","step":"...","part":null,"type":"...","message":"..."}]}
```
Không có trạng thái `partial` (không giới hạn thời gian): chưa xong thì lần chạy cuối còn mở (`runs[-1].ended_at` rỗng) hoặc `failed`;
chạy lại cùng lệnh để tiếp tục. `wall_hours` = tổng thời gian chạy qua mọi lần resume. `errors[].type = "warn"` cho cảnh báo không dừng.
`embed.chunks_per_s_avg` chỉ là dòng thông tin, không có quyết định/ngưỡng nào dựa vào nó. `peaks.ram_used_gb` = psutil RSS đỉnh của tiến trình
chính, `peaks.vram_peak_gb` = `torch.cuda.max_memory_allocated`. Dry-run ghi report riêng `index/dryrun/night_report.json` (cùng schema, không commit).
Không có khối `gpu`, không có ETA hay dự báo thời gian.

### 7.6 `results/valuation_metrics.json` (Phase 1 tạo `tabular` trên CPU, Phase 2 thêm `full` trên GPU bằng `price --variant full` sau `P1_DONE`)
`{schema_version, generated_at, git_sha, split:{train_months:["2025-06","2026-01"], test_months:["2026-02","2026-03"]}, rent_model:{trained:bool, n_rent:int, reason:str|null}, variants:{tabular:Metrics, full:Metrics|null, baseline_ward_median:Metrics}}`;
`Metrics = {n_train, n_test, mae_mil_m2, mape, r2, by_property_type:[{property_type,n,mae_mil_m2,mape,r2}], by_province_top10:[...], shap_top:[{feature,mean_abs_contribution}], model_path, train_seconds}`. Model: `models/valuation/xgb_{variant}_sale.json` (+ `encoders_{variant}.json`).

### 7.7 Eval (Phase 3)

`benchmark/questions.jsonl` (SSOT câu hỏi, commit, KHÔNG bị ghi đè bởi code):
`{"id":"LK01","category":"lookup","route_gold":"basic_rag","question":"...","gt_kind":"ids|value|refusal","gt_sql":"SELECT ... FROM v_listings ...","value_tolerance":0.05,"tags":["no_diacritics"]}`
- `category` ∈ `lookup, constraint, aggregate, multihop, temporal, comparison, project_street, valuation, unanswerable`.
- `route_gold` ∈ `basic_rag, graph, hybrid, valuation`.
- `gt_kind=ids`: SQL trả cột `listing_id`; `value`: SQL trả 1 dòng (cột đầu = đáp án chính, các cột sau là thành phần); `refusal`: `gt_sql` = null.
- `valuation_input` (thêm 2026-10-15, chỉ `category=valuation`, các câu khác bỏ trống/null): object theo `ValuationInput` (api-contract §3) mô tả BĐS giả định của câu hỏi (bắt buộc `property_type`, `province`, `district`, `area_m2`); GT = `area_m2 × median(price_m2_mil)` (tỷ đồng, cột `est_price_bn`) của tin bán cùng tỉnh × quận × loại, diện tích ±20%, `month` 2026-02…03, `value_tolerance=0.15`.
- Quy ước GT (`vre.eval.ground_truth`): `gt_value` = cột đầu (số thực làm tròn 2 chữ số), `gt_rows` = dòng đầy đủ; mọi ô khác null; cột cỡ nhóm tên `n` / `n_<nhóm>` phải ≥ 20; lookup ≤ 5 `dup_cluster_id`.

`results/benchmark_gt.json`: `{schema_version, generated_at, git_sha, data_fingerprint:{n_listings, price_vnd_sum}, items:[{id, gt_ids:[int]|null, gt_dup_clusters:[int]|null, gt_value:str|null, gt_rows:[object]|null, sql_seconds}]}`.

`results/eval/raw/answers.jsonl` (ignored, lớn): 1 dòng / (câu × hệ thống): `{id, system, answer, route, sources:[int], contexts:[str], cypher, error, fallback, unsupported_citations:[int], latency_ms, input_tokens, output_tokens, cached}`.

`results/eval_summary.json` (commit):
```json
{"schema_version":1,"generated_at":"...","git_sha":"...",
 "index":{"collection":"vre_chunks_e5_large","model_slug":"e5_large","n_points":0},
 "llm":{"generator":"openai/gpt-oss-120b","judge":"openai/gpt-oss-20b"},
 "systems":["llm_only","basic_rag","graph","hybrid"],"n_questions":100,
 "by_category":{"aggregate":{"hybrid":{"n":12,"accuracy":0.0,"mean_points":0.0,"ci95":[0.0,0.0]}}},
 "overall":{"hybrid":{"accuracy":0.0,"mean_points":0.0,"ci95":[0.0,0.0]}},
 "retrieval":{"basic_rag":{"recall_at_5":0.0,"recall_at_10":0.0,"mrr":0.0,"ndcg_at_10":0.0}},
 "ragas":{"basic_rag":{"faithfulness":0.0,"answer_relevancy":0.0,"context_precision":0.0}},
 "cost":{"hybrid":{"latency_ms_p50":0,"latency_ms_p95":0,"input_tokens_mean":0,"output_tokens_mean":0,"usd_per_question":0.0,"bad_citation_rate":0.0}},
 "pairwise":[{"a":"hybrid","b":"basic_rag","scope":"aggregate","metric":"accuracy","diff":0.0,"ci95":[0.0,0.0],"p_value":0.0,"n_boot":10000}],
 "router":{"accuracy":0.0,"confusion":{"graph":{"graph":0,"basic_rag":0,"hybrid":0,"valuation":0}}}}
```
`results/ablation.json`: `{schema_version, generated_at, git_sha, corpus:{n_listings, n_gt_listings, seed}, runs:[{name, embedding, chunk_size, retrieval:"dense|bm25|hybrid", prefilter:bool, rerank:bool, recall_at_5, recall_at_10, mrr, ndcg_at_10, n_questions, latency_ms_p50}]}`
— `embedding` luôn `"e5_large"` (không chạy model embedding khác; so sánh model lấy từ `embedding_benchmark.json` §7.4).
`results/error_analysis.json`: `{schema_version, generated_at, git_sha, counts:{<system>:{<cause>:n}}, examples:[{id, system, category, cause, question, answer_excerpt, gt_value, cypher}]}`;
`cause` ∈ `runtime_error, router_wrong, cypher_error, cypher_wrong_result, filter_wrong, retrieval_miss, llm_hallucination, refusal_missing, insufficient_context`.
Bootstrap: 10.000 mẫu lại theo câu hỏi (ghép cặp giữa hệ thống), seed 42, CI phân vị 2,5/97,5; `p_value` = tỷ lệ mẫu có dấu ngược `diff`.

### 7.8 `results/examples.json` (Phase 3, nguồn ví dụ cho notebook + slide)
`{schema_version, generated_at, git_sha, rag:[{question, filters:object, hits:[{listing_id, title, score, province, district, price_vnd}]}],
graph:[{question, cypher, columns:[str], rows:[object], latency_ms}], valuation:[{input:object, result:object}],
answers:[{id, system, question, answer, route, cited:[int]}]}` — ví dụ lấy từ lần chạy eval thật (không viết tay), mỗi mảng 3–8 phần tử.
Writer: `vre_pipeline.py report examples`.

### 7.9 Bảng tra writer/reader

| File | Writer | Reader |
|---|---|---|
| `ingest_report.json`, `eda_summary.json`, `eda/*.png`, `graph_report.json` | Phase 1 | Phase 3 (notebook 01/03, README), Phase 4 |
| `embedding_benchmark.json` | Phase 1 (`ingest`, sao chép thẳng từ `.cache/scratch/bench_final.jsonl`) | Phase 3 (notebook 02, ablation), Phase 4 |
| `night_report.json` | Phase 2 | Phase 3, Phase 4 |
| `valuation_metrics.json` | Phase 1 (`tabular`), Phase 2 (`full`) | Phase 3 (API, notebook 05), Phase 4 |
| `handoff/*.json` (gitignored) | p1, p2 (§7.10) | p1, p2, cổng Phase 3 |
| `benchmark_gt.json`, `eval_summary.json`, `ablation.json`, `error_analysis.json`, `examples.json` | Phase 3 | Phase 3 (API `/api/benchmark`, notebook 02/04), Phase 4 |

### 7.10 `results/handoff/<SIGNAL>.json` (tín hiệu giữa p1 và p2; gitignored)
Danh sách tín hiệu, payload từng loại, thứ tự gửi và cách chờ: [`coordination.md`](coordination.md) §3–§4 (SSOT). Vỏ chung:
```json
{"schema_version":1,"signal":"CLEAN_DATA_READY","from":"p1","seq":1,"at":"2026-10-15T10:00:00Z","payload":{}}
```
`signal` ∈ `SKELETON_READY, CLEAN_DATA_READY, INDEX_STARTED, INDEX_DONE, INDEX_FAILED, P1_DONE, P2_DONE, CHANGE_REQUEST, CHANGE_APPLIED, NOTE`;
`from` ∈ `p1, p2`. Tên file: `<signal>.json`; có `payload.id` ⇒ `<signal>-<id>.json`; `NOTE` ⇒ `NOTE-<from>-<yyMMddHHmmss>.json`.
Ghi nguyên tử bằng `vre_pipeline.py signal` (§9.3). Gửi lại cùng tín hiệu ghi đè và tăng `seq`.

---

## 8. Dữ liệu địa lý — `data/geo/centroids.json` (Phase 3 Lane C ghi, Lane B đọc)

`{schema_version, generated_at, source: "geoBoundaries gbOpen VNM ADM1/ADM2 (CC BY 4.0)", source_vintage: str,
items: [{level: "province"|"district", key: str, name: str, lat: float, lon: float}], unmatched: [{level, key}], coverage: {province: float, district: float}}`
— `key` dùng ĐÚNG khoá node Neo4j (`Province.key` = tên tỉnh; `District.key` = `province|district`, data-contract §6.1). Toạ độ WGS84.
Thư mục `data/geo/` bị gitignore (license nguồn yêu cầu ghi công; file sinh lại bằng `scripts/build_geo.py`).

---

## 9. Bề mặt code dùng chung (p1 tạo trong khung package; p2 viết code theo đúng các tên này)

Đổi bất kỳ tên nào ở đây ⇒ sửa mục này trước, rồi gửi `NOTE`/`CHANGE_REQUEST` (coordination.md §3).

### 9.1 `vre.config.Settings` (pydantic-settings, prefix `VRE_`)

| Khoá | Mặc định | Ghi chú |
|---|---|---|
| `data_root` | `D:\Vietnam-Real-Estates` | gốc DỮ LIỆU (không phải gốc code); các đường dẫn dưới đây là property tính từ đây |
| `raw_dir`, `clean_path`, `dup_map_path`, `index_dir`, `results_dir`, `models_dir`, `logs_dir`, `handoff_dir`, `features_dir`, `cache_dir`, `compose_file` | property | `data/raw/vietnam-real-estates`, `data/clean/listings.parquet`, `data/clean/dup_map.parquet`, `index`, `results`, `models`, `logs`, `results/handoff`, `data/features`, `.cache`, `docker-compose.yaml` dưới `data_root` |
| `duckdb_threads`, `duckdb_memory_gb` | `12`, `12` | chia CPU/RAM với p2 |
| `ingest_workers` | `8` | |
| `chunk_size`, `chunk_overlap`, `max_chunks_per_listing`, `header_max_tokens`, `part_size` | `400`, `100`, `64`, `96`, `50000` | |
| `embed_max_length`, `embed_token_budget`, `embed_max_batch`, `embed_min_token_budget` | `512`, `32768`, `256`, `4096` | `32768` cố định từ số đo 2026-10-15 — không đo lại; `embed_min_token_budget` = sàn khi OOM ⇒ ½ ngân sách |
| `tokenize_workers`, `tokenize_prefetch_parts`, `bm25_workers` | `2`, `3`, `4` | giảm 2026-10-15 (CR-04): commit (~42,8 GB) là giới hạn; GPU vẫn là nút cổ chai nên không kéo dài lượt chạy |
| `qdrant_url`, `qdrant_grpc_port`, `qdrant_upload_parallel`, `qdrant_upload_batch`, `qdrant_max_indexing_threads`, `qdrant_indexing_threshold`, `qdrant_green_timeout_s` | `http://127.0.0.1:6333`, `6334`, `1`, `256`, `12`, `20000`, `10800` | `qdrant_upload_parallel=1`: upload chạy trong tiến trình riêng với `wait=True`, 1 luồng gRPC đủ ~100–130 point/s; 8 luồng + `wait=False` làm quá tải gRPC khi HNSW đang xây (CR-02, dry-run 2026-10-15) |
| `collection`, `collection_alias`, `dryrun_collection`, `dryrun_parts` | `vre_chunks_e5_large`, `vre_chunks`, `vre_dryrun`, `2` | |
| `neo4j_uri`, `neo4j_user`, `neo4j_password` | env cũ | |
| `xgb_nthread`, `xgb_device_tabular`, `xgb_device_full`, `rent_min_rows` | `12`, `cpu`, `cuda`, `20000` | tabular trên CPU để không tranh GPU với p2 |
| `pca_dim`, `pca_fit_sample` | `32`, `300000` | |
| `min_free_disk_gb_start`, `min_free_disk_gb_part` | `45`, `5` | kiểm fail-fast |
| `night_max_attempts`, `night_retry_wait_s` | `3`, `120` | `night.ps1` đọc qua `vre_pipeline.py config-get` |
| `handoff_poll_s` | `300` | |
| tham số LLM/RAG/chat cũ | giữ tên env cũ qua `validation_alias` | Phase 3 dùng |

Hằng số module: `EMBED_MODELS = {"e5_large": ("intfloat/multilingual-e5-large", 1024, "mean", "passage: ", "query: ")}`, `EMBED_MODEL_SLUG = "e5_large"`
(một model duy nhất, không có logic chọn model). `.env` chỉ được đọc khi `VRE_LOAD_DOTENV != "0"`.

### 9.2 Kiểm tra fail-fast — `vre.checks` (p1)

Mỗi kiểm tra là hàm thuần trên dữ kiện + một collector (ai-driven-design). `require(settings, *names)` chạy các kiểm tra được nêu, in kết quả
và raise `PreconditionError` nếu có `fail` (không có module hay lệnh kiểm tra riêng, không ghi file report riêng).

| Kiểm tra | Fail khi |
|---|---|
| `raw_shards_present` | thiếu bất kỳ `shard_0000..0009.parquet` |
| `e5_large_cached` | snapshot `intfloat/multilingual-e5-large` không có trong `HF_HOME` (không tải — báo lỗi) |
| `gpu` | `torch.cuda.is_available()` False |
| `qdrant_healthy`, `neo4j_healthy` | `/readyz` ≠ 200; `RETURN 1` lỗi |
| `d_free_space` | ổ D trống < `min_free_disk_gb_start` |
| `groq_key_present` | `.env` không có `GROQ_API_KEY` (chỉ kiểm tồn tại) |
| `env_paths_on_d` | `envguard`: biến cache/TMP nào đó không bắt đầu `D:` |
| `clean_data_ready` | thiếu `listings.parquet` hoặc fingerprint ≠ `results/handoff/CLEAN_DATA_READY.json` |

### 9.3 CLI `scripts/vre_pipeline.py` (p1 viết; lệnh của p2 import lười bên trong thân lệnh)

| Lệnh | Kiểm tra đầu lệnh (`vre.checks.require`) | Gọi hàm | Chủ code |
|---|---|---|---|
| `ingest` | `raw_shards_present`, `env_paths_on_d`, `d_free_space`, `groq_key_present` | `vre.data.ingest.run(settings) -> dict` | p1 |
| `eda` | `clean_data_ready` | `vre.data.eda.run(settings) -> dict` | p1 |
| `graph` | `neo4j_healthy`, `clean_data_ready` | `vre.index.graph_import.run(settings) -> dict` | p1 |
| `price --variant tabular\|full` | `clean_data_ready`; `full` thêm `gpu` + có file PCA §4 | `vre.valuation.train.run(settings, variant: Literal["tabular","full"]) -> dict` | p1 |
| `night [--resume/--no-resume] [--dry-run]` | `e5_large_cached`, `gpu`, `qdrant_healthy`, `d_free_space`, `env_paths_on_d`, `clean_data_ready` | `vre.pipeline.night.run(settings, resume: bool, dry_run: bool) -> int` (0 = completed, 20 = lỗi cần người) | p2 |
| `status [--dry-run]` | — | `vre.pipeline.report.status(settings, dry_run: bool) -> dict` | p2 |
| `smoke-search QUERY [--collection NAME]` | `qdrant_healthy` | `vre.index.qdrant_store.smoke_search(settings, query: str, collection: str \| None = None, limit: int = 10) -> list[dict]` | p2 |
| `signal NAME --from p1\|p2 --payload JSON` | — | `vre.common.write_handoff(settings, name, sender, payload) -> Path` | p1 |
| `config-get KEY` | — | in giá trị một khoá Settings | p1 |
| `all --resume` | theo từng lệnh | `ingest → eda → graph → price tabular → night → price full` | p1 |

Tiện ích chung trong `vre.common` (p1): `write_json_atomic(path, obj)`, `git_sha() -> str`, `write_handoff(...)`, `scalar(...)`.
Mọi lệnh idempotent: artifact đã có + fingerprint khớp ⇒ in "đã xong", thoát 0. Script tự thêm `Path(__file__).resolve().parents[1] / "src"`
vào `sys.path` (không cài editable, coordination §7).
