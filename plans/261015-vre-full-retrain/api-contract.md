# API Contract — FastAPI (`api/`) ↔ React (`web/`)

SSOT cho mọi giao tiếp giữa backend và frontend ở Phase 3. Cả hai lane (api, web) code theo file này, KHÔNG
chép lại vào brief. Đổi hợp đồng ⇒ sửa file này trước, rồi sửa cả `api/schemas.py` và `web/src/api/types.ts`
trong cùng commit. Kiểm tra tự động: `tests/api/test_contract.py` sinh OpenAPI từ app và so khớp tên trường/enums
với bảng dưới; `web/src/api/types.ts` sinh bằng `openapi-typescript` từ `api/openapi.json` (commit file sinh).

## 0. Quy ước chung

- Base path: `/api`. JSON UTF-8. Mọi tên trường `snake_case` (cả request và response, cả payload SSE).
- Số: `listing_id`, `dup_cluster_id` là **số nguyên JSON** (≤ 2⁵³−1, an toàn cho JS `Number`). Tiền `*_vnd` là số thực VND;
  `*_m2_mil` là triệu đồng/m²; `*_m2` là m².
- Optional/null: trường ghi kiểu `T | null` LUÔN có mặt trong response với giá trị `null` khi không có dữ liệu.
  Trường ghi `T?` có thể vắng mặt (chỉ dùng trong REQUEST; server điền mặc định). Không bao giờ dùng `""`, `0` hay `NaN` thay cho null.
- Thời gian: ISO-8601 UTC (`2026-10-15T07:14:00Z`). Tháng: `YYYY-MM`.
- Auth: nếu server đặt `VRE_API_TOKEN`, mọi `/api/*` (trừ `/api/health`) yêu cầu header `X-API-Key: <token>`.
  KHÔNG nhận token qua query string. Web lưu token nhập tay trong `sessionStorage` (khoá `vre_api_key`).
- Thứ tự middleware (sửa bug cũ "rate-limit không chặn sai key"): request-id → body-size → **rate limit theo IP** → auth → handler.
- Mọi response có header `X-Request-Id` (8 ký tự hex). Lỗi luôn kèm `request_id` trong envelope.
- CORS: chỉ origin trong `VRE_CORS_ORIGINS` (mặc định `http://127.0.0.1:5173,http://localhost:5173`).
- Giới hạn: body ≤ 8 KiB; `question` 1–500 ký tự (sau `strip`); `VRE_CHAT_MAX_CONCURRENT` (4) câu đồng thời,
  chờ slot tối đa 60s rồi trả `busy`; `VRE_CHAT_RATE_PER_MIN` (20)/IP/phút; timeout tổng mỗi câu `VRE_CHAT_TIMEOUT_S` (120).

## 1. Enums

| Enum | Giá trị |
|---|---|
| `SystemId` | `llm_only`, `basic_rag`, `graph`, `hybrid` |
| `Route` | `basic_rag`, `graph`, `hybrid`, `valuation` (chỉ `hybrid` mới có route khác tên chính nó) |
| `PropertyType` | `Nhà`, `Đất`, `Căn hộ chung cư`, `Biệt thự/Nhà liền kề`, `Shophouse` |
| `DealType` | `sale`, `rent` |
| `ErrorCode` | `invalid_request`, `unauthorized`, `not_found`, `payload_too_large`, `unsupported_media_type`, `rate_limited`, `busy`, `system_unavailable`, `upstream_llm_error`, `timeout`, `internal_error` |
| `NodeLabel` | `Listing`, `PropertyType`, `Province`, `District`, `Ward`, `Street`, `Project`, `Month`, `Amenity`, `Cluster`, `MarketStat` |
| `RelType` | `OF_TYPE`, `IN_WARD`, `IN_DISTRICT`, `IN_PROVINCE`, `ON_STREET`, `IN_PROJECT`, `POSTED_IN`, `HAS_AMENITY`, `IN_CLUSTER`, `STAT_OF` |
| `GeoLevel` | `province`, `district`, `ward` |
| `BenchmarkCategory` | `lookup`, `constraint`, `aggregate`, `multihop`, `temporal`, `comparison`, `project_street`, `valuation`, `unanswerable` |

## 2. Kiểu dùng chung

```ts
QueryFilters {               // mọi trường nullable; null = không lọc
  province: string | null; district: string | null; ward: string | null;
  property_type: PropertyType | null; deal_type: DealType | null;
  bedrooms: number | null;               // số nguyên, so khớp bằng
  min_price_mil: number | null; max_price_mil: number | null;   // triệu đồng (giá tổng)
  min_area_m2: number | null;   max_area_m2: number | null;
  project: string | null; street: string | null;
  month_from: string | null;  month_to: string | null;          // "YYYY-MM", bao gồm hai đầu
}

Address { street: string | null; ward: string | null; district: string | null; province: string | null }

ListingCard {
  listing_id: number; dup_cluster_id: number;
  title: string | null; property_type: PropertyType; deal_type: DealType;
  price_vnd: number | null; price_m2_mil: number | null; area_m2: number | null;
  bedrooms: number | null; bathrooms: number | null;
  address: Address; project: string | null; published_month: string | null;
  description_excerpt: string;           // ≤ 400 ký tự, có thể ""
  amenities: string[];                   // khoá amenity (data-contract §3.2)
  score: number | null;                  // điểm truy xuất (RRF/cosine/rerank), null nếu từ Cypher
  cited: boolean                         // có được trích [Tin#id] trong câu trả lời
}

ListingDetail = ListingCard & {
  description: string | null; floors: number | null; frontage_m: number | null; road_width_m: number | null;
  house_direction: string | null; balcony_direction: string | null; published_at: string | null;
  n_raw_rows: number; similar_in_cluster: number   // số tin khác cùng dup_cluster_id
}

CypherResult {
  cypher: string;                        // câu Cypher cuối cùng đã chạy (đã thêm LIMIT)
  columns: string[]; rows: Array<Record<string, string | number | boolean | null>>;   // ≤ 50 dòng
  error: string | null;                  // thông điệp lỗi Neo4j/guard — HIỂN THỊ cho người dùng (sửa bug "ẩn lỗi Cypher")
  attempts: number                       // 1..3
}

ShapItem { feature: string; value: string | number | null; contribution_log: number }  // đóng góp vào log(giá/m²)

ErrorEnvelope { error: { code: ErrorCode; message: string; request_id: string; details: object | null } }
```

## 3. Endpoint REST

| Method + path | Request | 200 response | Lỗi có thể |
|---|---|---|---|
| `GET /api/health` | — | `{status: "ok" \| "degraded", qdrant: boolean, neo4j: boolean, llm_configured: boolean, version: string}` (không cần auth, không gọi LLM) | — |
| `GET /api/info` | — | `{systems: Array<{id: SystemId, name: string, available: boolean, unavailable_reason: string \| null}>, index: {collection: string, model_slug: string, n_listings: number, n_points: number}, graph_ok: boolean, data_period: {min_month: string, max_month: string}, limits: {max_question_chars: number, rate_per_min: number}}`. Thực tế luôn `index.collection = "vre_chunks_e5_large"`, `index.model_slug = "e5_large"` (embedding duy nhất đã chốt 2026-10-15; kiểu giữ `string`) | 401, 429 |
| `GET /api/filters/options` | — | `{provinces: string[], districts: Record<string, string[]>, property_types: PropertyType[], months: string[]}` (districts khoá theo tỉnh) | 401, 429 |
| `GET /api/listings/{listing_id}` | path int | `ListingDetail` | 400, 401, 404, 429 |
| `POST /api/valuate` | `ValuationInput` | `ValuationResult` | 400(422→400), 401, 429, 503 (model chưa có) |
| `GET /api/stats/overview` | — | `{n_listings: number, n_raw_rows: number, n_duplicates_removed: number, by_property_type: Array<{property_type: PropertyType, n: number, median_price_m2_mil: number \| null}>, by_province_top: Array<{province: string, n: number, median_price_m2_mil: number \| null}>}` | 401, 429 |
| `GET /api/stats/market?province=&district=&property_type=&deal_type=sale` | query: `province` bắt buộc, còn lại tuỳ chọn | `{scope: {province: string, district: string \| null, property_type: PropertyType \| null, deal_type: DealType}, monthly: Array<{month: string, n: number, median_price_m2_mil: number \| null}>, by_area: Array<{key: string, name: string, n: number, median_price_m2_mil: number \| null}>}` (by_area = các quận của tỉnh, hoặc các phường của quận; nguồn: MarketStat) | 400, 401, 404, 429 |
| `GET /api/geo/areas?level=district&province=` | `level: GeoLevel` bắt buộc; `province` bắt buộc khi level≠province | `{level: GeoLevel, items: Array<{key: string, name: string, lat: number \| null, lon: number \| null, n_listings: number, median_price_m2_mil: number \| null}>, coverage: number}` (lat/lon null khi không khớp GeoJSON; `coverage` = tỷ lệ item có toạ độ) | 400, 401, 429 |
| `GET /api/graph/subgraph?listing_id=` hoặc `?node_label=&key=&limit=40` | đúng MỘT trong hai dạng; `limit` 1..100 | `{nodes: Array<{id: string, label: NodeLabel, name: string, props: Record<string, string \| number \| boolean \| null>}>, edges: Array<{source: string, target: string, type: RelType}>, truncated: boolean}`; `id` = `"<Label>:<key>"` | 400, 401, 404, 429, 503 |
| `GET /api/benchmark` | — | nội dung `results/eval_summary.json` (data-contract §7.7) + `{ablation: <results/ablation.json> \| null, embedding_benchmark: <results/embedding_benchmark.json (mảng, §7.4)> \| null}` | 401, 404 (chưa chạy eval), 429 |

```ts
ValuationInput {
  property_type: PropertyType; deal_type?: DealType /* = "sale"; "rent" → 400 nếu không có model thuê */;
  province: string; district: string; ward?: string | null;
  area_m2: number;            // > 0, ≤ 100000
  bedrooms?: number | null; bathrooms?: number | null; floors?: number | null;
  frontage_m?: number | null; road_width_m?: number | null; house_direction?: string | null;
  month?: string | null;      // "YYYY-MM", mặc định tháng mới nhất trong dữ liệu
  amenities?: string[];       // khoá amenity
  asking_price_vnd?: number | null   // nếu có: trả verdict so với dự báo
}

ValuationResult {
  predicted_price_m2_mil: number; predicted_price_vnd: number;
  interval_price_vnd: { low: number; high: number };     // khoảng từ phân vị sai số trên tập test (p10/p90 theo loại)
  baseline_ward_median_price_m2_mil: number | null; comparables_n: number;
  verdict: "below_market" | "fair" | "above_market" | null;   // null nếu không có asking_price_vnd; fair = trong interval
  shap: ShapItem[];           // top 8 theo |contribution_log|
  model: { variant: "full" | "tabular"; path: string; trained_at: string; test_mape: number }
}
```

## 4. SSE: `POST /api/chat` và `POST /api/compare`

Request `Content-Type: application/json`; response `Content-Type: text/event-stream`, `Cache-Control: no-store`,
`X-Accel-Buffering: no`. Mỗi sự kiện: `event: <name>\ndata: <json một dòng>\n\n`. Heartbeat `: ping\n\n` mỗi 15s.
Lỗi xảy ra TRƯỚC khi mở stream (auth, rate limit, validate, busy) ⇒ trả HTTP lỗi + `ErrorEnvelope` (không SSE).
Lỗi SAU khi mở stream ⇒ sự kiện `error` rồi `done` (status HTTP vẫn 200).

```ts
ChatRequest {
  question: string;                 // 1..500 ký tự
  system?: SystemId;                // mặc định "hybrid"
  filters?: QueryFilters | null;    // bộ lọc chỉnh tay; trường khác null GHI ĐÈ trường LLM trích ra
  top_k?: number;                   // 1..20, mặc định 5
  use_rerank?: boolean;             // mặc định false
  stream_tokens?: boolean           // mặc định true; false ⇒ không có sự kiện token
}
CompareRequest { question: string; systems?: SystemId[] /* mặc định cả 4, không trùng */; filters?: QueryFilters | null; top_k?: number }
```

Mọi payload sự kiện có `request_id: string` và `system: SystemId`.

| Sự kiện | Payload (ngoài `request_id`, `system`) | Khi nào |
|---|---|---|
| `meta` | `{question: string, started_at: string}` | đầu tiên, 1 lần / hệ thống |
| `route` | `{route: Route, reason: string}` | chỉ `system="hybrid"`, sau router |
| `filters` | `{filters: QueryFilters, source: "llm" \| "manual" \| "merged", note: string \| null}` | `basic_rag`, `hybrid` (route có RAG); `note` = `"filter_dropped"` khi bỏ lọc vì 0 kết quả |
| `cypher` | `CypherResult` | `graph`, `hybrid` (route graph/hybrid) — gửi cả khi `error≠null` |
| `valuation` | `ValuationResult` | `hybrid` khi `route="valuation"` |
| `sources` | `{sources: ListingCard[]}` (≤ `top_k`·2, thứ tự: tin được trích trước, rồi theo điểm; `cited` cập nhật lại ở `answer`) | trước khi sinh câu trả lời; `llm_only` gửi `[]` |
| `token` | `{text: string}` | 0..n lần, mảnh văn bản câu trả lời |
| `answer` | `{answer: string, cited: number[], unsupported_citations: number[], fallback: boolean, route: Route \| null}` | 1 lần, văn bản cuối đã chuẩn hoá `[Tin#id]` |
| `usage` | `{latency_ms: number, input_tokens: number, output_tokens: number, usd: number, cached: boolean}` | sau `answer` |
| `error` | `{code: ErrorCode, message: string}` | lỗi trong lúc stream; KHÔNG kèm stack trace |
| `done` | `{status: "ok" \| "error"}` | cuối cùng của MỖI hệ thống |
| `compare_done` | `{systems_ok: SystemId[], systems_failed: SystemId[]}` (không có `system`) | chỉ `/api/compare`, sự kiện cuối cùng của stream |

Bất biến thứ tự (theo từng `system`): `meta` → [`route`] → [`filters`] → [`cypher`] → [`valuation`] → `sources` → `token`* → `answer` → `usage` → `done`;
hoặc bất kỳ lúc nào sau `meta`: `error` → `done`. Ở `/api/compare` các hệ thống chạy song song, sự kiện xen kẽ, phân biệt bằng `system`.

## 5. Mã trạng thái + envelope lỗi

| HTTP | `code` | Ghi chú |
|---|---|---|
| 400 | `invalid_request` | validate thất bại (FastAPI 422 được map về 400); `details` = `{fields: [{loc: string[], msg: string}]}` |
| 401 | `unauthorized` | thiếu/sai `X-API-Key` |
| 404 | `not_found` | listing/area/benchmark chưa có |
| 413 | `payload_too_large` | body > 8 KiB |
| 415 | `unsupported_media_type` | POST không phải `application/json` |
| 429 | `rate_limited` | kèm header `Retry-After` (giây) |
| 502 | `upstream_llm_error` | LLM lỗi sau khi đã retry/fallback (chỉ trước khi mở stream; trong stream dùng sự kiện `error`) |
| 503 | `busy` \| `system_unavailable` | hết slot sau 60s; hoặc hệ thống cần Neo4j/model mà chưa sẵn sàng (`message` nói rõ thiếu gì) |
| 504 | `timeout` | vượt `VRE_CHAT_TIMEOUT_S` trước khi mở stream |
| 500 | `internal_error` | `message` = "Lỗi khi xử lý (mã <request_id>)", chi tiết chỉ ghi log server |

Log server không ghi nội dung câu hỏi (chỉ độ dài), giữ chính sách của `app/server.py` cũ.

## 6. Chế độ fake cho test

`VRE_FAKE_BACKENDS=1` (chỉ dùng trong test/CI): API trả dữ liệu cố định từ `api/fixtures/*.json` cho mọi endpoint
và phát lại một chuỗi SSE ghi sẵn cho `/api/chat`, `/api/compare` — không cần Qdrant/Neo4j/LLM/GPU. Playwright E2E
chạy với chế độ này. Fixture PHẢI hợp lệ theo schema Pydantic (test `tests/api/test_fixtures_valid.py`).

## 7. Facade Python giữa backend lõi (Lane A) và API (Lane B)

API KHÔNG import trực tiếp `vre.rag`/`vre.graph`/`vre.hybrid`; chỉ gọi các hàm dưới đây (Lane A sở hữu, Lane B dùng).
Chữ ký là hợp đồng — đổi thì sửa file này trước.

```python
# src/vre/qa/__init__.py
SystemId = Literal["llm_only", "basic_rag", "graph", "hybrid"]

def stream_answer(system: SystemId, question: str, filters: dict | None = None, top_k: int = 5,
                  use_rerank: bool = False, stream_tokens: bool = True) -> Iterator[tuple[str, dict]]:
    """Sinh (event_name, payload) đúng bảng §4, KHÔNG gồm `request_id`/`system` (API tự thêm).
    `filters`: dict theo QueryFilters (§2) hoặc None. Sau sự kiện đầu tiên không raise nữa: mọi lỗi thành
    ("error", {"code", "message"}) rồi ("done", {"status": "error"}). Thread-safe (nhiều request song song)."""

def answer(system: SystemId, question: str, filters: dict | None = None, top_k: int = 5,
           use_rerank: bool = False) -> dict:
    """Bản không stream cho eval runner. Trả dict đúng các khoá của results/eval/raw/answers.jsonl
    (data-contract §7.7) trừ `id`."""

def system_status() -> dict[SystemId, tuple[bool, str | None]]:
    """(available, unavailable_reason) cho /api/info — vd graph cần Neo4j khớp manifest."""

# src/vre/valuation/predict.py
def valuate(payload: dict) -> dict:
    """payload theo ValuationInput (§3) → dict theo ValuationResult (§3). Raise ValueError (→ 400) khi input sai,
    RuntimeError (→ 503) khi chưa có model. Ưu tiên model `full`, lùi về `tabular` nếu `full` chưa có (ghi `model.variant`)."""
```
