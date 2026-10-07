# Vietnam Real Estates — từ RAG đến GraphRAG

Trợ lý hỏi đáp thị trường bất động sản Việt Nam trên dữ liệu tin đăng
[tinixai/vietnam-real-estates](https://huggingface.co/datasets/tinixai/vietnam-real-estates).
Dự án so sánh ba cách trả lời trên cùng dữ liệu và cùng bộ câu hỏi:

| Hệ thống | Cách tìm bằng chứng | Mạnh ở |
| --- | --- | --- |
| **Basic RAG** | LLM trích bộ lọc → dense (E5) + BM25 trên Qdrant gộp bằng RRF, lọc trước theo payload → LLM trả lời có `[Tin#ID]` | Tìm theo mô tả tự do: "gần trường", "có thang máy" |
| **Graph** | Text2Cypher: LLM sinh Cypher chỉ đọc trên đồ thị Neo4j → LLM diễn giải kết quả | Câu tổng hợp, đếm, xếp hạng, đa chặng |
| **Hybrid** | Router chọn Basic RAG, Graph hoặc gộp ngữ cảnh cả hai; đồ thị lỗi hoặc rỗng thì lui về Basic RAG | Câu vừa có điều kiện cấu trúc vừa có mô tả |

Mã nằm trong [src/](src/) và được dùng chung bởi 5 notebook trình bày và chatbot. Có hai chỉ mục cùng cấu trúc:

| Chỉ mục | Dữ liệu | Qdrant | Neo4j | Tạo bởi | Dùng cho |
| --- | --- | --- | --- | --- | --- |
| Mẫu | `N_ROWS` dòng đầu (20.000) | `listings_sample` | `graph_db` (7687) | notebook 01, 03 | Notebook, chatbot khi chưa có bản đầy đủ |
| Đầy đủ | ~3,5 triệu dòng | `listings_full` | `graph_db_full` (7688) | [scripts/index_full.py](scripts/index_full.py) | Chatbot |

## Mục lục

1. [Dữ liệu](#dữ-liệu)
2. [Cài đặt](#cài-đặt)
3. [Notebook trình bày](#notebook-trình-bày)
4. [Chatbot](#chatbot)
5. [Đánh giá](#đánh-giá)
6. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
7. [Kiểm thử](#kiểm-thử)
8. [Giới hạn](#giới-hạn)

## Dữ liệu

Dataset có khoảng 3,5 triệu tin, 19 cột: tiêu đề, mô tả, loại hình, tỉnh, quận, phường, đường, dự án,
giá (VND), diện tích (m²), số phòng, hướng, ngày đăng… Notebook 01 đọc theo luồng `N_ROWS` dòng đầu
(mặc định 20.000), không tải cả dataset. Tên cột thật được ánh xạ một lần trong `COL` ở
[src/config.py](src/config.py).

Làm sạch ([src/data.py](src/data.py)):
- Gộp khoảng trắng, giữ dấu tiếng Việt.
- Ẩn số điện thoại và email.
- Bỏ tin thiếu cả tiêu đề lẫn mô tả, bỏ tin trùng.
- Chuẩn hoá địa danh: `"8"` → `Quận 8`, `"P.5"` → `Phường 5`, `TP. Hồ Chí Minh` → `Hồ Chí Minh`.
- Sửa giá nhập theo nghìn đồng: tin ghi "10 tỷ" nhưng lưu 10.000.000 được nhân 1.000 nếu sau đó
  giá/m² hợp lý. Tin cho thuê hoặc không sửa được thì bỏ giá.
- `listing_id` là số thứ tự dòng trong dataset gốc.

Trên 20.000 dòng đầu: còn 19.925 tin sau làm sạch, thành 21.030 chunk. Đồ thị có 22.116 node và 41.978 cạnh.

## Cài đặt

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env                       # sửa N_ROWS, NEO4J_PASSWORD nếu cần
docker compose up -d --wait               # Qdrant (vector_db) + Neo4j mẫu (graph_db)
# Đặt GROQ_API_KEY trong .env (tạo khoá tại console.groq.com)
```

| Thành phần | Chọn |
| --- | --- |
| Embedding | `intfloat/multilingual-e5-small` (384 chiều), tiền tố `passage:`/`query:`; trên GPU chạy fp16 (nhanh ~3 lần, cosine với bản fp32 ≥ 0,999) |
| Chunk | Cửa sổ 400 token, chồng lấn 100; header (tiêu đề, loại, địa chỉ, giá, diện tích) lặp ở mọi chunk |
| Vector DB | Qdrant: vector dense trên đĩa + bản nén int8 trong RAM (tìm trên bản nén, lấy dư `QDRANT_OVERSAMPLING` lần rồi chấm lại bằng vector gốc), payload index để pre-filter. Số chiều collection lấy từ mô hình embedding và được kiểm khi mở lại |
| Từ khoá | BM25 dạng sparse vector trong cùng collection Qdrant (IDF tính phía server), tách từ tiếng Việt bằng `pyvi`. Dense và BM25 đi chung một lượt gọi Qdrant (`query_batch_points`) |
| Rerank | Cross-encoder `mmarco-mMiniLMv2`, tắt mặc định; `RERANK=1` để `retrieve()` lấy `RERANK_CANDIDATES` tin rồi chấm lại |
| Đồ thị | Neo4j: `(Listing)-[:IN_WARD]->(Ward)-[:IN_DISTRICT]->(District)-[:IN_PROVINCE]->(Province)`, `(Listing)-[:OF_TYPE]->(PropertyType)`; Listing có thêm `project`, `street`, `direction` (có index) để lọc theo dự án/đường; script [graph/load.cypher](graph/load.cypher) |
| LLM | Groq (API tương thích OpenAI): `openai/gpt-oss-120b` sinh, `openai/gpt-oss-20b` chấm và làm dự phòng khi quá tải; structured output cho bộ lọc, Cypher, router và chấm điểm; timeout `LLM_TIMEOUT`, đầu ra có cấu trúc bị cắt ở `max_tokens` báo lỗi rõ |

Text2Cypher có hai lớp chặn lệnh ghi: regex trong [src/graph.py](src/graph.py) (cả lệnh quản trị như
`SHOW`, `USE`, `CALL`), và giao dịch chỉ đọc của Neo4j (`execute_read`, lỗi `AccessMode` nếu câu lệnh có ghi).
Câu Cypher do LLM sinh còn được thêm `LIMIT` nếu thiếu, bị Neo4j huỷ sau `CYPHER_TIMEOUT` giây và chỉ đọc
tối đa `CYPHER_MAX_ROWS` dòng. Danh sách loại hình trong prompt đọc từ đồ thị đang dùng.

Câu trả lời được kiểm trích dẫn: mọi `[Tin#ID]` không nằm trong nguồn đã đưa vào ngữ cảnh được trả về ở
`unsupported_citations`. Khi bộ lọc quá chặt và phải bỏ lọc, prompt báo cho LLM rằng các tin chỉ gần đúng.

## Notebook trình bày

[notebooks/](notebooks/) dựng theo `Outline_chia_cell_code_RAG_KnowledgeGraph_BDS.pdf`: 5 notebook,
51 cell, gồm 19 LIVE (chạy trước lớp), 22 CACHE (chạy sẵn, trên lớp chỉ đọc từ đĩa) và 10 ẨN (gọi hàm
trong `src/`). Mỗi cell code có một cell markdown phía trên ghi mã, nhãn, slide, tiêu đề là câu kết luận
và mục tiêu. Cell toy có `assert` khớp số trên slide. Chỗ còn chờ số liệu thật ghi `[ĐIỀN SAU]`.

| Notebook | Người | Nội dung | Đầu ra |
| --- | --- | --- | --- |
| `00_setup` | 1 | Cấu hình, bảng ánh xạ cột `COL`, seed | Mọi notebook gọi `%run ./00_setup.ipynb` |
| `01_rag_indexing` | 1 | Khảo sát, làm sạch, chunk, embed, nạp Qdrant | `data/listings_clean.parquet`, Qdrant `listings_sample` |
| `02_retrieval_generation` | 2 | Bộ lọc LLM, dense, BM25, RRF, pre/post-filter, rerank, sinh câu trả lời | `src/rag.py` |
| `03_knowledge_graph` | 3 | Neo4j, entity resolution, Cypher, Leiden | Đồ thị, `src/graph.py` |
| `04_graphrag_eval` | 4 | Text2Cypher, router, hybrid, Recall@k, MRR, RAGAS, benchmark | `results/results.parquet`, `src/hybrid.py`, `src/eval.py` |

```bash
cd notebooks && ../.venv/bin/python -m jupyter lab
```

Chạy theo thứ tự 01 → 04. Lần đầu tính và lưu mọi cell CACHE vào `data/`, `index/` và `results/cache/`;
các lần sau chỉ đọc lại. Mỗi lượt gọi LLM được lưu theo nội dung prompt trong `results/cache/llm/`,
nên chạy lại cùng câu hỏi không tốn token. Riêng P4-10 luôn gọi API thật để đo đúng độ trễ và token.
Đặt `REBUILD=1` để tính lại mọi cell CACHE. Khi dữ liệu sạch đổi, notebook 03 tự xoá và nạp lại đồ thị.

## Chatbot

Chatbot dùng chung `src/` với notebook. Nó cần một trong hai chỉ mục: bản mẫu (chạy notebook 01 và 03)
hoặc bản đầy đủ.

```bash
.venv/bin/python app/server.py                  # http://127.0.0.1:8000, --index auto
.venv/bin/python app/server.py --index sample   # ép dùng bản mẫu
```

`--index auto` (mặc định) dùng bản đầy đủ khi `scripts/index_full.py` đã chạy hết dataset, ngược lại dùng bản mẫu.
Chatbot kiểm tra đồ thị có khớp chỉ mục không (cùng số tin, cùng tổng giá trong manifest `index/<collection>.json`).

### Lập chỉ mục toàn bộ dataset

```bash
docker compose --profile full up -d --wait      # thêm Neo4j graph_db_full (cổng 7688)
.venv/bin/python scripts/index_full.py --reset  # lần đầu; ~4–5 giờ trên RTX 3050
.venv/bin/python scripts/index_full.py          # chạy tiếp sau khi bị ngắt
```

Script đọc dataset theo lô 4.096 dòng. Mỗi lô đi qua các bước:
- làm sạch như notebook,
- bỏ tin trùng trên toàn bộ dataset,
- chunk và embed E5 (fp16 trên GPU); tách từ pyvi chạy song song trên CPU,
- upsert vào Qdrant,
- MERGE vào Neo4j.

Sau mỗi lô, script ghi checkpoint vào `index/listings_full_state.json`. Bị ngắt thì chạy lại cùng lệnh
để tiếp tục. `--limit N` để chạy thử N dòng đầu. Đo trên RTX 3050 Laptop: khoảng 200 dòng/s, gần hết
thời gian nằm ở bước embed. Dung lượng ước tính: Qdrant khoảng 10–12 GB đĩa và 2 GB RAM (vector nén int8).

- Chế độ **Tự động** (Hybrid, mặc định), **Basic RAG** và **Graph** chọn ở góc phải.
- Câu trả lời do LLM sinh. Mỗi `[Tin#ID]` thành liên kết tới thẻ tin nguồn bên dưới (giá, diện tích,
  phòng, địa chỉ, mô tả).
- Dòng ghi chú cho biết tuyến router đã chọn, bộ lọc trích từ câu hỏi, độ trễ và số token. Câu Cypher
  đã chạy nằm trong phần thu gọn.
- Link dạng `/?q=Căn hộ 2PN ở Thủ Đức dưới 3 tỷ&system=graph` hỏi ngay khi mở trang, tiện khi trình bày.
- Nếu Neo4j chưa chạy hoặc đồ thị chưa khớp dữ liệu sạch, chế độ Tự động và Graph bị tắt; Basic RAG
  vẫn dùng được. Neo4j mất kết nối giữa chừng thì chế độ Tự động lui về Basic RAG.
- Chế độ Tự động chạy truy xuất RAG song song với router và Text2Cypher, nên tuyến hybrid và đường lui
  không phải chờ thêm một lượt LLM (đổi lại, tuyến graph tốn thêm một lượt trích bộ lọc).
- Trích dẫn không có trong nguồn được cảnh báo ở dòng ghi chú.

API: `GET /api/info` trả danh sách chế độ và thống kê chỉ mục; `POST /api/chat` nhận
`{"query": "...", "system": "hybrid" | "basic_rag" | "graph"}` (Content-Type `application/json`).

Bảo vệ API (cấu hình trong `.env`):

| Biến | Mặc định | Tác dụng |
| --- | --- | --- |
| `CHAT_MAX_CONCURRENT` | 4 | Số câu hỏi xử lý đồng thời; chờ quá 60 s trả 503 |
| `CHAT_RATE_PER_MIN` | 20 | Số câu hỏi mỗi phút cho một IP (429 khi vượt); 0 = không giới hạn |
| `CHAT_MAX_QUERY_CHARS` | 500 | Độ dài câu hỏi tối đa; body tối đa 8 KB |
| `CHAT_API_TOKEN` | rỗng | Bắt buộc header `X-API-Key`; mở trang bằng `/?token=...` một lần. Bắt buộc khi `--host` không phải loopback |

Server ghi log mỗi câu hỏi (mã yêu cầu, chế độ, tuyến, độ trễ, token, số trích dẫn sai) nhưng không ghi
nội dung câu hỏi. Lỗi nội bộ chỉ trả mã yêu cầu cho giao diện, chi tiết nằm trong log. Trang có CSP
`default-src 'self'` và `X-Content-Type-Options: nosniff`.

## Đánh giá

[benchmark/questions.csv](benchmark/questions.csv) có 28 câu thuộc 5 loại: tra cứu, ràng buộc, tổng hợp,
đa chặng (mỗi loại 6 câu) và 4 câu cố ý không có đáp án trong dữ liệu (giá nhà trên sao Hỏa, dự báo năm 2030…).
Mỗi câu có nhãn tuyến đúng cho router. Đáp án chuẩn tính bằng pandas từ cột `gt_rule`
([src/eval.py](src/eval.py)), không gõ tay. Mọi câu được chấm thang 0–2 (cột `points`); `correct` là đạt 2 điểm:
- Câu tra cứu/ràng buộc có tập tin đúng `gt_ids`, chấm bằng Recall@5 (chia cho min(số tin đúng, 5)) và MRR.
  2 điểm khi Recall@5 = 1, 1 điểm khi có ít nhất một tin đúng trong top 5.
- Câu tổng hợp/đa chặng có giá trị đúng `gt_value`. LLM giám khảo chấm câu trả lời so với đáp án: 2 = đúng
  (sai số ≤ 5%), 1 = đúng một phần (lệch 5–20% hoặc chỉ đúng một phần của danh sách), 0 = sai.
- Câu không có đáp án đo tỷ lệ từ chối đúng: 2 = nói rõ không đủ dữ liệu, 1 = có cảnh báo nhưng vẫn đoán, 0 = bịa.

Benchmark chạy bốn hệ thống: baseline **LLM only** (`rag.ask_llm_only`: cùng mô hình nhưng không có ngữ cảnh),
Basic RAG, Graph và Hybrid. Nhờ baseline này, ta đo được RAG thêm được gì so với LLM trần. LLM only không trả về tin nào,
nên ở câu tra cứu/ràng buộc nó luôn được 0 điểm.

Tin đăng do người dùng tự nhập nên không được coi là đáng tin. `build_prompt` bọc từng tài liệu trong
`<tai_lieu>…</tai_lieu>` và vô hiệu hoá thẻ đóng giả mạo trong nội dung. System prompt yêu cầu coi nội dung trong thẻ
là dữ liệu và bỏ qua mọi chỉ thị nằm trong đó (chống prompt injection).

Notebook 04 còn tính Faithfulness, Answer Relevancy và Context Precision theo định nghĩa của RAGAS
(chấm bằng LLM giám khảo). Nó cũng lập bảng độ chính xác theo loại câu × hệ thống, chi phí, độ trễ, tỷ lệ câu trả lời
trích nguồn không có trong ngữ cảnh (`bad_citation_rate`) và phân tích lỗi. Đặt `JUDGE_MODEL` để chấm bằng
mô hình khác mô hình sinh (tránh tự chấm thiên vị); để trống thì dùng `LLM_MODEL`.

## Cấu trúc thư mục

```text
app/
  server.py             Server HTTP + API /api/chat
  static/               Giao diện HTML/CSS/JS
benchmark/questions.csv 28 câu hỏi đánh giá (4 câu không có đáp án), đáp án chuẩn tính bằng pandas
scripts/index_full.py   Lập chỉ mục toàn bộ dataset cho chatbot (Qdrant + Neo4j), chạy tiếp được
graph/load.cypher       Script nạp đồ thị Neo4j
notebooks/              5 notebook trình bày (00_setup → 04_graphrag_eval)
src/
  config.py             Đường dẫn, bảng ánh xạ cột COL, cấu hình mô hình và DB
  common.py             Cache cho cell CACHE, hiển thị, seed
  data.py               Nạp, làm sạch, chunk
  vector.py             Embedding E5
  qdrant_store.py       Collection Qdrant: dense + BM25 sparse, bộ lọc payload, manifest
  rag.py                Bộ lọc, dense, BM25, RRF, rerank, ask_rag
  graph.py              Nạp đồ thị, Cypher mẫu, Leiden, Text2Cypher, ask_graph
  hybrid.py             Router, ask_hybrid
  eval.py               Recall@k, MRR, RAGAS, chạy benchmark
  llm.py                Gọi LLM (Groq), structured output, cache đĩa
tests/                  Kiểm thử src/, cấu trúc notebook, chatbot
data/ index/ results/   Sinh ra khi chạy notebook, không đưa vào Git
docker-compose.yaml     Qdrant (vector_db), Neo4j mẫu (graph_db), Neo4j đầy đủ (graph_db_full, profile full)
```

## Kiểm thử

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Test không cần mạng, LLM, Qdrant hay Neo4j. Chúng kiểm tra:
- Các số toy trên slide.
- Làm sạch, entity resolution, sửa đơn vị giá.
- Chunk theo ngân sách token, bộ lọc và đơn vị tiền (cả "1 tỷ 50 triệu", "3.000.000.000", chuỗi không có số).
- Nạp đồ thị: khoá địa danh theo cấp cha.
- Chặn lệnh ghi/quản trị, thêm `LIMIT`, cơ chế thử lại của Text2Cypher, prompt lui về mặc định khi mất Neo4j.
- Hybrid search một lượt Qdrant, kết quả rỗng vẫn đủ cột, kiểm trích dẫn, báo bỏ lọc cho LLM.
- Đường lui của Hybrid (đồ thị rỗng hoặc Neo4j lỗi), cách tính token, luật tính đáp án chuẩn, căn độ dài phán quyết RAGAS.
- Khung notebook: đúng 51 cell, 19 LIVE · 22 CACHE · 10 ẨN, không lộ khoá API hay số điện thoại trong output.
- Route HTTP của chatbot, header bảo mật, và các lớp chặn: Content-Type, kích thước body, độ dài câu hỏi,
  tần suất, API key, không lộ lỗi nội bộ.

CI ([.github/workflows/tests.yml](.github/workflows/tests.yml)) chạy bộ test này trên mỗi push và pull request.

## Giới hạn

- Notebook chạy trên mẫu `N_ROWS` dòng đầu, không đại diện toàn thị trường; giá đăng không phải giá giao dịch.
- Chỉ mục đầy đủ cố định `avgdl` của BM25 từ lô đầu tiên. Tin trùng được nhận diện bằng tiêu đề + mô tả
  giống hệt nhau, không bắt được tin đăng lại có sửa chữ.
- Bộ lọc, câu trả lời, Cypher và router phụ thuộc LLM; cần khoá Groq khi chạy lần đầu.
- Đồ thị nạp trước khi Listing có `project`, `street`, `direction` vẫn qua bước kiểm khớp (cùng số tin, tổng giá)
  nhưng thiếu ba thuộc tính này: nạp lại bằng notebook 03 với `REBUILD=1`, hoặc `scripts/index_full.py --reset`.
- Kết quả benchmark trên 28 câu chỉ là so sánh tương đối giữa bốn hệ thống, chưa đủ để kết luận tổng quát.
