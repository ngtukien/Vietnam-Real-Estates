# Vietnam Real Estates — từ RAG đến GraphRAG

Trợ lý hỏi đáp thị trường bất động sản Việt Nam trên dữ liệu tin đăng
[tinixai/vietnam-real-estates](https://huggingface.co/datasets/tinixai/vietnam-real-estates).
Dự án so sánh ba cách trả lời trên cùng dữ liệu và cùng bộ câu hỏi:

| Hệ thống | Cách tìm bằng chứng | Mạnh ở |
| --- | --- | --- |
| **Basic RAG** | LLM trích bộ lọc → dense (E5) + BM25 trên Qdrant gộp bằng RRF, lọc trước theo payload → Claude trả lời có `[Tin#ID]` | Tìm theo mô tả tự do: "gần trường", "có thang máy" |
| **Graph** | Text2Cypher: Claude sinh Cypher chỉ đọc trên đồ thị Neo4j → Claude diễn giải kết quả | Câu tổng hợp, đếm, xếp hạng, đa chặng |
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
export ANTHROPIC_API_KEY=...               # hoặc `ant auth login`
```

| Thành phần | Chọn |
| --- | --- |
| Embedding | `intfloat/multilingual-e5-small` (384 chiều), tiền tố `passage:`/`query:`; trên GPU chạy fp16 (nhanh ~3 lần, cosine với bản fp32 ≥ 0,999) |
| Chunk | Cửa sổ 400 token, chồng lấn 100; header (tiêu đề, loại, địa chỉ, giá, diện tích) lặp ở mọi chunk |
| Vector DB | Qdrant: vector dense trên đĩa + bản nén int8 trong RAM, payload index để pre-filter |
| Từ khoá | BM25 dạng sparse vector trong cùng collection Qdrant (IDF tính phía server), tách từ tiếng Việt bằng `pyvi` |
| Đồ thị | Neo4j: `(Listing)-[:IN_WARD]->(Ward)-[:IN_DISTRICT]->(District)-[:IN_PROVINCE]->(Province)`, `(Listing)-[:OF_TYPE]->(PropertyType)`; script [graph/load.cypher](graph/load.cypher) |
| LLM | `claude-opus-5-5` qua Anthropic SDK, structured output cho bộ lọc, Cypher, router và chấm điểm |

Text2Cypher có hai lớp chặn lệnh ghi: regex trong [src/graph.py](src/graph.py), và giao dịch chỉ đọc
của Neo4j (`execute_read`, lỗi `AccessMode` nếu câu lệnh có ghi).

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
các lần sau chỉ đọc lại. Mỗi lượt gọi Claude được lưu theo nội dung prompt trong `results/cache/llm/`,
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
- Câu trả lời do Claude sinh. Mỗi `[Tin#ID]` thành liên kết tới thẻ tin nguồn bên dưới (giá, diện tích,
  phòng, địa chỉ, mô tả).
- Dòng ghi chú cho biết tuyến router đã chọn, bộ lọc trích từ câu hỏi, độ trễ và số token. Câu Cypher
  đã chạy nằm trong phần thu gọn.
- Link dạng `/?q=Căn hộ 2PN ở Thủ Đức dưới 3 tỷ&system=graph` hỏi ngay khi mở trang, tiện khi trình bày.
- Nếu Neo4j chưa chạy hoặc đồ thị chưa khớp dữ liệu sạch, chế độ Tự động và Graph bị tắt; Basic RAG
  vẫn dùng được.

API: `GET /api/info` trả danh sách chế độ và thống kê chỉ mục; `POST /api/chat` nhận
`{"query": "...", "system": "hybrid" | "basic_rag" | "graph"}`.

## Đánh giá

[benchmark/questions.csv](benchmark/questions.csv) có 24 câu thuộc 4 loại: tra cứu, ràng buộc, tổng hợp,
đa chặng. Mỗi câu có nhãn tuyến đúng cho router. Đáp án chuẩn tính bằng pandas từ cột `gt_rule`
([src/eval.py](src/eval.py)), không gõ tay:
- Câu tra cứu/ràng buộc có tập tin đúng `gt_ids`, chấm bằng Recall@5 (chia cho min(số tin đúng, 5)) và MRR.
- Câu tổng hợp/đa chặng có giá trị đúng `gt_value`, do Claude chấm câu trả lời so với đáp án (cho phép sai số 5%).

Notebook 04 còn tính Faithfulness, Answer Relevancy và Context Precision theo định nghĩa của RAGAS
(chấm bằng Claude). Nó cũng lập bảng độ chính xác theo loại câu × hệ thống, chi phí, độ trễ và phân tích lỗi.

## Cấu trúc thư mục

```text
app/
  server.py             Server HTTP + API /api/chat
  static/               Giao diện HTML/CSS/JS
benchmark/questions.csv 24 câu hỏi đánh giá, đáp án chuẩn tính bằng pandas
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
  llm.py                Gọi Claude, structured output, cache đĩa
tests/                  Kiểm thử src/, cấu trúc notebook, chatbot
data/ index/ results/   Sinh ra khi chạy notebook, không đưa vào Git
docker-compose.yaml     Qdrant (vector_db), Neo4j mẫu (graph_db), Neo4j đầy đủ (graph_db_full, profile full)
```

## Kiểm thử

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Test không cần mạng, Claude, Qdrant hay Neo4j. Chúng kiểm tra:
- Các số toy trên slide.
- Làm sạch, entity resolution, sửa đơn vị giá.
- Chunk theo ngân sách token, bộ lọc và đơn vị tiền.
- Nạp đồ thị: khoá địa danh theo cấp cha.
- Chặn lệnh ghi và cơ chế thử lại của Text2Cypher.
- Đường lui của Hybrid, luật tính đáp án chuẩn.
- Khung notebook: đúng 51 cell, 19 LIVE · 22 CACHE · 10 ẨN, không lộ khoá API hay số điện thoại trong output.
- Route HTTP của chatbot.

## Giới hạn

- Notebook chạy trên mẫu `N_ROWS` dòng đầu, không đại diện toàn thị trường; giá đăng không phải giá giao dịch.
- Chỉ mục đầy đủ cố định `avgdl` của BM25 từ lô đầu tiên. Tin trùng được nhận diện bằng tiêu đề + mô tả
  giống hệt nhau, không bắt được tin đăng lại có sửa chữ.
- Bộ lọc, câu trả lời, Cypher và router phụ thuộc Claude; cần khoá API khi chạy lần đầu.
- Kết quả benchmark trên 24 câu chỉ là so sánh tương đối giữa ba hệ thống, chưa đủ để kết luận tổng quát.
