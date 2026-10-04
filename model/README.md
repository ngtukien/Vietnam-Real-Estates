# Bốn mô hình — bốn thư mục

| Thư mục | Notebook | Flow |
| --- | --- | --- |
| `TraditionalRAG/` | [Traditional RAG](TraditionalRAG/traditional_rag.ipynb) | Làm sạch → chunk → E5 embedding → pgvector → truy xuất tin liên quan |
| `TraditionalKG/` | [Traditional KG](TraditionalKG/traditional_kg.ipynb) | Làm sạch → thực thể/quan hệ → PostgreSQL graph store → graph retrieval → facts → context/prompt |
| `RAG_KG/` | [RAG + KG](RAG_KG/rag_kg.ipynb) | Chỉ mục vector + KG → truy xuất hai nhánh → RRF → context/prompt |
| `GraphRAG/` | [GraphRAG](GraphRAG/graphrag.ipynb) | Chunk/vector + KG → Louvain → báo cáo cộng đồng → embedding báo cáo → local/global retrieval → context/prompt |

Mã chạy trực tiếp trong từng cell; không có hàm tự định nghĩa hoặc khối gom tham số cấu hình. Mỗi mô hình độc lập, không có thư mục `shared` và không import code của mô hình khác. Mỗi thư mục chứa notebook, `requirements.txt` và README; toàn bộ mã mô hình nằm trong notebook, không dùng script Python hỗ trợ. Traditional RAG hiển thị kết quả ngay tại bước Test. Các notebook còn lại có cell xuất kết quả vào `outputs/result.json` của chính mô hình đó. Dữ liệu CSV và Docker Compose dùng chung ở thư mục gốc.

`DATA_URL` trong `.env.example` trỏ tới [tinixai/vietnam-real-estates](https://huggingface.co/datasets/tinixai/vietnam-real-estates).
Cả bốn notebook đọc Parquet qua `datasets.load_dataset(..., streaming=True)` và dừng sau số dòng mẫu,
không tải toàn bộ dataset. Cũng hỗ trợ link HTTP(S) tải trực tiếp CSV UTF-8. Để trống `DATA_URL` để dùng
`data/vietnam-real-estates.csv` cục bộ; CSV không đưa vào Git. Notebook tìm thư mục gốc qua `app/`
và `model/`, nên vẫn chạy được khi máy không có file CSV.

## Chạy

Ví dụ Traditional RAG, từ thư mục gốc:

```bash
.venv/bin/python -m pip install -r model/TraditionalRAG/requirements.txt
docker compose up -d vector_db
.venv/bin/python -m jupyter lab
```

Chọn kernel có dependency đã cài. Mở notebook và chạy từ trên xuống. Có thể mở từ thư mục riêng của mô hình; notebook tìm thư mục gốc qua vị trí dataset. Các bước chia thành **A. Lập chỉ mục** và **B. Truy vấn**. Traditional RAG có 7 bước, kết thúc bằng các tin truy xuất tại bước Test; các notebook còn lại có thêm context/prompt.

Mặc định đọc 1.000 tin đầu CSV. E5 tải ở lần đầu và chạy CPU, dùng `passage:` cho chunk và `query:` cho câu hỏi. Chunk tối đa 384 token, overlap 48 token; header tính trong token budget. PostgreSQL đọc `.env`, mặc định host `127.0.0.1`; kernel trong container Compose cần `POSTGRES_HOST=vector_db`.

Bảng vector theo số chiều embedding, KG dùng nodes/edges với khóa ngoại. Mỗi thí nghiệm có collection theo dữ liệu/cấu hình. Upsert tránh bản ghi trùng khi chạy lại. Demo dùng cosine search chính xác, chưa bật HNSW.

## Trình bày

Notebook hiển thị kết quả truy xuất, context và prompt, không sinh câu trả lời. GraphRAG dùng báo cáo thống kê cộng đồng và giữ ID nguồn.

Để trình bày: giới thiệu flow, xem một bản ghi nguồn, xem từng đầu ra lập chỉ mục, chạy truy xuất rồi kiểm tra context và ID nguồn. Các notebook đọc cùng 1.000 dòng đầu CSV; dùng cùng câu hỏi để đối chiếu. Traditional RAG nhập câu hỏi ngay tại bước Test và tìm theo độ tương đồng vector; các notebook còn lại có thêm bộ lọc `filters`. Giá dùng VND, diện tích dùng m²; tên tỉnh/loại hình phải khớp CSV. Điều kiện số truyền bằng bộ lọc, chưa tự phân tích từ câu hỏi.

KG lấy quan hệ từ các cột CSV; chưa trích từ mô tả bằng LLM. GraphRAG là biến thể minh họa dùng Louvain, chưa có Leiden phân cấp/global map-reduce của Microsoft GraphRAG. Báo cáo cộng đồng tổng hợp trước bộ lọc truy vấn; nguồn local mới áp dụng bộ lọc. Dữ liệu đầu CSV không đại diện toàn thị trường và chưa xác nhận tin còn hiệu lực.

## Kiểm tra

```bash
python3 -m unittest discover -s tests -v
```

Kiểm thử token budget/coverage, nguồn, bộ lọc, đồ thị, cộng đồng, RRF và cấu trúc notebook. Kiểm thử dùng tokenizer giả để kiểm tra ranh giới chunk, không đo embedding thật. Tải model E5 và kiểm tra lưu trữ PostgreSQL cần môi trường tương ứng; chưa có tập nhãn đo Recall@k/MRR/nDCG.

Tài liệu: [E5](https://huggingface.co/intfloat/multilingual-e5-small), [pgvector Python](https://github.com/pgvector/pgvector-python), [Louvain](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html), [GraphRAG dataflow](https://microsoft.github.io/graphrag/index/default_dataflow/).
