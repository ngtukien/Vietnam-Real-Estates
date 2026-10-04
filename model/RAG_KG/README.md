# RAG + KG

Mở [rag_kg.ipynb](rag_kg.ipynb), chạy các cell từ trên xuống.

Notebook có flow **A. Lập chỉ mục**, **B. Truy vấn**, rồi tạo context/prompt. Các bước có giải thích và đầu ra trung gian.

Từ thư mục gốc dự án:

```bash
.venv/bin/python -m pip install -r model/RAG_KG/requirements.txt
docker compose up -d vector_db
.venv/bin/python -m jupyter lab
```

Toàn bộ mã nằm trong notebook, viết trực tiếp trong các cell chạy từng bước, không có hàm tự định nghĩa hay khối gom tham số, kèm giải thích và đầu ra trung gian. Không cần mở script Python khi trình bày. Dữ liệu nguồn và database dùng chung ở gốc dự án.

Mặc định đọc 1.000 tin đầu CSV. Embedding E5 được tải ở lần đầu chạy. Cell xuất kết quả lưu trong `outputs/result.json` của thư mục này.

Nguồn mặc định đặt trực tiếp trong cell tải dữ liệu (không cần `.env`): [tinixai/vietnam-real-estates](https://huggingface.co/datasets/tinixai/vietnam-real-estates). `DATA_URL` trong `.env` hỗ trợ dataset Hugging Face (Parquet streaming) hoặc link CSV trực tiếp. Notebook đọc theo luồng và dừng sau số dòng mẫu. Để trống để dùng `data/vietnam-real-estates.csv` cục bộ.
