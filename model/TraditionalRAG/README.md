# Traditional RAG

Mở [traditional_rag.ipynb](traditional_rag.ipynb), chọn kernel có dependency rồi chạy từ trên xuống.

Notebook có 7 bước: load dữ liệu → kiểm tra và làm sạch → chia chunk → khởi tạo embedding → chuyển chunk thành vector → lưu PostgreSQL → Test truy xuất.

```bash
.venv/bin/python -m pip install -r model/TraditionalRAG/requirements.txt
docker compose up -d vector_db
.venv/bin/python -m jupyter lab
```

Mã nằm trực tiếp trong từng cell; không có khối cấu hình thí nghiệm hay hàm tự định nghĩa. E5 là model embedding để chuyển văn bản thành vector, được tải ở lần đầu chạy. Đây là phần truy xuất trong Traditional RAG: đầu ra là các tin liên quan, chưa có bước sinh câu trả lời.

Notebook đọc 10 dòng đầu CSV và hiển thị dạng bảng, chia đoạn tối đa 384 token với overlap 48 token và lưu chunk/vector/metadata trong PostgreSQL. Sửa câu hỏi ngay tại bước Test để tìm tối đa 5 tin khác nhau; không cần xuất file hoặc khai báo bộ lọc riêng.

Nguồn mặc định trong `.env.example`: [tinixai/vietnam-real-estates](https://huggingface.co/datasets/tinixai/vietnam-real-estates). `DATA_URL` trong `.env` hỗ trợ dataset Hugging Face (Parquet streaming) hoặc link CSV trực tiếp. Notebook đọc theo luồng và dừng sau số dòng mẫu. Để trống để dùng `data/vietnam-real-estates.csv` cục bộ.
