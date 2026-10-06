"""Cấu hình dùng chung cho 5 notebook (cell 00-01, 00-02). Đổi tên cột hay mô hình chỉ sửa ở đây."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)

# ---------------------------------------------------------------- đường dẫn (mục 2.1 của outline)
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
GRAPH_DIR = ROOT / "graph"
BENCHMARK_DIR = ROOT / "benchmark"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"

CLEAN_PATH = DATA_DIR / "listings_clean.parquet"
CHUNKS_PATH = INDEX_DIR / "chunks.parquet"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
LOAD_CYPHER = GRAPH_DIR / "load.cypher"
QUESTIONS_PATH = BENCHMARK_DIR / "questions.csv"
RESULTS_PATH = RESULTS_DIR / "results.parquet"

# ---------------------------------------------------------------- bảng ánh xạ cột
# Khoá: tên dùng trong notebook. Giá trị: tên cột thật của dataset tinixai/vietnam-real-estates.
COL = {
    "title": "name",
    "description": "description",
    "property_type": "property_type_name",
    "province": "province_name",
    "district": "district_name",
    "ward": "ward_name",
    "street": "street_name",
    "project": "project_name",
    "price": "price",  # VND
    "area": "area",  # m²
    "floors": "floor_count",
    "frontage": "frontage_width",
    "depth": "house_depth",
    "road_width": "road_width",
    "bedrooms": "bedroom_count",
    "bathrooms": "bathroom_count",
    "direction": "house_direction",
    "balcony_direction": "balcony_direction",
    "published_at": "published_at",
}
TEXT_COLS = ["title", "description", "property_type", "province", "district", "ward", "street",
             "project", "direction", "balcony_direction", "published_at"]
NUMERIC_COLS = ["price", "area", "floors", "frontage", "depth", "road_width", "bedrooms", "bathrooms"]

# ---------------------------------------------------------------- dữ liệu, tái lập
SEED = 42
DATASET = os.getenv("DATASET", "tinixai/vietnam-real-estates")
# Số dòng đầu của dataset dùng cho notebook; toàn bộ (~3,5 triệu) không cần cho buổi trình bày.
N_ROWS = int(os.getenv("N_ROWS", "20000"))
# Đặt REBUILD=1 để chạy lại mọi cell CACHE thay vì đọc file đã lưu.
REBUILD = os.getenv("REBUILD", "0") == "1"

# ---------------------------------------------------------------- embedding, chunk, tìm kiếm
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
CHUNK_SIZE, CHUNK_OVERLAP = 400, 100  # token, khớp slide 7 (S = 400, O = 100)
RRF_K = 60
TOP_K = 5
# RERANK=1: retrieve() lấy RERANK_CANDIDATES tin từ hybrid search rồi cross-encoder chọn TOP_K (chậm hơn, chính xác hơn).
RERANK = os.getenv("RERANK", "0") == "1"
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
# Dense search trên vector nén int8: lấy dư `oversampling` lần rồi chấm lại bằng vector gốc.
QDRANT_OVERSAMPLING = float(os.getenv("QDRANT_OVERSAMPLING", "2.0"))

# ---------------------------------------------------------------- LLM (khoá đọc từ biến môi trường)
# Mặc định dùng Groq (API tương thích OpenAI), đọc GROQ_API_KEY. Tuỳ chọn khác: LLM_PROVIDER=anthropic
# (ANTHROPIC_API_KEY hoặc `ant auth login`), =openai (OPENAI_API_KEY). Không ghi khoá vào notebook.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
_DEFAULTS = {  # provider: (mô hình sinh, mô hình dự phòng khi 429/503, mô hình chấm, giá USD/1M token vào, ra)
    "groq": ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-20b", 0.15, 0.60),
    "openai": ("gpt-5.4-mini", "", None, 4.00, 20.00),
    "anthropic": ("claude-opus-5-5", "", None, 4.00, 20.00),
}
_D = _DEFAULTS.get(LLM_PROVIDER, _DEFAULTS["groq"])
LLM_MODEL = os.getenv("LLM_MODEL", _D[0])
# Model thử tiếp khi model chính quá tải (503) hoặc vượt hạn mức (429); dùng cho groq/openai.
LLM_FALLBACK_MODELS = [m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", _D[1]).split(",") if m.strip()]
LLM_EFFORT = os.getenv("LLM_EFFORT", "low")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))  # giây cho mỗi lượt gọi API
# Mô hình chấm điểm ở eval.py; khác mô hình sinh để tránh tự chấm thiên vị. Rỗng: dùng LLM_MODEL.
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or _D[2]
# USD cho 1 triệu token vào/ra của LLM_MODEL, dùng để tính chi phí ở P4-12. Kiểm tra lại bảng giá khi đổi model.
LLM_PRICE_PER_MTOK = {"input": float(os.getenv("LLM_PRICE_INPUT", _D[3])),
                      "output": float(os.getenv("LLM_PRICE_OUTPUT", _D[4]))}

# ---------------------------------------------------------------- vector DB (Qdrant)
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
SAMPLE_COLLECTION = "listings_sample"  # N_ROWS dòng đầu, dùng cho notebook
FULL_COLLECTION = "listings_full"  # toàn bộ dataset, dùng cho chatbot (scripts/index_full.py)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", SAMPLE_COLLECTION)

# ---------------------------------------------------------------- đồ thị
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")  # mẫu, notebook 03
NEO4J_FULL_URI = os.getenv("NEO4J_FULL_URI", "bolt://127.0.0.1:7688")  # toàn bộ, chatbot
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "local_dev_password")
# Cypher do LLM sinh: Neo4j huỷ giao dịch quá CYPHER_TIMEOUT giây, chỉ đọc tối đa CYPHER_MAX_ROWS dòng.
CYPHER_TIMEOUT = float(os.getenv("CYPHER_TIMEOUT", "10"))
CYPHER_MAX_ROWS = int(os.getenv("CYPHER_MAX_ROWS", "50"))

# ---------------------------------------------------------------- chatbot (app/server.py)
CHAT_MAX_CONCURRENT = int(os.getenv("CHAT_MAX_CONCURRENT", "4"))  # số câu hỏi xử lý đồng thời
CHAT_RATE_PER_MIN = int(os.getenv("CHAT_RATE_PER_MIN", "20"))  # số câu hỏi mỗi phút cho một IP; 0 = không giới hạn
CHAT_MAX_QUERY_CHARS = int(os.getenv("CHAT_MAX_QUERY_CHARS", "500"))
CHAT_API_TOKEN = os.getenv("CHAT_API_TOKEN", "")  # đặt khi mở server ra ngoài 127.0.0.1
