"""Chatbot: server HTTP phục vụ giao diện HTML/CSS và API hỏi đáp trên src/ (cùng code với notebook).

Hai chỉ mục, chọn bằng --index:
- sample: N_ROWS dòng đầu do notebook 01 (Qdrant listings_sample) và 03 (Neo4j graph_db) tạo.
- full: toàn bộ dataset do scripts/index_full.py tạo (Qdrant listings_full, Neo4j graph_db_full).
- auto (mặc định): full nếu đã lập chỉ mục xong, ngược lại sample.

    .venv/bin/python app/server.py            # http://127.0.0.1:8000
"""

import argparse
import json
import mimetypes
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for key, value in {"TQDM_DISABLE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_VERBOSITY": "error",
                   "TRANSFORMERS_VERBOSITY": "error", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(key, value)  # log khởi động gọn

from src import graph, qdrant_store, rag, vector  # noqa: E402
from src.common import fmt_vnd  # noqa: E402
from src.config import (FULL_COLLECTION, INDEX_DIR, NEO4J_FULL_URI, NEO4J_URI,  # noqa: E402
                        SAMPLE_COLLECTION)
from src.data import text  # noqa: E402
from src.hybrid import ask_hybrid  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
SYSTEMS = {
    "hybrid": ("Tự động", ask_hybrid),  # router chọn Basic RAG, Graph hoặc gộp cả hai
    "basic_rag": ("Basic RAG", rag.ask_rag),
    "graph": ("Graph (Text2Cypher)", graph.ask_graph),
}
NEEDS_GRAPH = {"hybrid", "graph"}
MAX_SOURCES = 5


INDEXES = {"sample": (SAMPLE_COLLECTION, NEO4J_URI, "chạy notebook 01 và 03"),
           "full": (FULL_COLLECTION, NEO4J_FULL_URI, "chạy scripts/index_full.py")}


def number(value):
    return None if value is None or value != value else float(value)  # value != value: NaN


def pick_index(choice: str) -> str:
    """auto: dùng chỉ mục toàn bộ khi scripts/index_full.py đã chạy hết dataset."""
    if choice != "auto":
        return choice
    state = INDEX_DIR / f"{FULL_COLLECTION}_state.json"
    return "full" if state.exists() and json.loads(state.read_text())["done"] else "sample"


def listing_card(row: dict) -> dict:
    """Thông tin một tin nguồn (payload Qdrant) để giao diện hiển thị thành thẻ."""
    description = text(row.get("description")) or ""
    return dict(
        id=int(row["listing_id"]),
        title=text(row.get("title")) or "Không có tiêu đề",
        property_type=text(row.get("property_type")) or "Chưa rõ loại",
        price=fmt_vnd(number(row.get("price"))) if number(row.get("price")) else "Chưa rõ giá",
        area=f"{row['area']:g} m²".replace(".", ",") if number(row.get("area")) else "Chưa rõ diện tích",
        bedrooms=number(row.get("bedrooms")),
        bathrooms=number(row.get("bathrooms")),
        address=", ".join(p for p in (text(row.get(k)) for k in ("street", "ward", "district", "province")) if p),
        project=text(row.get("project")) or "",
        published_at=(text(row.get("published_at")) or "")[:10],
        description=description[:400] + ("…" if len(description) > 400 else ""),
    )


class App:
    """Trạng thái dùng chung: chỉ mục Qdrant và kết nối đồ thị (nếu có) của bản sample hoặc full."""

    def __init__(self, index: str):
        collection, uri, how = INDEXES[index]
        qdrant_store.use(collection)
        graph.use(uri)
        manifest = qdrant_store.read_manifest()
        if manifest is None or not qdrant_store.count():
            raise SystemExit(f"Chưa có chỉ mục {index} trong Qdrant ({collection}): {how}.")
        vector.embed_query("khởi động")  # tải E5 sẵn để câu hỏi đầu không phải chờ
        self.stats = dict(index=index, listings=manifest["listings"], chunks=qdrant_store.count())
        try:
            self.graph_ok = graph.graph_matches_counts(manifest["listings"], manifest["price_sum"])
            self.graph_error = None if self.graph_ok else f"Đồ thị chưa khớp chỉ mục: {how}"
        except Exception as error:  # Neo4j chưa chạy: vẫn phục vụ Basic RAG
            self.graph_ok, self.graph_error = False, f"Không kết nối được Neo4j {uri} ({type(error).__name__})"
        self.lock = threading.Lock()  # mô hình embedding và client dùng chung giữa các luồng

    def systems(self) -> dict:
        return {key: dict(name=name, available=self.graph_ok or key not in NEEDS_GRAPH)
                for key, (name, _) in SYSTEMS.items()}

    def ask(self, question: str, system: str) -> dict:
        with self.lock:
            result = SYSTEMS[system][1](question)
        # Ưu tiên tin được trích trong câu trả lời, sau đó tới các tin đã đưa vào ngữ cảnh.
        wanted = list(dict.fromkeys(rag.cited_ids(result["answer"]) + [int(i) for i in result["sources"]]))
        cards = qdrant_store.listings(wanted[:MAX_SOURCES * 2])
        ids = [i for i in wanted if i in cards][:MAX_SOURCES]
        return dict(
            system=system, system_name=SYSTEMS[system][0], answer=result["answer"],
            cited=rag.cited_ids(result["answer"]),
            sources=[listing_card(cards[i]) for i in ids],
            route=result.get("route"), fallback=bool(result.get("fallback")),
            filters=result.get("filters") or {}, note=result.get("note") or "",
            cypher=result.get("cypher"), error=result.get("error"),
            elapsed_ms=round(result["latency"] * 1000),
            tokens=result["input_tokens"] + result["output_tokens"],
        )


class Handler(BaseHTTPRequestHandler):
    app: App

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urlsplit(self.path).path  # bỏ query string (?q=...)
        if route == "/api/info":
            return self.send_json(dict(systems=self.app.systems(), stats=self.app.stats,
                                       graph_error=self.app.graph_error))
        name = "index.html" if route in ("/", "/index.html") else route.removeprefix("/static/")
        path = (STATIC / name).resolve()
        if path.parent != STATIC or not path.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", (mimetypes.guess_type(path.name)[0] or "text/plain") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/chat":
            return self.send_error(HTTPStatus.NOT_FOUND)
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            query = str(payload.get("query", "")).strip()
            system = payload.get("system", "hybrid")
            if not query or system not in SYSTEMS:
                return self.send_json({"error": "Thiếu câu hỏi hoặc chế độ không hợp lệ"}, HTTPStatus.BAD_REQUEST)
            if not self.app.systems()[system]["available"]:
                return self.send_json({"error": self.app.graph_error}, HTTPStatus.SERVICE_UNAVAILABLE)
            self.send_json(self.app.ask(query, system))
        except json.JSONDecodeError as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # lỗi API Claude, Neo4j…: trả JSON để giao diện hiển thị
            self.send_json({"error": f"Lỗi khi trả lời: {type(error).__name__}: {error}"},
                           HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format, *args):
        if self.path.startswith("/api/chat"):
            super().log_message(format, *args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--index", choices=("auto", "sample", "full"), default="auto")
    args = parser.parse_args()
    index = pick_index(args.index)
    print(f"Chỉ mục {index}: kết nối Qdrant, Neo4j và tải E5…")
    Handler.app = App(index)
    print(f"Chỉ mục: {Handler.app.stats}" + (f" | {Handler.app.graph_error}" if Handler.app.graph_error else ""))
    print(f"Mở http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
