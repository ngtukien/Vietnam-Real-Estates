"""Chatbot: server HTTP phục vụ giao diện HTML/CSS và API hỏi đáp trên src/ (cùng code với notebook).

Hai chỉ mục, chọn bằng --index:
- sample: N_ROWS dòng đầu do notebook 01 (Qdrant listings_sample) và 03 (Neo4j graph_db) tạo.
- full: toàn bộ dataset do scripts/index_full.py tạo (Qdrant listings_full, Neo4j graph_db_full).
- auto (mặc định): full nếu đã lập chỉ mục xong, ngược lại sample.

    .venv/bin/python app/server.py            # http://127.0.0.1:8000

Bảo vệ API: tối đa CHAT_MAX_CONCURRENT câu hỏi chạy đồng thời, CHAT_RATE_PER_MIN câu hỏi/phút cho mỗi IP,
body tối đa MAX_BODY byte. Đặt CHAT_API_TOKEN để bắt buộc header X-API-Key (mở trang bằng /?token=...).
"""

import argparse
import hmac
import ipaddress
import json
import logging
import mimetypes
import os
import sys
import threading
import uuid
from collections import defaultdict, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for key, value in {"TQDM_DISABLE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_VERBOSITY": "error",
                   "TRANSFORMERS_VERBOSITY": "error", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(key, value)  # log khởi động gọn

from src import graph, qdrant_store, rag, vector  # noqa: E402
from src.common import fmt_vnd  # noqa: E402
from src.config import (CHAT_API_TOKEN, CHAT_MAX_CONCURRENT, CHAT_MAX_QUERY_CHARS,  # noqa: E402
                        CHAT_RATE_PER_MIN, FULL_COLLECTION, INDEX_DIR, NEO4J_FULL_URI, NEO4J_URI,
                        SAMPLE_COLLECTION)
from src.data import text  # noqa: E402
from src.hybrid import ask_hybrid  # noqa: E402

log = logging.getLogger("chatbot")
STATIC = Path(__file__).resolve().parent / "static"
SYSTEMS = {
    "hybrid": ("Tự động", ask_hybrid),  # router chọn Basic RAG, Graph hoặc gộp cả hai
    "basic_rag": ("Basic RAG", rag.ask_rag),
    "graph": ("Graph (Text2Cypher)", graph.ask_graph),
}
NEEDS_GRAPH = {"hybrid", "graph"}
MAX_SOURCES = 5
MAX_BODY = 8 * 1024
QUEUE_TIMEOUT = 60  # giây chờ một chỗ trống khi đã đủ CHAT_MAX_CONCURRENT câu hỏi đang chạy
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


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


class RateLimiter:
    """Cửa sổ trượt 60 giây: mỗi khoá (IP) tối đa `per_minute` lượt; per_minute = 0 là không giới hạn."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = monotonic()
        with self.lock:
            hits = self.hits[key]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            return True


SLOTS = threading.BoundedSemaphore(max(1, CHAT_MAX_CONCURRENT))
LIMITER = RateLimiter(CHAT_RATE_PER_MIN)


class Busy(Exception):
    """Đã đủ số câu hỏi chạy đồng thời và chờ quá QUEUE_TIMEOUT."""


class App:
    """Trạng thái dùng chung: chỉ mục Qdrant và kết nối đồ thị (nếu có) của bản sample hoặc full.
    Embedding, pyvi, reranker và client LLM tự khoá bên trong src/, nên nhiều câu hỏi chạy song song được."""

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

    def systems(self) -> dict:
        return {key: dict(name=name, available=self.graph_ok or key not in NEEDS_GRAPH)
                for key, (name, _) in SYSTEMS.items()}

    def ask(self, question: str, system: str) -> dict:
        if not SLOTS.acquire(timeout=QUEUE_TIMEOUT):
            raise Busy
        try:
            result = SYSTEMS[system][1](question)
        finally:
            SLOTS.release()
        sources = [int(i) for i in result["sources"]]
        cited, unsupported = rag.check_citations(result["answer"], sources)
        # Ưu tiên tin được trích trong câu trả lời, sau đó tới các tin đã đưa vào ngữ cảnh.
        wanted = list(dict.fromkeys(cited + sources))
        cards = qdrant_store.listings(wanted[:MAX_SOURCES * 2])
        ids = [i for i in wanted if i in cards][:MAX_SOURCES]
        return dict(
            system=system, system_name=SYSTEMS[system][0], answer=result["answer"],
            cited=cited, unsupported_citations=unsupported,
            sources=[listing_card(cards[i]) for i in ids],
            route=result.get("route"), fallback=bool(result.get("fallback")),
            filters=result.get("filters") or {}, note=result.get("note") or "",
            cypher=result.get("cypher"), error=result.get("error"),
            elapsed_ms=round(result["latency"] * 1000),
            tokens=result["input_tokens"] + result["output_tokens"],
        )


class Handler(BaseHTTPRequestHandler):
    app: App
    api_token: str = CHAT_API_TOKEN

    def end_headers(self):
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        super().end_headers()

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def authorized(self) -> bool:
        if not self.api_token:
            return True
        given = self.headers.get("X-API-Key", "")
        return hmac.compare_digest(given.encode(), self.api_token.encode())

    def do_GET(self):
        route = urlsplit(self.path).path  # bỏ query string (?q=...)
        if route == "/api/info":
            if not self.authorized():
                return self.send_json({"error": "Thiếu hoặc sai API key"}, HTTPStatus.UNAUTHORIZED)
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

    def read_payload(self) -> dict:
        """Body JSON của POST; ValueError (-> 400/413/415) nếu sai định dạng hoặc quá lớn."""
        if self.headers.get_content_type() != "application/json":
            raise ValueError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type phải là application/json")
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            raise ValueError(HTTPStatus.BAD_REQUEST, "Content-Length không hợp lệ") from None
        if length < 0 or length > MAX_BODY:
            raise ValueError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"Body tối đa {MAX_BODY} byte")
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(HTTPStatus.BAD_REQUEST, f"JSON không hợp lệ: {error}") from None
        if not isinstance(payload, dict):
            raise ValueError(HTTPStatus.BAD_REQUEST, "Body phải là object JSON")
        return payload

    def do_POST(self):
        if urlsplit(self.path).path != "/api/chat":
            return self.send_error(HTTPStatus.NOT_FOUND)
        if not self.authorized():
            return self.send_json({"error": "Thiếu hoặc sai API key"}, HTTPStatus.UNAUTHORIZED)
        if not LIMITER.allow(self.client_address[0]):
            return self.send_json({"error": "Quá nhiều câu hỏi, thử lại sau ít phút"}, HTTPStatus.TOO_MANY_REQUESTS)
        try:
            payload = self.read_payload()
        except ValueError as error:
            status, message = error.args
            return self.send_json({"error": message}, status)
        query = str(payload.get("query", "")).strip()
        system = payload.get("system", "hybrid")
        if not query or system not in SYSTEMS:
            return self.send_json({"error": "Thiếu câu hỏi hoặc chế độ không hợp lệ"}, HTTPStatus.BAD_REQUEST)
        if len(query) > CHAT_MAX_QUERY_CHARS:
            return self.send_json({"error": f"Câu hỏi tối đa {CHAT_MAX_QUERY_CHARS} ký tự"}, HTTPStatus.BAD_REQUEST)
        if not self.app.systems()[system]["available"]:
            return self.send_json({"error": self.app.graph_error}, HTTPStatus.SERVICE_UNAVAILABLE)

        request_id = uuid.uuid4().hex[:8]
        try:
            out = self.app.ask(query, system)
        except Busy:
            return self.send_json({"error": "Server đang bận, thử lại sau"}, HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception:  # lỗi API LLM, Neo4j…: ghi đủ vào log, giao diện chỉ nhận mã yêu cầu
            log.exception("request=%s system=%s lỗi khi trả lời", request_id, system)
            return self.send_json({"error": f"Lỗi khi trả lời (mã {request_id}), xem log server"},
                                  HTTPStatus.INTERNAL_SERVER_ERROR)
        # Không ghi nội dung câu hỏi vào log (có thể chứa thông tin cá nhân), chỉ ghi độ dài.
        log.info("request=%s system=%s route=%s fallback=%s chars=%d sources=%d bad_cites=%d ms=%d tokens=%d",
                 request_id, system, out["route"], out["fallback"], len(query), len(out["sources"]),
                 len(out["unsupported_citations"]), out["elapsed_ms"], out["tokens"])
        self.send_json({**out, "request_id": request_id})

    def log_message(self, format, *args):
        pass  # do_POST đã ghi log có cấu trúc; bỏ log truy cập file tĩnh


def is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--index", choices=("auto", "sample", "full"), default="auto")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if not is_loopback(args.host) and not CHAT_API_TOKEN:
        raise SystemExit(f"Mở server ra {args.host} cần đặt CHAT_API_TOKEN trong .env để chặn người lạ gọi LLM.")
    index = pick_index(args.index)
    print(f"Chỉ mục {index}: kết nối Qdrant, Neo4j và tải E5…")
    Handler.app = App(index)
    print(f"Chỉ mục: {Handler.app.stats}" + (f" | {Handler.app.graph_error}" if Handler.app.graph_error else ""))
    print(f"Mở http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
