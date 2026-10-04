"""Pha B. Chatbot: server HTTP phục vụ giao diện HTML/CSS và API truy vấn.

Cần lập chỉ mục trước (`python app/index.py`). Chạy từ thư mục gốc dự án:
    .venv/bin/python app/server.py            # http://127.0.0.1:8000
"""

import argparse
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from engine import MODELS, Index

STATIC = Path(__file__).resolve().parent / "static"
FILTER_KEYS = ("province_name", "property_type_name", "min_price", "max_price", "min_area", "max_area")


def format_vnd(value):
    if value is None:
        return "Chưa rõ giá"
    for unit, name in ((1e9, "tỷ"), (1e6, "triệu")):
        if value >= unit:
            return f"{value / unit:.2f}".rstrip("0").rstrip(".").replace(".", ",") + f" {name}"
    return f"{value:,.0f} đ"


def format_area(value):
    return "Chưa rõ diện tích" if value is None else f"{value:g}".replace(".", ",") + " m²"


def score_label(model, hit):
    if model == "traditional_rag":
        return f"cosine {hit['score']:.3f}"
    if model == "traditional_kg":
        return f"điểm KG {hit['score']:.3f}"
    ranks = hit.get("ranks", {})
    parts = [f"{name} #{ranks[branch]}" for branch, name in ((0, "vector"), (1, "KG")) if branch in ranks]
    return f"RRF {hit['score']:.4f} ({', '.join(parts)})"


def most_common(counts, n=2):
    return ", ".join(sorted(counts, key=counts.get, reverse=True)[:n])


def best_answer(index, result):
    """Rút gọn kết quả truy xuất thành một tin tốt nhất (và một báo cáo khu vực với GraphRAG)."""
    model = result["model"]
    answer = dict(
        model=model, model_name=MODELS[model], matched_entities=result["matched_entities"],
        filters_ignored=model == "traditional_rag" and bool(result["filters"]),
        elapsed_ms=round(result["elapsed"] * 1000), best=None, area=None,
    )
    if result["hits"]:
        hit = result["hits"][0]
        listing = index.listing(hit["listing_id"])
        meta = listing["metadata"]
        description = meta.get("description") or ""
        answer["best"] = dict(
            id=listing["id"],
            title=listing["title"],
            property_type=meta.get("property_type_name") or "Chưa rõ loại",
            price=format_vnd(meta.get("price")),
            area=format_area(meta.get("area")),
            address=", ".join(meta[k] for k in ("street_name", "ward_name", "district_name", "province_name")
                              if meta.get(k)),
            project=meta.get("project_name") or "",
            bedrooms=meta.get("bedroom_count"),
            bathrooms=meta.get("bathroom_count"),
            published_at=(meta.get("published_at") or "")[:10],
            description=description[:600] + ("…" if len(description) > 600 else ""),
            score=score_label(model, hit),
            evidence=hit["content"],
        )
    elif model == "traditional_kg" and not result["matched_entities"]:
        answer["message"] = ("Không nhận diện được tỉnh, quận, phường, đường, dự án hay loại hình nào trong câu hỏi, "
                             "nên Traditional KG không có điểm xuất phát để duyệt đồ thị.")
    else:
        answer["message"] = "Không tìm thấy tin phù hợp với câu hỏi và bộ lọc hiện tại."
    if result["reports"]:
        stats = result["reports"][0]["stats"]
        price = stats["price"]
        answer["area"] = dict(
            id=result["reports"][0]["id"],
            summary=(f"{stats['listing_count']} tin ở {most_common(stats['district_name']) or '—'}, {most_common(stats['province_name']) or '—'}; "
                     f"chủ yếu {most_common(stats['property_type_name']) or '—'}; giá "
                     + (f"{format_vnd(price['min'])} – {format_vnd(price['max'])}" if price["known_count"] else "chưa rõ")),
        )
    return answer


def parse_filters(raw):
    filters = {}
    for key in FILTER_KEYS:
        value = raw.get(key)
        if value in (None, "", 0):
            continue
        filters[key] = str(value) if key in ("province_name", "property_type_name") else float(value)
    return filters


class Handler(BaseHTTPRequestHandler):
    index: Index
    lock = threading.Lock()  # encoder và kết nối DB dùng chung giữa các luồng

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/info":
            return self.send_json(dict(models=MODELS, stats=self.index.stats, provinces=self.index.provinces,
                                       property_types=self.index.property_types))
        name = "index.html" if self.path in ("/", "/index.html") else self.path.removeprefix("/static/")
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
        if self.path != "/api/chat":
            return self.send_error(HTTPStatus.NOT_FOUND)
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            query = str(payload.get("query", "")).strip()
            model = payload.get("model", "rag_kg")
            if not query or model not in MODELS:
                return self.send_json({"error": "Thiếu câu hỏi hoặc mô hình không hợp lệ"}, HTTPStatus.BAD_REQUEST)
            filters = parse_filters(payload.get("filters") or {})
            with self.lock:
                result = self.index.search(model, query, filters, top_k=1, report_k=1)
            self.send_json(best_answer(self.index, result))
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # lỗi DB… trả về JSON để giao diện hiển thị
            self.send_json({"error": f"Lỗi khi truy vấn: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format, *args):
        if self.path.startswith("/api/chat"):
            super().log_message(format, *args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print("Đang đọc chỉ mục từ PostgreSQL và tải E5…")
    Handler.index = Index.load()
    Handler.index.connection()  # mở sẵn kết nối DB để câu hỏi đầu tiên không phải chờ
    print(f"Chỉ mục: {Handler.index.stats}")
    print(f"Mở http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
