"""Kiểm thử chatbot app/server.py với hệ thống hỏi đáp giả (không cần Qdrant, Neo4j hay Claude)."""

import importlib.util
import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("chat_server", ROOT / "app" / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


CARDS = {
    7: dict(listing_id=7, title="Căn hộ 2PN", description="view sông", property_type="Căn hộ chung cư",
            price=2.9e9, area=56.0, bedrooms=2.0, ward="Cát Lái", district="Thủ Đức", province="Hồ Chí Minh",
            published_at="2025-06-01T00:00"),
    8: dict(listing_id=8, title="Nhà", property_type="Nhà", province="Hà Nội"),  # payload bỏ trường thiếu
}


def fake_app(graph_ok=True):
    app = server.App.__new__(server.App)
    app.stats = {"index": "sample", "listings": 2, "chunks": 3}
    app.graph_ok, app.graph_error = graph_ok, None if graph_ok else "Không kết nối được Neo4j"
    app.lock = threading.Lock()
    return app


def fake_system(question):
    return dict(answer="Có căn [Tin#7] và [Tin#5].", sources=[8, 7, 999], route="graph", fallback=False,
                cypher="MATCH", latency=0.25, input_tokens=100, output_tokens=20)


def failing_system(question):
    raise RuntimeError("secret internal detail")


class ChatbotTests(unittest.TestCase):
    def setUp(self):
        self.systems, self.listings = server.SYSTEMS, server.qdrant_store.listings
        server.SYSTEMS = {key: (name, fake_system) for key, (name, _) in self.systems.items()}
        server.qdrant_store.listings = lambda ids: {i: CARDS[i] for i in ids if i in CARDS}

    def tearDown(self):
        server.SYSTEMS, server.qdrant_store.listings = self.systems, self.listings

    def test_answer_sources_cited_first_and_unknown_dropped(self):
        out = fake_app().ask("q", "hybrid")
        self.assertEqual([s["id"] for s in out["sources"]], [7, 8])
        self.assertEqual((out["cited"], out["unsupported_citations"]), ([7], [5]))  # Tin#5 không có trong nguồn
        self.assertEqual((out["elapsed_ms"], out["tokens"]), (250, 120))
        card7, card8 = out["sources"]
        self.assertEqual((card7["price"], card7["area"], card7["address"]), ("2,90 tỷ", "56 m²", "Cát Lái, Thủ Đức, Hồ Chí Minh"))
        self.assertEqual((card8["price"], card8["area"], card8["project"]), ("Chưa rõ giá", "Chưa rõ diện tích", ""))
        json.dumps(out)  # trả được về giao diện

    def test_graph_modes_disabled_without_neo4j(self):
        systems = fake_app(graph_ok=False).systems()
        self.assertEqual({k: v["available"] for k, v in systems.items()},
                         {"hybrid": False, "basic_rag": True, "graph": False})

    def test_http_routes(self):
        server.Handler.app = fake_app()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            self.assertIn(b"Chatbot", urllib.request.urlopen(base + "/?q=abc").read())
            info = json.load(urllib.request.urlopen(base + "/api/info"))
            self.assertIs(info["systems"]["graph"]["available"], True)
            request = urllib.request.Request(base + "/api/chat", data=json.dumps({"query": "q", "system": "graph"}).encode(),
                                             headers={"Content-Type": "application/json"})
            response = urllib.request.urlopen(request)
            self.assertEqual(json.load(response)["answer"], "Có căn [Tin#7] và [Tin#5].")
            self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
            self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(base + "/static/../server.py")
            self.assertEqual(ctx.exception.code, 404)
        finally:
            httpd.shutdown()
            httpd.server_close()


class GuardTests(unittest.TestCase):
    """Các lớp chặn của /api/chat: Content-Type, kích thước, độ dài câu hỏi, tần suất, API key, lỗi nội bộ."""

    def setUp(self):
        self.saved = server.SYSTEMS, server.LIMITER, server.Handler.api_token, server.qdrant_store.listings
        server.SYSTEMS = {key: (name, fake_system) for key, (name, _) in server.SYSTEMS.items()}
        server.LIMITER = server.RateLimiter(0)
        server.qdrant_store.listings = lambda ids: {i: CARDS[i] for i in ids if i in CARDS}
        server.Handler.app = fake_app()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/api/chat"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        server.SYSTEMS, server.LIMITER, server.Handler.api_token, server.qdrant_store.listings = self.saved

    def post(self, body=None, headers=None, raw=None):
        data = raw if raw is not None else json.dumps(body or {"query": "q", "system": "basic_rag"}).encode()
        request = urllib.request.Request(self.url, data=data,
                                         headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            with error:
                return error.code, json.load(error)

    def test_rejects_bad_requests(self):
        self.assertEqual(self.post(headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.post(raw=b"{" * (server.MAX_BODY + 1))[0], 413)
        self.assertEqual(self.post(raw=b"[1, 2]")[0], 400)
        self.assertEqual(self.post(raw=b"{not json")[0], 400)
        self.assertEqual(self.post({"query": "x" * (server.CHAT_MAX_QUERY_CHARS + 1), "system": "graph"})[0], 400)
        self.assertEqual(self.post({"query": "q", "system": "unknown"})[0], 400)

    def test_rate_limit(self):
        server.LIMITER = server.RateLimiter(2)
        self.assertEqual([self.post()[0] for _ in range(3)], [200, 200, 429])

    def test_api_key(self):
        server.Handler.api_token = "secret"
        self.assertEqual(self.post()[0], 401)
        self.assertEqual(self.post(headers={"X-API-Key": "wrong"})[0], 401)
        self.assertEqual(self.post(headers={"X-API-Key": "secret"})[0], 200)

    def test_internal_error_is_not_leaked(self):
        server.SYSTEMS = {key: (name, failing_system) for key, (name, _) in server.SYSTEMS.items()}
        with self.assertLogs("chatbot", level="ERROR"):
            status, body = self.post()
        self.assertEqual(status, 500)
        self.assertNotIn("secret internal detail", body["error"])


class PickIndexTests(unittest.TestCase):
    def test_explicit_choice_is_kept(self):
        self.assertEqual(server.pick_index("full"), "full")
        self.assertEqual(server.pick_index("sample"), "sample")

    def test_loopback_detection(self):
        self.assertTrue(server.is_loopback("127.0.0.1"))
        self.assertTrue(server.is_loopback("localhost"))
        self.assertFalse(server.is_loopback("0.0.0.0"))


if __name__ == "__main__":
    unittest.main()
