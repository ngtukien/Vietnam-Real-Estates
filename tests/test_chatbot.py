"""Kiểm thử chatbot app/server.py với hệ thống hỏi đáp giả (không cần Chroma, Neo4j hay Claude)."""

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
    return dict(answer="Có căn [Tin#7].", sources=[8, 7, 999], route="graph", fallback=False, cypher="MATCH",
                latency=0.25, input_tokens=100, output_tokens=20)


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
        self.assertEqual(out["cited"], [7])
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
            self.assertEqual(json.load(urllib.request.urlopen(request))["answer"], "Có căn [Tin#7].")
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(base + "/static/../server.py")
            self.assertEqual(ctx.exception.code, 404)
        finally:
            httpd.shutdown()


if __name__ == "__main__":
    unittest.main()


class PickIndexTests(unittest.TestCase):
    def test_explicit_choice_is_kept(self):
        self.assertEqual(server.pick_index("full"), "full")
        self.assertEqual(server.pick_index("sample"), "sample")
