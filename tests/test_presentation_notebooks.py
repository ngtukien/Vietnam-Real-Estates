"""Kiểm thử src/ và cấu trúc 5 notebook trình bày theo outline (không cần mạng, LLM hay Neo4j)."""

import json
import re
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import nbformat
import pandas as pd

from src import data, eval as ev, graph, hybrid, llm, qdrant_store, rag

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"


class FakeTokenizer:
    """Mỗi ký tự là một token, offset là vị trí ký tự."""

    def encode(self, text, add_special_tokens=False):
        return list(text)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


def listing(**over):
    row = dict(listing_id=1, title="Nhà", description="x" * 1000, property_type="Nhà", province="Hà Nội",
               district="Cầu Giấy", ward="Dịch Vọng", street="Xuân Thủy", price=5e9, area=50.0, bedrooms=3.0)
    row.update(over)
    return row


class ToyCellTests(unittest.TestCase):
    """Các số trên slide (Phụ lục A) tính bằng code của src/."""

    def test_chunk_count_matches_formula(self):
        self.assertEqual(len(data.chunk_starts(2000, 400, 100)), 7)
        self.assertEqual(data.n_chunks_formula(2000, 400, 100), 7)
        for L in (1, 400, 401, 1234, 5000):
            self.assertEqual(len(data.chunk_starts(L, 300, 50)), data.n_chunks_formula(L, 300, 50))

    def test_cosine_rrf_modularity(self):
        self.assertEqual([round(rag.cosine((3, 4), d), 2) for d in [(6, 8), (4, 3), (-4, 3)]], [1.0, 0.96, 0.0])
        toy = rag.rrf([["X", "Y", "Z"], ["Y", "Z", "X"]])
        self.assertEqual([d for d, _ in toy], ["Y", "X", "Z"])
        self.assertEqual([f"{s:.5f}" for _, s in toy], ["0.03252", "0.03227", "0.03200"])
        edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
        self.assertAlmostEqual(graph.modularity(edges, [{0, 1, 2}, {3, 4, 5}]), 5 / 14)

    def test_metrics(self):
        self.assertAlmostEqual(ev.recall_at_k(["d4", "d2", "d1", "d9", "d3"], {"d1", "d4", "d7"}, 5), 2 / 3)
        runs = [(["a", "x", "y"], {"a"}), (["x", "y", "b"], {"b"}), (["x", "c", "y"], {"c"})]
        self.assertAlmostEqual(ev.mrr(runs), (1 + 1 / 3 + 1 / 2) / 3)
        self.assertEqual(ev.faithfulness([True, True, True, False]), 0.75)
        # capped: 10 tin đúng, lấy được 5 trong top-5 -> 1,0
        self.assertEqual(ev.recall_at_k(list(range(5)), set(range(10)), 5, capped=True), 1.0)

    def test_read_only_guard(self):
        self.assertTrue(graph.is_read_only("MATCH (l:Listing) RETURN count(l) AS so_tin"))
        for bad in ("MATCH (l) DETACH DELETE l", "MATCH (l) SET l.x = 1", "CALL db.labels()",
                    "LOAD CSV FROM 'x' AS r RETURN r", "merge (n:X)"):
            self.assertFalse(graph.is_read_only(bad), bad)


class DataTests(unittest.TestCase):
    def test_normalize_place(self):
        cases = [("Q.7", "district", "Quận 7"), ("q7", "district", "Quận 7"), ("8", "district", "Quận 8"),
                 ("Quận 07", "district", "Quận 7"), ("Huyện Củ Chi", "district", "Củ Chi"),
                 ("Thủ Đức", "district", "Thủ Đức"), ("P.5", "ward", "Phường 5"), ("5", "ward", "Phường 5"),
                 ("Xã Tân Thông Hội", "ward", "Tân Thông Hội"), ("TP. Hồ Chí Minh", "province", "Hồ Chí Minh"),
                 ("None", "district", None), (None, "ward", None)]
        for value, kind, expected in cases:
            self.assertEqual(data.normalize_place(value, kind), expected, value)

    def test_mask_pii(self):
        self.assertEqual(data.mask_pii("LH 0912 345 678 hoặc a.b@mail.com"), "LH [phone_number] hoặc [email]")
        self.assertEqual(data.mask_pii("giá 7450000000"), "giá 7450000000")
        self.assertIsNone(data.mask_pii(None))

    def test_clean(self):
        raw = pd.DataFrame([
            dict(listing_id=1, name="Nhà  đẹp", description="gọi 0912345678", district_name="7", ward_name="P.5",
                 province_name="TP. Hồ Chí Minh", price=5e9, area=50.0),
            dict(listing_id=2, name="nhà đẹp", description="Gọi 0912345678", district_name="7", ward_name="5",
                 province_name="Hồ Chí Minh", price=-1, area=0),
            dict(listing_id=3, name=None, description=None, district_name=None, ward_name=None,
                 province_name=None, price=None, area=None),
            dict(listing_id=4, name="Nhà Q10 giá 3,2 tỷ", description="", district_name="10", ward_name=None,
                 province_name="Hồ Chí Minh", price=3.2e6, area=64.0),
            dict(listing_id=5, name="Cho thuê nhà", description="15 triệu/tháng", district_name="10", ward_name=None,
                 province_name="Hồ Chí Minh", price=1.5e7, area=16.0),
        ])
        for col in data.COL.values():
            if col not in raw:
                raw[col] = None
        df, steps = data.clean(raw)
        self.assertEqual(df["listing_id"].tolist(), [1, 4, 5])  # dòng 2 trùng, dòng 3 rỗng
        self.assertEqual(list(steps.values()), [5, 4, 3, 3, 3])
        # giá nhập theo nghìn đồng được nhân 1.000; tin cho thuê bị bỏ giá
        self.assertEqual(df.set_index("listing_id").loc[4, "price"], 3.2e9)
        self.assertTrue(pd.isna(df.set_index("listing_id").loc[5, "price"]))
        row = df.iloc[0]
        self.assertEqual((row.title, row.district, row.ward, row.province), ("Nhà đẹp", "Quận 7", "Phường 5", "Hồ Chí Minh"))
        self.assertEqual(row.description, "gọi [phone_number]")
        self.assertAlmostEqual(row.price_m2_mil, 100.0)

    def test_chunk_listing_budget_and_overlap(self):
        tok = FakeTokenizer()
        row = listing()
        chunks = data.chunk_listing(row, tok, size=300, overlap=50)
        budget = data.body_budget(row, tok, 300)
        self.assertEqual(len(chunks), data.n_chunks_formula(1000, budget, 50))
        self.assertTrue(all(c["n_tokens"] <= 300 for c in chunks))
        self.assertTrue(all(c["text"].startswith(data.listing_header(row)) for c in chunks))
        self.assertEqual(len(data.chunk_listing(row, tok, strategy="whole")), 1)


class FilterTests(unittest.TestCase):
    def test_money_and_filters(self):
        self.assertEqual(rag.parse_money_mil("5 tỷ"), 5000)
        self.assertEqual(rag.parse_money_mil("8,5 tỷ"), 8500)
        self.assertEqual(rag.parse_money_mil("1 tỷ 2"), 1200)
        self.assertEqual(rag.parse_money_mil("850 triệu"), 850)
        f = rag.QueryFilters.from_extracted({"district": "Q.7", "property_type": "Căn hộ chung cư", "bedrooms": 2,
                                             "max_price": "4 tỷ", "min_price": None})
        self.assertEqual((f.district, f.max_price_mil), ("Quận 7", 4000))
        must = {c.key: c for c in qdrant_store.to_filter(f).must}
        self.assertEqual(must["district"].match.value, "Quận 7")
        self.assertEqual((must["price"].range.gte, must["price"].range.lte), (None, 4e9))
        self.assertEqual((must["bedrooms"].range.gte, must["bedrooms"].range.lte), (2, 2))
        self.assertIsNone(qdrant_store.to_filter(rag.QueryFilters()))

    def test_bm25_sparse_vectors(self):
        doc = qdrant_store.bm25_document(["sổ", "hồng", "sổ"], avgdl=3.0)
        weights = dict(zip(doc.indices, doc.values))
        tf2 = 2 * 2.2 / (2 + 1.2)  # tf = 2, dl = avgdl -> chuẩn hoá độ dài bằng 1
        self.assertAlmostEqual(weights[qdrant_store.token_id("sổ")], tf2)
        self.assertAlmostEqual(weights[qdrant_store.token_id("hồng")], 2.2 / 2.2)
        query = qdrant_store.bm25_query(["sổ", "sổ", "hồng"])
        self.assertEqual((len(query.indices), set(query.values)), (2, {1.0}))

    def test_post_filter_matches_qdrant_filter(self):
        df = pd.DataFrame([listing(listing_id=1, district="Quận 7", bedrooms=2.0, price=3e9),
                           listing(listing_id=2, district="Quận 7", bedrooms=2.0, price=None),
                           listing(listing_id=3, district="Quận 8", bedrooms=2.0, price=3e9)])
        f = rag.QueryFilters(district="Quận 7", bedrooms=2, max_price_mil=4000)
        self.assertEqual(df[rag.matches(df, f)].listing_id.tolist(), [1])

    def test_llm_schema_is_closed(self):
        schema = llm._schema(rag.ExtractedFilters)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertNotIn("default", json.dumps(schema))


class GraphTests(unittest.TestCase):
    def test_load_blocks_and_rows(self):
        blocks = graph.load_blocks()
        self.assertEqual(set(blocks), {"constraints", "provinces", "districts", "wards", "property_types", "listings",
                                       "of_type", "in_ward", "in_district", "in_province"})
        self.assertEqual(len(blocks["constraints"]), 8)
        df = pd.DataFrame([listing(listing_id=1, price_bn=5.0, price_m2_mil=100.0, bathrooms=None, floors=None,
                                   published_at="2025-06-01"),
                           listing(listing_id=2, ward=None, price_bn=5.0, price_m2_mil=float("nan"), bathrooms=None,
                                   floors=None, published_at="2025-06-01"),
                           listing(listing_id=3, province="Đà Nẵng", district="Cầu Giấy", ward="Dịch Vọng", price_bn=5.0,
                                   price_m2_mil=100.0, bathrooms=None, floors=None, published_at="2025-06-01")])
        rows = graph.graph_rows(df)
        # Cầu Giấy ở hai tỉnh khác nhau là hai District khác nhau.
        self.assertEqual(len(rows["districts"]), 2)
        self.assertEqual([r["listing_id"] for r in rows["in_district"]], [2])
        self.assertIsNone(rows["listings"][1]["props"]["price_m2"])
        expected = graph.expected_counts(df).set_index("loai")["pandas"]
        self.assertEqual(expected["IN_DISTRICT"], 1 + 2)  # 1 tin thiếu phường + 2 phường

    def test_mini_graph(self):
        G, triples = graph.mini_graph(listing())
        self.assertEqual((G.number_of_nodes(), len(triples)), (6, 5))

    def test_text2cypher_blocks_then_retries(self):
        replies = iter([{"cypher": "MATCH (l) DETACH DELETE l"}, {"cypher": "MATCH (l:Listing) RETURN 1 AS x"}])
        fake = lambda *a, **k: llm.LLMResult("", next(replies), 10, 5, False)
        with mock.patch.object(graph, "ask_llm", fake), \
                mock.patch.object(graph, "run_cypher", return_value=pd.DataFrame({"x": [1]})) as run:
            out = graph.text2cypher("câu hỏi")
        self.assertIsNone(out["error"])
        self.assertEqual(len(out["attempts"]), 2)
        self.assertIn("bị chặn", out["attempts"][0]["error"])
        run.assert_called_once_with("MATCH (l:Listing) RETURN 1 AS x")
        self.assertEqual(out["usage"], [20, 10])


class HybridTests(unittest.TestCase):
    def test_graph_route_falls_back_to_rag_when_empty(self):
        hits = pd.DataFrame({"listing_id": [7], "text": ["Căn hộ"]})
        with mock.patch.object(hybrid, "router", return_value={"route": "graph", "reason": "", "input_tokens": 1,
                                                              "output_tokens": 1}), \
                mock.patch.object(hybrid, "text2cypher", return_value={"error": None, "rows": pd.DataFrame(),
                                                                       "cypher": "MATCH", "usage": [2, 2]}), \
                mock.patch.object(hybrid, "retrieve", return_value={"hits": hits, "calls": []}), \
                mock.patch.object(hybrid, "generate", return_value=llm.LLMResult("ok [Tin#7]", None, 3, 3, False)):
            out = hybrid.ask_hybrid("q")
        self.assertTrue(out["fallback"])
        self.assertEqual(out["sources"], [7])
        self.assertEqual((out["input_tokens"], out["output_tokens"]), (6, 6))


class BenchmarkTests(unittest.TestCase):
    def test_ground_truth_rules(self):
        df = pd.DataFrame({"listing_id": [1, 2, 3, 4], "district": ["A", "A", "B", "B"],
                           "price_m2_mil": [10.0, 20.0, 50.0, None], "type": ["x", "y", "x", "x"]})
        self.assertEqual(ev.ground_truth(df, {"query": "type == 'x'"}), ([1, 3, 4], None))
        self.assertEqual(ev.ground_truth(df, {"query": "type == 'x'", "agg": "count"})[1], "3")
        rule = {"groupby": ["district"], "value": "price_m2_mil", "agg": "mean", "pick": "min"}
        self.assertEqual(ev.ground_truth(df, rule)[1], "A (15.00)")
        self.assertEqual(ev.ground_truth(df, {**rule, "min_count": 2})[1], "A (15.00)")
        self.assertEqual(ev.ground_truth(df, {"groupby": ["district"], "agg": "count", "pick": "top5"})[1],
                         "A (2); B (2)")

    def test_questions_file(self):
        q = pd.read_csv(ROOT / "benchmark" / "questions.csv", keep_default_na=False)
        self.assertEqual(list(q.columns[:5]), ["id", "type", "route", "question", "gt_rule"])
        self.assertTrue({"gt_ids", "gt_value"} <= set(q.columns))
        self.assertEqual(Counter(q.type), {"lookup": 6, "constraint": 6, "aggregate": 6, "multihop": 6})
        self.assertTrue(set(q.route) <= set(hybrid.ROUTES))
        for rule in q.gt_rule:
            json.loads(rule)


class NotebookStructureTests(unittest.TestCase):
    """Khung cell khớp outline: 51 cell, 19 LIVE · 22 CACHE · 10 ẨN, đúng số theo từng notebook."""

    EXPECTED = {"00_setup.ipynb": (0, 0, 3), "01_rag_indexing.ipynb": (4, 6, 1),
                "02_retrieval_generation.ipynb": (5, 4, 2), "03_knowledge_graph.ipynb": (6, 6, 1),
                "04_graphrag_eval.ipynb": (4, 6, 3)}

    def test_cell_labels(self):
        total = Counter()
        for name, (live, cache, hidden) in self.EXPECTED.items():
            nb = nbformat.read(NOTEBOOKS / name, as_version=4)
            labels = [re.match(r"# ((?:P\d|00)-\d\d) · (LIVE|CACHE|ẨN)", c.source) for c in nb.cells
                      if c.cell_type == "code"]
            labels = [m.groups() for m in labels if m]
            ids = [cid for cid, _ in labels]
            self.assertEqual(ids, sorted(ids), name)
            self.assertEqual(len(ids), len(set(ids)), name)
            counts = Counter(label for _, label in labels)
            self.assertEqual((counts["LIVE"], counts["CACHE"], counts["ẨN"]), (live, cache, hidden), name)
            total.update(counts)
            if name != "00_setup.ipynb":
                self.assertEqual(nb.cells[1].source, "%run ./00_setup.ipynb")
        self.assertEqual((sum(total.values()), total["LIVE"], total["CACHE"], total["ẨN"]), (51, 19, 22, 10))

    def test_no_secrets_or_phone_numbers_in_outputs(self):
        for name in self.EXPECTED:
            text = (NOTEBOOKS / name).read_text(encoding="utf-8")
            self.assertNotRegex(text, r"sk-ant-[\w-]{10,}", name)
            nb = nbformat.read(NOTEBOOKS / name, as_version=4)
            outputs = json.dumps([c.get("outputs", []) for c in nb.cells if c.cell_type == "code"], ensure_ascii=False)
            self.assertIsNone(data.PHONE.search(outputs), name)


if __name__ == "__main__":
    unittest.main()
