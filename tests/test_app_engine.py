import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'app'))

if not importlib.util.find_spec('networkx'):
    raise unittest.SkipTest('requires networkx')

import engine  # noqa: E402
from test_notebook_flow import OffsetTokenizer  # noqa: E402


def document(id, title, province, district, kind, price, area, description='', ward='A'):
    return dict(id=id, title=title, description=description, metadata=dict(
        province_name=province, district_name=district, ward_name=ward, street_name='',
        project_name='', property_type_name=kind, price=price, area=area))


class EngineTests(unittest.TestCase):
    def test_iter_documents_cleans_rows_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            path.write_text('name,description,province_name,district_name,price,area\n'
                            ',,,,,\n" Nhà  đẹp ",mô tả,Hà Nội,A,NaN,40\nNhà 3,x,Hà Nội,A,1,2\n', encoding='utf-8')
            documents = list(engine.iter_documents(path))
            self.assertEqual([d['id'] for d in documents], ['2', '3'])
            self.assertEqual(documents[0]['title'], 'Nhà đẹp')
            self.assertIsNone(documents[0]['metadata']['price'])
            self.assertEqual([d['id'] for d in engine.iter_documents(path, start_after=2)], ['3'])
            self.assertEqual(engine.count_rows(path), 3)

    def test_entities_scope_keys_and_parents(self):
        hanoi = engine.listing_entities(document('1', 'x', 'Hà Nội', 'Trung Tâm', 'Nhà', 1, 1))
        danang = engine.listing_entities(document('2', 'x', 'Đà Nẵng', 'Trung Tâm', 'Nhà', 1, 1))
        district = next(e for e in hanoi if e['kind'] == 'district_name')
        self.assertEqual(district['key'], 'district_name:ha noi|trung tam')
        self.assertEqual(district['parent'], 'province_name:ha noi')
        self.assertNotEqual(district['key'], next(e for e in danang if e['kind'] == 'district_name')['key'])
        ward = next(e for e in hanoi if e['kind'] == 'ward_name')
        self.assertEqual(ward['parent'], district['key'])
        self.assertIsNone(next(e for e in hanoi if e['kind'] == 'property_type_name')['parent'])
        self.assertNotIn('street_name', {e['kind'] for e in hanoi})  # giá trị rỗng không thành thực thể

    def test_match_entities_requires_whole_phrase(self):
        entities = {1: ('province_name', 'Hà Nội', 10), 2: ('property_type_name', 'Nhà', 20),
                    3: ('district_name', 'Nội', 5), 4: ('ward_name', 'Hà', 5), 5: ('street_name', 'Thang', 1)}
        index = engine.build_phrase_index(entities)
        self.assertCountEqual(engine.match_entities(index, 'Nhà ở Hà Nội'), [1, 2, 3, 4])
        self.assertEqual(engine.match_entities(index, 'thang máy'), [5])
        self.assertEqual(engine.match_entities(index, 'nhanoi'), [])
        self.assertLess(engine.kg_weight(1000), engine.kg_weight(10))

    def test_detect_communities_groups_connected_listings(self):
        # Tin 1, 2 cùng phường 10; tin 3 ở phường 11; tin 4 không có liên kết.
        links = [(1, 10), (2, 10), (3, 11)]
        communities = engine.detect_communities([1, 2, 3, 4], links, parents={})
        self.assertIn([1, 2], communities)
        self.assertIn([4], communities)
        self.assertCountEqual([i for c in communities for i in c], [1, 2, 3, 4])
        self.assertEqual(communities, engine.detect_communities([1, 2, 3, 4], links, parents={}))

    def test_community_stats_keeps_top_values(self):
        rows = [dict(province_name='Hà Nội', district_name=f'Q{i}', property_type_name='Nhà', project_name='',
                     price=float(i), area=None) for i in range(1, 16)]
        stats = engine.community_stats(rows)
        self.assertEqual(stats['listing_count'], 15)
        self.assertEqual(len(stats['district_name']), engine.REPORT_TOP)
        self.assertEqual(stats['price'], dict(known_count=15, min=1.0, max=15.0))
        self.assertEqual(stats['area']['known_count'], 0)
        self.assertEqual(stats['project_name'], {})

    def test_rrf_merges_branches_by_listing(self):
        vector = [{'id': '3:0', 'listing_id': '3', 'content': 'a'}, {'id': '1:0', 'listing_id': '1', 'content': 'b'},
                  {'id': '1:1', 'listing_id': '1', 'content': 'c'}]
        graph = [{'id': 'listing:1', 'listing_id': '1', 'content': 'facts'}]
        hits = engine.reciprocal_rank_fusion(vector, graph)
        self.assertEqual(hits[0]['listing_id'], '1')
        self.assertAlmostEqual(hits[0]['score'], 1 / 62 + 1 / 61)
        self.assertEqual(hits[0]['ranks'], {0: 2, 1: 1})
        self.assertIn('facts', hits[0]['content'])

    def test_chunks_respect_budget(self):
        tokenizer = OffsetTokenizer()
        doc = document('1', 'Nhà Hà Nội', 'Hà Nội', 'Trung Tâm', 'Nhà', 2e9, 50,
                       ' '.join(f't{i}' for i in range(900)))
        chunks = engine.chunk_documents([doc], tokenizer)
        self.assertGreater(len(chunks), 1)
        covered = set()
        for chunk in chunks:
            self.assertLessEqual(len(tokenizer.encode('passage: ' + chunk['text'])), engine.CHUNK_TOKENS)
            covered.update(range(chunk['metadata']['token_start'], chunk['metadata']['token_end']))
        self.assertEqual(covered, set(range(900)))

    def test_prompt_cites_sources_and_scopes_reports(self):
        hit = {'id': '1:0', 'listing_id': '1', 'content': 'Giá 2 tỷ', 'metadata': {}}
        report = {'id': 'community:0', 'listing_id': None, 'content': '{}', 'metadata': {'listing_count': 31}}
        _, prompt = engine.build_prompt('Giá?', {}, [hit], [report])
        self.assertIn('[Nguồn 1:0; tin 1]', prompt)
        self.assertIn('[Nguồn community:0; 31 tin]', prompt)
        self.assertIn('thống kê trước bộ lọc', prompt)


class FakeIndex:
    def listing(self, listing_id):
        return dict(id=listing_id, title='Nhà Hà Nội', metadata=dict(
            province_name='Hà Nội', district_name='Trung Tâm', ward_name='A', street_name='', project_name='',
            property_type_name='Nhà', price=7.45e9, area=37.5, bedroom_count=3.0, description='mô tả'))


class ServerTests(unittest.TestCase):
    def setUp(self):
        import server
        self.server = server

    def test_best_answer_returns_single_formatted_listing(self):
        hit = {'id': 'listing:1', 'listing_id': '1', 'score': 0.0318, 'content': 'facts', 'ranks': {0: 5, 1: 1}}
        report = {'id': 'community:3', 'stats': {'listing_count': 31, 'district_name': {'Hoàng Mai': 31},
                  'province_name': {'Hà Nội': 31}, 'property_type_name': {'Nhà': 19, 'Đất': 3},
                  'price': {'known_count': 31, 'min': 2.98e9, 'max': 75e9}}}
        result = dict(model='graphrag', filters={}, matched_entities=['Hà Nội'], hits=[hit, dict(hit)],
                      reports=[report], elapsed=0.03)
        answer = self.server.best_answer(FakeIndex(), result)
        self.assertEqual(answer['best']['id'], '1')
        self.assertEqual(answer['best']['price'], '7,45 tỷ')
        self.assertEqual(answer['best']['area'], '37,5 m²')
        self.assertEqual(answer['best']['score'], 'RRF 0.0318 (vector #5, KG #1)')
        self.assertIn('31 tin ở Hoàng Mai, Hà Nội', answer['area']['summary'])
        self.assertNotIn('hits', answer)

    def test_empty_kg_answer_explains_missing_entities(self):
        result = dict(model='traditional_kg', filters={}, matched_entities=[], hits=[], reports=[], elapsed=0)
        answer = self.server.best_answer(FakeIndex(), result)
        self.assertIsNone(answer['best'])
        self.assertIn('Không nhận diện', answer['message'])

    def test_parse_filters_drops_empty_values(self):
        self.assertEqual(self.server.parse_filters({'province_name': 'Hà Nội', 'max_price': '8e9', 'min_area': ''}),
                         {'province_name': 'Hà Nội', 'max_price': 8e9})


if __name__ == '__main__':
    unittest.main()
