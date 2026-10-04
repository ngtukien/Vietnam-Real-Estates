import csv
import tempfile
import unittest
from pathlib import Path
try:
    from model import Corpus, SearchFilters, build_models
except ImportError:
    raise unittest.SkipTest('Baseline TF-IDF cũ đã được thay bằng notebook semantic trong các thư mục riêng')


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            dict(name='Nhà Hà Nội', description='thang máy gần trường', province_name='Hà Nội',
                 district_name='Trung Tâm', ward_name='A', property_type_name='Nhà', price='2000000000', area='50'),
            dict(name='Căn hộ Hà Nội', description='hồ bơi', province_name='Hà Nội',
                 district_name='Trung Tâm', ward_name='B', property_type_name='Căn hộ', price='3000000000', area='70'),
            dict(name='Nhà Đà Nẵng', description='gần biển', province_name='Đà Nẵng',
                 district_name='Trung Tâm', ward_name='A', property_type_name='Nhà', price='', area='60'),
        ]
        self.corpus = Corpus(self.rows)
        self.models = build_models(self.corpus)

    def test_text_retrieves_description_without_entity(self):
        self.assertEqual(self.models['traditional_rag'].search('thang máy')['results'][0]['id'], '1')
        self.assertEqual(self.models['traditional_kg'].search('thang máy')['results'], [])

    def test_graph_provenance_and_scoped_nodes(self):
        self.assertEqual(len([e for e in self.corpus.entities if e.startswith('district_name:')]), 2)
        output = self.models['traditional_kg'].search('Ha Noi')
        self.assertEqual({item['id'] for item in output['results']}, {'1', '2'})
        self.assertTrue(output['results'][0]['evidence'])

    def test_filters_apply_to_all_models_and_missing_values(self):
        for model in self.models.values():
            output = model.search('Nhà Hà Nội Đà Nẵng', filters=SearchFilters(max_price=2500000000))
            self.assertEqual([item['id'] for item in output['results']], ['1'])

    def test_community_report_has_sources(self):
        output = self.models['graphrag'].search('Hà Nội', filters=SearchFilters(province='Hà Nội'))
        report = output['community_reports'][0]
        self.assertEqual(report['source_ids'], ['1', '2'])
        self.assertIn('2 tin', report['summary'])
        self.assertEqual(len(self.corpus.communities), 2)

    def test_unknown_query_and_empty_corpus(self):
        for model in self.models.values():
            self.assertEqual(model.search('xyzunknown')['results'], [])
        for model in build_models(Corpus([])).values():
            self.assertEqual(model.search('Hà Nội')['results'], [])

    def test_csv_limit_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            with path.open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            self.assertEqual(len(Corpus.from_csv(path, 2).rows), 2)
            path.write_text('wrong\nvalue\n')
            with self.assertRaises(ValueError):
                Corpus.from_csv(path)

    def test_deterministic_comparison(self):
        for model in self.models.values():
            self.assertEqual(model.search('Nhà Hà Nội'), model.search('Nhà Hà Nội'))


if __name__ == '__main__':
    unittest.main()
