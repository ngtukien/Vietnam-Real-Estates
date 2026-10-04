import ast
import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import tempfile
import unicodedata
import unittest
from collections import Counter
from contextlib import redirect_stdout
from itertools import islice
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / 'model').glob('*/*.ipynb'))


def code_cells(path):
    notebook = json.loads(path.read_text())
    return [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']


def step(path, fragment):
    return next(source for source in code_cells(path)
                if fragment in source or fragment in ast.unparse(ast.parse(source)))


def execute(source, namespace):
    with redirect_stdout(io.StringIO()):
        exec(compile(source, '<notebook-cell>', 'exec'), namespace)


def namespace(**values):
    return dict(csv=csv, hashlib=hashlib, json=json, math=math, re=re,
                unicodedata=unicodedata, Counter=Counter, islice=islice,
                display=lambda value: None, **values)


class OffsetTokenizer:
    """Word offsets exercise chunk coverage without downloading E5."""
    is_fast = True

    def encode(self, text, add_special_tokens=True):
        return list(range(len(re.findall(r'\S+', text)) + (2 if add_special_tokens else 0)))

    def __call__(self, text, **kwargs):
        return {'offset_mapping': [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]}


class NotebookFlowTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            {'id': '1', 'title': 'Nhà Hà Nội', 'description': ' '.join('t' + str(i) for i in range(900)),
             'metadata': {'province_name': 'Hà Nội', 'district_name': 'Trung Tâm',
                          'ward_name': 'A', 'price': 2e9, 'area': 50, 'property_type_name': 'Nhà'}},
            {'id': '2', 'title': 'Nhà Đà Nẵng', 'description': '',
             'metadata': {'province_name': 'Đà Nẵng', 'district_name': 'Trung Tâm',
                          'ward_name': 'A', 'price': None, 'area': 60, 'property_type_name': 'Nhà'}},
        ]

    def test_notebooks_use_direct_cells_without_configuration_blocks(self):
        self.assertEqual(len(NOTEBOOKS), 4)
        self.assertEqual(list((ROOT / 'model').rglob('*.py')), [])
        removed = {'LIMIT', 'CHUNK_TOKENS', 'CHUNK_OVERLAP', 'EMBEDDING_MODEL',
                   'TOP_K', 'CANDIDATES', 'USE_LLM', 'LLM_MODEL', 'OLLAMA_URL',
                   'EXPORT', 'CLOSE_DB', 'MODEL_NAME', 'MODEL_FOLDER'}
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text())
                source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
                self.assertIn('A. Lập chỉ mục', source)
                self.assertIn('B. Truy vấn', source)
                self.assertNotIn('USE_LLM', source)
                self.assertNotIn('11434', source)
                self.assertNotIn('requests', source)
                if path.stem == 'traditional_rag':
                    self.assertEqual(len(code_cells(path)), 8)  # Preparation + seven steps.
                    self.assertNotIn('LLM', source)
                    self.assertNotIn('prompt', source)
                    self.assertNotIn('filters', source)
                    self.assertNotIn('batch_size', source)
                for index, code in enumerate(code_cells(path)):
                    tree = ast.parse(code)
                    compile(code, f'{path}:{index}', 'exec')
                    self.assertFalse(any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                       ast.Lambda, ast.ClassDef)) for n in ast.walk(tree)))
                    self.assertFalse(removed & {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})

    def test_csv_cleaning_preserves_source_ids_and_handles_missing_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'data/vietnam-real-estates.csv').write_text(
                'name,description,province_name,district_name,price,area\n'
                ',,,,,\n" Nhà  đẹp ",mô tả,Hà Nội,A,NaN,40\n', encoding='utf-8')
            for path in NOTEBOOKS:
                with self.subTest(notebook=path.name):
                    ns = namespace(root=root)
                    if path.stem == 'traditional_rag':
                        execute(step(path, 'raw_rows = '), ns)
                        execute(step(path, 'documents = []'), ns)
                    else:
                        execute(step(path, 'documents = []'), ns)
                    self.assertEqual(ns['documents'][0]['id'], '2')
                    self.assertEqual(ns['documents'][0]['title'], 'Nhà đẹp')
                    self.assertIsNone(ns['documents'][0]['metadata']['price'])
                    self.assertEqual(ns['documents'][0]['metadata']['area'], 40)

    def test_chunk_coverage_budget_and_sources(self):
        tokenizer = OffsetTokenizer()
        for path in NOTEBOOKS:
            if path.stem == 'traditional_kg':
                continue
            with self.subTest(notebook=path.name):
                tree = ast.parse(step(path, 'chunks = []'))
                # Inject a deterministic tokenizer instead of loading Hugging Face.
                tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)
                             and not (isinstance(node, ast.Assign) and any(
                                 isinstance(t, ast.Name) and t.id == 'tokenizer' for t in node.targets))]
                ns = namespace(documents=self.documents, tokenizer=tokenizer)
                execute(ast.unparse(tree), ns)
                coverage = set()
                for chunk in ns['chunks']:
                    self.assertLessEqual(len(tokenizer.encode('passage: ' + chunk['text'])), 384)
                    self.assertEqual(chunk['metadata']['source_ids'], [chunk['listing_id']])
                    if chunk['listing_id'] == '1':
                        coverage.update(range(chunk['metadata']['token_start'], chunk['metadata']['token_end']))
                self.assertEqual(coverage, set(range(900)))
                self.assertEqual(len([c for c in ns['chunks'] if c['listing_id'] == '2']), 1)
                self.assertEqual(len({c['id'] for c in ns['chunks']}), len(ns['chunks']))

    @unittest.skipUnless(importlib.util.find_spec('networkx'), 'requires networkx')
    def test_graph_scoping_filters_and_missing_values(self):
        for path in NOTEBOOKS:
            if path.stem == 'traditional_rag':
                continue
            with self.subTest(notebook=path.name):
                ns = namespace(documents=self.documents, query='Nha Ha Noi Da Nang', filters={'max_price': 3e9})
                execute(step(path, "listing = 'listing:'"), ns)
                districts = [n for n, p in ns['graph'].nodes(data=True) if p['kind'] == 'district_name']
                self.assertEqual(len(districts), 2)
                execute(step(path, 'normalized_query ='), ns)
                self.assertEqual([hit['listing_id'] for hit in ns['graph_hits']], ['1'])
                self.assertEqual(ns['query'], 'Nha Ha Noi Da Nang')
                if path.stem == 'graphrag':
                    execute(step(path, 'community_graph ='), ns)
                    ids = [key for report in ns['reports'] for key in report['metadata']['source_ids']]
                    self.assertCountEqual(ids, ['1', '2'])
                    for report in ns['reports']:
                        self.assertEqual(json.loads(report['text'])['listing_count'], len(report['metadata']['source_ids']))
                    execute(step(path, 'summary_method'), ns)
                    for report in ns['reports']:
                        self.assertEqual(report['metadata']['summary_method'], 'structured_statistics')
                        self.assertEqual(report['text'], report['metadata']['facts'])

    def test_rrf_deduplicates_listing_ranks_and_merges_evidence(self):
        for path in NOTEBOOKS:
            if path.stem not in ('rag_kg', 'graphrag'):
                continue
            with self.subTest(notebook=path.name):
                ns = namespace(vector_hits=[{'id': '1:0', 'listing_id': '1', 'content': 'chunk'},
                                            {'id': '1:1', 'listing_id': '1', 'content': 'second'},
                                            {'id': '2:0', 'listing_id': '2', 'content': 'other'}],
                               graph_hits=[{'id': 'listing:1', 'listing_id': '1', 'content': 'graph facts'}],
                               selected_reports=[])
                execute(step(path, 'for ranking in'), ns)
                self.assertEqual(ns['hits'][0]['listing_id'], '1')
                self.assertAlmostEqual(ns['hits'][0]['score'], 2 / 61)
                self.assertAlmostEqual(ns['hits'][1]['score'], 1 / 62)
                self.assertIn('graph facts', ns['hits'][0]['content'])

    @unittest.skipUnless(importlib.util.find_spec('psycopg'), 'requires psycopg')
    def test_sql_local_filters_and_global_report_scope(self):
        for path in NOTEBOOKS:
            if path.stem == 'traditional_kg':
                continue
            with self.subTest(notebook=path.name):
                conn = MagicMock()
                cursor = conn.cursor.return_value.__enter__.return_value
                cursor.fetchall.return_value = []
                ns = namespace(conn=conn, query_vector=[0.1, 0.2], collection='sample',
                               vector_table_name='notebook_vectors_2', dimension=2,
                               encoder=MagicMock(), np=MagicMock(), query='Nhà Hà Nội',
                               filters={'province_name': 'Hà Nội', 'max_price': 3e9})
                if path.stem == 'traditional_rag':
                    import psycopg
                    from psycopg.rows import dict_row
                    ns.update(sql=psycopg.sql, dict_row=dict_row)
                    tree = ast.parse(step(path, 'WITH nearest_chunks'))
                    start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'statement')
                    end = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.With)) + 1
                    source = ast.unparse(ast.Module(body=tree.body[start:end], type_ignores=[]))
                    execute(source, ns)
                    sql, args = cursor.execute.call_args.args
                    self.assertIn('DISTINCT ON (listing_id)', sql.as_string())
                    self.assertIn('ORDER BY score DESC, id', sql.as_string())
                    self.assertIn('LIMIT 5', sql.as_string())
                    self.assertNotIn('metadata->>', sql.as_string())
                    self.assertEqual(args, ([0.1, 0.2], 'sample', [0.1, 0.2]))
                    self.assertEqual(ns['hits'], [])
                    continue
                source = step(path, 'conditions, args =')
                execute(source, ns)
                sql, args = cursor.execute.call_args.args
                self.assertIn("metadata->>'province_name' = %s", sql.as_string())
                self.assertIn("(metadata->>'price')::double precision <= %s", sql.as_string())
                self.assertEqual(args[3:5], ['Hà Nội', 3e9])
                self.assertEqual(args[-1], 20)
                if path.stem == 'graphrag':
                    execute(step(path, "[query_vector, collection, 'community_report']"), ns)
                    sql, args = cursor.execute.call_args.args
                    self.assertNotIn('metadata->>', sql.as_string())
                    self.assertEqual(args[2], 'community_report')

    def test_prompt_keeps_citations(self):
        for path in NOTEBOOKS:
            if path.stem == 'traditional_rag':
                continue
            with self.subTest(notebook=path.name):
                ns = namespace(query='Giá?', filters={},
                               hits=[{'id': '1:0', 'listing_id': '1', 'content': 'Giá 2 tỷ', 'metadata': {}}])
                source = step(path, 'context = ')
                execute(source, ns)
                self.assertIn('[Nguồn 1:0; tin 1]', ns['prompt'])
                self.assertIn('Câu hỏi: Giá?', ns['prompt'])

    def test_exports_retrieval_without_a_generated_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in NOTEBOOKS:
                if path.stem == 'traditional_rag':
                    continue
                with self.subTest(notebook=path.name):
                    ns = namespace(root=Path(directory), collection='sample', query='Giá?',
                                   filters={}, hits=[], context='', prompt='Câu hỏi: Giá?')
                    execute(step(path, 'output_path = '), ns)
                    output = json.loads(ns['output_path'].read_text())
                    self.assertEqual(output['query'], 'Giá?')
                    self.assertEqual(output['sources'], [])
                    self.assertEqual(output['prompt'], 'Câu hỏi: Giá?')
                    self.assertNotIn('answer', output)


if __name__ == '__main__':
    unittest.main()
