import io
import os
import sys
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import data_source
import engine
import index as indexer
from test_notebook_flow import NOTEBOOKS, code_cells, execute, namespace, step

URL = "https://example.test/listings.csv"
HF_URL = "https://huggingface.co/datasets/tinixai/vietnam-real-estates"
HEADER = "name,description,province_name,district_name,price,area\n"
CSV = ("\ufeff" + HEADER + ',,,,,\n" Nhà  đẹp ","mô tả\n nhiều dòng",Hà Nội,A,NaN,40\n'
       + "Nhà 3,x,Hà Nội,A,1,2\n")


class Response(io.BytesIO):
    def __init__(self, body, content_type="text/csv"):
        super().__init__(body.encode("utf-8"))
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.bytes_read = 0

    def close(self):
        if not self.closed:
            self.bytes_read = self.tell()
        super().close()


class StreamingDataset:
    column_names = ["name", "description", "province_name", "district_name", "price", "area"]

    def __init__(self):
        self.consumed = 0
        self.closed = False

    def __iter__(self):
        try:
            for position in range(10_000):
                self.consumed += 1
                yield dict(name=" Nhà  đẹp " if position else "", description="mô tả\n nhiều dòng" if position else None,
                           province_name="Hà Nội", district_name="A", price=None if position == 1 else 0.0,
                           area=40.0)
        finally:
            self.closed = True


class DataSourceTests(unittest.TestCase):
    def test_huggingface_streaming_limit_resume_nulls_numbers_and_cleanup(self):
        dataset = StreamingDataset()
        with patch("datasets.load_dataset", return_value=dataset) as load:
            documents = list(engine.iter_documents(HF_URL, limit=4, start_after=1))
        load.assert_called_once_with("tinixai/vietnam-real-estates", split="train", streaming=True)
        self.assertEqual([doc["id"] for doc in documents], ["2", "3", "4"])
        self.assertEqual(documents[0]["title"], "Nhà đẹp")
        self.assertEqual(documents[0]["description"], "mô tả nhiều dòng")
        self.assertIsNone(documents[0]["metadata"]["price"])
        self.assertEqual(documents[1]["metadata"]["price"], 0.0)
        self.assertEqual(documents[0]["metadata"]["area"], 40.0)
        self.assertEqual(dataset.consumed, 4)
        self.assertTrue(dataset.closed)

    def test_huggingface_missing_schema_closes_stream(self):
        dataset = StreamingDataset()
        dataset.column_names = ["wrong"]
        def wrong_rows():
            try:
                yield {"wrong": 1}
            finally:
                dataset.closed = True
        with patch("datasets.load_dataset", return_value=wrong_rows()):
            with self.assertRaisesRegex(ValueError, "thiếu cột"):
                list(engine.iter_documents(HF_URL))
        self.assertTrue(dataset.closed)

    def test_url_stream_stops_at_limit_and_closes_response(self):
        response = Response(CSV + "x" * 1_000_000)
        with patch.object(data_source, "urlopen", return_value=response) as request:
            documents = list(engine.iter_documents(URL, limit=2))
        self.assertEqual([doc["id"] for doc in documents], ["2"])
        self.assertEqual(documents[0]["title"], "Nhà đẹp")
        self.assertEqual(documents[0]["description"], "mô tả nhiều dòng")
        self.assertIsNone(documents[0]["metadata"]["price"])
        self.assertTrue(response.closed)
        self.assertLess(response.bytes_read, 1_000_000)
        self.assertEqual(request.call_args.args[0].full_url, URL)
        self.assertEqual(request.call_args.kwargs["timeout"], 60)

    def test_url_resume_and_count_csv_records_including_multiline(self):
        with patch.object(data_source, "urlopen", side_effect=lambda *a, **k: Response(CSV)):
            self.assertEqual([doc["id"] for doc in engine.iter_documents(URL, start_after=2)], ["3"])
            self.assertEqual(engine.count_rows(URL), 3)

    def test_environment_url_takes_priority_and_explicit_file_overrides_it(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, DATA_URL=URL):
            path = Path(directory) / "data.csv"
            path.write_text(CSV, encoding="utf-8")
            with patch.object(data_source, "urlopen", side_effect=lambda *a, **k: Response(CSV)) as request:
                self.assertEqual(len(engine.load_documents(3)), 2)
                self.assertEqual(engine.count_rows(path), 3)
                self.assertEqual(request.call_count, 1)

    def test_bad_url_html_schema_and_http_errors_do_not_fall_back(self):
        with patch.dict(os.environ, DATA_URL="file:///tmp/data.csv"):
            with self.assertRaisesRegex(ValueError, "HTTP"):
                data_source.resolve_source()
        for body, content_type, error in (("<html>login</html>", "text/html", "HTML"),
                                           ("wrong\nvalue\n", "text/csv", "thiếu cột")):
            response = Response(body, content_type)
            with self.subTest(content_type=content_type), patch.object(data_source, "urlopen", return_value=response):
                with self.assertRaisesRegex(ValueError, error):
                    list(engine.iter_documents(URL))
                self.assertTrue(response.closed)
        with patch.object(data_source, "urlopen", side_effect=HTTPError(URL, 404, "Not found", {}, None)):
            with self.assertRaises(HTTPError):
                list(engine.iter_documents(URL))


class NotebookRemoteTests(unittest.TestCase):
    def test_all_notebooks_default_to_huggingface_without_env_or_local_csv(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            os.environ.pop("DATA_URL", None)
            for path in NOTEBOOKS:
                dataset = StreamingDataset()
                with self.subTest(notebook=path.name), patch("datasets.load_dataset", return_value=dataset) as load:
                    ns = namespace(root=Path(directory))
                    if path.stem == "traditional_rag":
                        execute(step(path, "raw_rows = "), ns)
                    execute(step(path, "documents = []"), ns)
                    load.assert_called_once_with("tinixai/vietnam-real-estates", split="train", streaming=True)
                    self.assertEqual(ns["documents"][0]["id"], "2")
                    self.assertIsNone(ns["documents"][0]["metadata"]["price"])
                    self.assertEqual(dataset.consumed, 10 if path.stem == "traditional_rag" else 1000)
                    self.assertTrue(dataset.closed)

    def test_all_notebooks_read_url_without_local_csv_and_stop_at_sample(self):
        body = CSV + ("Nhà,x,Hà Nội,A,1,2\n" * 100_000)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, DATA_URL=URL):
            root = Path(directory)
            for path in NOTEBOOKS:
                response = Response(body)
                with self.subTest(notebook=path.name), patch("urllib.request.urlopen", return_value=response):
                    ns = namespace(root=root)
                    if path.stem == "traditional_rag":
                        execute(step(path, "raw_rows = "), ns)
                    execute(step(path, "documents = []"), ns)
                    self.assertEqual(ns["documents"][0]["id"], "2")
                    self.assertEqual(ns["documents"][0]["description"], "mô tả nhiều dòng")
                    self.assertEqual(len(ns["documents"]), 9 if path.stem == "traditional_rag" else 999)
                    self.assertTrue(response.closed)
                    self.assertLess(response.bytes_read, len(body.encode("utf-8")))

    def test_root_discovery_and_standalone_rag_cleaning_without_csv(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, DATA_URL=URL):
            root = Path(directory)
            (root / "app").mkdir()
            (root / "model").mkdir()
            for path in NOTEBOOKS:
                folder = root / "model" / path.parent.name
                folder.mkdir()
                with self.subTest(notebook=path.name), patch.object(Path, "cwd", return_value=folder):
                    ns = namespace()
                    execute(code_cells(path)[0], ns)
                    self.assertEqual(ns["root"], root)
                    if path.stem == "traditional_rag":
                        with patch("urllib.request.urlopen", return_value=Response(CSV)):
                            execute(step(path, "documents = []"), ns)
                        self.assertEqual([doc["id"] for doc in ns["documents"]], ["2", "3"])


class IndexRemoteTests(unittest.TestCase):
    def test_full_remote_index_reads_once_and_resumes_from_saved_source_id(self):
        state = {"loaded_through": 2}
        conn, encoder, pool = MagicMock(), MagicMock(), MagicMock()
        pool.submit.return_value.result.return_value = []

        def save_batch(conn, documents, chunks, vectors, entities):
            state["loaded_through"] = int(documents[-1]["id"])

        with patch.object(indexer, "get_state", side_effect=lambda conn, key, default=None: state.get(key, default)), \
                patch.object(indexer, "set_state", side_effect=lambda conn, key, value: state.update({key: value})), \
                patch.object(indexer, "ProcessPoolExecutor") as executor, \
                patch.object(indexer, "write_batch", side_effect=save_batch) as write, \
                patch.object(indexer, "log"), \
                patch.object(data_source, "urlopen", side_effect=lambda *a, **k: Response(CSV)) as request:
            executor.return_value.__enter__.return_value = pool
            indexer.load_listings(conn, encoder, None, 2, 1, URL)
            self.assertEqual(request.call_count, 1)
            self.assertEqual([doc["id"] for doc in write.call_args.args[1]], ["3"])
            self.assertEqual(state["loaded_through"], 3)
            self.assertTrue(state["source_complete"])
            indexer.load_listings(conn, encoder, None, 2, 1, URL)
            self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
