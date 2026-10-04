import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))


def load_script(name):
    specification = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


embed = load_script("embed")
verify_image = load_script("verify_embedding_image")


class EmbeddingImageTests(unittest.TestCase):
    def test_snapshot_parts_round_trip_and_remove_stale_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            snapshot = output / "snapshot.gz"
            snapshot.write_bytes(bytes(range(256)) * 7 + b"last bytes")
            names = embed.prepare_image(snapshot, output, part_size=500)
            self.assertGreater(len(names), 1)
            self.assertEqual(b"".join((output / "parts" / name).read_bytes() for name in names), snapshot.read_bytes())
            self.assertTrue(all((output / "parts" / name).stat().st_size <= 500 for name in names))
            dockerfile = (output / "Dockerfile").read_text()
            self.assertEqual(dockerfile.count("COPY parts/"), len(names))
            snapshot.write_bytes(b"new, smaller snapshot")
            new_names = embed.prepare_image(snapshot, output, part_size=500)
            self.assertEqual(sorted(path.name for path in (output / "parts").iterdir()), new_names)
            self.assertEqual((output / "parts" / new_names[0]).read_bytes(), snapshot.read_bytes())

    def test_default_and_all_export_entire_dataset_while_limit_is_explicit(self):
        # Kiểm tra cả lệnh index thực tế và manifest: mặc định không cắt nguồn ở 1.000 dòng.
        for selection, expected_limit in (([], None), (["--all"], None), (["--limit", "7"], 7)):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as directory:
                before, after = MagicMock(), MagicMock()
                before.execute.return_value.fetchone.return_value = (None,)
                after.execute.return_value.fetchone.return_value = ({"documents": 7},)

                def write_snapshot(command, destination, environment):
                    destination.write_bytes(gzip.compress(b"test database snapshot"))

                with patch.object(sys, "argv", ["embed.py", "--db-container", "stage", "--output", directory, *selection]), \
                        patch.object(embed, "connect", side_effect=[before, after]), \
                        patch.object(embed, "load_dotenv"), patch.object(embed.subprocess, "run") as index_run, \
                        patch.object(embed, "export_snapshot", side_effect=write_snapshot), patch("builtins.print"):
                    before.__enter__.return_value = before
                    after.__enter__.return_value = after
                    embed.main()
                command = index_run.call_args.args[0]
                if expected_limit is None:
                    self.assertNotIn("--limit", command)
                else:
                    self.assertEqual(command[command.index("--limit") + 1], str(expected_limit))
                manifest = json.loads((Path(directory) / "manifest.json").read_text())
                self.assertEqual(manifest["limit"], expected_limit)

    def test_dump_contains_only_app_tables_and_no_password_or_old_owners(self):
        environment = dict(POSTGRES_USER="stage_user", POSTGRES_PASSWORD="secret_for_test", POSTGRES_DB="stage",
                           POSTGRES_PORT="6543", POSTGRES_HOST="127.0.0.1", PGPASSWORD="secret_for_test")
        command = embed.dump_command("stage_container", environment)
        self.assertNotIn("secret_for_test", " ".join(command))
        self.assertIn("--no-owner", command)
        self.assertIn("--no-privileges", command)
        self.assertIn("--port=5432", command)  # cổng bên trong container, không phải cổng host
        self.assertEqual({item for item in command if item.startswith("--table=")},
                         {"--table=public." + name for name in embed.TABLES})

    def test_export_gzip_round_trip_without_loading_entire_dump(self):
        body = b"CREATE TABLE sample(id int);\n" * 100_000
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.sql"
            source.write_bytes(body)
            destination = Path(directory) / "snapshot.sql.gz"
            embed.export_snapshot([sys.executable, "-c", "import sys; sys.stdout.buffer.write(open(sys.argv[1], 'rb').read())",
                                   str(source)], destination, None)
            self.assertEqual(gzip.decompress(destination.read_bytes()), body)
            self.assertFalse(destination.with_name(destination.name + ".tmp").exists())

    def test_failed_dump_preserves_previous_snapshot_and_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "snapshot.sql.gz"
            destination.write_bytes(b"previous valid snapshot")
            with self.assertRaisesRegex(RuntimeError, "pg_dump"):
                embed.export_snapshot([sys.executable, "-c", "import sys; print('partial'); sys.exit(9)"],
                                      destination, None)
            self.assertEqual(destination.read_bytes(), b"previous valid snapshot")
            self.assertFalse(destination.with_name(destination.name + ".tmp").exists())

    def test_image_verification_rejects_missing_or_mismatched_tables(self):
        stats = dict(documents=10, chunks=15, entities=12, links=40, communities=3)
        conn = MagicMock()
        conn.execute.return_value.fetchone.side_effect = [(stats,), (9,)]
        with self.assertRaisesRegex(RuntimeError, "app_listings"):
            verify_image.verify(conn, dict(stats=stats, dimension=384))

    def test_image_verification_checks_counts_vectors_and_hnsw(self):
        stats = dict(documents=10, chunks=15, entities=12, links=40, communities=3)
        conn = MagicMock()
        conn.execute.return_value.fetchone.side_effect = [(stats,), (10,), (15,), (12,), (40,), (3,), (384,), (384,), (2,)]
        verify_image.verify(conn, dict(stats=stats, dimension=384))
        conn.execute.return_value.fetchone.side_effect = [(stats,), (10,), (15,), (12,), (40,), (3,), (384,), (384,), (1,)]
        with self.assertRaisesRegex(RuntimeError, "HNSW"):
            verify_image.verify(conn, dict(stats=stats, dimension=384))


if __name__ == "__main__":
    unittest.main()
