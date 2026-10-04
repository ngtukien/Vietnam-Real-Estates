"""Khởi động image bằng volume trống và kiểm tra chỉ mục đã được khôi phục đầy đủ.

Container kiểm tra dùng user khác user tạo dump để xác nhận snapshot không phụ thuộc owner cũ.
Chỉ xóa container/volume kiểm tra do script này tạo ra.
"""

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path

import psycopg


def verify(conn, manifest):
    expected = manifest["stats"]
    ready = conn.execute("SELECT value FROM app_state WHERE key='ready'").fetchone()
    if not ready or ready[0] != expected:
        raise RuntimeError("Trạng thái ready trong image không khớp manifest")
    for name, table in (("documents", "app_listings"), ("chunks", "app_chunks"), ("entities", "app_entities"),
                        ("links", "app_listing_entities"), ("communities", "app_reports")):
        actual = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        if actual != expected[name]:
            raise RuntimeError(f"{table}: mong đợi {expected[name]} hàng, nhận {actual}")
    for table in ("app_chunks", "app_reports"):
        dimensions = conn.execute(f"SELECT vector_dims(embedding::vector) FROM {table} LIMIT 1").fetchone()
        if dimensions is None or dimensions[0] != manifest["dimension"]:
            raise RuntimeError(f"{table}: vector thiếu hoặc sai số chiều")
    indexes = conn.execute("SELECT count(*) FROM pg_indexes WHERE schemaname='public' "
                           "AND indexname IN ('app_chunks_embedding', 'app_reports_embedding')").fetchone()[0]
    if indexes != 2:
        raise RuntimeError("Image chưa khôi phục đủ hai index vector HNSW")
    print(f"Image đã khôi phục đúng chỉ mục: {expected}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--manifest", type=Path, default=Path("build/embeddings/manifest.json"))
    parser.add_argument("--timeout", type=int, default=21600,
                        help="số giây chờ khôi phục snapshot; mặc định 6 giờ cho toàn bộ dataset")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    name = "vre-embedding-check-" + uuid.uuid4().hex[:12]
    # Cổng được Docker chọn tự động, chỉ mở trên loopback để không va chạm database của người dùng.
    subprocess.run(["docker", "run", "--detach", "--name", name, "--shm-size=1g",
                    "--publish", "127.0.0.1::5432", "--env", "POSTGRES_USER=image_check",
                    "--env", "POSTGRES_PASSWORD=image_check_password", "--env", "POSTGRES_DB=image_check",
                    args.image], check=True, stdout=subprocess.DEVNULL)
    try:
        embedded_manifest = json.loads(subprocess.check_output(
            ["docker", "exec", name, "cat", "/opt/embeddings/manifest.json"], text=True))
        # Hỗ trợ cả image dump đơn trước đây và image toàn bộ dataset dùng nhiều layer.
        digest_command = (["bash", "-o", "pipefail", "-c",
                           "cat /opt/embeddings/parts/snapshot-*.part | sha256sum"]
                          if "snapshot_parts" in manifest else
                          ["sha256sum", "/docker-entrypoint-initdb.d/02-embeddings.sql.gz"])
        digest = subprocess.check_output(["docker", "exec", name, *digest_command], text=True).split()[0]
        if embedded_manifest != manifest or digest != manifest["snapshot_sha256"]:
            raise RuntimeError("Manifest hoặc SHA-256 snapshot trong image không khớp bản vừa embedding")
        address = subprocess.check_output(["docker", "port", name, "5432/tcp"], text=True).strip()
        host, port = address.rsplit(":", 1)
        deadline = time.monotonic() + args.timeout
        while True:
            try:
                conn = psycopg.connect(host=host, port=int(port), user="image_check", password="image_check_password",
                                       dbname="image_check", sslmode="disable", connect_timeout=2, autocommit=True)
                break
            except psycopg.OperationalError:
                if time.monotonic() >= deadline:
                    subprocess.run(["docker", "logs", "--tail", "80", name], check=False)
                    raise RuntimeError("Hết thời gian chờ PostgreSQL khôi phục snapshot") from None
                time.sleep(1)
        with conn:
            verify(conn, manifest)
    finally:
        subprocess.run(["docker", "rm", "--force", "--volumes", name], check=False, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
