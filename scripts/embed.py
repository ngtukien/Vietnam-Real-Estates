"""Tạo embedding + KG + GraphRAG và xuất snapshot để đóng thành Docker image.

Ví dụ (PostgreSQL staging đã chạy, thông tin kết nối lấy từ .env):
    python scripts/embed.py --db-container <container-id>  # toàn bộ dataset
    python scripts/embed.py --limit 1000 --db-container <container-id>  # chạy thử

Script dùng lại app/index.py để image có đúng chỉ mục mà chatbot cần.
Không cần pg_dump trên máy nếu truyền --db-container; dùng pg_dump trong container PostgreSQL 17.
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from dotenv import load_dotenv

from engine import CHUNK_OVERLAP, CHUNK_TOKENS, DIMENSION, EMBEDDING_MODEL, connect
from index import TABLES

DATA_URL = "https://huggingface.co/datasets/tinixai/vietnam-real-estates"


def dump_command(container, environment):
    """Chỉ xuất các bảng app, không xuất role, mật khẩu hoặc bảng khác trong database."""
    command = ["pg_dump", "--no-owner", "--no-privileges", "--no-password", "--format=plain",
               "--host=127.0.0.1" if container else "--host=" + environment.get("POSTGRES_HOST", "127.0.0.1"),
               "--port=5432" if container else "--port=" + environment.get("POSTGRES_PORT", "5432"),
               "--username=" + environment.get("POSTGRES_USER", "real_estates"),
               "--dbname=" + environment.get("POSTGRES_DB", "real_estates")]
    for table in TABLES:
        command.append("--table=public." + table)
    if container:
        # Docker lấy PGPASSWORD từ môi trường của tiến trình, không đưa mật khẩu vào tham số lệnh.
        return ["docker", "exec", "--env", "PGPASSWORD", container, *command]
    if not shutil.which("pg_dump"):
        raise RuntimeError("Cần pg_dump 17 trên máy hoặc --db-container <container-id>")
    return command


def export_snapshot(command, destination, environment):
    """Nén trực tiếp luồng pg_dump; không giữ toàn bộ SQL trong RAM và không để lại dump dở dang."""
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, env=environment) as process:
            try:
                with gzip.open(temporary, "wb", compresslevel=6) as handle:
                    shutil.copyfileobj(process.stdout, handle, length=1024 * 1024)
            except BaseException:
                process.kill()
                raise
            if process.wait():
                raise RuntimeError("pg_dump thất bại; không tạo snapshot để build image")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_image(snapshot, destination, part_size=None):
    """Chia dump nén thành nhiều layer nhỏ để snapshot lớn publish được lên GHCR."""
    if part_size is None:
        # Dùng ít nhất 256 MiB/phần và tối đa 96 phần để chừa layer cho image nền và bootstrap.
        part_size = max(256 * 1024 * 1024, (snapshot.stat().st_size + 95) // 96)
        if part_size > 8 * 1024 * 1024 * 1024:
            raise RuntimeError("Snapshot quá lớn để đóng gói trong giới hạn layer GHCR")
    if part_size <= 0:
        raise ValueError("part_size phải lớn hơn 0")
    parts = destination / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    # Xóa các phần của lần đóng gói trước để không ghép lẫn snapshot cũ khi dump mới nhỏ hơn.
    for previous in parts.glob("snapshot-*.part"):
        previous.unlink()
    names = []
    with snapshot.open("rb") as source:
        while block := source.read(min(part_size, 1024 * 1024)):
            name = f"snapshot-{len(names):06d}.part"
            with (parts / name).open("wb") as handle:
                handle.write(block)
                remaining = part_size - len(block)
                while remaining and (block := source.read(min(remaining, 1024 * 1024))):
                    handle.write(block)
                    remaining -= len(block)
            names.append(name)
    if not names:
        raise RuntimeError("Snapshot rỗng; không đóng gói image")

    template = (ROOT / "docker" / "embeddings" / "Dockerfile").read_text(encoding="utf-8")
    # Mỗi COPY là một layer, không COPY cả thư mục parts trong một lệnh.
    layers = "\n".join(f"COPY parts/{name} /opt/embeddings/parts/{name}" for name in names)
    (destination / "Dockerfile").write_text(template.replace("# SNAPSHOT_LAYERS", layers), encoding="utf-8")
    for source, name in ((ROOT / "docker/postgres/init/01-pgvector.sql", "01-pgvector.sql"),
                         (ROOT / "docker/embeddings/02-restore.sh", "02-restore.sh"),
                         (ROOT / ".dockerignore", ".dockerignore")):
        shutil.copyfile(source, destination / name)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="chỉ giới hạn số bản ghi khi muốn chạy thử; mặc định đọc toàn bộ")
    selection.add_argument("--all", action="store_true", help="embedding toàn bộ dataset (đã là mặc định)")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "embeddings")
    parser.add_argument("--db-container", help="ID/tên container PostgreSQL staging để chạy pg_dump")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    # Không truyền --limit thì index.py đọc luồng dữ liệu đến hết nguồn Hugging Face.
    limit = args.limit
    if (limit is not None and limit <= 0) or args.batch_size <= 0 or args.workers <= 0:
        parser.error("--limit, --batch-size và --workers phải lớn hơn 0; dùng --all để đọc toàn bộ")

    load_dotenv(ROOT / ".env", override=False)
    environment = os.environ.copy()
    environment.setdefault("DATA_URL", DATA_URL)
    environment["PGPASSWORD"] = environment.get("POSTGRES_PASSWORD", "local_dev_password")
    command = dump_command(args.db_container, environment)  # kiểm tra công cụ trước khi embedding

    # Dùng database staging riêng. Không reset hoặc xóa dữ liệu có sẵn trong script đóng gói.
    with connect() as conn:
        if conn.execute("SELECT to_regclass('app_state')").fetchone()[0]:
            row = conn.execute("SELECT value FROM app_state WHERE key='loaded_through'").fetchone()
            if limit is not None and row and row[0] > limit:
                parser.error("Database đã chứa nhiều dòng hơn --limit; dùng database staging mới để tạo image đúng mẫu")

    index_command = [sys.executable, str(ROOT / "app" / "index.py"), "--batch-size", str(args.batch_size),
                     "--workers", str(args.workers)]
    if limit is not None:
        index_command += ["--limit", str(limit)]
    subprocess.run(index_command, check=True, cwd=ROOT, env=environment)

    with connect() as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key='ready'").fetchone()
        if not row or not row[0].get("documents"):
            raise RuntimeError("Chỉ mục chưa sẵn sàng hoặc không có tin hợp lệ")
        stats = row[0]

    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / "02-embeddings.sql.gz"
    export_snapshot(command, snapshot, environment)
    with snapshot.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    parts = prepare_image(snapshot, args.output)
    # Manifest giúp kiểm tra image khôi phục đủ dữ liệu và xác định cấu hình tạo vector.
    manifest = dict(format_version=2, created_at=datetime.now(timezone.utc).isoformat(),
                    embedding_model=EMBEDDING_MODEL, dimension=DIMENSION,
                    chunk_tokens=CHUNK_TOKENS, chunk_overlap=CHUNK_OVERLAP,
                    limit=limit, stats=stats, snapshot_sha256=digest, snapshot_parts=parts)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Snapshot: {snapshot} ({snapshot.stat().st_size:,} bytes)", flush=True)
    print(f"Chỉ mục đóng gói: {stats}", flush=True)


if __name__ == "__main__":
    main()
