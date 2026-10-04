"""Đọc theo luồng từ Hugging Face, CSV qua HTTP(S) hoặc CSV cục bộ."""

import csv
import os
from contextlib import closing, contextmanager
from io import TextIOWrapper
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "vietnam-real-estates.csv"
REQUIRED_COLUMNS = {"name", "description", "province_name", "district_name", "price", "area"}


def huggingface_repo(source: Path | str) -> str | None:
    if isinstance(source, Path):
        return None
    parts = urlsplit(source)
    segments = parts.path.strip("/").split("/")
    if parts.hostname == "huggingface.co" and len(segments) == 3 and segments[0] == "datasets":
        return "/".join(segments[1:])
    return None


def resolve_source(path: Path | str | None = None) -> Path | str:
    if path is None:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        data_url = os.getenv("DATA_URL", "").strip()
        if not data_url:
            return DATA_PATH
        parts = urlsplit(data_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("DATA_URL phải là link dataset Hugging Face hoặc link HTTP(S) tải CSV")
        return data_url
    if isinstance(path, str) and urlsplit(path).scheme in {"http", "https"}:
        return path
    return Path(path)


@contextmanager
def open_csv(path: Path | str | None = None):
    source = resolve_source(path)
    if isinstance(source, Path):
        with source.open(encoding="utf-8-sig", newline="") as handle:
            yield handle
    else:
        request = Request(source, headers={"Accept-Encoding": "identity"})
        with urlopen(request, timeout=60) as response:
            if response.headers.get_content_type() == "text/html":
                raise ValueError("DATA_URL trả về trang HTML; cần link tải trực tiếp CSV")
            with TextIOWrapper(response, encoding="utf-8-sig", newline="") as handle:
                yield handle


@contextmanager
def open_rows(path: Path | str | None = None):
    source = resolve_source(path)
    repo = huggingface_repo(source)
    if repo:
        from datasets import load_dataset

        dataset = load_dataset(repo, split="train", streaming=True)
        with closing(iter(dataset)) as iterator:
            # Chuyển kiểu số/null của Parquet về cùng dạng đầu vào với csv.DictReader.
            rows = ({key: "" if value is None else str(value) for key, value in row.items()}
                    for row in iterator)
            with closing(rows):
                yield rows
    else:
        with open_csv(source) as handle:
            reader = csv.DictReader(handle)
            if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
                raise ValueError("CSV thiếu cột bắt buộc")
            yield reader
