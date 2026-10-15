"""Tiện ích dùng chung (data-contract §9.3): ghi JSON nguyên tử, git sha, file handoff, chuyển scalar, fingerprint."""

from __future__ import annotations

import json
import math
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vre.config import Settings

CODE_ROOT = Path(__file__).resolve().parents[2]
SIGNALS = ("SKELETON_READY", "CLEAN_DATA_READY", "INDEX_STARTED", "INDEX_DONE", "INDEX_FAILED",
           "P1_DONE", "P2_DONE", "CHANGE_REQUEST", "CHANGE_APPLIED", "NOTE")
SENDERS = ("p1", "p2")
FINGERPRINT_RTOL = 1e-9
FINGERPRINT_THREADS = 4  # một cột, đọc tuần tự: không cần nhiều tài nguyên
FINGERPRINT_MEMORY = "1GB"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def scalar(value: Any) -> Any:
    """numpy/pandas scalar → kiểu Python thuần cho JSON; NaN/NaT/NA → None."""
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        import pandas as pd

        if value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except ImportError:
        pass
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def write_json_atomic(path: Path | str, obj: Any) -> Path:
    """Ghi `<path>.tmp` rồi `os.replace` — người đọc không bao giờ thấy file dở dang."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=scalar), encoding="utf-8")
    os.replace(tmp, path)
    return path


def git_sha(root: Path = CODE_ROOT) -> str:
    """7 ký tự đầu của HEAD; `"dirty"` nếu file đã theo dõi có thay đổi chưa commit."""
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()

    if run("status", "--porcelain", "--untracked-files=no"):
        return "dirty"
    return run("rev-parse", "--short=7", "HEAD")


def result_header(schema_version: int = 1) -> dict[str, Any]:
    """Ba trường bắt buộc của mọi JSON trong results/ (data-contract quy ước chung)."""
    return {"schema_version": schema_version, "generated_at": utc_now(), "git_sha": git_sha()}


def handoff_filename(name: str, sender: str, payload: dict[str, Any], now: datetime | None = None) -> str:
    if name == "NOTE":
        stamp = (now or datetime.now(UTC)).strftime("%y%m%d%H%M%S")
        return f"NOTE-{sender}-{stamp}.json"
    if "id" in payload:
        return f"{name}-{payload['id']}.json"
    return f"{name}.json"


def write_handoff(settings: Settings, name: str, sender: str, payload: dict[str, Any]) -> Path:
    """Ghi tín hiệu phối hợp `results/handoff/<file>.json` (data-contract §7.10); gửi lại ⇒ tăng `seq`."""
    if name not in SIGNALS:
        raise ValueError(f"Tín hiệu lạ {name!r}; hợp lệ: {', '.join(SIGNALS)}")
    if sender not in SENDERS:
        raise ValueError(f"Người gửi lạ {sender!r}; hợp lệ: p1, p2")
    path = settings.handoff_dir / handoff_filename(name, sender, payload)
    seq = 1
    if path.exists():
        seq = int(json.loads(path.read_text(encoding="utf-8")).get("seq", 0)) + 1
    envelope = {"schema_version": 1, "signal": name, "from": sender, "seq": seq, "at": utc_now(), "payload": payload}
    return write_json_atomic(path, envelope)


def read_handoff(settings: Settings, filename: str) -> dict[str, Any] | None:
    path = settings.handoff_dir / filename
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def listings_fingerprint(clean_path: Path) -> dict[str, Any]:
    """`{n_listings, price_vnd_sum}` của bảng sạch; tổng Kahan để kết quả không đổi theo thứ tự luồng."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads = {FINGERPRINT_THREADS}")
    con.execute(f"SET memory_limit = '{FINGERPRINT_MEMORY}'")
    try:
        n, total = con.execute("SELECT count(*), kahan_sum(price_vnd) FROM read_parquet(?)",
                               [str(clean_path)]).fetchone()
    finally:
        con.close()
    return {"n_listings": int(n), "price_vnd_sum": float(total or 0.0)}


def fingerprints_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (int(a["n_listings"]) == int(b["n_listings"])
            and math.isclose(float(a["price_vnd_sum"]), float(b["price_vnd_sum"]), rel_tol=FINGERPRINT_RTOL))
