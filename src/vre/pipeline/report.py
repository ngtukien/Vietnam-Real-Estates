"""`night_report.json` (data-contract §7.5) và lệnh `status`.

Report luôn được ghi (kể cả khi lỗi). `embed.chunks_per_s_avg` chỉ là dòng thông tin — không có quyết định, ngưỡng hay ETA nào dựa vào nó.
"""

import ctypes
import datetime as dt
import json
import shutil
import sys
import threading
import time
from pathlib import Path

import psutil
import pyarrow.compute as pc
import pyarrow.parquet as pq

from vre.common import git_sha, write_json_atomic
from vre.config import EMBED_MODEL_SLUG
from vre.pipeline.state import layout_for, now_iso

STEPS = ("checks", "chunk", "bm25", "embed", "upload", "hnsw", "verify", "pca", "report")
MIN_FREE_COMMIT_GB, COMMIT_WAIT_S = 4.0, 1800  # yêu cầu của p1: luôn để ≥ 4 GB commit trống
COUNTERS = ("parts_total", "parts_embedded", "parts_uploaded", "chunks", "points_in_qdrant",
            "chunks_truncated_listings", "oom_retries", "upload_retries", "batch_size_final")


class DiskLowError(RuntimeError):
    """Ổ dữ liệu sắp đầy / thiếu commit memory: dừng ở checkpoint, không tự xoá gì."""


def require_resources(settings, part: int, log=print) -> None:
    """Trước mỗi part: ổ D đủ chỗ (không ⇒ dừng ở checkpoint) và commit memory còn ≥ MIN_FREE_COMMIT_GB.

    Commit (không phải RAM vật lý) là giới hạn thật trên máy này: hai lần sập 2026-10-15 là do cạn commit. Thiếu commit ⇒
    chờ (phiên p1 có watchdog nhường bộ nhớ) thay vì thoát, để không phải nạp lại model; quá COMMIT_WAIT_S ⇒ dừng ở checkpoint.
    """
    free = shutil.disk_usage(Path(settings.data_root)).free / 1e9
    if free < settings.min_free_disk_gb_part:
        raise DiskLowError(f"ổ dữ liệu còn {free:.1f} GB < {settings.min_free_disk_gb_part} GB trước part {part:04d}")
    deadline = time.monotonic() + COMMIT_WAIT_S
    while (commit := psutil.virtual_memory().available / 1e9 if sys.platform != "win32" else free_commit_gb()) < MIN_FREE_COMMIT_GB:
        if time.monotonic() > deadline:
            raise DiskLowError(f"commit memory còn {commit:.1f} GB < {MIN_FREE_COMMIT_GB} GB sau {COMMIT_WAIT_S}s chờ (part {part:04d})")
        log(f"commit memory còn {commit:.1f} GB < {MIN_FREE_COMMIT_GB} GB: chờ trước part {part:04d}")
        time.sleep(30)


def free_commit_gb() -> float:
    if sys.platform != "win32":  # GlobalMemoryStatusEx chỉ có trên Windows; require_resources dùng psutil ở nơi khác
        raise OSError("free_commit_gb chỉ chạy trên Windows")
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), *[(n, ctypes.c_ulonglong) for n in (
            "total_phys", "avail_phys", "total_commit", "avail_commit", "total_virtual", "avail_virtual", "avail_ext")]]

    status = MemoryStatus(dwLength=ctypes.sizeof(MemoryStatus))
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.avail_commit / 1e9


def _wall_hours(runs: list[dict], now: dt.datetime) -> float:
    """Tổng giờ chạy; lần chạy bị kill (không có ended_at) tính tới lúc lần sau bắt đầu, lần cuối còn mở tính tới `now`."""
    ts = lambda v: dt.datetime.fromisoformat(v.replace("Z", "+00:00"))  # noqa: E731
    ends = [r.get("ended_at") or (runs[i + 1]["started_at"] if i + 1 < len(runs) else None) for i, r in enumerate(runs)]
    return sum(((ts(e) if e else now) - ts(r["started_at"])).total_seconds() for r, e in zip(runs, ends, strict=True)) / 3600


class Report:
    def __init__(self, layout, settings):
        self.layout, self._lock = layout, threading.Lock()
        self.steps = {name: {"name": name, "status": None, "started_at": None, "finished_at": None, "seconds": 0.0,
                             "items_done": 0, "items_total": 0, "note": None} for name in STEPS}
        self.counters = dict.fromkeys(COUNTERS, 0)
        self.errors: list[dict] = []
        self.qdrant = {"status": None, "indexed_vectors": 0, "points": 0}
        self.token_budget, self.avgdl = settings.embed_token_budget, None
        self.embed_chunks, self.embed_seconds = 0, 0.0
        self.ram_peak_gb, self.vram_peak_gb = 0.0, 0.0
        self._t0: dict[str, float] = {}

    def begin(self, step: str, total: int = 0) -> None:
        with self._lock:
            self.steps[step].update(started_at=now_iso(), items_total=total)
            self._t0[step] = time.monotonic()

    def progress(self, step: str, done: int) -> None:
        with self._lock:
            self.steps[step]["items_done"] = done

    def end(self, step: str, ok: bool = True, note: str | None = None) -> None:
        with self._lock:
            s = self.steps[step]
            s.update(status="done" if ok else "failed", finished_at=now_iso(), note=note,
                     seconds=round(time.monotonic() - self._t0.get(step, time.monotonic()), 1))

    def error(self, step: str, exc: BaseException | str, part: int | None = None, warn: bool = False) -> None:
        kind = "warn" if warn else (type(exc).__name__ if isinstance(exc, BaseException) else "error")
        with self._lock:
            self.errors.append({"at": now_iso(), "step": step, "part": part, "type": kind, "message": str(exc)})

    def sample_peaks(self) -> None:
        rss = psutil.Process().memory_info().rss / 1e9
        try:
            import torch

            vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        except ImportError:
            vram = 0.0
        with self._lock:
            self.ram_peak_gb, self.vram_peak_gb = max(self.ram_peak_gb, rss), max(self.vram_peak_gb, vram)

    def write(self, state: dict, status: str) -> dict:
        now = dt.datetime.now(dt.UTC)
        runs = state.get("runs", [])
        report = {
            "schema_version": 1, "generated_at": now_iso(), "git_sha": git_sha(), "status": status,
            "model_slug": EMBED_MODEL_SLUG, "collection": state["collection"],
            "started_at": runs[0]["started_at"] if runs else None, "finished_at": now_iso(),
            "wall_hours": round(_wall_hours(runs, now), 3), "resumed_runs": max(len(runs) - 1, 0),
            "steps": [s for s in self.steps.values() if s["status"] is not None],
            "counters": self.counters,
            "embed": {"token_budget": self.token_budget,
                      "chunks_per_s_avg": round(self.embed_chunks / self.embed_seconds, 1) if self.embed_seconds else None},
            "bm25": {"avgdl": state.get("bm25_avgdl")},
            "peaks": {"ram_used_gb": round(self.ram_peak_gb, 2), "vram_peak_gb": round(self.vram_peak_gb, 2)},
            "qdrant": self.qdrant, "errors": self.errors,
        }
        if not self.embed_seconds:  # lần chạy này không embed (resume sau embed): giữ số đo của report trước, không ghi 0 giả
            previous = _read(self.layout.report_path) or {}
            report["embed"] = previous.get("embed") or report["embed"]
            report["peaks"] = previous.get("peaks") or {"ram_used_gb": None, "vram_peak_gb": None}
        self.layout.report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.layout.report_path, report)
        return report


def truncated_listings(layout, n_parts: int, cap: int) -> int:
    """Số tin chạm trần `max_chunks_per_listing` (có chunk_index = cap − 1), đếm lại từ `chunks/` (không cần state phụ)."""
    total = 0
    for part in range(n_parts):
        path = layout.chunks(part)
        if path.exists():
            index = pq.read_table(path, columns=["chunk_index"]).column("chunk_index")
            total += int(pc.sum(pc.equal(index, cap - 1)).as_py() or 0)
    return total


def fill_counters(job) -> None:
    """Bộ đếm §7.5 lấy từ state (qua mọi lần resume) + encoder của lần chạy này."""
    c, ledger = job.report.counters, job.ledger
    c.update(parts_total=job.n_parts, parts_embedded=ledger.count("embedded"), parts_uploaded=ledger.count("uploaded"),
             chunks=ledger.n_chunks(), upload_retries=job.upload_counters["upload_retries"],
             chunks_truncated_listings=truncated_listings(job.layout, job.n_parts, job.settings.max_chunks_per_listing))
    if job.encoder is not None:
        c.update(oom_retries=job.encoder.oom_retries, batch_size_final=job.encoder.last_batch_size)
        job.report.token_budget = job.encoder.budget


def finish(job, status: str) -> dict:
    """Cuối mỗi lần chạy (cả khi lỗi): bộ đếm, đóng run trong state, ghi report."""
    try:
        if job.ledger is not None:
            fill_counters(job)
    except Exception as exc:
        job.report.error("report", exc, warn=True)
    if job.ledger is not None and job.ledger.running:
        job.ledger.end_run(status == "completed")
    job.report.begin("report")
    job.report.end("report")
    state = job.ledger.state if job.ledger is not None else {"collection": job.layout.collection, "runs": []}
    return job.report.write(state, status)


def _read(path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def status(settings, dry_run: bool) -> dict:
    """Trạng thái hiện tại từ state + report + Qdrant (không ETA)."""
    layout = layout_for(settings, dry_run)
    state, report = _read(layout.state_path), _read(layout.report_path)
    if state is None:
        return {"status": "not_started", "state_path": str(layout.state_path)}
    parts = state["parts"].values()
    runs = state.get("runs", [])
    running = bool(runs) and runs[-1].get("ended_at") is None
    out = {"status": "running" if running else (report or {}).get("status", "unknown"),
           "collection": state["collection"], "parts_total": state["n_parts"],
           **{f"parts_{s}": sum(bool(p.get(s)) for p in parts) for s in ("chunked", "bm25", "embedded", "uploaded")},
           "n_chunks_known": sum(int(p.get("n_chunks", 0)) for p in parts), "bm25_avgdl": state.get("bm25_avgdl"),
           **{k: state.get(k) for k in ("hnsw_green", "alias_set", "pca_done")}, "runs": len(runs),
           "peaks": (report or {}).get("peaks"), "errors": len((report or {}).get("errors", []))}
    try:
        from vre.index import qdrant_store

        out["qdrant"] = qdrant_store.info(qdrant_store.client(settings, timeout=10), state["collection"])
        out["points_in_qdrant"] = out["qdrant"]["points"]
    except Exception as exc:  # status chỉ để xem: Qdrant tắt thì báo, không làm hỏng lệnh
        out["qdrant"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out
