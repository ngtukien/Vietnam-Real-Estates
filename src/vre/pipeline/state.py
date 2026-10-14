"""Checkpoint của lượt chạy dài — `night_state.json` (data-contract §4.1) và bố cục thư mục artifact.

Ghi nguyên tử sau MỖI part của MỖI stage (`vre.common.write_json_atomic`). Dry-run có bố cục riêng dưới `index/dryrun/`
(state, report, artifact) nên không bao giờ chạm checkpoint của lượt thật.
"""

import datetime as dt
import json
import threading
from dataclasses import dataclass
from pathlib import Path

from vre.common import fingerprints_match, write_json_atomic
from vre.config import EMBED_MODEL_SLUG

STAGES = ("chunked", "bm25", "embedded", "uploaded")
FLAGS = ("hnsw_enabled", "hnsw_green", "alias_set", "pca_done")


class StateError(RuntimeError):
    """Checkpoint không dùng được (fingerprint lệch, khác collection/model)."""


@dataclass(frozen=True)
class Layout:
    root: Path
    state_path: Path
    report_path: Path
    collection: str
    dry_run: bool

    def chunks(self, part: int) -> Path:
        return self.root / "chunks" / f"part_{part:04d}.parquet"

    def tok(self, part: int) -> Path:
        return self.root / "tok" / f"part_{part:04d}.npz"

    def bm25(self, part: int) -> Path:
        return self.root / "bm25" / f"part_{part:04d}.parquet"

    def emb(self, part: int) -> Path:
        return self.root / "emb" / EMBED_MODEL_SLUG / f"part_{part:04d}.npy"

    @property
    def manifest_path(self) -> Path:
        return self.root / f"manifest_{self.collection}.json"


def layout_for(settings, dry_run: bool) -> Layout:
    index = Path(settings.index_dir)
    if dry_run:
        root = index / "dryrun"
        return Layout(root, root / "night_state.json", root / "night_report.json", settings.dryrun_collection, True)
    return Layout(index, index / "state" / "night_state.json", Path(settings.results_dir) / "night_report.json",
                  settings.collection, False)


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_state(collection: str, n_parts: int, fingerprint: dict) -> dict:
    return {"schema_version": 1, "model_slug": EMBED_MODEL_SLUG, "collection": collection, "n_parts": n_parts,
            "listings_fingerprint": fingerprint, "parts": {}, "bm25_avgdl": None,
            **{flag: False for flag in FLAGS}, "runs": []}


def check_compatible(state: dict, collection: str, fingerprint: dict) -> None:
    """Không trộn chỉ mục của hai lần ingest hoặc hai collection/model khác nhau."""
    if state["model_slug"] != EMBED_MODEL_SLUG or state["collection"] != collection:
        raise StateError(f"checkpoint thuộc {state['model_slug']}/{state['collection']}, "
                         f"lượt này là {EMBED_MODEL_SLUG}/{collection}")
    if not fingerprints_match(state["listings_fingerprint"], fingerprint):
        raise StateError(f"fingerprint bảng sạch đã đổi: checkpoint {state['listings_fingerprint']} ≠ hiện tại "
                         f"{fingerprint}. Bảng sạch được ingest lại ⇒ chỉ mục cũ không dùng được; "
                         "xoá checkpoint (chạy --no-resume) để làm lại toàn bộ.")


class Ledger:
    """State dùng chung giữa các luồng: mọi thay đổi đi qua đây, dưới khoá, rồi ghi nguyên tử."""

    def __init__(self, path: Path, state: dict):
        self.path, self.state, self._lock = path, state, threading.Lock()

    @classmethod
    def open(cls, layout: Layout, n_parts: int, fingerprint: dict, resume: bool) -> "Ledger":
        if resume and layout.state_path.exists():
            state = json.loads(layout.state_path.read_text(encoding="utf-8"))
            check_compatible(state, layout.collection, fingerprint)
        else:
            state = new_state(layout.collection, n_parts, fingerprint)
        ledger = cls(layout.state_path, state)
        ledger.save()
        return ledger

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.path, self.state)

    def part(self, part: int) -> dict:
        with self._lock:
            return dict(self.state["parts"].get(f"{part:04d}", {}))

    def done(self, part: int, stage: str) -> bool:
        return bool(self.part(part).get(stage))

    def mark(self, part: int, stage: str, value: bool = True, **extra) -> None:
        if stage not in STAGES:
            raise ValueError(stage)
        with self._lock:
            entry = self.state["parts"].setdefault(f"{part:04d}", {s: False for s in STAGES})
            entry[stage] = value
            entry.update(extra)
            self.save()

    def set(self, **fields) -> None:
        with self._lock:
            self.state.update(fields)
            self.save()

    def count(self, stage: str) -> int:
        with self._lock:
            return sum(bool(p.get(stage)) for p in self.state["parts"].values())

    def n_chunks(self) -> int:
        with self._lock:
            return sum(int(p.get("n_chunks", 0)) for p in self.state["parts"].values())

    @property
    def running(self) -> bool:
        """Lần chạy cuối còn mở (đã `start_run`, chưa `end_run`)."""
        runs = self.state["runs"]
        return bool(runs) and runs[-1].get("ended_at") is None

    def start_run(self) -> None:
        self.set(runs=[*self.state["runs"], {"started_at": now_iso(), "ended_at": None, "exit": None}])

    def end_run(self, ok: bool) -> None:
        with self._lock:
            self.state["runs"][-1].update(ended_at=now_iso(), exit="ok" if ok else "error")
            self.save()
