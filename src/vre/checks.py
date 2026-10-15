"""Kiểm tra fail-fast mức dự án (data-contract §9.2).

Mỗi kiểm tra = collector (đọc hệ thống → dict dữ kiện) + hàm thuần `check_*(facts) -> (ok, detail)`.
`require(settings, *names)` chạy các kiểm tra được nêu, in kết quả, raise `PreconditionError` nếu có fail.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vre import envguard
from vre.common import fingerprints_match, listings_fingerprint, read_handoff
from vre.config import EMBED_MODEL_SLUG, EMBED_MODELS, Settings

N_SHARDS = 10
Facts = dict[str, Any]
Verdict = tuple[bool, str]


class PreconditionError(RuntimeError):
    """Một hoặc nhiều kiểm tra fail-fast không đạt."""


# ---------------------------------------------------------------- hàm thuần trên dữ kiện
def check_raw_shards_present(f: Facts) -> Verdict:
    return (not f["missing"], "đủ 10 shard" if not f["missing"] else f"thiếu: {', '.join(f['missing'])}")


def check_e5_large_cached(f: Facts) -> Verdict:
    return (f["n_snapshots"] > 0, f"{f['repo_dir']} ({f['n_snapshots']} snapshot)")


def check_gpu(f: Facts) -> Verdict:
    return (bool(f["cuda_available"]), f.get("device") or "torch.cuda.is_available() = False")


def check_qdrant_healthy(f: Facts) -> Verdict:
    return (f["status"] == 200, f"{f['url']}/readyz → {f['status']} {f.get('error') or ''}".strip())


def check_neo4j_healthy(f: Facts) -> Verdict:
    return (f["ok"], f"{f['uri']} RETURN 1 → {'ok' if f['ok'] else f.get('error')}")


def check_d_free_space(f: Facts) -> Verdict:
    ok = f["drive"].upper().startswith("D:") and f["free_gb"] >= f["min_gb"]
    return (ok, f"ổ {f['drive']} trống {f['free_gb']:.1f} GB (cần ≥ {f['min_gb']} GB trên D:)")


def check_groq_key_present(f: Facts) -> Verdict:
    return (f["present"], "GROQ_API_KEY có giá trị" if f["present"] else "thiếu GROQ_API_KEY trong .env/env")


def check_env_paths_on_d(f: Facts) -> Verdict:
    bad = f["violations"]
    return (not bad, "mọi cache/TMP trên D:" if not bad else "ngoài D: " + ", ".join(f"{k}={v}" for k, v in bad.items()))


def check_clean_data_ready(f: Facts) -> Verdict:
    if not f["exists"]:
        return (False, f"thiếu {f['path']}")
    if f["signal"] is None:
        return (False, "chưa có results/handoff/CLEAN_DATA_READY.json")
    ok = fingerprints_match(f["actual"], f["signal"])
    return (ok, f"fingerprint {f['actual']} {'khớp' if ok else '≠'} tín hiệu {f['signal']}")


# ---------------------------------------------------------------- collector (đọc hệ thống)
def collect_raw_shards_present(s: Settings) -> Facts:
    names = [f"shard_{i:04d}.parquet" for i in range(N_SHARDS)]
    return {"missing": [n for n in names if not (s.raw_dir / n).is_file()]}


def collect_e5_large_cached(s: Settings) -> Facts:
    hf_home = Path(os.environ.get("HF_HOME", s.cache_dir / "huggingface"))
    repo = EMBED_MODELS[EMBED_MODEL_SLUG][0].replace("/", "--")
    snapshots = hf_home / "hub" / f"models--{repo}" / "snapshots"
    n = sum(1 for p in snapshots.iterdir() if p.is_dir() and any(p.iterdir())) if snapshots.is_dir() else 0
    return {"repo_dir": str(snapshots.parent), "n_snapshots": n}


def collect_gpu(s: Settings) -> Facts:
    import torch

    available = torch.cuda.is_available()
    return {"cuda_available": available, "device": torch.cuda.get_device_name(0) if available else None}


def collect_qdrant_healthy(s: Settings) -> Facts:
    url = s.qdrant_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{url}/readyz", timeout=5) as resp:
            return {"url": url, "status": resp.status}
    except OSError as exc:
        return {"url": url, "status": None, "error": str(exc)}


def collect_neo4j_healthy(s: Settings) -> Facts:
    from neo4j import GraphDatabase

    try:
        with GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password)) as driver:
            driver.execute_query("RETURN 1")
        return {"uri": s.neo4j_uri, "ok": True}
    except Exception as exc:  # noqa: BLE001 — mọi lỗi kết nối đều là dữ kiện "không healthy"
        return {"uri": s.neo4j_uri, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def collect_d_free_space(s: Settings) -> Facts:
    usage = shutil.disk_usage(s.data_root)
    return {"drive": s.data_root.drive or str(s.data_root), "free_gb": usage.free / 1e9, "min_gb": s.min_free_disk_gb_start}


def collect_groq_key_present(s: Settings) -> Facts:
    return {"present": bool(s.groq_api_key.strip())}


def collect_env_paths_on_d(s: Settings) -> Facts:
    return {"violations": envguard.violations(os.environ)}


def collect_clean_data_ready(s: Settings) -> Facts:
    exists = s.clean_path.is_file()
    signal = read_handoff(s, "CLEAN_DATA_READY.json")
    return {"path": str(s.clean_path), "exists": exists,
            "actual": listings_fingerprint(s.clean_path) if exists and signal else None,
            "signal": signal["payload"]["fingerprint"] if signal else None}


CHECKS: dict[str, tuple[Callable[[Settings], Facts], Callable[[Facts], Verdict]]] = {
    name: (globals()[f"collect_{name}"], globals()[f"check_{name}"])
    for name in ("raw_shards_present", "e5_large_cached", "gpu", "qdrant_healthy", "neo4j_healthy",
                 "d_free_space", "groq_key_present", "env_paths_on_d", "clean_data_ready")
}


def run_checks(settings: Settings, *names: str) -> dict[str, Verdict]:
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        raise KeyError(f"Kiểm tra không tồn tại: {unknown}; có: {list(CHECKS)}")
    results: dict[str, Verdict] = {}
    for name in names:
        collect, check = CHECKS[name]
        results[name] = check(collect(settings))
    return results


def require(settings: Settings, *names: str, echo: Callable[[str], None] = print) -> dict[str, Verdict]:
    results = run_checks(settings, *names)
    for name, (ok, detail) in results.items():
        echo(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    failed = [n for n, (ok, _) in results.items() if not ok]
    if failed:
        raise PreconditionError(f"Kiểm tra không đạt: {', '.join(failed)}")
    return results
