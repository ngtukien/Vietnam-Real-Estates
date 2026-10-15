"""Kiểm tra 6 notebook trình bày theo bố cục bản cũ (phase-03-lanes §4c, bug B11).

Kiểm tĩnh (mọi job): đúng bộ file; mỗi ô code có ô markdown `### Px-yy · …` ngay trước, cùng mã, dòng đầu `# Px-yy · ẨN|CACHE|LIVE`;
notebook đã lưu khớp bản sinh lại từ `notebooks/build_notebooks.py` (heading điền từ JSON, không gõ tay); đoạn giải thích không có
số gõ tay; không `[ĐIỀN SAU]`; đã chạy, không có output lỗi; output và cache cell không lộ số điện thoại / khoá / đường dẫn máy.
Kiểm chạy (marker `notebooks`): chạy cả 6 notebook bằng nbclient ở chế độ offline (`VRE_NB_OFFLINE=1`: không Docker, GPU hay
LLM — mọi cell đọc cache), không lỗi, tổng < 2 phút.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import nbformat
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = REPO_ROOT / "notebooks"
NB_CACHE = REPO_ROOT / "results" / "nb_cache"
sys.path.insert(0, str(NB_DIR))
import build_notebooks as builder  # noqa: E402

NOTEBOOKS = list(builder.NOTEBOOKS)
PLACEHOLDER = "[ĐIỀN SAU]"
TIME_BUDGET_S = 120
CELL_ID = r"(?:P\d|\d\d)-\d\d"
CODE_HEAD = re.compile(rf"^# ({CELL_ID}) · (ẨN|CACHE|LIVE)$")
MD_HEAD = re.compile(rf"^### ({CELL_ID}) · \S.*\n\*\*(ẨN|CACHE|LIVE)\*\*\n\n\S", re.DOTALL)
SECRET_PATTERNS = {
    "phone": re.compile(r"(?<![\d.,])(?:\+84|0)(?:\d{9}|\d{2,3}[ .]\d{3}[ .]\d{3,4})(?![\d.,])"),
    "groq_key": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    "openai_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "local_path": re.compile(r"[A-Za-z]:[\\/]+(?:Users|Vietnam-Real-Estates)", re.IGNORECASE),
}
# Chuỗi chỉ có trong DỮ LIỆU MẪU tự đặt của cell P1-05 (minh hoạ mask_pii) — được phép xuất hiện ở cột "văn bản gốc".
DEMO_PII = {"0901 234 567"}


def _read(name: str) -> nbformat.NotebookNode:
    return nbformat.read(NB_DIR / name, as_version=4)


def _code(nb: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [c for c in nb.cells if c.cell_type == "code"]


def _output_text(nb: nbformat.NotebookNode) -> str:
    parts: list[str] = []
    for cell in _code(nb):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                parts.append(out.get("text", ""))
            elif out.get("output_type") == "error":
                parts.append("\n".join(out.get("traceback", [])))
            for mime, value in out.get("data", {}).items():
                if mime.startswith("text/"):
                    parts.append(value if isinstance(value, str) else "".join(value))
    return "\n".join(parts)


def _leaks(text: str) -> dict[str, list[str]]:
    found = {}
    for label, pattern in SECRET_PATTERNS.items():
        hits = [h for h in pattern.findall(text) if h not in DEMO_PII]
        if hits:
            found[label] = hits[:3]
    return found


def test_notebook_set_matches_builder() -> None:
    assert sorted(p.name for p in NB_DIR.glob("*.ipynb")) == sorted(NOTEBOOKS)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_code_cell_has_its_heading(name: str) -> None:
    cells = _read(name).cells
    for i, cell in enumerate(cells):
        if cell.cell_type != "code" or cell.source == builder.RUN_SETUP:
            continue
        head = CODE_HEAD.match(cell.source.splitlines()[0])
        assert head, f"{name} ô {i}: dòng đầu phải là '# Px-yy · ẨN|CACHE|LIVE':\n{cell.source[:80]}"
        prev = cells[i - 1]
        md = MD_HEAD.match(prev.source) if prev.cell_type == "markdown" else None
        assert md, f"{name} ô {i}: thiếu markdown '### {head.group(1)} · <kết luận>' + '**{head.group(2)}**' + giải thích ngay trước"
        assert md.groups() == head.groups(), f"{name} ô {i}: heading {md.groups()} không khớp ô code {head.groups()}"
    if name != "00_setup.ipynb":
        assert any(c.source == builder.RUN_SETUP for c in _code(_read(name))), f"{name}: thiếu {builder.RUN_SETUP}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_saved_notebook_equals_regenerated(name: str) -> None:
    """Heading điền từ JSON: notebook lệch bản sinh lại = số gõ tay hoặc notebook cũ hơn results/ (chạy build_notebooks.py)."""
    saved = builder.sources(_read(name))
    fresh = builder.sources(builder.render(name))
    diff = [(i, a[1][:120], b[1][:120]) for i, (a, b) in enumerate(zip(saved, fresh, strict=False)) if a != b]
    assert len(saved) == len(fresh) and not diff, (
        f"{name}: khác bản sinh lại ở {len(diff)} ô (đầu tiên: {diff[:1]}). Chạy `python notebooks/build_notebooks.py` "
        "trên cùng results/ rồi commit; không sửa tay notebook.")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_explanations_have_no_hand_typed_numbers(name: str) -> None:
    for cell in _read(name).cells:
        if cell.cell_type != "markdown":
            continue
        # dòng heading (`### Px-yy · …`, `# 0N · …`) do builder điền từ JSON; phần còn lại là giải thích viết tay
        body = cell.source.split("\n", 1)[1] if cell.source.startswith(("### ", "# ")) else cell.source
        body = re.sub(r"`[^`\n]*`", "", re.sub(r"```.*?```", "", body, flags=re.DOTALL))
        found = re.findall(r"(?<![\w.])\d+(?:[.,]\d+)*", body)
        assert not found, f"{name}: số gõ tay trong giải thích {found}:\n{cell.source[:200]}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_placeholder(name: str) -> None:
    nb = _read(name)
    assert PLACEHOLDER not in "\n".join(c.source for c in nb.cells) and PLACEHOLDER not in _output_text(nb)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_saved_with_outputs_and_no_errors(name: str) -> None:
    cells = _code(_read(name))
    assert all(c.get("execution_count") for c in cells), f"{name}: chưa được chạy trước khi lưu"
    errors = [o for c in cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"{name}: output lỗi {[e.get('ename') for e in errors]}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_outputs_leak_nothing(name: str) -> None:
    assert not _leaks(_output_text(_read(name))), f"{name}: output lộ {_leaks(_output_text(_read(name)))}"


def test_cell_caches_are_valid_and_clean() -> None:
    files = sorted(NB_CACHE.glob("*.json"))
    assert files, "chưa có cache cell nào trong results/nb_cache/"
    known = {s.id for _, steps in builder.NOTEBOOKS.values() for s in steps}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["_meta"]["cell"] == path.stem and path.stem in known, f"{path.name}: cache của cell không tồn tại"
        assert not _leaks(json.dumps(data, ensure_ascii=False)), f"{path.name}: lộ {_leaks(json.dumps(data, ensure_ascii=False))}"


@pytest.mark.notebooks
def test_notebooks_run_offline_within_budget() -> None:
    from nbclient import NotebookClient

    saved = {k: os.environ.get(k) for k in ("VRE_NB_OFFLINE", "VRE_NB_STRICT", "VRE_LOAD_DOTENV")}
    os.environ.update({"VRE_NB_OFFLINE": "1", "VRE_NB_STRICT": "1", "VRE_LOAD_DOTENV": "0"})  # kernel con thừa hưởng
    before = {p.name: p.read_bytes() for p in NB_CACHE.glob("*.json")}
    start = time.monotonic()
    try:
        for name in NOTEBOOKS:
            nb = _read(name)
            NotebookClient(nb, timeout=TIME_BUDGET_S, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}}).execute()
            assert not _leaks(_output_text(nb)), f"{name} (offline): output lộ {_leaks(_output_text(nb))}"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    elapsed = time.monotonic() - start
    assert {p.name: p.read_bytes() for p in NB_CACHE.glob("*.json")} == before, "chế độ offline không được ghi cache cell"
    assert elapsed < TIME_BUDGET_S, f"chạy offline 6 notebook mất {elapsed:.0f}s ≥ {TIME_BUDGET_S}s"
