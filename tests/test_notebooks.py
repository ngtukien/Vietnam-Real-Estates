"""Kiểm tra 5 notebook trình bày (Phase 3 Lane D, bug B11).

Kiểm tĩnh (chạy ở mọi job): đủ 5 file, chỉ import thư viện được phép, không magic/shell, không `[ĐIỀN SAU]`, không số gõ tay
trong ô markdown, đã lưu kèm output, không có output lỗi, không lộ số điện thoại/khoá API/đường dẫn máy trong output.
Kiểm chạy (marker `notebooks`): chạy bằng nbclient trên `results/` thật và trên bộ kết quả eval giả đúng hình data-contract
§7.7–§7.8 (để mọi nhánh "đã có dữ liệu" của notebook 02/03/04 được chạy trước khi eval thật xong), tổng < 2 phút.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import time
from pathlib import Path

import nbformat
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = REPO_ROOT / "notebooks"
NOTEBOOKS = ["01_eda.ipynb", "02_rag.ipynb", "03_knowledge_graph.ipynb", "04_graphrag_eval.ipynb", "05_valuation.ipynb"]
ALLOWED_IMPORTS = {"json", "pathlib", "os", "pandas", "matplotlib", "IPython"}
PLACEHOLDER = "[ĐIỀN SAU]"
TIME_BUDGET_S = 120
# Số điện thoại VN (10 số liền hoặc có dấu chấm/cách), khoá API phổ biến, đường dẫn tuyệt đối của máy.
SECRET_PATTERNS = {
    "phone": re.compile(r"(?<![\d.,])(?:\+84|0)(?:\d{9}|\d{2,3}[ .]\d{3}[ .]\d{3,4})(?![\d.,])"),
    "groq_key": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    "openai_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "local_path": re.compile(r"[A-Za-z]:[\\/](?:Users|Vietnam-Real-Estates)", re.IGNORECASE),
}


def _read(name: str) -> nbformat.NotebookNode:
    return nbformat.read(NB_DIR / name, as_version=4)


def _code_cells(nb: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [c for c in nb.cells if c.cell_type == "code"]


def _output_text(nb: nbformat.NotebookNode) -> str:
    """Toàn bộ output dạng chữ (bỏ ảnh base64)."""
    parts: list[str] = []
    for cell in _code_cells(nb):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                parts.append(out.get("text", ""))
            elif out.get("output_type") == "error":
                parts.append("\n".join(out.get("traceback", [])))
            for mime, value in out.get("data", {}).items():
                if mime.startswith("text/") or mime == "application/json":
                    parts.append(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
    return "\n".join(parts)


def _strip_code(markdown: str) -> str:
    """Bỏ khối ``` ``` và `inline` (tên file, lệnh, định danh không phải số kết quả)."""
    return re.sub(r"`[^`\n]*`", "", re.sub(r"```.*?```", "", markdown, flags=re.DOTALL))


def test_exactly_five_notebooks() -> None:
    assert sorted(p.name for p in NB_DIR.glob("*.ipynb")) == NOTEBOOKS


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_imports_are_allowed(name: str) -> None:
    for cell in _code_cells(_read(name)):
        src = cell.source
        assert not re.search(r"^\s*[!%]", src, re.MULTILINE), f"{name}: magic/shell không được phép:\n{src}"
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                roots = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                roots = {(node.module or "").split(".")[0]}
            else:
                continue
            assert roots <= ALLOWED_IMPORTS, f"{name}: import ngoài danh sách cho phép: {roots - ALLOWED_IMPORTS}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_placeholder(name: str) -> None:
    nb = _read(name)
    sources = "\n".join(c.source for c in nb.cells)
    assert PLACEHOLDER not in sources and PLACEHOLDER not in _output_text(nb)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_markdown_has_no_hand_typed_numbers(name: str) -> None:
    for cell in _read(name).cells:
        if cell.cell_type == "markdown":
            found = re.findall(r"(?<![\w.])\d+(?:[.,]\d+)*", _strip_code(cell.source))
            assert not found, f"{name}: số gõ tay trong markdown {found} — in từ JSON bằng ô code:\n{cell.source}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_saved_with_outputs_and_no_errors(name: str) -> None:
    cells = _code_cells(_read(name))
    assert cells and all(c.get("execution_count") for c in cells), f"{name}: chưa được chạy trước khi lưu"
    errors = [o for c in cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"{name}: output lỗi {[e.get('ename') for e in errors]}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_outputs_leak_nothing(name: str) -> None:
    text = _output_text(_read(name))
    for label, pattern in SECRET_PATTERNS.items():
        hits = pattern.findall(text)
        assert not hits, f"{name}: output chứa {label}: {hits[:3]}"


# ------------------------------------------------------------------------------------------------ chạy thật


def _fake_eval_results(results: Path) -> None:
    """Kết quả eval giả, đúng khoá data-contract §7.7–§7.8 (chỉ để chạy mọi nhánh của notebook)."""
    systems = ["llm_only", "basic_rag", "graph", "hybrid"]
    head = {"schema_version": 1, "generated_at": "2026-01-01T00:00:00Z", "git_sha": "test"}
    cats = ["lookup", "aggregate", "unanswerable"]
    metric = {"accuracy": 0.5, "mean_points": 1.0, "ci95": [0.4, 0.6]}
    eval_summary = {
        **head,
        "index": {"collection": "vre_chunks_e5_large", "model_slug": "e5_large", "n_points": 10},
        "llm": {"generator": "gen", "judge": "judge"},
        "systems": systems,
        "n_questions": 3,
        "by_category": {c: {s: {"n": 1, **metric} for s in systems} for c in cats},
        "overall": {s: dict(metric, accuracy=0.2 + 0.1 * i) for i, s in enumerate(systems)},
        "retrieval": {"basic_rag": {"recall_at_5": 0.5, "recall_at_10": 0.6, "mrr": 0.4, "ndcg_at_10": 0.5}},
        "ragas": {"basic_rag": {"faithfulness": 0.8, "answer_relevancy": 0.7, "context_precision": 0.6}},
        "cost": {s: {"latency_ms_p50": 100, "latency_ms_p95": 300, "input_tokens_mean": 50, "output_tokens_mean": 20,
                     "usd_per_question": 0.001, "bad_citation_rate": 0.0} for s in systems},
        "pairwise": [{"a": "hybrid", "b": "basic_rag", "scope": "aggregate", "metric": "accuracy", "diff": 0.1,
                      "ci95": [0.02, 0.2], "p_value": 0.01, "n_boot": 10000}],
        "router": {"accuracy": 0.75, "confusion": {"graph": {"graph": 1, "basic_rag": 0, "hybrid": 0, "valuation": 0},
                                                   "basic_rag": {"graph": 0, "basic_rag": 1, "hybrid": 1, "valuation": 0}}},
    }
    ablation = {**head, "corpus": {"n_listings": 100, "n_gt_listings": 10, "seed": 42},
                "runs": [{"name": n, "embedding": "e5_large", "chunk_size": 400, "retrieval": r, "prefilter": True, "rerank": False,
                          "recall_at_5": v, "recall_at_10": v, "mrr": v, "ndcg_at_10": v, "n_questions": 3, "latency_ms_p50": 10}
                         for n, r, v in [("dense", "dense", 0.4), ("hybrid", "hybrid", 0.6)]]}
    error_analysis = {**head, "counts": {"hybrid": {"retrieval_miss": 1}, "graph": {"cypher_error": 2}},
                      "examples": [{"id": "AG01", "system": "graph", "category": "aggregate", "cause": "cypher_error",
                                    "question": "q", "answer_excerpt": "a", "gt_value": "1", "cypher": "MATCH (n) RETURN n"}]}
    examples = {**head,
                "rag": [{"question": "q", "filters": {"province": "Hà Nội"},
                         "hits": [{"listing_id": 1, "title": "t", "score": 0.9, "province": "Hà Nội", "district": "d",
                                   "price_vnd": 1}]}],
                "graph": [{"question": "q", "cypher": "MATCH (p:Province) RETURN p.name AS name", "columns": ["name"],
                           "rows": [{"name": "Hà Nội"}], "latency_ms": 12}],
                "valuation": [{"input": {"area_m2": 50}, "result": {"price_m2_mil": 80}}],
                "answers": [{"id": "LK01", "system": "hybrid", "question": "q", "answer": "a [Tin#1]", "route": "hybrid",
                             "cited": [1]}]}
    for name, obj in [("eval_summary.json", eval_summary), ("ablation.json", ablation),
                      ("error_analysis.json", error_analysis), ("examples.json", examples)]:
        (results / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _execute(name: str, data_root: Path) -> nbformat.NotebookNode:
    from nbclient import NotebookClient

    nb = _read(name)
    old = os.environ.get("VRE_DATA_ROOT")
    os.environ["VRE_DATA_ROOT"] = str(data_root)  # kernel con thừa hưởng môi trường
    try:
        NotebookClient(nb, timeout=TIME_BUDGET_S, kernel_name="python3",
                       resources={"metadata": {"path": str(NB_DIR)}}).execute()
    finally:
        if old is None:
            os.environ.pop("VRE_DATA_ROOT", None)
        else:
            os.environ["VRE_DATA_ROOT"] = old
    return nb


@pytest.fixture(scope="module")
def data_roots(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    real = Path(os.environ.get("VRE_DATA_ROOT") or REPO_ROOT)
    full = tmp_path_factory.mktemp("nb_full")
    shutil.copytree(real / "results", full / "results", ignore=shutil.ignore_patterns("cache", "eval", "handoff", "*.parquet"))
    _fake_eval_results(full / "results")
    return {"real": real, "full": full}


@pytest.mark.notebooks
def test_notebooks_execute_within_budget(data_roots: dict[str, Path]) -> None:
    start = time.monotonic()
    for variant, root in data_roots.items():
        for name in NOTEBOOKS:
            nb = _execute(name, root)
            text = _output_text(nb)
            for label, pattern in SECRET_PATTERNS.items():
                assert not pattern.findall(text), f"{name} ({variant}): output chứa {label}"
            if variant == "full":
                assert "chưa có" not in text, f"{name}: còn nhánh 'chưa có' dù đủ kết quả"
    elapsed = time.monotonic() - start
    assert elapsed < TIME_BUDGET_S, f"chạy 5 notebook × 2 bộ kết quả mất {elapsed:.0f}s ≥ {TIME_BUDGET_S}s"
