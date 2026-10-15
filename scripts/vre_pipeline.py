"""CLI của pipeline VRE (data-contract §9.3). Chỉ nối lệnh → hàm; logic nằm trong package `vre`.

    . D:\\Vietnam-Real-Estates\\scripts\\vre_env.ps1
    & $VRE_PY scripts\\vre_pipeline.py --help

Mỗi lệnh chạy `vre.checks.require(...)` rồi import LƯỜI hàm đích (lệnh của p2 chưa có code ⇒ ImportError rõ ràng).
"""

import os
import sys
from pathlib import Path

# OpenBLAS cấp bộ đệm theo số luồng (32 lõi ⇒ ~0,8 GB commit ngay khi import numpy); 4 luồng đủ cho pipeline.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vre import envguard  # noqa: E402

envguard.apply()

import json  # noqa: E402
from enum import StrEnum  # noqa: E402
from typing import Any  # noqa: E402

import typer  # noqa: E402

from vre.checks import require  # noqa: E402
from vre.config import get_settings  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Pipeline VRE: ingest → eda → graph → price → night.")

INGEST_CHECKS = ("raw_shards_present", "env_paths_on_d", "d_free_space", "groq_key_present")
EDA_CHECKS = ("clean_data_ready",)
GRAPH_CHECKS = ("neo4j_healthy", "clean_data_ready")
PRICE_CHECKS = ("clean_data_ready",)
NIGHT_CHECKS = ("e5_large_cached", "gpu", "qdrant_healthy", "d_free_space", "env_paths_on_d", "clean_data_ready")
SEARCH_CHECKS = ("qdrant_healthy",)


class Variant(StrEnum):
    tabular = "tabular"
    full = "full"


class SystemChoice(StrEnum):
    llm_only = "llm_only"
    basic_rag = "basic_rag"
    graph = "graph"
    hybrid = "hybrid"


ASK_CHECKS = {"llm_only": ("groq_key_present",), "basic_rag": ("groq_key_present", "qdrant_healthy"),
              "graph": ("groq_key_present", "neo4j_healthy"),
              "hybrid": ("groq_key_present", "qdrant_healthy", "neo4j_healthy")}
GT_CHECKS = ("clean_data_ready",)
eval_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Benchmark: đáp án chuẩn, chạy, chấm.")
app.add_typer(eval_app, name="eval")


class Sender(StrEnum):
    p1 = "p1"
    p2 = "p2"
    p3 = "p3"
    p4 = "p4"


def _echo(result: Any) -> None:
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, default=str))


@app.command()
def ingest() -> None:
    """Shard thô → bảng sạch (dedup, chuẩn hoá) + ingest_report + embedding_benchmark."""
    settings = get_settings()
    require(settings, *INGEST_CHECKS)
    from vre.data.ingest import run

    _echo(run(settings))


@app.command()
def eda() -> None:
    """EDA trên bảng sạch → results/eda_summary.json + results/eda/*.png."""
    settings = get_settings()
    require(settings, *EDA_CHECKS)
    from vre.data.eda import run

    _echo(run(settings))


@app.command()
def graph() -> None:
    """Xuất CSV + neo4j-admin import + post_import.cypher → results/graph_report.json."""
    settings = get_settings()
    require(settings, *GRAPH_CHECKS)
    from vre.index.graph_import import run

    _echo(run(settings))


@app.command()
def price(variant: Variant = typer.Option(..., "--variant", help="tabular (CPU) | full (GPU, cần PCA)")) -> None:
    """Huấn luyện model định giá XGBoost → results/valuation_metrics.json."""
    settings = get_settings()
    checks = PRICE_CHECKS + (("gpu",) if variant is Variant.full else ())
    require(settings, *checks)
    from vre.valuation.features import pca_path

    pca_file = pca_path(settings)
    if variant is Variant.full and not pca_file.is_file():
        raise typer.BadParameter(f"Thiếu file PCA {pca_file} (Phase 2 sinh)")
    from vre.valuation.train import run

    _echo(run(settings, variant.value))


@app.command()
def night(resume: bool = typer.Option(True, "--resume/--no-resume"),
          dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Lượt chạy dài chỉ mục e5-large (p2). Mã thoát 0 = xong, 20 = lỗi cần người."""
    settings = get_settings()
    require(settings, *NIGHT_CHECKS)
    from vre.pipeline.night import run

    raise typer.Exit(run(settings, resume=resume, dry_run=dry_run))


@app.command()
def status(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Trạng thái lượt chạy dài (p2)."""
    from vre.pipeline.report import status as report_status

    _echo(report_status(get_settings(), dry_run=dry_run))


@app.command("smoke-search")
def smoke_search(query: str, collection: str | None = typer.Option(None, "--collection"),
                 limit: int = typer.Option(10, "--limit")) -> None:
    """Truy vấn thử Qdrant (p2)."""
    settings = get_settings()
    require(settings, *SEARCH_CHECKS)
    from vre.index.qdrant_store import smoke_search as search

    _echo(search(settings, query, collection=collection, limit=limit))


@app.command()
def ask(question: str, system: SystemChoice = typer.Option(SystemChoice.hybrid, "--system"),
        top_k: int = typer.Option(5, "--top-k"), rerank: bool = typer.Option(False, "--rerank")) -> None:
    """Hỏi một câu qua `vre.qa` (in từng sự kiện SSE dạng JSON, không in token)."""
    settings = get_settings()
    if settings.llm_provider == "groq":
        require(settings, *ASK_CHECKS[system.value])
    import logging

    from vre.qa import stream_answer

    logging.basicConfig(format="[%(name)s] %(message)s")
    logging.getLogger("vre.graph.text2cypher").setLevel(logging.INFO)  # in độ trễ Cypher
    for name, payload in stream_answer(system.value, question, top_k=top_k, use_rerank=rerank, stream_tokens=False):
        typer.echo(f"event: {name}")
        _echo(payload)


@eval_app.command("gt")
def eval_gt() -> None:
    """Chạy SQL đáp án chuẩn → results/benchmark_gt.json (code của Lane A2: `vre.eval.ground_truth`)."""
    settings = get_settings()
    require(settings, *GT_CHECKS)
    from vre.eval.ground_truth import run

    _echo(run(settings))


EVAL_CHECKS = ("qdrant_healthy", "neo4j_healthy", "clean_data_ready")


def _llm_checks(settings: Any) -> tuple[str, ...]:
    """Kiểm khoá chỉ khi provider là groq (check có sẵn); provider khác: client báo lỗi rõ khi thiếu khoá."""
    return ("groq_key_present",) if settings.llm_provider == "groq" else ()
report_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Ví dụ + khối kết quả README từ results/.")
app.add_typer(report_app, name="report")


def _eval_logging() -> None:
    import logging

    logging.basicConfig(format="[%(name)s] %(levelname)s %(message)s", level=logging.WARNING)
    logging.getLogger("vre.eval").setLevel(logging.INFO)


def _systems(values: list[SystemChoice] | None) -> list[str] | None:
    return [v.value for v in values] if values else None


@eval_app.command("run")
def eval_run(limit: int | None = typer.Option(None, "--limit", help="chỉ N câu (xen kẽ theo nhóm) — lượt thử"),
             systems: list[SystemChoice] | None = typer.Option(None, "--systems", help="lặp lại để chọn nhiều"),
             per_category: int | None = typer.Option(None, "--per-category", help="chỉ N câu đầu mỗi nhóm"),
             workers: int = typer.Option(4, "--workers", min=1, max=8),
             rule_based: bool = typer.Option(False, "--rule-based", help="sẽ chấm theo luật: không kiểm B2"),
             retry_errors: bool = typer.Option(False, "--retry-errors", help="chạy lại cả dòng lỗi (không chỉ 429)")) -> None:
    """Câu benchmark × hệ thống qua vre.qa.answer → results/eval/raw/answers.jsonl (resume theo id, system)."""
    settings = get_settings()
    require(settings, *_llm_checks(settings), *EVAL_CHECKS)
    _eval_logging()
    from vre.eval.runner import run_answers

    _echo(run_answers(settings, limit=limit, systems=_systems(systems), per_category=per_category, workers=workers,
                      llm_judge=not rule_based, retry_errors=retry_errors))


@eval_app.command("score")
def eval_score(ragas: bool = typer.Option(True, "--ragas/--no-ragas", help="faithfulness (basic_rag/graph/hybrid)"),
               offline: bool = typer.Option(False, "--offline", help="chấm theo luật, không gọi LLM")) -> None:
    """Chấm answers.jsonl (judge 0–2 hoặc luật `--offline`, truy xuất theo cụm dup) → results/eval/raw/scores.jsonl."""
    settings = get_settings()
    require(settings, *(() if offline else _llm_checks(settings)), "clean_data_ready")
    _eval_logging()
    from vre.eval.runner import run_scores

    _echo(run_scores(settings, ragas=ragas, offline=offline))


@eval_app.command("report")
def eval_report() -> None:
    """eval_summary.json (CI bootstrap, so sánh cặp, router, chi phí) + error_analysis.json."""
    settings = get_settings()
    _eval_logging()
    from vre.eval import errors, summary

    doc = summary.run(settings)
    analysis = errors.run(settings)
    _echo({"eval_summary": {k: doc[k] for k in ("n_questions", "complete", "systems", "overall", "router", "notes")},
           "error_counts": analysis["counts"], "n_error_examples": len(analysis["examples"])})


@eval_app.command("all")
def eval_all(limit: int | None = typer.Option(None, "--limit"),
             systems: list[SystemChoice] | None = typer.Option(None, "--systems")) -> None:
    """eval run → eval score → eval report."""
    eval_run(limit=limit, systems=systems, per_category=None, workers=4, rule_based=False, retry_errors=False)
    eval_score(ragas=True, offline=False)
    eval_report()


@report_app.command("examples")
def report_examples() -> None:
    """results/examples.json từ lượt eval thật (data-contract §7.8)."""
    settings = get_settings()
    require(settings, *EVAL_CHECKS)  # không cần khoá LLM: ví dụ không gọi LLM
    _eval_logging()
    from vre.eval.report import examples

    doc = examples(settings)
    _echo({k: len(doc[k]) for k in ("rag", "graph", "valuation", "answers")} | {"notes": doc.get("notes")})


@report_app.command("readme")
def report_readme() -> None:
    """Thay khối giữa <!-- results:start --> và <!-- results:end --> trong README bằng bảng sinh từ results/*.json."""
    from vre.eval.report import readme

    _echo(readme(get_settings()))


@app.command()
def ablation(limit: int | None = typer.Option(None, "--limit", help="chỉ N câu")) -> None:
    """Ablation truy xuất e5-large → results/ablation.json (code của lane ablation: `vre.eval.ablation`)."""
    settings = get_settings()
    require(settings, "qdrant_healthy", "clean_data_ready")
    from vre.eval.ablation import run

    _echo(run(settings, limit=limit))


@app.command()
def signal(name: str, sender: Sender = typer.Option(..., "--from"),
           payload: str = typer.Option("{}", "--payload", help="JSON object")) -> None:
    """Ghi tín hiệu phối hợp results/handoff/<NAME>.json (coordination §3–§4)."""
    from vre.common import write_handoff

    body = json.loads(payload)
    if not isinstance(body, dict):
        raise typer.BadParameter("--payload phải là JSON object")
    typer.echo(str(write_handoff(get_settings(), name, sender.value, body)))


@app.command("config-get")
def config_get(key: str) -> None:
    """In giá trị một khoá Settings (hoặc property đường dẫn)."""
    settings = get_settings()
    if not hasattr(settings, key):
        raise typer.BadParameter(f"Không có khoá {key!r}")
    typer.echo(str(getattr(settings, key)))


@app.command("all")
def run_all(resume: bool = typer.Option(True, "--resume/--no-resume")) -> None:
    """ingest → eda → graph → price tabular → night → price full (mỗi bước idempotent)."""
    ingest()
    eda()
    graph()
    price(Variant.tabular)
    settings = get_settings()
    require(settings, *NIGHT_CHECKS)
    from vre.pipeline.night import run

    code = run(settings, resume=resume, dry_run=False)
    if code != 0:
        raise typer.Exit(code)
    price(Variant.full)


if __name__ == "__main__":
    app()
