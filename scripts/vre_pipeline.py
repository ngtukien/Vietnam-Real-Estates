"""CLI của pipeline VRE (data-contract §9.3). Chỉ nối lệnh → hàm; logic nằm trong package `vre`.

    . D:\\Vietnam-Real-Estates\\scripts\\vre_env.ps1
    & $VRE_PY scripts\\vre_pipeline.py --help

Mỗi lệnh chạy `vre.checks.require(...)` rồi import LƯỜI hàm đích (lệnh của p2 chưa có code ⇒ ImportError rõ ràng).
"""

import sys
from pathlib import Path

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
    from vre.qa import stream_answer

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
