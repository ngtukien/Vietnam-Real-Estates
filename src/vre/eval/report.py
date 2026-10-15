"""A5: `results/examples.json` (data-contract §7.8) và khối kết quả trong README (giữa hai dấu `<!-- results:… -->`).

Ví dụ lấy từ lần chạy eval thật, không viết tay:
- `rag` / `graph`: chạy lại `vre.qa.stream_answer` trên câu benchmark đã được chấm đúng (LLM trúng cache của lượt eval;
  truy xuất/Cypher chạy thật) để có bộ lọc, điểm và hàng kết quả mà answers.jsonl không lưu.
- `valuation`: `vre.valuation.predict.valuate(valuation_input)` của câu định giá trong benchmark.
- `answers`: dòng answers.jsonl đạt 2 điểm, xen kẽ hệ thống.
README: mọi con số sinh từ `results/*.json`; thiếu dấu mốc ⇒ lỗi (không ghi đè README).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from vre.common import CODE_ROOT, result_header, write_json_atomic
from vre.config import Settings

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_EXAMPLES = 5
MIN_EXAMPLES = 3
START, END = "<!-- results:start -->", "<!-- results:end -->"
README_PATH = CODE_ROOT / "README.md"


def _load(settings: Settings) -> tuple[dict, dict, dict]:
    from vre.eval.questions import load_questions
    from vre.eval.runner import answers_path, read_jsonl, scores_path

    return ({q.id: q for q in load_questions()}, read_jsonl(answers_path(settings)), read_jsonl(scores_path(settings)))


def _pick(scores: dict, questions: dict, system: str, predicate: Any, n: int = MAX_EXAMPLES) -> list[str]:
    """Id câu theo thứ tự: đạt 2 điểm trước, rồi còn lại (vẫn là câu thật đã chạy) — tối đa n."""
    ids = sorted({i for (i, s) in scores if s == system and predicate(questions[i])})
    good = [i for i in ids if scores[(i, system)].get("points") == 2]
    return (good + [i for i in ids if i not in good])[:n]


def _events(system: str, question: str) -> dict[str, dict]:
    from vre.qa import stream_answer

    events = stream_answer(system, question, stream_tokens=False)  # type: ignore[arg-type]  # system ∈ SYSTEMS
    return {name: payload for name, payload in events}


def rag_examples(questions: dict, scores: dict) -> list[dict]:
    out = []
    for qid in _pick(scores, questions, "basic_rag", lambda q: q.gt_kind == "ids"):
        events = _events("basic_rag", questions[qid].question)
        if "error" in events:
            continue
        hits = [{"listing_id": c["listing_id"], "title": c.get("title"), "score": c.get("score"),
                 "province": c["address"].get("province"), "district": c["address"].get("district"),
                 "price_vnd": c.get("price_vnd")} for c in events.get("sources", {}).get("sources", [])]
        out.append({"question": questions[qid].question, "filters": events.get("filters", {}).get("filters", {}),
                    "hits": hits})
    return out


def graph_examples(questions: dict, scores: dict) -> list[dict]:
    out = []
    for qid in _pick(scores, questions, "graph", lambda q: q.route_gold == "graph" and q.category != "unanswerable"):
        events = _events("graph", questions[qid].question)
        cypher = events.get("cypher")
        if not cypher or cypher.get("error"):
            continue
        out.append({"question": questions[qid].question, "cypher": cypher["cypher"], "columns": cypher["columns"],
                    "rows": cypher["rows"], "latency_ms": events.get("usage", {}).get("latency_ms")})
    return out


def valuation_examples(questions: dict) -> list[dict]:
    from vre.valuation.predict import valuate

    out = []
    for q in [q for q in questions.values() if q.category == "valuation"][:MAX_EXAMPLES]:
        try:
            out.append({"input": q.valuation_input, "result": valuate(dict(q.valuation_input or {}))})
        except (ValueError, RuntimeError) as error:
            log.warning("định giá %s lỗi: %s", q.id, error)
    return out


def answer_examples(questions: dict, answers: dict, scores: dict) -> list[dict]:
    from vre.rag.citations import cited_ids

    good = sorted(k for k, s in scores.items() if s.get("points") == 2 and k in answers and answers[k].get("answer"))
    picked: list[tuple[str, str]] = []
    for system in ("hybrid", "basic_rag", "graph", "llm_only", "hybrid", "basic_rag", "graph", "hybrid"):
        key = next((k for k in good if k[1] == system and k not in picked and k[0] not in {p[0] for p in picked}), None)
        if key:
            picked.append(key)
    return [{"id": i, "system": s, "question": questions[i].question, "answer": answers[(i, s)]["answer"],
             "route": answers[(i, s)].get("route"), "cited": cited_ids(answers[(i, s)]["answer"])} for i, s in picked]


def examples(settings: Settings) -> dict:
    """`report examples` → `results/examples.json`. Mảng có < 3 phần tử ⇒ cảnh báo trong `notes` (không bịa thêm)."""
    questions, answers, scores = _load(settings)
    if not scores:
        raise FileNotFoundError("Chưa có scores.jsonl; chạy `eval run` + `eval score` trước")
    doc: dict[str, Any] = {**result_header(SCHEMA_VERSION), "rag": rag_examples(questions, scores),
                           "graph": graph_examples(questions, scores), "valuation": valuation_examples(questions),
                           "answers": answer_examples(questions, answers, scores)}
    short = [k for k in ("rag", "graph", "valuation", "answers") if len(doc[k]) < MIN_EXAMPLES]
    if short:
        doc["notes"] = [f"ít hơn {MIN_EXAMPLES} ví dụ ở: {', '.join(short)} (lượt eval chưa đủ câu)"]
    write_json_atomic(settings.results_dir / "examples.json", doc)
    return doc


# ---- README

def _fmt(x: Any, pct: bool = False, nd: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%" if pct else f"{x:.{nd}f}"


def _ci(ci: list | None) -> str:
    return "—" if not ci else f"[{ci[0] * 100:.0f}–{ci[1] * 100:.0f}%]"


def _read(settings: Settings, name: str) -> dict | None:
    path = settings.results_dir / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def results_block(settings: Settings) -> str:
    """Markdown sinh từ results/*.json (không số gõ tay)."""
    lines: list[str] = []
    summary = _read(settings, "eval_summary.json")
    if summary:
        systems = summary["systems"]
        lines += [f"### Benchmark ({summary['n_questions']}/{summary.get('n_questions_total', summary['n_questions'])} câu, "
                  f"sinh `{summary['llm'].get('generator')}`, chấm `{summary['llm'].get('judge')}`, "
                  f"`{summary['git_sha']}`)", "",
                  "| Nhóm | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
        rows = {**summary["by_category"], "**Tổng**": summary["overall"]}
        for name, cells in rows.items():
            lines.append(f"| {name} | " + " | ".join(
                f"{_fmt(cells[s]['accuracy'], pct=True)} {_ci(cells[s]['ci95'])}" if s in cells else "—"
                for s in systems) + " |")
        lines += ["", "Độ chính xác (2/2 điểm), CI 95% bootstrap 10.000 mẫu theo câu hỏi, seed 42.", ""]
        pairs = [p for p in summary.get("pairwise", []) if p["metric"] == "accuracy"
                 and p["scope"] in ("overall", "aggregate", "multihop")]
        if pairs:
            lines += ["| So sánh | Phạm vi | Chênh lệch | CI 95% | p | n |", "|---|---|---|---|---|---|"]
            lines += [f"| {p['a']} − {p['b']} | {p['scope']} | {p['diff'] * 100:+.1f} điểm % | "
                      f"[{p['ci95'][0] * 100:+.1f}, {p['ci95'][1] * 100:+.1f}] | {p['p_value']:.3f} | {p['n']} |"
                      for p in pairs]
            lines.append("")
        cost = summary.get("cost", {})
        if cost:
            lines += ["| Hệ thống | Độ trễ p50 (ms) | p95 (ms) | Token vào/ra TB | USD/câu | Trích dẫn sai |",
                      "|---|---|---|---|---|---|"]
            lines += [f"| {s} | {_fmt(c['latency_ms_p50'], nd=0)} | {_fmt(c['latency_ms_p95'], nd=0)} | "
                      f"{_fmt(c['input_tokens_mean'], nd=0)}/{_fmt(c['output_tokens_mean'], nd=0)} | "
                      f"{_fmt(c['usd_per_question'], nd=5)} | {_fmt(c['bad_citation_rate'], pct=True)} |"
                      for s, c in cost.items()]
            lines.append("")
        router = summary.get("router")
        if router:
            lines += [f"Router (hybrid): {_fmt(router['accuracy'], pct=True)} trên {router['n']} câu; "
                      f"{_fmt(router['accuracy_excl_unanswerable'], pct=True)} khi bỏ nhóm unanswerable.", ""]
        lines += [f"> {n}" for n in summary.get("notes", [])] + [""]
    valuation = _read(settings, "valuation_metrics.json")
    if valuation:
        lines += ["### Định giá (tập kiểm tra)", "", "| Biến thể | MAE (tr/m²) | MAPE | R² |", "|---|---|---|---|"]
        lines += [f"| {name} | {_fmt(v.get('mae_mil_m2'))} | {_fmt(v.get('mape'), pct=True)} | {_fmt(v.get('r2'))} |"
                  for name, v in valuation["variants"].items()]
        lines.append("")
    ablation = _read(settings, "ablation.json")
    if ablation and ablation.get("runs"):
        lines += ["### Ablation truy xuất (e5-large)", "", "| Run | Recall@5 | MRR | nDCG@10 | n |", "|---|---|---|---|---|"]
        lines += [f"| {r['name']} | {_fmt(r.get('recall_at_5'))} | {_fmt(r.get('mrr'))} | {_fmt(r.get('ndcg_at_10'))} | "
                  f"{r.get('n_questions')} |" for r in ablation["runs"]]
        lines.append("")
    if not lines:
        lines = ["_Chưa có kết quả trong `results/`._", ""]
    return "\n".join(lines).rstrip() + "\n"


def replace_block(text: str, block: str) -> str:
    """Thay ĐÚNG phần giữa hai dấu mốc; thiếu/sai thứ tự ⇒ ValueError (không đoán chỗ chèn)."""
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if text.count(START) != 1 or text.count(END) != 1 or not pattern.search(text):
        raise ValueError(f"README cần đúng một cặp {START} … {END}")
    return pattern.sub(lambda _m: f"{START}\n{block}{END}", text)


def readme(settings: Settings, path: Path | None = None) -> dict:
    """`report readme`: ghi lại khối kết quả trong README (giữ nguyên phần còn lại, giữ kiểu xuống dòng)."""
    path = path or README_PATH
    raw = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    block = results_block(settings)
    updated = replace_block(raw.replace("\r\n", "\n"), block).replace("\n", newline)
    changed = updated != raw
    if changed:
        path.write_bytes(updated.encode("utf-8"))
    return {"path": str(path), "changed": changed, "lines": block.count("\n")}
