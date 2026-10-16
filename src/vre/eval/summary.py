"""`results/eval_summary.json` (A3.6, data-contract §7.7): độ chính xác theo nhóm × hệ thống có CI bootstrap, truy xuất,
faithfulness, chi phí/độ trễ, so sánh cặp (ghép cặp, mọi nhóm + tổng), router.

Chỉ đọc `results/eval/raw/{answers,scores}.jsonl`. Dòng chưa chấm được (`points=None`) bị loại khỏi mọi trung bình
và đếm ở `n_unscored`; so sánh cặp chỉ dùng câu cả hai hệ thống đều có điểm.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from statistics import mean
from typing import Any

from vre.common import result_header, write_json_atomic
from vre.config import Settings
from vre.eval import bootstrap
from vre.eval.metrics import percentile
from vre.eval.questions import ALLOCATION, Question

SCHEMA_VERSION = 1
SYSTEM_ORDER = ("llm_only", "basic_rag", "graph", "hybrid")
PAIRS = (("hybrid", "basic_rag"), ("basic_rag", "llm_only"), ("hybrid", "graph"))
PAIR_METRICS = ("accuracy", "mean_points")
ROUTES = ("basic_rag", "graph", "hybrid", "valuation")
RAGAS_NOTE = "ragas: answer_relevancy và context_precision bỏ để tiết kiệm lời gọi LLM (chỉ đo faithfulness)"
RAGAS_OFF_NOTE = "ragas: không chạy (chấm theo luật, không LLM giám khảo — chọn để tiết kiệm chi phí); faithfulness để trống"


def _round(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)


def _mean(values: Iterable[Any]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return _round(mean(clean)) if clean else None


def _metric_value(score: dict, metric: str) -> float:
    return float(score["points"] == 2) if metric == "accuracy" else float(score["points"])


def _cell(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("points") is not None]
    acc = [_metric_value(r, "accuracy") for r in scored]
    return {"n": len(scored), "accuracy": _mean(acc), "mean_points": _mean(r["points"] for r in scored),
            "ci95": bootstrap.mean_ci(acc), "n_unscored": len(rows) - len(scored)}


def _pairwise(scores: dict[tuple[str, str], dict], questions: dict[str, Question], systems: list[str]) -> list[dict]:
    scopes = ["overall", *[c for c in ALLOCATION if any(q.category == c for q in questions.values())]]
    out = []
    for a, b in PAIRS:
        if a not in systems or b not in systems:
            continue
        for scope in scopes:
            ids = sorted(i for i, q in questions.items() if scope in ("overall", q.category)
                         and scores.get((i, a), {}).get("points") is not None
                         and scores.get((i, b), {}).get("points") is not None)
            for metric in PAIR_METRICS:
                result = bootstrap.paired([_metric_value(scores[(i, a)], metric) for i in ids],
                                          [_metric_value(scores[(i, b)], metric) for i in ids])
                if result is not None:
                    out.append({"a": a, "b": b, "scope": scope, "metric": metric, **result})
    return out


def _router(answers: dict[tuple[str, str], dict], questions: dict[str, Question]) -> dict | None:
    """Router của hybrid so với `route_gold`. Báo cả không tính `unanswerable` (route_gold nhóm này là phán đoán)."""
    rows = [(questions[i], a.get("route")) for (i, s), a in answers.items() if s == "hybrid" and i in questions]
    if not rows:
        return None
    confusion: dict[str, dict[str, int]] = {g: {p: 0 for p in ROUTES} for g in ROUTES}
    for q, route in rows:
        confusion[q.route_gold][route or "none"] = confusion[q.route_gold].get(route or "none", 0) + 1

    def acc(subset: list[tuple[Question, Any]]) -> float | None:
        return _mean(float(route == q.route_gold) for q, route in subset) if subset else None

    answerable = [(q, r) for q, r in rows if q.category != "unanswerable"]
    by_category = {c: {"n": len(sub), "accuracy": acc(sub)} for c in ALLOCATION
                   if (sub := [(q, r) for q, r in rows if q.category == c])}
    return {"accuracy": acc(rows), "n": len(rows), "accuracy_excl_unanswerable": acc(answerable),
            "n_excl_unanswerable": len(answerable), "by_category": by_category, "confusion": confusion}


def _cost(rows: list[dict], price_in: float, price_out: float) -> dict:
    tokens_in = [int(r.get("input_tokens") or 0) for r in rows]
    tokens_out = [int(r.get("output_tokens") or 0) for r in rows]
    usd = [(i * price_in + o * price_out) / 1e6 for i, o in zip(tokens_in, tokens_out, strict=True)]
    return {"n": len(rows), "latency_ms_p50": _round(percentile([r.get("latency_ms") for r in rows], 50), 1),
            "latency_ms_p95": _round(percentile([r.get("latency_ms") for r in rows], 95), 1),
            "input_tokens_mean": _mean(tokens_in), "output_tokens_mean": _mean(tokens_out),
            "usd_per_question": _round(mean(usd), 6) if usd else None,
            "bad_citation_rate": _mean(float(bool(r.get("unsupported_citations"))) for r in rows),
            "error_rate": _mean(float(bool(r.get("error"))) for r in rows),
            "fallback_rate": _mean(float(r.get("fallback") is True) for r in rows),
            "cached_rate": _mean(float(bool(r.get("cached"))) for r in rows)}


def _scoring(scores: dict[tuple[str, str], dict]) -> str:
    kinds = {r.get("scoring", "llm_judge") for r in scores.values()}
    return kinds.pop() if len(kinds) == 1 else "mixed"


def build(settings: Settings, answers: dict[tuple[str, str], dict], scores: dict[tuple[str, str], dict],
          questions: dict[str, Question], llm: dict, index: dict | None) -> dict:
    systems = [s for s in SYSTEM_ORDER if any(k[1] == s for k in answers)]
    answered = sorted({i for i, _ in answers if i in questions})
    common = [i for i in answered if all(scores.get((i, s), {}).get("points") is not None for s in systems)]
    by_category: dict[str, dict] = {}
    coverage: dict[str, dict] = {}
    for category in ALLOCATION:  # mọi nhóm đều có mặt; nhóm chưa chạy ⇒ n=0 và chỉ số null (không phải 0.0)
        ids = [i for i in answered if questions[i].category == category]
        by_category[category] = {s: _cell([scores[(i, s)] for i in ids if (i, s) in scores]) for s in systems}
        coverage[category] = {"n_benchmark": sum(q.category == category for q in questions.values()),
                              "n_answered_any": len(ids), "n_answered_all_systems": sum(i in common for i in ids)}
    overall = {s: _cell([scores[(i, s)] for i in answered if (i, s) in scores]) for s in systems}
    overall_common = {s: _cell([scores[(i, s)] for i in common]) for s in systems}
    retrieval, ragas, cost = {}, {}, {}
    for s in systems:
        ids_rows = [r for (i, sys), r in scores.items() if sys == s and r.get("recall_at_5") is not None]
        if ids_rows and s != "llm_only":
            retrieval[s] = {"n": len(ids_rows), "top_k": settings.top_k, **{
                key: _mean(r.get(src) for r in ids_rows) for key, src in
                (("recall_at_5", "recall_at_5"), ("recall_at_10", "recall_at_10"), ("mrr", "rr"), ("ndcg_at_10", "ndcg_at_10"))}}
        faith = [r.get("faithfulness") for (i, sys), r in scores.items() if sys == s and r.get("faithfulness") is not None]
        if s != "llm_only":
            ragas[s] = {"faithfulness": _mean(faith), "answer_relevancy": None, "context_precision": None, "n": len(faith)}
        cost[s] = _cost([a for (i, sys), a in answers.items() if sys == s],
                        settings.llm_price_input or 0.0, settings.llm_price_output or 0.0)
    total = len(questions)
    scoring = _scoring(scores)
    ragas_ran = any(v.get("n") for v in ragas.values())
    notes = [RAGAS_NOTE if ragas_ran else RAGAS_OFF_NOTE,
             f"retrieval: nguồn = top_k={settings.top_k} tin mà hệ thống đưa vào ngữ cảnh; recall_at_10 vì thế ≤ "
             "recall_at_5 khi có hơn 5 cụm đúng (truy xuất thuần ở ablation.json)",
             f"benchmark đầy đủ có {total} câu; bản tóm tắt này chấm {len(answered)} câu (benchmark_subset)",
             "pairwise chỉ dùng câu cả hai hệ thống đều có điểm; overall_common chỉ dùng câu mọi hệ thống đều có điểm; "
             "mọi CI có n kèm theo"]
    doc: dict[str, Any] = {**result_header(SCHEMA_VERSION), "index": index, "llm": llm, "systems": systems,
                           "scoring": scoring, "n_questions": len(answered), "n_questions_scored": len(common),
                           "n_questions_total": total, "benchmark_subset": answered,
                           "complete": bool(answered) and len(common) == len(answered),
                           "coverage": coverage, "by_category": by_category, "overall": overall,
                           "overall_common": overall_common, "retrieval": retrieval, "ragas": ragas, "cost": cost,
                           "pairwise": _pairwise(scores, {i: questions[i] for i in answered}, systems),
                           "router": _router(answers, questions), "notes": notes}
    if scoring == "rule_based":
        from vre.eval.rule_score import RULES

        doc["scoring_rules"] = RULES
    return doc


def index_info(settings: Settings) -> dict | None:
    path = settings.index_dir / f"manifest_{settings.collection}.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {k: manifest.get(k) for k in ("collection", "model_slug", "n_points")}


def run(settings: Settings) -> dict:
    """`eval report` (phần tổng hợp): ghi `results/eval_summary.json`."""
    from vre.eval.judge import check_judge_config
    from vre.eval.questions import load_questions
    from vre.eval.runner import answers_path, read_jsonl, scores_path

    answers = read_jsonl(answers_path(settings))
    scores = read_jsonl(scores_path(settings))
    if not answers or not scores:
        raise FileNotFoundError("Chưa có answers.jsonl/scores.jsonl; chạy `eval run` rồi `eval score`")
    llm = check_judge_config(settings, judge_used=_scoring(scores) != "rule_based")
    meta_path = answers_path(settings).with_name("run_meta.json")
    if meta_path.is_file():  # model thật của lượt `eval run` (môi trường lúc report có thể khác)
        llm = {**llm, **json.loads(meta_path.read_text(encoding="utf-8")).get("llm", {})}
    doc = build(settings, answers, scores, {q.id: q for q in load_questions()}, llm, index_info(settings))
    write_json_atomic(settings.results_dir / "eval_summary.json", doc)
    return doc
