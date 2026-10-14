"""Benchmark: đáp án chuẩn bằng pandas, chạy hệ thống, chấm điểm, bảng tổng hợp, ca lỗi
(chuyển nguyên từ `src/eval.py`)."""

from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import pandas as pd

from vre.config import get_settings
from vre.eval.judge import judge_answer
from vre.eval.metrics import recall_at_k, reciprocal_rank

RETRIEVAL_TYPES = ("lookup", "constraint")  # câu có tập tin đúng (gt_ids)
VALUE_TYPES = ("aggregate", "multihop")  # câu có giá trị đúng (gt_value)
UNANSWERABLE = "unanswerable"  # câu cố ý không có đáp án trong dữ liệu: đúng khi hệ thống từ chối
TYPES = RETRIEVAL_TYPES + VALUE_TYPES + (UNANSWERABLE,)


def _fmt(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}".replace(",", " ")
    return str(value)


def _name(key) -> str:
    return ", ".join(map(str, key)) if isinstance(key, tuple) else str(key)


def ground_truth(df: pd.DataFrame, rule: dict) -> tuple[list[int], str | None]:
    """rule (JSON trong cột gt_rule):
    - {"query": ...}: tập tin đúng = các listing_id thoả query (pandas, engine python).
    - thêm "agg" (count | mean | median | nunique) và "value": một giá trị trên tập đó.
    - thêm "groupby", "pick" (min | max | top5) và "min_count": nhóm thắng và giá trị của nhóm.
    - {"unanswerable": true}: không có đáp án trong dữ liệu."""
    if rule.get("unanswerable"):
        return [], None
    d = df.query(rule["query"], engine="python") if rule.get("query") else df
    agg, value = rule.get("agg"), rule.get("value")
    if not agg:
        return sorted(int(x) for x in d["listing_id"]), None
    if "groupby" not in rule:
        result = len(d) if agg == "count" else d[value].agg(agg)
        return [], _fmt(result)
    keys = rule["groupby"]
    d = d.dropna(subset=keys + ([value] if value else []))
    sizes = d.groupby(keys).size()
    series = sizes if agg == "count" else d.groupby(keys)[value].agg(agg)
    series = series[sizes >= rule.get("min_count", 1)]
    pick = rule.get("pick", "max")
    if pick == "top5":
        return [], "; ".join(f"{_name(k)} ({_fmt(v)})" for k, v in series.nlargest(5).items())
    key = series.idxmin() if pick == "min" else series.idxmax()
    return [], f"{_name(key)} ({_fmt(series[key])})"


def fill_ground_truth(questions: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    out = questions.copy()
    truths = [ground_truth(df, json.loads(rule)) for rule in out["gt_rule"]]
    out["gt_ids"] = [ids for ids, _ in truths]
    out["gt_value"] = [value for _, value in truths]
    return out


def run_benchmark(questions: pd.DataFrame, systems: dict) -> pd.DataFrame:
    """Chạy từng câu qua từng hệ thống; lỗi của một câu được ghi lại, không dừng cả lượt chạy."""
    rows = []
    for q in questions.itertuples():
        for name, ask in systems.items():
            start = perf_counter()
            try:
                result = ask(q.question)
            except Exception as exc:  # ghi lỗi để phân tích ca lỗi
                result = dict(answer="", sources=[], contexts=[], error=f"{type(exc).__name__}: {exc}",
                              latency=perf_counter() - start, input_tokens=0, output_tokens=0)
            rows.append({"id": q.id, "type": q.type, "route_gold": q.route, "system": name,
                         "question": q.question, "answer": result.get("answer", ""),
                         "sources": [int(s) for s in result.get("sources", [])],
                         "contexts": list(result.get("contexts", [])),
                         "route": result.get("route", name), "cypher": result.get("cypher"),
                         "unsupported_citations": list(result.get("unsupported_citations", [])),
                         "error": result.get("error"), "fallback": bool(result.get("fallback", False)),
                         "latency": result["latency"], "input_tokens": result["input_tokens"],
                         "output_tokens": result["output_tokens"]})
    return pd.DataFrame(rows)


def score(results: pd.DataFrame, questions: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    """Điểm 0–2 cho mọi câu; đúng (correct) = 2 điểm. `k=None`: dùng `Settings.top_k`.
    - Tra cứu/ràng buộc: Recall@k (capped), RR; 2 = Recall@k đủ 1,0, 1 = có ít nhất 1 tin đúng trong top-k.
    - Tổng hợp/đa chặng: LLM chấm so với gt_value (2 đúng, 1 đúng một phần, 0 sai).
    - Không có đáp án: LLM chấm việc từ chối (2 từ chối rõ, 1 từ chối nửa vời, 0 bịa)."""
    k = get_settings().top_k if k is None else k
    truth = questions.set_index("id")
    out = results.copy()
    recall, rr, points = [], [], []
    for r in out.itertuples():
        t = truth.loc[r.id]
        if r.type in RETRIEVAL_TYPES:
            relevant = set(t.gt_ids)
            rec = recall_at_k(list(r.sources), relevant, k, capped=True)
            recall.append(rec)
            rr.append(reciprocal_rank(list(r.sources)[:k], relevant))
            points.append(2 if rec >= 1 else int(rec > 0))
        else:
            recall.append(np.nan)
            rr.append(np.nan)
            gt = None if r.type == UNANSWERABLE else t.gt_value
            points.append(judge_answer(r.question, r.answer, gt)["score"] if r.answer else 0)
    return out.assign(recall_at_k=recall, rr=rr, points=points, correct=[p == 2 for p in points])


def accuracy_table(scored: pd.DataFrame, values: str = "correct") -> pd.DataFrame:
    """Tỷ lệ đúng (values="correct") hoặc điểm trung bình 0–2 (values="points") theo loại câu × hệ thống."""
    table = scored.pivot_table(index="type", columns="system", values=values, aggfunc="mean")
    return table.reindex([t for t in TYPES if t in table.index]).round(2)


def cost_table(scored: pd.DataFrame) -> pd.DataFrame:
    s = get_settings()
    t = scored.groupby("system").agg(latency_s=("latency", "mean"), input_tokens=("input_tokens", "mean"),
                                     output_tokens=("output_tokens", "mean"))
    if "unsupported_citations" in scored:  # tỷ lệ câu trả lời trích [Tin#ID] không có trong ngữ cảnh
        bad = scored["unsupported_citations"].map(len) > 0
        t["bad_citation_rate"] = bad.groupby(scored["system"]).mean()
    t["usd_per_question"] = (t["input_tokens"] * s.llm_price_input + t["output_tokens"] * s.llm_price_output) / 1e6
    return t.round({"latency_s": 2, "input_tokens": 0, "output_tokens": 0, "usd_per_question": 4,
                    "bad_citation_rate": 2})


def error_cases(scored: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Gán nguyên nhân cho câu sai: lỗi khi chạy, router nhầm, Cypher lỗi, Cypher sai kết quả hoặc truy xuất trượt.
    Bỏ baseline llm_only: nó sai câu tra cứu/ràng buộc theo thiết kế (không có nguồn), sẽ lấn hết các ví dụ."""
    wrong = scored[~scored["correct"] & (scored["system"] != "llm_only")].copy()

    def cause(r):
        if r.error and not r.answer:  # exception trong run_benchmark: hỏng hệ thống, không phải bịa
            return f"lỗi khi chạy: {str(r.error)[:80]}"
        if r.type == UNANSWERABLE:
            return "không từ chối: trả lời câu không có dữ liệu"
        if r.system == "hybrid" and r.route != r.route_gold:
            return f"router nhầm: {r.route} thay vì {r.route_gold}"
        if r.error:
            return f"Cypher lỗi: {str(r.error)[:80]}"
        if r.cypher:
            return "Cypher chạy được nhưng sai kết quả"
        return "truy xuất trượt hoặc ngữ cảnh không đủ để tính"

    wrong["nguyên nhân"] = [cause(r) for r in wrong.itertuples()]
    return wrong[["id", "type", "system", "question", "route", "nguyên nhân"]].head(n)
