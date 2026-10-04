"""Đánh giá (Notebook 4): Recall@k, MRR, Faithfulness, đáp án chuẩn bằng pandas và chạy benchmark."""

import json
from time import perf_counter

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.config import LLM_PRICE_PER_MTOK, TOP_K
from src.llm import ask_llm

RETRIEVAL_TYPES = ("lookup", "constraint")  # câu có tập tin đúng (gt_ids)
VALUE_TYPES = ("aggregate", "multihop")  # câu có giá trị đúng (gt_value)

# ---------------------------------------------------------------- chỉ số (Phụ lục A.6)


def recall_at_k(retrieved, relevant, k, capped: bool = False):
    """Tỷ lệ tin đúng nằm trong top-k. capped=True chia cho min(|relevant|, k): khi có hơn k tin đúng,
    lấy đủ k tin đúng vẫn được 1,0 (dùng cho câu ràng buộc có hàng chục tin thoả điều kiện)."""
    if not relevant:
        return np.nan
    denominator = min(len(relevant), k) if capped else len(relevant)
    return len(set(retrieved[:k]) & set(relevant)) / denominator


def reciprocal_rank(retrieved, relevant):
    for pos, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1 / pos
    return 0.0


def mrr(runs):
    return sum(reciprocal_rank(r, rel) for r, rel in runs) / len(runs)


def faithfulness(claim_supported):
    return sum(claim_supported) / len(claim_supported) if claim_supported else np.nan


# ---------------------------------------------------------------- đáp án chuẩn (P4-02)


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
    - thêm "groupby", "pick" (min | max | top5) và "min_count": nhóm thắng và giá trị của nhóm."""
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


# ---------------------------------------------------------------- chạy benchmark (P4-10)


def run_benchmark(questions: pd.DataFrame, systems: dict) -> pd.DataFrame:
    """Chạy từng câu qua từng hệ thống; lỗi của một câu được ghi lại, không dừng cả lượt chạy."""
    rows = []
    for q in questions.itertuples():
        for name, ask in systems.items():
            start = perf_counter()
            try:
                result = ask(q.question)
            except Exception as exc:  # ghi lỗi để phân tích ở P4-13
                result = dict(answer="", sources=[], contexts=[], error=f"{type(exc).__name__}: {exc}",
                              latency=perf_counter() - start, input_tokens=0, output_tokens=0)
            rows.append({"id": q.id, "type": q.type, "route_gold": q.route, "system": name,
                         "question": q.question, "answer": result.get("answer", ""),
                         "sources": [int(s) for s in result.get("sources", [])],
                         "contexts": list(result.get("contexts", [])),
                         "route": result.get("route", name), "cypher": result.get("cypher"),
                         "error": result.get("error"), "fallback": bool(result.get("fallback", False)),
                         "latency": result["latency"], "input_tokens": result["input_tokens"],
                         "output_tokens": result["output_tokens"]})
    return pd.DataFrame(rows)


class Verdict(BaseModel):
    correct: bool
    reason: str


JUDGE_SYSTEM = """Bạn chấm câu trả lời theo ĐÁP ÁN CHUẨN. Đúng khi câu trả lời nêu cùng thực thể/giá trị với
đáp án chuẩn (cho phép làm tròn, sai số tương đối tối đa 5%, cách viết khác nhau). Sai khi nêu thực thể khác,
giá trị lệch hơn 5%, hoặc nói không đủ dữ liệu."""


def judge_answer(question: str, answer: str, gt_value: str) -> dict:
    result = ask_llm(f"CÂU HỎI: {question}\nĐÁP ÁN CHUẨN: {gt_value}\nCÂU TRẢ LỜI: {answer}",
                     system=JUDGE_SYSTEM, output=Verdict)
    return result.data


def score(results: pd.DataFrame, questions: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """Câu tra cứu/ràng buộc: Recall@k (capped), RR và đúng = có ít nhất 1 tin đúng trong top-k.
    Câu tổng hợp/đa chặng: đúng theo LLM chấm so với gt_value."""
    truth = questions.set_index("id")
    out = results.copy()
    recall, rr, correct = [], [], []
    for r in out.itertuples():
        t = truth.loc[r.id]
        if r.type in RETRIEVAL_TYPES:
            relevant = set(t.gt_ids)
            recall.append(recall_at_k(list(r.sources), relevant, k, capped=True))
            rr.append(reciprocal_rank(list(r.sources)[:k], relevant))
            correct.append(bool(set(list(r.sources)[:k]) & relevant))
        else:
            recall.append(np.nan)
            rr.append(np.nan)
            correct.append(bool(r.answer) and judge_answer(r.question, r.answer, t.gt_value)["correct"])
    return out.assign(recall_at_k=recall, rr=rr, correct=correct)


def accuracy_table(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.pivot_table(index="type", columns="system", values="correct", aggfunc="mean").round(2)


def cost_table(scored: pd.DataFrame) -> pd.DataFrame:
    price = LLM_PRICE_PER_MTOK
    t = scored.groupby("system").agg(latency_s=("latency", "mean"), input_tokens=("input_tokens", "mean"),
                                     output_tokens=("output_tokens", "mean"))
    t["usd_per_question"] = (t["input_tokens"] * price["input"] + t["output_tokens"] * price["output"]) / 1e6
    return t.round({"latency_s": 2, "input_tokens": 0, "output_tokens": 0, "usd_per_question": 4})


def error_cases(scored: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Gán nguyên nhân cho câu sai: router nhầm, Cypher lỗi, Cypher sai kết quả hoặc truy xuất trượt."""
    wrong = scored[~scored["correct"]].copy()

    def cause(r):
        if r.system == "hybrid" and r.route != r.route_gold:
            return f"router nhầm: {r.route} thay vì {r.route_gold}"
        if r.error:
            return f"Cypher lỗi: {str(r.error)[:80]}"
        if r.cypher:
            return "Cypher chạy được nhưng sai kết quả"
        return "truy xuất trượt hoặc ngữ cảnh không đủ để tính"

    wrong["nguyên nhân"] = [cause(r) for r in wrong.itertuples()]
    return wrong[["id", "type", "system", "question", "route", "nguyên nhân"]].head(n)


# ---------------------------------------------------------------- RAGAS (P4-09), chấm bằng Claude


class Claims(BaseModel):
    claims: list[str]


class ClaimCheck(BaseModel):
    supported: list[bool]


class Questions(BaseModel):
    questions: list[str]


class ContextVerdicts(BaseModel):
    relevant: list[bool]


def claims_supported(answer: str, contexts: list[str]) -> list[bool]:
    """Faithfulness theo định nghĩa RAGAS: tách câu trả lời thành nhận định, kiểm từng nhận định
    có được ngữ cảnh hỗ trợ không."""
    claims = ask_llm(f"Tách câu trả lời sau thành các nhận định độc lập, ngắn:\n{answer}", output=Claims).data["claims"]
    if not claims:
        return []
    listed = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    check = ask_llm(f"NGỮ CẢNH:\n{chr(10).join(contexts)}\n\nNHẬN ĐỊNH:\n{listed}\n\n"
                    "Với mỗi nhận định theo thứ tự, trả true nếu ngữ cảnh hỗ trợ trực tiếp.", output=ClaimCheck)
    return check.data["supported"][:len(claims)]


def answer_relevancy(question: str, answer: str, n: int = 3) -> float:
    """RAGAS: sinh n câu hỏi ngược từ câu trả lời, lấy cosine trung bình với câu hỏi gốc (E5)."""
    from src.vector import embed_query

    generated = ask_llm(f"Viết {n} câu hỏi mà câu trả lời sau trả lời được:\n{answer}",
                        output=Questions).data["questions"][:n]
    if not generated:
        return 0.0
    q = embed_query(question)
    return float(np.mean([q @ embed_query(g) for g in generated]))


def context_precision(question: str, contexts: list[str], reference: str) -> float:
    """RAGAS: trung bình precision@i tại các vị trí ngữ cảnh liên quan (ngữ cảnh đúng nên đứng đầu)."""
    if not contexts:
        return 0.0
    listed = "\n\n".join(f"[{i + 1}] {c[:1500]}" for i, c in enumerate(contexts))
    verdicts = ask_llm(f"CÂU HỎI: {question}\nĐÁP ÁN THAM CHIẾU: {reference}\n\nNGỮ CẢNH:\n{listed}\n\n"
                       "Với mỗi ngữ cảnh theo thứ tự, trả true nếu nó hữu ích để ra đáp án tham chiếu.",
                       output=ContextVerdicts).data["relevant"][:len(contexts)]
    hits, total = 0, 0.0
    for i, relevant in enumerate(verdicts, start=1):
        if relevant:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0
