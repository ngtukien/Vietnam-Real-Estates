"""Eval runner Phase 3 (A3.1): câu benchmark × 4 hệ thống qua `vre.qa.answer` → `results/eval/raw/answers.jsonl`,
rồi chấm → `results/eval/raw/scores.jsonl`.

- Ghi từng dòng NGAY khi xong; resume theo (id, system): dòng đã có được bỏ qua, trừ dòng lỗi 429 (chạy lại). Người đọc
  lấy dòng CUỐI của mỗi khoá.
- ThreadPool 4, cache LLM đĩa (`vre.llm.cache`), backoff khi Groq trả 429 (theo `Retry-After` nếu có, không thì
  luỹ thừa 2, trần 120 s). SDK OpenAI đã tự thử lại 4 lần theo `Retry-After`; backoff ở đây là tầng ngoài khi SDK bỏ cuộc.
- Encoder truy vấn e5-large là singleton của `vre.rag.query_encoder` ⇒ nạp một lần cho cả tiến trình.
- RAGAS: chỉ `faithfulness` (quyết định của lead 2026-10-15: bỏ answer_relevancy/context_precision để tiết kiệm lời gọi LLM).

Phần cuối file giữ API pandas cũ (`ground_truth`, `score`, `error_cases`…) vì `tests/unit/test_metrics.py` còn dùng;
xoá khi test đó chuyển sang API mới (GT đã chuyển sang SQL ở `vre.eval.ground_truth`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from vre.config import Settings, get_settings
from vre.eval import judge as J
from vre.eval import metrics as M
from vre.eval.judge import judge_answer
from vre.eval.metrics import recall_at_k, reciprocal_rank
from vre.eval.questions import Question, load_questions

log = logging.getLogger(__name__)

SYSTEMS = ("llm_only", "basic_rag", "graph", "hybrid")
RAGAS_SYSTEMS = ("basic_rag", "graph", "hybrid")
WORKERS = 4
MAX_ATTEMPTS = 6
BACKOFF_BASE_S = 5.0
BACKOFF_CAP_S = 120.0
RAGAS_MAX_CONTEXTS = 10


# ---- tệp jsonl thô (resume theo khoá)

def raw_dir(settings: Settings) -> Path:
    return settings.results_dir / "eval" / "raw"


def answers_path(settings: Settings) -> Path:
    return raw_dir(settings) / "answers.jsonl"


def scores_path(settings: Settings) -> Path:
    return raw_dir(settings) / "scores.jsonl"


def read_jsonl(path: Path) -> dict[tuple[str, str], dict]:
    """(id, system) → dòng CUỐI (dòng sau ghi đè dòng trước). Dòng hỏng (tiến trình bị giết giữa lúc ghi) bị bỏ qua
    có cảnh báo."""
    rows: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return rows
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            log.warning("%s:%d: dòng JSON hỏng, bỏ qua", path.name, lineno)
            continue
        rows[(str(row["id"]), str(row["system"]))] = row
    return rows


class JsonlWriter:
    """Ghi thêm một dòng, flush ngay; khoá để nhiều luồng ghi không xen nhau."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False, default=str)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        self._file.close()


# ---- backoff 429

def is_rate_limited(error: Any) -> bool:
    text = str(error or "")
    return "RateLimitError" in text or "429" in text


def retry_after_seconds(error: BaseException | None) -> float | None:
    """Giây chờ từ header `retry-after-ms` / `retry-after` của lỗi openai; None nếu không có."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        if headers.get("retry-after-ms"):
            return float(headers["retry-after-ms"]) / 1000
        if headers.get("retry-after"):
            return float(headers["retry-after"])
    except (TypeError, ValueError):
        return None
    return None


def backoff_delay(attempt: int, retry_after: float | None) -> float:
    """Chờ ≥ Retry-After; không có thì 5·2^attempt s; trần 120 s; + jitter ≤ 1 s để 4 luồng không dội cùng lúc."""
    exponential = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** attempt)
    wait = max(retry_after, exponential) if retry_after is not None else exponential
    return min(wait, BACKOFF_CAP_S) + random.random()


class _RateLimitTrap(logging.Filter):
    """`vre.qa.answer` bắt mọi lỗi và chỉ trả chuỗi `error`; lỗi gốc (mang header Retry-After) chỉ xuất hiện trong
    `log.warning(..., exc_info=True)` của logger `vre.qa`. Filter này giữ lỗi 429 cuối cùng THEO LUỒNG để backoff
    đọc được Retry-After. Không chặn bản ghi log."""

    def __init__(self) -> None:
        super().__init__()
        self._local = threading.local()

    def filter(self, record: logging.LogRecord) -> bool:
        error = record.exc_info[1] if record.exc_info else None
        if error is not None and is_rate_limited(type(error).__name__):
            self._local.error = error
        return True

    def take(self) -> BaseException | None:
        error = getattr(self._local, "error", None)
        self._local.error = None
        return error


_trap = _RateLimitTrap()
logging.getLogger("vre.qa").addFilter(_trap)


def with_backoff(fn: Callable[[], Any], *, sleep: Callable[[float], None] = time.sleep,
                 max_attempts: int = MAX_ATTEMPTS) -> Any:
    """Gọi `fn`; khi raise lỗi 429 (openai.RateLimitError) thì chờ rồi gọi lại, tối đa `max_attempts` lần."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as error:
            if not is_rate_limited(type(error).__name__) or attempt == max_attempts - 1:
                raise
            delay = backoff_delay(attempt, retry_after_seconds(error))
            log.warning("429 từ LLM, chờ %.1f s (lần %d/%d)", delay, attempt + 1, max_attempts)
            sleep(delay)
    raise AssertionError("unreachable")


# ---- chạy hệ thống

def select_questions(questions: list[Question], limit: int | None) -> list[Question]:
    """`limit` câu xen kẽ theo nhóm (lookup, constraint, …) để lượt thử nhỏ vẫn phủ nhiều nhóm; None ⇒ tất cả."""
    if limit is None or limit >= len(questions):
        return list(questions)
    by_category: dict[str, list[Question]] = {}
    for q in questions:
        by_category.setdefault(q.category, []).append(q)
    picked: list[Question] = []
    while len(picked) < limit:
        for group in by_category.values():
            if group and len(picked) < limit:
                picked.append(group.pop(0))
    return picked


def _error_row(system: str, error: BaseException, started: float) -> dict:
    return {"system": system, "answer": "", "route": None, "sources": [], "contexts": [], "cypher": None,
            "error": f"runtime_error: {type(error).__name__}: {error}", "fallback": False,
            "unsupported_citations": [], "latency_ms": round((perf_counter() - started) * 1000, 1),
            "input_tokens": 0, "output_tokens": 0, "cached": False}


def answer_one(q: Question, system: str, answer_fn: Callable[..., dict], top_k: int, *,
               sleep: Callable[[float], None] = time.sleep, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Một dòng answers.jsonl. Lỗi 429 ⇒ chờ (Retry-After) rồi hỏi lại; hết lượt ⇒ ghi dòng lỗi (resume chạy lại)."""
    row: dict = {}
    for attempt in range(max_attempts):
        _trap.take()
        started = perf_counter()
        try:
            row = answer_fn(system, q.question, top_k=top_k)
        except Exception as error:  # vd ValueError trước sự kiện đầu: ghi lại, không dừng cả lượt
            row = _error_row(system, error, started)
        if not is_rate_limited(row.get("error")) or attempt == max_attempts - 1:
            break
        delay = backoff_delay(attempt, retry_after_seconds(_trap.take()))
        log.warning("%s/%s: 429, chờ %.1f s (lần %d/%d)", q.id, system, delay, attempt + 1, max_attempts)
        sleep(delay)
    return {"id": q.id, **row, "system": system}


def _cache_entries() -> int:
    """Số file cache LLM: mỗi lời gọi API thật (không trúng cache) thêm một file ⇒ hiệu trước/sau ≈ số lời gọi."""
    from vre.llm.cache import cache_dir

    directory = cache_dir()
    return sum(1 for _ in directory.glob("*.json")) if directory.exists() else 0


def run_answers(settings: Settings, *, limit: int | None = None, systems: Iterable[str] | None = None,
                workers: int = WORKERS, answer_fn: Callable[..., dict] | None = None,
                sleep: Callable[[float], None] = time.sleep) -> dict:
    """`eval run`: hỏi mọi (câu, hệ thống) chưa có trong answers.jsonl, ghi từng dòng. Trả thống kê lượt chạy."""
    llm = J.check_judge_config(settings)  # B2: từ chối trước khi tốn lời gọi nào
    chosen = tuple(systems or SYSTEMS)
    unknown = set(chosen) - set(SYSTEMS)
    if unknown:
        raise ValueError(f"hệ thống lạ {sorted(unknown)}; chọn trong {SYSTEMS}")
    if answer_fn is None:
        from vre.qa import answer as qa_answer
        answer_fn = qa_answer
    questions = select_questions(load_questions(), limit)
    path = answers_path(settings)
    done = {k for k, r in read_jsonl(path).items() if not is_rate_limited(r.get("error"))}
    tasks = [(q, s) for q in questions for s in chosen if (q.id, s) not in done]
    cache_before = _cache_entries()
    started = perf_counter()
    stats: dict[str, Any] = {"n_questions": len(questions), "systems": list(chosen),
                             "skipped": len(questions) * len(chosen) - len(tasks), "written": 0, "errors": 0,
                             "rate_limited": 0, "input_tokens": 0, "output_tokens": 0, "cached_rows": 0, "llm": llm}
    writer = JsonlWriter(path)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(answer_one, q, s, answer_fn, settings.top_k, sleep=sleep) for q, s in tasks]
            for future in as_completed(futures):
                row = future.result()
                writer.write(row)
                stats["written"] += 1
                stats["errors"] += bool(row.get("error"))
                stats["rate_limited"] += is_rate_limited(row.get("error"))
                stats["input_tokens"] += int(row.get("input_tokens") or 0)
                stats["output_tokens"] += int(row.get("output_tokens") or 0)
                stats["cached_rows"] += bool(row.get("cached"))
                log.info("%s/%s xong (%s ms)%s", row["id"], row["system"], row.get("latency_ms"),
                         f" lỗi {row['error']}" if row.get("error") else "")
    finally:
        writer.close()
    wall = perf_counter() - started
    n_new = len({q.id for q, _ in tasks})
    stats.update(wall_s=round(wall, 1), s_per_question=round(wall / n_new, 2) if n_new else None,
                 llm_api_calls=_cache_entries() - cache_before, answers_path=str(path))
    return stats


# ---- chấm điểm

def cluster_map(settings: Settings, ids: Iterable[int]) -> dict[int, int]:
    """listing_id → dup_cluster_id (bảng sạch). Một truy vấn DuckDB nhỏ (1 GB, 2 luồng: tiến trình eval chia máy)."""
    wanted = sorted({int(i) for i in ids})
    if not wanted:
        return {}
    from vre.data.duck import connect

    con = connect(settings, views=False)
    try:
        con.execute("SET memory_limit = '1GB'")
        con.execute("SET threads = 2")
        rows = con.execute("SELECT listing_id, dup_cluster_id FROM read_parquet(?) WHERE listing_id IN "
                           "(SELECT unnest(?::BIGINT[]))", [str(settings.clean_path), wanted]).fetchall()
    finally:
        con.close()
    return {int(a): int(b) for a, b in rows}


def answer_digest(row: dict) -> str:
    """Dấu vân tay câu trả lời: điểm đã chấm còn hợp lệ khi câu trả lời không đổi."""
    payload = json.dumps([row.get("answer"), row.get("sources"), row.get("error")], ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def score_one(q: Question, gt: dict, row: dict, cluster_of: dict[int, int], *, ragas: bool = True,
              call: Callable[[Callable[[], Any]], Any] = with_backoff) -> dict:
    """Một dòng scores.jsonl: điểm 0–2, đúng/sai, chỉ số truy xuất, faithfulness. Lỗi giám khảo (không phải 429) ⇒
    `points=None` + `judge_error` (summary loại khỏi trung bình, resume chấm lại)."""
    out: dict[str, Any] = {"id": q.id, "system": row["system"], "category": q.category, "gt_kind": q.gt_kind,
                           "answer_digest": answer_digest(row), "points": 0, "reason": None, "predicted_value": None,
                           "faithfulness": None, "judge_error": None}
    answer = (row.get("answer") or "").strip()
    sources = [int(s) for s in row.get("sources") or []]
    try:
        if q.gt_kind == "ids":
            out.update(M.retrieval_scores(sources, gt.get("gt_dup_clusters") or [], cluster_of))
            out["points"] = M.retrieval_points(out["recall_at_5"])
        elif not answer:
            out["reason"] = "câu trả lời rỗng"
        elif q.category == "valuation":
            predicted = call(lambda: J.extract_price_bn(answer))
            out["predicted_value"] = predicted
            out["points"] = M.value_points(predicted, float(gt["gt_value"]), float(q.value_tolerance or 0.15))
        else:
            verdict = call(lambda: J.judge_answer(q.question, answer, gt.get("gt_value"), q.value_tolerance,
                                                  gt.get("gt_rows")))
            out["points"], out["reason"] = int(verdict["score"]), verdict.get("reason")
        contexts = list(row.get("contexts") or [])[:RAGAS_MAX_CONTEXTS]
        if ragas and row["system"] in RAGAS_SYSTEMS and q.category != "unanswerable" and answer and contexts:
            supported = call(lambda: J.claims_supported(answer, contexts))
            out["faithfulness"] = float(M.faithfulness(supported)) if supported else None
    except Exception as error:
        log.warning("chấm %s/%s lỗi: %s", q.id, row["system"], type(error).__name__)
        out["points"], out["judge_error"] = None, f"{type(error).__name__}: {str(error)[:200]}"
    out["correct"] = None if out["points"] is None else out["points"] == 2
    return out


def run_scores(settings: Settings, *, ragas: bool = True, workers: int = WORKERS) -> dict:
    """`eval score`: chấm mọi dòng answers.jsonl chưa có điểm hợp lệ (cùng `answer_digest`, không `judge_error`)."""
    from vre.eval.ground_truth import load_gt

    llm = J.check_judge_config(settings)
    answers = read_jsonl(answers_path(settings))
    if not answers:
        raise FileNotFoundError(f"Chưa có {answers_path(settings)}; chạy `eval run` trước")
    questions = {q.id: q for q in load_questions()}
    gt = load_gt(settings)
    answered = {i for i, _ in answers}
    missing = sorted((answered - set(questions)) | (answered - set(gt)))
    if missing:
        raise RuntimeError(f"answers.jsonl có câu không còn trong benchmark/GT: {missing}")
    path = scores_path(settings)
    old = read_jsonl(path)
    todo = [r for k, r in answers.items()
            if not (k in old and old[k].get("answer_digest") == answer_digest(r) and not old[k].get("judge_error"))]
    cluster_of = cluster_map(settings, (s for r in todo for s in r.get("sources") or []))
    cache_before = _cache_entries()
    started = perf_counter()
    stats: dict[str, Any] = {"rows": len(answers), "scored": 0, "judge_errors": 0, "llm": llm}
    writer = JsonlWriter(path)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(score_one, questions[r["id"]], gt[r["id"]], r, cluster_of, ragas=ragas) for r in todo]
            for future in as_completed(futures):
                row = future.result()
                writer.write(row)
                stats["scored"] += 1
                stats["judge_errors"] += bool(row["judge_error"])
    finally:
        writer.close()
    stats.update(skipped=len(answers) - len(todo), wall_s=round(perf_counter() - started, 1),
                 llm_api_calls=_cache_entries() - cache_before, scores_path=str(path))
    return stats


# ---- API pandas cũ (chỉ `tests/unit/test_metrics.py` còn dùng; xem docstring đầu file)

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
    series = sizes if agg == "count" else d.groupby(keys)[str(value)].agg(agg)
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
    for item in out.itertuples():
        r: Any = item
        t: Any = truth.loc[r.id]
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
    price_in, price_out = s.llm_price_input or 0.0, s.llm_price_output or 0.0
    t["usd_per_question"] = (t["input_tokens"] * price_in + t["output_tokens"] * price_out) / 1e6
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
