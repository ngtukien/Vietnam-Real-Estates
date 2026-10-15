"""vre.eval.runner: resume theo (id, system), backoff 429 theo Retry-After, chọn câu, chấm từng dòng. Không gọi LLM."""

from types import SimpleNamespace

import pytest

from vre.config import load_settings
from vre.eval import judge, runner
from vre.eval.questions import load_questions


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv(judge.ALLOW_SAME_JUDGE_ENV, "0")
    s = load_settings(data_root=tmp_path, llm_provider="groq", llm_model="", judge_model="")
    monkeypatch.setattr(runner, "_cache_entries", lambda: 0)
    return s


def _ok(system, question, top_k=5):
    return {"system": system, "answer": f"trả lời {question[:5]}", "route": None, "sources": [1], "contexts": ["c"],
            "cypher": None, "error": None, "fallback": False, "unsupported_citations": [], "latency_ms": 1.0,
            "input_tokens": 10, "output_tokens": 5, "cached": False}


def test_select_questions_round_robin_covers_categories():
    picked = runner.select_questions(load_questions(), 5)
    assert len(picked) == 5 and len({q.category for q in picked}) == 5
    assert runner.select_questions(load_questions(), None) == load_questions()


def test_run_answers_writes_and_resumes(settings):
    calls = []

    def fake(system, question, top_k=5):
        calls.append(system)
        return _ok(system, question)

    first = runner.run_answers(settings, limit=2, systems=["llm_only", "basic_rag"], answer_fn=fake)
    assert first["written"] == 4 and first["skipped"] == 0
    rows = runner.read_jsonl(runner.answers_path(settings))
    assert len(rows) == 4 and all(r["id"] and r["system"] in ("llm_only", "basic_rag") for r in rows.values())
    second = runner.run_answers(settings, limit=2, systems=["llm_only", "basic_rag"], answer_fn=fake)
    assert second["written"] == 0 and second["skipped"] == 4 and len(calls) == 4


def test_rate_limited_rows_are_rerun_on_resume(settings):
    def limited(system, question, top_k=5):
        return {**_ok(system, question), "answer": "", "error": "upstream_llm_error: RateLimitError"}

    runner.run_answers(settings, limit=1, systems=["llm_only"], answer_fn=limited, sleep=lambda _s: None)
    stats = runner.run_answers(settings, limit=1, systems=["llm_only"], answer_fn=_ok)
    assert stats["written"] == 1
    (row,) = runner.read_jsonl(runner.answers_path(settings)).values()
    assert row["error"] is None  # dòng sau ghi đè dòng 429


def test_answer_one_backs_off_with_retry_after():
    q = load_questions()[0]
    replies = iter([{**_ok("graph", "x"), "error": "upstream_llm_error: RateLimitError"}, _ok("graph", "x")])
    slept = []

    class RateLimitError(Exception):
        response = SimpleNamespace(headers={"retry-after": "7"})

    def fake(system, question, top_k=5):
        runner.log.warning("x")  # trap chỉ nhận bản ghi của logger vre.qa
        import logging

        try:
            raise RateLimitError()
        except RateLimitError:
            logging.getLogger("vre.qa").warning("QA lỗi", exc_info=True)
        return next(replies)

    row = runner.answer_one(q, "graph", fake, 5, sleep=slept.append)
    assert row["error"] is None and row["id"] == q.id
    assert len(slept) == 1 and 7 <= slept[0] < 8  # Retry-After 7 s (+ jitter < 1 s) thắng 5·2^0


def test_with_backoff_retries_only_rate_limits():
    class RateLimitError(Exception):
        response = SimpleNamespace(headers={})

    attempts = iter([RateLimitError(), RateLimitError(), "ok"])
    slept = []

    def fn():
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    assert runner.with_backoff(fn, sleep=slept.append) == "ok" and len(slept) == 2
    with pytest.raises(KeyError):
        runner.with_backoff(lambda: {}["x"], sleep=slept.append)


def test_backoff_delay_caps():
    assert 5 <= runner.backoff_delay(0, None) < 6
    assert runner.backoff_delay(10, None) < runner.BACKOFF_CAP_S + 1
    assert runner.backoff_delay(0, 1000) < runner.BACKOFF_CAP_S + 1


def test_b2_refuses_before_any_call(tmp_path, monkeypatch):
    monkeypatch.delenv(judge.ALLOW_SAME_JUDGE_ENV, raising=False)
    s = load_settings(data_root=tmp_path, llm_provider="openai", llm_model="m", judge_model="m")
    with pytest.raises(RuntimeError):
        runner.run_answers(s, limit=1, answer_fn=lambda *a, **k: pytest.fail("không được gọi"))


def _q(category):
    return next(q for q in load_questions() if q.category == category)


def test_score_one_ids_uses_clusters():
    q = _q("lookup")
    gt = {"gt_dup_clusters": [100], "gt_ids": [101]}
    out = runner.score_one(q, gt, {**_ok("basic_rag", "x"), "sources": [101, 5]}, {101: 100}, ragas=False)
    assert out["points"] == 2 and out["correct"] is True and out["recall_at_5"] == 1.0


def test_score_one_valuation_is_numeric(monkeypatch):
    q = _q("valuation")
    monkeypatch.setattr(judge, "extract_price_bn", lambda answer: 5.3)
    out = runner.score_one(q, {"gt_value": "5.12"}, _ok("hybrid", "x"), {}, ragas=False)
    assert out["points"] == 2 and out["predicted_value"] == 5.3
    monkeypatch.setattr(judge, "extract_price_bn", lambda answer: None)
    assert runner.score_one(q, {"gt_value": "5.12"}, _ok("hybrid", "x"), {}, ragas=False)["points"] == 0


def test_score_one_judge_error_is_unscored(monkeypatch):
    q = _q("aggregate")

    def boom(*_a, **_k):
        raise RuntimeError("bad json")

    monkeypatch.setattr(judge, "judge_answer", boom)
    out = runner.score_one(q, {"gt_value": "1"}, _ok("graph", "x"), {}, ragas=False)
    assert out["points"] is None and out["correct"] is None and "RuntimeError" in out["judge_error"]


def test_score_one_faithfulness_only(monkeypatch):
    q = _q("aggregate")
    monkeypatch.setattr(judge, "judge_answer", lambda *a, **k: {"score": 2, "reason": "ok"})
    monkeypatch.setattr(judge, "claims_supported", lambda answer, contexts: [True, False])
    out = runner.score_one(q, {"gt_value": "1"}, _ok("graph", "x"), {})
    assert out["faithfulness"] == 0.5 and "answer_relevancy" not in out
    llm_only = runner.score_one(q, {"gt_value": "1"}, _ok("llm_only", "x"), {})
    assert llm_only["faithfulness"] is None
