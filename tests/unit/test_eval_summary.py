"""vre.eval.summary + vre.eval.errors trên answers/scores giả: khoá §7.7, CI có 2 số, router, nguyên nhân lỗi."""

from vre.config import load_settings
from vre.eval import errors, summary
from vre.eval.questions import load_questions

QUESTIONS = {q.id: q for q in load_questions()}
SYSTEMS = ("llm_only", "basic_rag", "graph", "hybrid")


def _answer(qid, system, **kw):
    base = {"id": qid, "system": system, "answer": "x", "route": "graph" if system == "hybrid" else None,
            "sources": [], "contexts": [], "cypher": None, "error": None, "fallback": False,
            "unsupported_citations": [], "latency_ms": 100.0, "input_tokens": 1000, "output_tokens": 100,
            "cached": False}
    return {**base, **kw}


def _fixture():
    answers, scores = {}, {}
    for qid, q in QUESTIONS.items():
        for i, s in enumerate(SYSTEMS):
            answers[(qid, s)] = _answer(qid, s)
            points = 2 if (s == "hybrid" or (sum(map(ord, qid)) + i) % 3 == 0) else 0
            scores[(qid, s)] = {"id": qid, "system": s, "category": q.category, "points": points,
                                "correct": points == 2, "faithfulness": 0.8 if s != "llm_only" else None,
                                **({"recall_at_5": 1.0, "recall_at_10": 1.0, "rr": 1.0, "ndcg_at_10": 1.0}
                                   if q.gt_kind == "ids" else {})}
    return answers, scores


def test_summary_shape_and_ci(tmp_path):
    answers, scores = _fixture()
    doc = summary.build(load_settings(data_root=tmp_path), answers, scores, QUESTIONS, {"generator": "g", "judge": "j"},
                        None)
    assert doc["systems"] == list(SYSTEMS) and doc["n_questions"] == 100 and doc["complete"] is True
    assert set(doc["by_category"]) == {q.category for q in QUESTIONS.values()}
    for cell in (c for cat in doc["by_category"].values() for c in cat.values()):
        assert len(cell["ci95"]) == 2 and cell["ci95"][0] <= cell["accuracy"] <= cell["ci95"][1]
    assert doc["overall"]["hybrid"]["accuracy"] == 1.0
    assert doc["ragas"]["basic_rag"] == {"faithfulness": 0.8, "answer_relevancy": None, "context_precision": None,
                                         "n": 100}
    assert summary.RAGAS_NOTE in doc["notes"]
    assert doc["cost"]["hybrid"]["usd_per_question"] > 0 and doc["cost"]["hybrid"]["latency_ms_p50"] == 100.0
    scopes = {(p["a"], p["b"], p["scope"]) for p in doc["pairwise"]}
    assert ("hybrid", "basic_rag", "aggregate") in scopes and ("hybrid", "graph", "overall") in scopes
    assert all(p["n_boot"] == 10_000 and len(p["ci95"]) == 2 for p in doc["pairwise"])
    router = doc["router"]
    assert router["n"] == 100 and 0 < router["accuracy"] < 1
    assert router["confusion"]["graph"]["graph"] == sum(q.route_gold == "graph" for q in QUESTIONS.values())
    assert set(router["by_category"]) == set(doc["by_category"])


def test_summary_partial_run_is_flagged(tmp_path):
    answers, scores = _fixture()
    keep = {"LK01", "AG01"}
    answers = {k: v for k, v in answers.items() if k[0] in keep}
    scores = {k: v for k, v in scores.items() if k[0] in keep}
    scores[("LK01", "graph")] = {**scores[("LK01", "graph")], "points": None, "correct": None}
    doc = summary.build(load_settings(data_root=tmp_path), answers, scores, QUESTIONS, {}, None)
    assert doc["complete"] is False and doc["n_questions"] == 2 and doc["n_questions_scored"] == 1
    assert doc["benchmark_subset"] == ["AG01", "LK01"] and doc["n_questions_total"] == 100
    assert doc["by_category"]["lookup"]["graph"]["n_unscored"] == 1
    empty = doc["by_category"]["valuation"]["hybrid"]  # nhóm chưa chạy: n=0, chỉ số null, không phải 0.0
    assert empty["n"] == 0 and empty["accuracy"] is None and empty["ci95"] is None
    assert doc["coverage"]["lookup"] == {"n_benchmark": 12, "n_answered_any": 1, "n_answered_all_systems": 0}
    assert doc["overall_common"]["hybrid"]["n"] == 1  # chỉ AG01 có điểm ở mọi hệ thống


def test_summary_marks_rule_based(tmp_path):
    answers, scores = _fixture()
    scores = {k: {**v, "scoring": "rule_based"} for k, v in scores.items()}
    doc = summary.build(load_settings(data_root=tmp_path), answers, scores, QUESTIONS, {}, None)
    assert doc["scoring"] == "rule_based" and doc["scoring_rules"] and doc["complete"] is True


def _q(category):
    return next(q for q in QUESTIONS.values() if q.category == category)


def test_assign_cause_rules():
    lookup, agg, una = _q("lookup"), _q("aggregate"), _q("unanswerable")
    bad = {"points": 0, "recall_at_5": 0.0}
    assert errors.assign_cause(agg, _answer(agg.id, "graph", points=2), {"points": 2}) is None
    assert errors.assign_cause(agg, _answer(agg.id, "graph", answer="", error="x"), bad) == "runtime_error"
    assert errors.assign_cause(una, _answer(una.id, "basic_rag"), bad) == "refusal_missing"
    wrong_route = "basic_rag" if agg.route_gold != "basic_rag" else "graph"
    assert errors.assign_cause(agg, _answer(agg.id, "hybrid", route=wrong_route), bad) == "router_wrong"
    assert errors.assign_cause(agg, _answer(agg.id, "graph", cypher="MATCH", error="CypherSyntaxError"),
                               bad) == "cypher_error"
    assert errors.assign_cause(lookup, _answer(lookup.id, "basic_rag"), bad, lambda q: True) == "filter_wrong"
    assert errors.assign_cause(lookup, _answer(lookup.id, "basic_rag"), bad, lambda q: False) == "retrieval_miss"
    assert errors.assign_cause(lookup, _answer(lookup.id, "llm_only"), bad) == "insufficient_context"
    assert errors.assign_cause(agg, _answer(agg.id, "basic_rag", unsupported_citations=[1]), bad) == "llm_hallucination"
    assert errors.assign_cause(agg, _answer(agg.id, "graph", cypher="MATCH"), bad) == "cypher_wrong_result"
    assert errors.assign_cause(agg, _answer(agg.id, "basic_rag"), bad) == "insufficient_context"
    unfaithful = {**bad, "faithfulness": 0.0}
    assert errors.assign_cause(agg, _answer(agg.id, "basic_rag"), unfaithful) == "llm_hallucination"
    refused = _answer(agg.id, "basic_rag", answer="Không đủ dữ liệu để tính")
    assert errors.assign_cause(agg, refused, unfaithful) == "insufficient_context"


def test_error_build_counts_and_examples():
    answers, scores = _fixture()
    doc = errors.build(QUESTIONS, {}, answers, scores, lambda q: False)
    assert set(doc["counts"]) <= set(SYSTEMS) and "hybrid" not in doc["counts"]  # hybrid đúng hết
    causes = {c for per in doc["counts"].values() for c in per}
    assert causes <= set(errors.CAUSES)
    for cause in causes:
        examples = [e for e in doc["examples"] if e["cause"] == cause]
        occurrences = sum(per.get(cause, 0) for per in doc["counts"].values())
        assert len(examples) >= min(2, occurrences)
    first = doc["examples"][0]
    assert set(first) == {"id", "system", "category", "cause", "question", "answer_excerpt", "gt_value", "cypher"}
