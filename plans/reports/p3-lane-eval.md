# Phase 3 Lane A3 + A5 (eval runner, metrics, judge, bootstrap, errors, summary, report): report

**Status:** DONE_WITH_CONCERNS (the concerns are listed at the end; none blocks the full run)

Branch `feat/p3-eval` (worktree `D:\Vietnam-Real-Estates-wt\eval`). HEAD `ba0d1b4`, which matches `origin/feat/p3-eval` (checked with `git ls-remote`). The working tree is clean. No PR is open; the lead integrates the lanes. `team-tools-unavailable: applied-from-prompt`.

Commits, on top of `490bd45`:
- `e515620` feat(eval): metrics, bootstrap, judge, B2
- `c6540bd` feat(eval): runner (resume, 429 backoff, scoring)
- `10d511f` feat(eval): errors + eval_summary
- `94e6780` feat(eval): report examples/readme + CLI
- `ba0d1b4` fix(eval): an answer that says "không đủ dữ liệu" is classed `insufficient_context`, not `llm_hallucination`

## Checklist

| # | Item | Status | Where |
|---|---|---|---|
| 1 | Runner: questions × 4 systems through `vre.qa.answer` only. ThreadPool 4, LLM cache reused. Each row is written to `results/eval/raw/answers.jsonl` as soon as it finishes. | ✓ | `runner.run_answers`, `answer_one` |
| 2 | Resume by (id, system). Rows that ended in a 429 are run again; readers take the last row per key. | ✓ | `read_jsonl`, `test_run_answers_writes_and_resumes`, `test_rate_limited_rows_are_rerun_on_resume` |
| 3 | Groq 429 backoff that honors `retry-after` / `retry-after-ms` (otherwise 5·2^n s, capped at 120 s, plus up to 1 s jitter). This covers both answers and judge calls. | ✓ | `with_backoff`, `_RateLimitTrap` (see D1), `test_answer_one_backs_off_with_retry_after` |
| 4 | Metrics: `ndcg_at_k`; matching by `dup_cluster_id` (reposts of one cluster count once); Recall@5/10 (capped); MRR. | ✓ | `metrics.py`, `test_eval_metrics.py` |
| 5 | Judge on the 0–2 scale, with each question's own tolerance and the `gt_rows` details; refusal judge with the 06/2025–03/2026 period. | ✓ | `judge.judge_answer`, `judge_system` |
| 6 | Valuation is scored numerically: the judge only *extracts* the forecast in tỷ; points are 2 within `value_tolerance`, 1 within 2×tolerance, else 0. | ✓ | `judge.extract_price_bn`, `metrics.value_points` |
| 7 | B2: eval refuses to run when `JUDGE_MODEL == LLM_MODEL`, unless `VRE_ALLOW_SAME_JUDGE=1`. The flag is recorded in `eval_summary.llm.same_judge_allowed`. | ✓ | `judge.check_judge_config`, `tests/unit/test_judge_config.py`, `test_b2_refuses_before_any_call` |
| 8 | RAGAS per the lead's scope cut: `faithfulness` only (basic_rag, graph, hybrid; not on unanswerable). `answer_relevancy` and `context_precision` are `null`, with a `notes` entry. | ✓ | `runner.score_one`, `summary.RAGAS_NOTE` |
| 9 | Bootstrap: 10,000 resamples, seed 42, paired by question, 2.5/97.5 percentiles. `p_value` is the share of resamples with the opposite sign (1.0 when diff = 0). | ✓ | `bootstrap.py`, `test_eval_bootstrap.py` |
| 10 | Pairwise comparisons: hybrid–basic_rag, basic_rag–llm_only, hybrid–graph. Scopes: overall plus every category. Metrics: accuracy and mean_points. | ✓ | `summary._pairwise` |
| 11 | Error analysis: the 9-value cause enum, filter_wrong checked by SQL on `v_listings`, ≥ 2 examples per cause that occurs, written to `results/error_analysis.json`. | ✓ | `errors.py`, `test_assign_cause_rules`, `test_error_build_counts_and_examples` |
| 12 | `eval_summary.json` per §7.7: index, llm, by_category × system with `ci95`, overall, retrieval, ragas, cost (p50/p95, tokens, USD, bad_citation_rate), pairwise, router. | ✓ | `summary.py`, `test_summary_shape_and_ci` |
| 13 | Router accuracy overall, per category, and without `unanswerable`, plus the confusion matrix. | ✓ | `summary._router` |
| 14 | USD is computed from `Settings.llm_price_input/output`, which already exist (0.15 / 0.60 for Groq). `vre/config.py` was not edited. | ✓ | `summary._cost` |
| 15 | A5 `report examples`: writes `results/examples.json` (§7.8) from real eval rows. | ✓ code, not run (smoke scope) | `report.examples` |
| 16 | A5 `report readme`: replaces only the text between the two markers; aborts if the markers are missing or duplicated; keeps CRLF. | ✓ code, not run (smoke scope) | `report.readme`, `test_eval_report.py` |
| 17 | CLI: `eval run [--limit N] [--systems ...]`, `eval score [--no-ragas]`, `eval report`, `eval all`, `report examples`, `report readme`, and `ablation [--limit]`, which lazily calls `vre.eval.ablation.run(settings, limit=...)`. | ✓ | `scripts/vre_pipeline.py` |
| 18 | Memory: one process, the encoder is the `vre.rag.query_encoder` singleton, DuckDB is capped at 1 GB / 2 threads, and every heavy step ran through `run_heavy.ps1`. | ✓ | — |

## Verify (real outputs)

- `pytest -q tests/unit` (through `run_heavy.ps1`): **293 passed in 41.89s**.
- `ruff check src/vre/eval tests/unit scripts`: **All checks passed!**
- `mypy src/vre/eval`: **0 errors in `src/vre/eval/*`**. Mypy also reports 7 errors in other modules it follows imports into (`index/embed`, `index/qdrant_store`, `valuation/features`, `valuation/search_cache`, `valuation/train`); those are not owned by this lane.
- Synthetic bootstrap tests:
  - A known +0.2 difference over n=400 gives a CI that contains 0.2 and excludes 0, with p < 0.01.
  - The CI narrows as n grows 50 → 200 → 800.
  - A negative difference gives the mirrored result.
  - A zero difference gives p = 1.0 and CI [0, 0].

### Smoke run: `eval run --limit 5` → `eval score` → `eval report`

Scope: 5 questions (LK01, CS01, AG01, MH01, TM01, one each from lookup, constraint, aggregate, multihop and temporal) × 4 systems, real Groq. Every step went through the heavy-job lock; free commit was 6.2 GB at the start.

| Step | Wall-clock | Real LLM API calls* | Tokens | Errors |
|---|---|---|---|---|
| `eval run` | 245.4 s (**49.1 s per question** for all 4 systems; 279 s including lock and checks) | 35 | 63,285 in / 9,335 out | 0 errors, 0 rows lost to 429 |
| `eval score` | 190.8 s | 41 | — | 0 judge errors. 5 judge 429s were each recovered by backoff (waits 8.5–28 s, Retry-After honored). |
| `eval report` | ~70 s | ~0 (filter re-extraction hits the cache) | — | 0 |

\* "Real calls" means new entries in the LLM disk cache, so a call answered from the cache is not counted.

Per-system latency p50 / p95 (ms):
- llm_only: 1,724 / 3,317
- basic_rag: 21,872 / 37,912
- graph: 75,813 / 93,481
- hybrid: 89,974 / 149,564 (CS01/hybrid took 161 s)

Cost per question, from Groq prices:
- llm_only $0.00007
- basic_rag $0.00069
- graph $0.00078
- hybrid $0.00147

These latencies include waiting inside the OpenAI SDK's own 429 retries, so they overstate the systems' speed cost.

Smoke results (5 questions, **not conclusive**; the summary records `complete: false`):

| System | Accuracy |
|---|---|
| llm_only | 0.0 |
| basic_rag | 0.2 (CI [0, 0.6]) |
| graph | 1.0 |
| hybrid | 1.0 |

- Router accuracy: 0.6. The hybrid router sent lookup and constraint (gold `basic_rag`) to `hybrid`.
- Faithfulness: basic_rag 0.77, graph 1.0, hybrid 0.9.
- Retrieval: basic_rag R@5 = 0.6.
- Error counts: basic_rag `insufficient_context` 3, `retrieval_miss` 1; llm_only `insufficient_context` 5.

**Extrapolated full run (100 questions):**
- `eval run` takes about 100 × 49 s ≈ **82 min**, with about 700 real calls.
- `eval score` takes about 20 × 38 s ≈ **64 min**, with about 820 calls.
- `eval report` adds about 2 min.
- Total is **about 2.5 h** of wall-clock and about 1,500 Groq calls; resume makes it safe to interrupt.

The smoke artifacts are in the main checkout and are not committed, because they come from a 5-question smoke run: `results/eval/raw/{answers,scores}.jsonl` (gitignored), `results/eval_summary.json`, `results/error_analysis.json`. The full run appends to answers.jsonl and skips the 20 rows already there; delete `results/eval/raw/` first if you want a clean full run.

## Decisions taken (autonomous, conservative)

- **D1 — Retry-After for answers.** `vre.qa.answer` swallows exceptions. The original `openai.RateLimitError` (which carries the headers) appears only in the `log.warning(..., exc_info=True)` of logger `vre.qa`. A logging filter keeps the last 429 per thread, so the backoff can read `Retry-After`. Judge calls raise directly, so headers are read straight from the exception.
- **D2 — Scoring artifact.** Scores go to `results/eval/raw/scores.jsonl`: one row per (id, system), keyed by `answer_digest` so a changed answer is re-scored. This file is not in the contract; it is raw and gitignored with `results/eval/raw/`. Rows the judge failed on (not 429) get `points=null`, are excluded from means, counted in `n_unscored`, and re-scored on the next `eval score`.
- **D3 — Points for `gt_kind=ids`.** Kept the old scale: 2 when capped Recall@5 by cluster is 1, 1 when at least one cluster is hit, else 0. Sources are the top_k=5 listings the system actually used. As a result recall_at_10 can be ≤ recall_at_5; this is stated in `notes`, and pure retrieval numbers belong to `ablation.json`.
- **D4 — Valuation partial credit.** 1 point when the error is within 2× tolerance (by analogy with the old 5%/20% bands).
- **D5 — `--limit N`.** Picks questions round-robin across categories, so a smoke run covers N categories.
- **D6 — Extra summary keys** beyond the §7.7 example:
  - `n_questions_total`, `complete`, `notes`
  - per cell: `n_unscored`
  - cost: `error_rate`, `cached_rate`
  - retrieval: `n`, `top_k`
  - pairwise: `n`
  - router: `n`, `accuracy_excl_unanswerable`, `by_category`
  - llm: `provider`, `generator_fallbacks`, `same_judge_allowed`, `note`

  If a route is missing, a `none` column appears in the confusion matrix.
- **D7 — Legacy pandas API kept.** `ground_truth`, `score`, `error_cases` etc. stay at the bottom of `runner.py` because `tests/unit/test_metrics.py` (not owned by this lane) imports them. Delete them once that test migrates. This makes runner.py longer than the 200-line guideline.
- **D8 — `filter_wrong`.** Re-extracts the filters with `vre.rag.filters.parse_filters`, using the same prompt so it hits the eval's LLM cache. It is classed `filter_wrong` when the SQL WHERE on `v_listings` keeps 0 of the GT listings. This reads `vre.rag` for analysis only; it does not run a system.
- **D9 — Examples.** `report examples` re-runs `vre.qa.stream_answer` on already-answered benchmark questions to get the filters, hits and Cypher rows that answers.jsonl does not store. Valuation examples call `valuate(valuation_input)`. Neither A5 command was run in this smoke; they are left for after the full run.

## Concerns (for the lead)

- **C1 — The `graph` system echoes the context wrapper.** Its answers contain the raw `<tai_lieu> KẾT QUẢ TRUY VẤN: …` (e.g. LK01, CS01, AG01, TM01). The judge still scored them correctly, but the text is poor for the web UI and examples. This sits in Lane A1's graph answer prompt or generation path (`vre.graph.text2cypher` / `qa.pipeline`), not this lane.
- **C2 — The judge model is also the generator's fallback.** `openai/gpt-oss-20b` is in `generator_fallbacks`. When gpt-oss-120b returns 429, some answers may be generated by the judge model. This is recorded in `eval_summary.llm.note`. A clean fix is a different fallback model, or none, for the eval run.
- **C3 — Router behavior.** Hybrid sends lookup/constraint questions to the `hybrid` route instead of `basic_rag` (2 of 2 in the smoke). This counts as `router_wrong` only when the answer is also wrong.
- **C4 — Latency is dominated by Groq.** 429 waits inside the SDK inflate the graph/hybrid p50 (76–90 s). Expect the full run to take about 2.5 h.

## Not done (per scope)

- The full 100-question run, and `report examples` / `report readme`, which wait for the full run.
- `vre.eval.ablation`, which belongs to the parallel lane; only the CLI hook exists here.
