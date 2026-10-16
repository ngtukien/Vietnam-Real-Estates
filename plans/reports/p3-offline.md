# Phase 3: offline/rule-based eval tooling, Gemini provider, ablation — report

**Status:** DONE_WITH_CONCERNS. The lead's final scope cut was applied: no examples, README, notebooks or P3_RESULTS_READY from this run; the flash-lite outputs are parked.

Branch `feat/p3-offline` (worktree `D:\Vietnam-Real-Estates-wt\offline`). HEAD `26d1020` = `origin/feat/p3-offline` (checked with `git ls-remote`). The working tree is clean. No PR is open; the lead integrates.

## Commits (on top of `1b266b9`)

| SHA | What |
|---|---|
| `cbfae96` | `VRE_LLM_OFFLINE=1` / `cache_only()` in `vre.llm.client`; `LLMCacheMiss`; the notebooks now import that one implementation |
| `a65f9fb` | rule-based scorer (`eval score --offline`); summary coverage/subset; offline-safe error analysis and examples |
| `3adc8a6` | `gemini` provider; CLI `eval run --per-category/--workers/--rule-based`, `eval score --offline`; `P3_RESULTS_READY` from p3/p4 |
| `5636d5d` | ablation in cache-only mode (fails on any real LLM call) |
| `94bd359` | slide-oriented examples (tagged graph examples, llm_only vs basic_rag pair, contrast case, `notes` when a case is absent); P4-11 cell tolerates a null CI |
| `0897cbf` | gemini has no fallback model; `eval run --retry-errors` |
| `d997a90` | backoff reads Gemini's in-body retry delay |
| `7dc0785` | rule-scorer fixes found by spot-checks (see below) |
| `b5b4f46` | `rules_version`, so a rule change re-scores every row (this commit had a broken string literal) |
| `7d18336` | fixes that literal |
| `26d1020` | README block labels rule-based scoring and lists the rules; `results/ablation.json` + `results/benchmark_gt.json` committed |

## Checklist (from the brief and its revisions)

1. **Cache-only switch** ✓
   - `VRE_LLM_OFFLINE=1` or `with cache_only():` serves answers from the disk cache; a miss raises `LLMCacheMiss`.
   - Guarded in `ask_llm`, `stream_llm` and `client()`, so no path reaches the network.
   - `notebooks/build_notebooks.py` now imports this one implementation; it had its own copy.
   - Tests: `tests/unit/test_llm_offline.py`.
2. **Rule-based scorer** ✓ `src/vre/eval/rule_score.py`
   - ids: capped cluster Recall@5; 2 points if ≥ 0.8, 1 if > 0.
   - Numeric values: within tolerance of GT. Negative GT needs a decrease word in the answer.
   - Text GT: diacritic-folded contains-match. For comparisons, the winner must appear before the first "cao/đắt hơn".
   - Valuation: first amount or range midpoint in tỷ.
   - Refusal: refusal phrases.
   - faithfulness = null.
   - The rules are written into `eval_summary.scoring_rules`, and `rules_version` forces a re-score after any rule change.
   - Summary carries `scoring`, `benchmark_subset`, `n_questions`, `n_questions_scored`, `n_questions_total=100`, `complete`, `coverage` per category, and `overall_common`. Empty categories show n=0 and null metrics. Pairwise uses only questions both systems answered, with n beside every CI.
   - Tests: `test_eval_rule_score.py`, `test_eval_summary.py`.
3. **Error analysis without an LLM** ✓
   - `filter_wrong` re-reads the filters from the cache; on a miss it is recorded as "không kiểm được" in `notes`.
   - `llm_hallucination` is triggered only by unsupported_citations in rule mode.
4. **Examples without `stream_answer`** ✓
   - rag: Qdrant hybrid search with filters read only from the cache.
   - graph: Cypher strings from answers.jsonl, re-executed on Neo4j.
   - valuation: `valuate()`.
   - answers: rows from answers.jsonl.
5. **Gemini provider** ✓
   - OpenAI SDK pointed at the `generativelanguage…/v1beta/openai/` endpoint; uses the `system` role and `max_tokens`.
   - Strict json_schema structured output was verified with 2 probe calls: structured call 3.0 s, stream with usage 8.3 s.
   - Groq was never called.
6. **Ablation (model-independent)** ✓ Full index, cache-only, **0 LLM calls**, 27/27 ids questions had cached filters (the filter cache key is the .env gpt-oss model), 87 s. Rerank runs were omitted because the model is not in the HF cache. Results:

   | Run | Recall@5 | nDCG@10 |
   |---|---|---|
   | dense | 0.291 | 0.281 |
   | dense+prefilter | 0.504 | 0.514 |
   | bm25 | 0.269 | 0.258 |
   | bm25+prefilter | **0.575** | **0.596** |
   | hybrid | 0.358 | 0.297 |
   | hybrid+prefilter | 0.528 | 0.551 |

7. **Signal sender** ✓ `vre.common.SENDERS` now includes p3/p4, `P3_RESULTS_READY` is added, and the CLI `--from p3|p4` works (`test_handoff_senders.py`). The signal was **not** sent, per the lead's latest instruction.
8. **Skipped per lead:** examples, README and notebooks from the flash-lite run, and the signal. A queued `report readme` was stopped and the README reverted. Flash-lite raw data was renamed so the gemini-3.8-flash rerun starts clean.

## The flash-lite run (parked, not committed)

- 36 questions (first 4 per category in file order; none unusable) × 4 systems = **144 rows, 0 errors after retries**. That is 182 + 24 + 12 (smoke) calls plus some in the killed retry, about 250 Gemini calls in total.
- Raw data is renamed to `D:\Vietnam-Real-Estates\results\eval\raw\{answers,scores}_flashlite.jsonl` and `run_meta_flashlite.json`. The gpt-oss rows are in `answers_gptoss_partial.jsonl`.
- `D:\Vietnam-Real-Estates\results\{eval_summary,error_analysis,examples}.json` still hold the flash-lite outputs (data root only, not committed). The rerun will overwrite them.
- Rule-based accuracy overall: graph 0.78 [0.64–0.92], hybrid 0.67, basic_rag 0.22, llm_only 0.11 (n=36). hybrid − basic_rag = +0.44 [0.28, 0.61]. Router 0.92.

## Issues found and fixed

- `gemini-2.5-flash-lite` (the fallback) returns 404 "no longer available to new users". When the primary model was rate-limited, the fallback produced 34 failed rows. Gemini now has no fallback, and `--retry-errors` re-runs failed rows.
- Gemini puts the retry delay in the error body ("Please retry in 54s" / `retryDelay`), not in a header. The backoff now parses it.
- The memory guard killed one retry run (exit -1). The lead attributed this to concurrent jobs; resume recovered the rows.
- Spot-checks of the rule scorer found three flaws, now fixed and tested:
  - comparisons scored 2 whenever the answer named both options;
  - "thấp hơn 13,46%" failed against a GT of -13.46;
  - "3,5 đến 6,5 tỷ" was read as 6.5.

## For the gemini-3.8-flash rerun / key rotation

- Run with: `eval run --per-category 4 --workers 1 --rule-based`, then `eval score --offline`, `eval report`, `report examples`, `report readme`. Every step goes through `run_heavy.ps1`. Set `LLM_FALLBACK_MODELS=` (empty) for no fallback. `VRE_LLM_OFFLINE=1` applies to the scoring, report and examples steps only.
- **Key rotation hook (`GEMINI_API_KEYS`)** goes in `src/vre/llm/client.py`:
  1. Add a `gemini_api_keys` Settings field.
  2. In `_new_client()`'s gemini branch, build one client per key instead of the single cached `_client` (per-key client cache keyed by index).
  3. In `_create_with_fallback()`, catch `openai.RateLimitError` whose body says `RESOURCE_EXHAUSTED`/per-day quota, then advance to the next key before advancing models.
  4. Per-minute 429s can keep the current key, because the runner's backoff already honors the in-body delay.

  Thread-safety: rotate under `_client_lock`.

## Concerns

- **C1.** The rule-based scorer is an approximation, spot-checked on 6 cells. Treat it as rule-based in slides: `scoring_rules` is in the summary and the README block.
- **C2.** Valuation scored 0 for every system on the subset. hybrid's model forecasts differ from GT by more than 15% (e.g. 4.58 vs 7.33 tỷ, using the tabular/full model). This is a real result, not a scoring bug; I checked VA01 and VA02.
- **C3.** The full unit suite gave 360 passed plus 1 that fails only when the run sets `VRE_LLM_OFFLINE=1` globally (it mocks the network). It passes without the variable, which is how CI runs: 361/361 green.
