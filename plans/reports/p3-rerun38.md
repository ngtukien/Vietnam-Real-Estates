# P3 rerun: gemini-3.8-flash attempt (stopped), then flash-lite subset published

**Status:** DONE_WITH_CONCERNS. The user decided the official eval is the finished **gemini-flash-lite-latest** run (36 questions × 4 systems, 0 errors). The gemini-3.8-flash rerun was stopped and its rows parked.

Branch `feat/p3-integration` (worktree `D:\Vietnam-Real-Estates-wt\int`). HEAD `fcb41f2c416f36e67ac63158117f5640b923f46c` = `origin/feat/p3-integration` (checked with `git ls-remote`).

## Commits

| SHA | What |
|---|---|
| `8130077` | Gemini key rotation. `GEMINI_API_KEYS` (comma list) followed by `GEMINI_API_KEY`, with duplicates dropped. One client per key. A per-day `RESOURCE_EXHAUSTED` switches to the next key under `_client_lock`; a per-minute 429 keeps the key and is raised to the runner backoff. Logs carry only key indices. Tests: `tests/unit/test_gemini_key_rotation.py` (mocks; checks that no key appears in the logs). |
| `49b87d3` | Scorer fixes from the lead. TM02: `MM/YYYY`, `M/YYYY`, `tháng M/YYYY`, `tháng M năm YYYY` and `YYYY-MM` all match a `YYYY-MM` ground truth. CP02: a refusal on an answerable question (value or valuation) scores 0 before any matching is done. The comparison category always uses the winner rule, because CP02 writes "q3", so the question never names "Quận 3" literally. `rules_version` changed, so all 288 score rows were re-scored. `cost.<system>.fallback_rate` was added to eval_summary, plus a "Fallback" column in the README cost table. A unit test covers each. |
| `79fe53b` | Gemini SDK `max_retries=0`. Failed calls (429 and 503) count against the 20/day free quota; see below. |
| `fcb41f2` | Published the results: `results/{eval_summary,error_analysis,examples}.json`, README results block, notebooks 00/02/03/04, and `results/nb_cache/*`. The P4-09 heading now handles a rule-based summary that has no faithfulness. `ablation.json` and `benchmark_gt.json` were unchanged (identical copies). |

## gemini-3.8-flash attempt (parked, not published)

- **Smoke:** 1 question × 4 systems gave 4 rows, 0 errors and 0 runner 429s. That took 6 API calls and 310 s.
- **Full run** (`--per-category 4 --workers 2`): I stopped it after 15 rows.
- **`--per-category 1 --retry-errors` run:** most rows failed.
- **Measured limits.** The free tier for gemini-3.8-flash allows 5 RPM and **20 requests/day per key**, under quotaId `GenerateRequestsPerDayPerProjectPerModel-FreeTier`. The quota resets about 5 h later (~07:00). The model was also overloaded: most failed calls were 503 "high demand" or timeouts.
- **Why quota ran out so fast.** The OpenAI SDK's own retries (4 per 503 or 429) used up keys 1–6 in about 15 minutes. That is what `79fe53b` fixes.
- **Keys rotated:** 6 rotations in the first run and 8 in the second. The second process starts again at key 1, so it re-detects the keys that were already exhausted. Keys 1–9 were exhausted at the stop; key 10 was still answering.
- **Rows:** 44 lines saved to `results/eval/raw/answers_38flash_partial.jsonl` plus `run_meta_38flash_partial.json`. There is no scores file for them. Groq was never called. No more Gemini calls were made after the user's decision.

## Restored flash-lite raw data and offline steps

- **Restored files:** `answers.jsonl`, `scores.jsonl` and `run_meta.json` were copied back from the `*_flashlite.*` files; the originals are kept as well.
- **Offline steps** (`VRE_LLM_OFFLINE=1`): `eval score --offline`, `eval report`, `report examples` (rag 5, graph 5, valuation 5, answers 8; no graph example for `aggregate_lowest`), `report readme` (changed, 75 lines), notebooks 02, 03, 04 and 00.

### Accuracy (rule-based, n=36, generator `gemini-flash-lite-latest`, from `eval_summary.json`)

| System | Accuracy | CI 95% | fallback_rate | error_rate |
|---|---|---|---|---|
| llm_only | 0.111 | [0.028, 0.222] | 0.0 | 0.0 |
| basic_rag | 0.222 | [0.083, 0.361] | 0.0 | 0.0 |
| graph | **0.833** | [0.694, 0.944] | 0.0 | 0.0 |
| hybrid | 0.694 | [0.528, 0.833] | 0.056 | 0.0 |

Router accuracy is 0.917.

| Pair | Difference | CI 95% |
|---|---|---|
| hybrid − basic_rag | +0.472 | [0.306, 0.639] |
| basic_rag − llm_only | +0.111 | [−0.083, 0.306] |
| hybrid − graph | −0.139 | [−0.250, −0.028] |

Other facts from the summary: `complete=true`, 36/36 questions scored, and 11 scoring rules listed. Graph and hybrid each moved by about +0.03 compared with the earlier flash-lite numbers, because of the scorer fixes.

TM02 and CP02 after the fixes:

| System | TM02 | CP02 |
|---|---|---|
| basic_rag | 2 | 0 (refusal) |
| graph | 2 | 2 |
| hybrid | 2 | 0 (refusal) |
| llm_only | 0 (refusal) | 2 |

## Verification

| Check | Result |
|---|---|
| `pytest -q tests/unit` (no `VRE_LLM_OFFLINE`) | 382 passed |
| `pytest -q tests/test_notebooks.py` | 39 passed. Notebook 00 was stale (its regenerated output differed in 1 cell), so I rebuilt it. |
| `ruff check src tests scripts api` (CI scope) | clean |
| `mypy src/vre api` | 0 errors in 99 files |

## Signal

`D:\Vietnam-Real-Estates\results\handoff\P3_RESULTS_READY.json` was written through the CLI (`signal P3_RESULTS_READY --from p3`, seq 1). Its payload is `eval: "subset36_gemini-flash-lite-latest_rule_based"`, `n_questions: 36`, the overall table above, and commit `fcb41f2…`.

## Concerns

- **C1.** Notebook 04 fails when run in live mode (`VRE_NB_OFFLINE` unset, `VRE_NB_STRICT=1`). In cell P4-06, `except LLMCacheMiss` does not catch the `LLMCacheMiss` raised by `vre.llm.client`; this looks like a class-identity mismatch inside the kernel. I built 04 and 00 with `VRE_NB_OFFLINE=1`, which reads cache cells the way CI does. 02 and 03 were built live. I have not investigated further.
- **C2.** The new refusal-first rule also zeroes answers that hedge ("không có dữ liệu…, có thể khoảng X"). That is intentional per the lead, and is recorded in `scoring_rules`.
- **C3.** `eval_summary.git_sha` is `79fe53b`, the code SHA when the report was generated. The published commit is `fcb41f2`.
