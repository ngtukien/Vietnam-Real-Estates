# Phase 3 integration fix-ups — report

**Status:** DONE_WITH_CONCERNS
**Branch:** `feat/p3-fixups` (worktree `D:\Vietnam-Real-Estates-wt\fixups`). HEAD `77225404dc3e3838cf078e11473766736f139044` matches `origin/feat/p3-fixups`. Working tree clean. No PR opened, because the lead merges at integration.
**Concern:** `tests/integration/test_index_complete.py::test_dense_bm25_and_province_filter` and `tests/gpu/test_embed_parity.py` crash the Python process with a Windows access violation while the e5 encoder loads. The crash happens inside transformers/torch. I changed nothing in that code path, and the crash is outside my files. Details are in §3.

## 1. Fixes (commit 7722540)

| # | Item | Change | Status |
|---|---|---|---|
| 1 | `pyproject.toml` marker | Registered `"notebooks: chạy 5 notebook bằng nbclient (< 2 phút)"`. The "unknown marker" warning is gone from the notebooks run. | ✓ |
| 2 | `valuate` serves `tabular` | `src/vre/valuation/predict.py`: the signature is now `valuate(payload, settings=None)`. It always uses `SERVING_VARIANT = "tabular"` and records it in `model.variant`. Removed the `variant` argument, the "legacy default" branch and `resolve_variant`. `_features` no longer builds the NaN-PCA `full` path. I grepped every caller (`api/services/backend.py`, `src/vre/qa/pipeline.py`, `tests/unit/test_qa.py`, `tests/unit/test_valuation_features.py`, `tests/api/test_valuate.py`). Only `test_qa.py` passed `variant` or used `resolve_variant`, and I updated it. `test_valuation_features.py` already asserts `model.variant == "tabular"` after both `tabular` and `full` are trained. It now proves the serving choice, so it needed no change. `settings` stays as a test-injection parameter, and the API calls `valuate(payload)`. | ✓ |
| 3 | Geo attribution | `README.md` now reads: geoBoundaries gbOpen VNM, ADM1 Public Domain (2008), ADM2 OCHA ROAP [CC BY 3.0 IGO](https://creativecommons.org/licenses/by/3.0/igo/) (2020). The URL returned 200. The step C text in `phase-03-system-eval-webapp.md` got the same correction (data-contract §8). | ✓ |
| 4 | `Dockerfile.api` | Added `OPENBLAS_NUM_THREADS=4` to the first `ENV` block, before any pip/numpy step. | ✓ |
| 5 | B3 `_clean` in `eda.py` | Removed `_clean`. `_rows` now calls `vre.common.scalar` directly and rounds floats with the named constant `ROUND_DIGITS = 4`, so output is unchanged. I also widened the gate `tests/unit/test_graph_schema.py::test_no_duplicate_scalar_helpers` from rag/graph/hybrid/qa/llm to the whole `src/vre` tree, excluding only `common.py`, with a non-empty-scan guard. Proof: `grep -rnE "def (_value\|_clean\|scalar)\b" src api` → only `src/vre/common.py:29:def scalar`. | ✓ |
| 6 | CI web job | `.github/workflows/ci.yml`: removed the hand-started API, which ran without `VRE_CHAT_RATE_PER_MIN`, so the 20/min chat limit could throttle e2e. `playwright.config.ts` `webServer` now starts the fake API from `VRE_API_DIR=${{ github.workspace }}` with `VRE_PY=python` (job env), so the API env lives in one place: the config. The final step now uploads `web/playwright-report/` on failure instead of printing `api.log`. Not run on GitHub yet; the local e2e run uses the same config path. | ✓ (local) |

Interpretation recorded: I kept the `settings` parameter on `valuate` because tests need it to inject `data_root`. The contract only fixes `valuate(payload)` as the API call form.

## 2. Verification (real outputs)

After the coordinator's 22:04 message, every heavy step ran through `run_heavy.ps1` (global lock, free commit ≥ 6 GB). Two rows below come from runs before the lock existed, and they are labelled.

| Check | Result |
|---|---|
| `pytest -q tests/unit` | `260 passed in 50.39s` |
| `pytest tests/api` (+ test_valuation_features, test_qa, test_graph_schema), after the edits, before the lock existed | `135 passed in 14.17s` |
| `pytest -q tests/test_notebooks.py` (static + `notebooks` execution) | `27 passed, 1 warning in 35.93s`. The warning is zmq Proactor on Windows; the unknown-marker warning is gone. |
| `pytest` integration: clean_invariants + neo4j_invariants + qdrant_roundtrip | `9 passed in 16.73s` |
| `pytest tests/integration/test_index_complete.py` | **crash** `exit -1073741819` (access violation), see §3. Once in isolation with `HF_DEACTIVATE_ASYNC_LOAD=1` I got `3 passed in 17.23s`. It crashed again under the lock, with and without that variable. |
| `pytest -m gpu` (`tests/gpu/test_embed_parity.py`) | **crash** `exit -1073740791` during model load. The GPU was idle (1%, 875 MiB), so this is not a busy-GPU skip. |
| Earlier `-m "not gpu"` run, before the lock, at free commit ≈ 4.9 GB | `1 failed, 400 passed`. The failure was `XGBoostError bad_malloc: Failed to allocate 2572304 bytes`, a memory-pressure symptom. It passed when re-run alone (in the 135 above and in the 260 under the lock). |
| `ruff check src tests scripts api notebooks` | `All checks passed!` |
| `mypy src/vre api` | `Found 48 errors in 12 files (checked 91 source files)`. None are in my files. Errors per file (counting only `: error:` lines): pipeline/night.py 24, eval/runner.py 7, eval/judge.py 6, valuation/features.py 2, valuation/search_cache.py 2, and 1 each in api/services/backend.py, data/clean.py, data/ingest.py, index/embed.py, index/graph_export.py, index/qdrant_store.py and valuation/train.py. Total 48. |
| `python -m api.export_openapi --check` | exit 0 |
| `npm --prefix web ci` | `added 698 packages, and audited 699 packages in 28s` |
| `npm run lint` / `typecheck` / `test` / `build` | exit 0 / 0 / `Test Files 3 passed (3) · Tests 22 passed (22)` / `built in 705ms` |
| `npm run gen:types` → git diff | `git diff --exit-code web/src/api/types.ts` → 0, so there is no diff. `git status` showed the file as `M` only because the generator writes LF where autocrlf has CRLF. The content is identical. |
| `npm run e2e -- --workers=1` | `11 passed (19.6s)`. Ports 8000 and 5173 were free afterwards. |
| `docker compose --profile app config --quiet` | exit 0 |

### Real-mode API smoke (no fake mode, uvicorn :8000, stopped afterwards; port 8000 free after stop)
- `GET /api/health` → 200: `{"status":"ok","qdrant":true,"neo4j":true,"llm_configured":true,"version":"0.3.0"}`
- `GET /api/info` → 200: all 4 systems `available:true`, index `vre_chunks_e5_large` with `n_listings 2537920`, `n_points 2685246`, `graph_ok:true`, period 2025-06..2026-03.
- `POST /api/chat` (hybrid, `stream_tokens:false`) → 200 in 17.4 s. The router picked `graph`. Cypher ran on the first attempt against `MarketStat` for District 7, returning `median_price_m2_mil 75.17` over `n_listings 35972`. The answer cites `[Cypher]`. Usage was `input_tokens 4868`, `output_tokens 620`, `usd 0.001102`. At most 3 Groq calls (route, text2cypher, answer), within the cap of 4.
- `POST /api/valuate` (apartment, Phú Mỹ, District 7, 70 m², 2 bedrooms, asking 4 billion VND) → 200: `predicted_price_m2_mil 51.52`, interval 3.07–4.31 billion VND, verdict `fair`, ward median 62.0 over `comparables_n 6191`, 8 SHAP rows, `model.variant "tabular"` (path `models/valuation/xgb_tabular_sale.json`, test_mape 0.240). This confirms item 2 end to end.
- The question text was mangled in transit (`Cn h�…`). PowerShell 5.1 read the smoke script, which has no BOM, as ANSI. This is a test-harness artifact, not an API bug. The district/province keys in the Cypher were also mangled for the same reason, yet the query still matched.

## 3. Problems outside my files (reported, not edited)

1. **Encoder load crash (Windows access violation).** Stack: `transformers/core_model_loading.py:_materialize_copy → torch/storage.py:471 __getitem__`, called from `sentence_transformers … from_pretrained`. It hits `tests/integration/test_index_complete.py::test_dense_bm25_and_province_filter` (through `vre.index.embed.get_encoder`) and `tests/gpu/test_embed_parity.py`. It is intermittent: one isolated pass with `HF_DEACTIVATE_ASYNC_LOAD=1`, crashes otherwise, including under the lock at free commit 6.2–6.9 GB. The real-mode API did load the encoder successfully for `/api/info` and `/api/chat`. Likely causes are memory/commit pressure (vmmemWSL ≈ 16 GB) or a transformers 5.x safetensors-materialize bug on Windows. The fix belongs to whoever owns `src/vre/index/embed.py`. One option is to load with `HF_DEACTIVATE_ASYNC_LOAD=1` set, or a lower-copy load path, and then re-measure.
2. **mypy:** 48 errors, mostly in `pipeline/night.py` (24), `eval/runner.py` (7) and `eval/judge.py` (6). Files owned by the parallel lanes (`src/vre/eval/**`, `scripts/vre_pipeline.py`) were not touched. `api/services/backend.py:74` passes `str` where the `SystemId` Literal is expected; that belongs to the API lane.
3. **CI lint job** runs `ruff check src tests scripts api` without `notebooks`. The notebooks are clean locally. I left the lint job unchanged because it was not in scope.
4. **Memory:** free commit fell to 4.9 GB before the coordinator's lock existed, producing the xgboost bad_malloc above. Docker's WSL VM holds about 16 GB.

## 4. Next steps
- Owner of `vre.index.embed`: make encoder loading robust on Windows, then re-run `tests/integration/test_index_complete.py` and `-m gpu` under the lock.
- The lead merges `feat/p3-fixups`. The first GitHub CI run validates the reworked web job on ubuntu.
