# P3 step E.2: webapp on the real backend + `e2e:real`

Branch `feat/p3-e2ereal` (worktree `D:\Vietnam-Real-Estates-wt\e2ereal`). Two commits, both pushed: `c56cdfca0ce4458049d75b26c4466b7d88b94290` (scenario + config) and `eb3bf64e64b92b62066d4f86db4b6a2995c75c81` (assert structure, not generator wording). No PR opened (not asked for). The working tree is clean and the lockfile is unchanged.

## What was added
- `web/e2e/real.spec.ts`: 4 scenarios against the real backend. They assert structure only, so they keep passing if the generator model changes:
  - **Chat hybrid AG01** (aggregate): route badge, Cypher panel containing "Cầu Giấy", non-empty answer, usage says "từ cache", no run-error.
  - **Chat hybrid LK01** (lookup): answer cites `Tin#…`, source cards shown, and the ground-truth listing `3092923782945354` is among them, "từ cache".
  - **Compare on AG01**: 4 non-empty columns with no errors, then compare-done.
  - **Valuation** (HCM / Quận 7 / 70 m² / asking price 3 tỷ, no LLM): result and verdict.
- `web/playwright.real.config.ts`: runs only `real.spec.ts` and starts no webServer. baseURL is `VRE_WEB_URL` (default `http://127.0.0.1:4173`, the vite preview, which proxies `/api` to :8000).
- `web/package.json`: `"e2e:real": "playwright test -c playwright.real.config.ts"`.
- `web/playwright.config.ts`: `testIgnore: "real.spec.ts"`, so the default `e2e` run still lists 11 tests in 2 files.
- `web/tsconfig.json`: includes the new config. Typecheck and lint both exit 0.

## Final run (lead GO, model gemini-flash-lite-latest)
Environment: `VRE_LLM_OFFLINE=1`, `LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-flash-lite-latest`, `OPENBLAS_NUM_THREADS=4`, `CHAT_RATE_PER_MIN=1000`. Started directly, not via run_heavy, as the lead instructed. Free commit before start was 8.9 GB.

| | URL | PID |
|---|---|---|
| API (uvicorn) | http://127.0.0.1:8000 | 12468, killed at about 19:12:30Z; restarted as **37592** |
| Web (vite preview via cmd wrapper, killed with `/T`) | http://127.0.0.1:4173 | **42772** |

```
Running 4 tests using 1 worker
  ok 1 › thật — chat hybrid, câu aggregate: route graph + Cypher + câu trả lời từ cache (12.2s)
  ok 2 › thật — chat hybrid, câu lookup: route basic_rag + nguồn + trích dẫn tin (1.2s)
  ok 3 › thật — so sánh 4 hệ trên AG01: 4 cột, cả 4 trả lời từ cache (1.5s)
  ok 4 › thật — định giá (không dùng LLM): kết quả + nhận định giá chào (6.1s)
  4 passed (21.9s)
```
Zero LLM network calls:
- Every chat assertion checks for "từ cache".
- `api.err.log` has 0 `LLMCacheMiss` and 0 tracebacks.

**Signal:** `D:\Vietnam-Real-Estates\results\handoff\P3_WEBAPP_READY.json`
- seq **2** was written at 2026-10-15T19:11:52Z. **Re-signalled as seq 3** at 19:25:02Z with `api_pid` 37592 and the same `until`, **2026-10-15T20:11:52Z**.
- Written atomically (tmp then `os.replace`), with a plain JSON array.
- Payload also carries `llm_model`, `api_pid` and `web_pid`.
- The servers are held until `until`, then stopped by the script's `finally` (taskkill /T).

Session script, logs and the `stop.flag` early-stop switch are in the session scratchpad: `C:\Users\Nghaiz\AppData\Local\Temp\claude\d--Vietnam-Real-Estates\78375574-07f7-4335-811e-4c4bc63b3baf\scratchpad\` (`locked_session.ps1`, `write_signal.py`, `session.log`, `e2e-real.1.log`, `api.err.log`).

**API kill:** uvicorn 12468 stopped answering about a minute after the e2e passed. The keep-alive loop in the session script logged "Unable to connect" from 19:12:56Z, and `api.err.log` ends cleanly with no traceback and no shutdown line, so the process was killed from outside. The likely cause is the machine memory guard, as at 00:58. Free commit afterwards was 8.9 GB. I restarted it with the same environment (`restart_api.ps1`); a cached chat through the 4173 proxy returned `cached:true`. I stop 37592 at `until` myself, because the session script's `finally` only knew the old PID.

## Cached questions (in the signal)
| id | system | question |
|---|---|---|
| AG01 | hybrid (route graph) | Giá trung vị mỗi m2 căn hộ chung cư ở Cầu Giấy là bao nhiêu? |
| LK01 | hybrid (route basic_rag) | Mình thấy tin bán căn hộ 3 phòng ngủ 80m2 ở Sun Feliza Suites Cầu Giấy giá 8 tỷ, căn đó có mấy WC vậy? |
| AG01 | llm_only / basic_rag / graph | (same AG01 text; used by compare) |

Any of the 36 ids in `run_meta_flashlite.json` should also hit the cache with these settings (top_k 5, no rerank, effort low).

## History
1. The first API start (PID 22072) died with `MemoryError` while loading the query-encoder tokenizer. Commit memory was nearly exhausted by parallel lanes; the lead's guard also killed a uvicorn at 00:58 local. One query-encoder load takes the API process from about 1.6 GB to 4.1 GB.
2. Session 1 ran under run_heavy on the flash-lite cache and passed 4/4 (API 26564, web 26032, signal seq 1). It was stopped at 18:18Z on the lead's HOLD, and that signal was renamed to `P3_WEBAPP_READY.flashlite-superseded.json`. That file can be deleted.
3. Spec made model-independent (commit `eb3bf64`). A 3.8-flash waiter (PID 8252) was queued, then killed when the lead sent GO for flash-lite.

## Backend bugs (reported, not patched)
1. **`/api/health` reports `llm_configured:false` for gemini.** `api/services/backend.py:108` maps the provider to a key with only groq/openai/anthropic. `gemini_api_key` is missing, although `vre.qa._llm_configured` includes it. Chat works because it uses `qa`, but health tells the UI that the LLM is not configured.

## Coverage gaps
- The chat SSE-error and Cypher-error paths are covered only by the mocked suite, not on the real backend.
- Map, market and benchmark pages are not in `e2e:real`.

**Status:** DONE_WITH_CONCERNS
**Summary:** `e2e:real` passes 4/4 against the real backend with zero LLM network calls. Signal seq 2 is written, the servers are up until 20:11:52Z and then stopped, and the commits are pushed.
**Concerns:** backend bug 1 (health ignores the gemini key).
