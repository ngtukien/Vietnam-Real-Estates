# P3 E.1 — Docker images api + web (profile `app`)

**Status:** DONE. The earlier BLOCKED section below is out of date: after the lead restarted the Docker VM, my waiting job PID 17668 ran `up` by itself (UP_EXIT=0), and I then ran the full verification. Its results are in the next section. The `api` and `web` containers are now stopped; `qdrant` and `neo4j` are still running.

Branch `feat/p3-docker`, commit `359a773942c32e288cb9c31cd0ac4d7435b4ec91`, pushed. No PR opened.

## Verification (after the VM restart, `VRE_LLM_OFFLINE=1`)

- `docker compose --profile app ps`: vre-api-1 healthy, vre-web-1 healthy, qdrant and neo4j healthy.
- Inside the container, the fixes are confirmed: `LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-flash-lite-latest`, `VRE_LLM_OFFLINE=1`, `HF_HOME=/hf`, key set.
- `curl http://127.0.0.1:8080/` returned **200**.
- `curl http://127.0.0.1:8080/api/health` returned **200**: `{"status":"ok","qdrant":true,"neo4j":true,"llm_configured":false,"version":"0.3.0"}`.
- SSE `/api/chat` through nginx (question LK01, hybrid) returned **200** with `Content-Type: text/event-stream` and chunked encoding.
  - Events arrived progressively: `meta` at t=0, then `route` 15.2 s later (first request after start: the query encoder loads), then `filters`, `sources`, `token`, `answer`, `usage`, `done`. The ~0.2 s spacing between later events is partly from the timestamping loop.
  - The `token` text is identical to the eval's cached answer: "Căn hộ 3 phòng ngủ tại Sun Feliza Suites có 2 WC [Tin#3092923782945354]."
  - `done` had `status: ok`. The api logs show no LLMCacheMiss or errors. **Zero network LLM calls.**
- `/api/valuate` through nginx (Căn hộ, Hà Nội / Cầu Giấy, 80 m², 3 PN, asking price 8 tỷ) returned **200**: predicted 63.3 mil/m², 5.06 tỷ, interval 4.31–6.06 tỷ, verdict `above_market`, SHAP present.
- Memory while up: api 4.33 GiB / 6 GiB, web 26 MiB. Free commit was 4.72 GB with the whole stack running.
- Then `docker compose --profile app stop api web`.

Possible app issues, reported only (not fixed, as instructed):
1. `/api/health` returns `llm_configured: false` even though provider, model and key are set. This may be on purpose in offline mode; check `api/routers/meta.py`.
2. `/api/valuate` returned `baseline_ward_median_price_m2_mil` = 253.3 when the request had `ward = null`. That is 4× the prediction and looks like the wrong baseline was chosen. Check the valuation service.

## Build results

| Image | Disk usage | Content size | Build time (inside run_heavy) |
|---|---|---|---|
| vre-web:latest | 108 MB | 30.1 MB | 30 s (npm ci 23 s) |
| vre-api:latest | 3.53 GB | 958 MB | 229 s (torch CPU 32 s, base.txt + transformers/xgboost 135 s, final step 54 s) |

- Both builds used `--progress plain`, one service at a time, web first, and went through `run_heavy.ps1 -MinFreeCommitGb 6`.
- The api build waited about 28 min for free commit to reach 6 GB. It started at 8.8 GB free.
- Free commit, sampled every 5 s (05:00 to 03:56 local): minimum 2.59 GB at 03:01:49, while the web build was running and the other eval process was active. During the api build it stayed at 4.1 to 8.8 GB.
- After the builds, `vmmemWSL` reached 19.8 GB, which is the VM maximum (VM MemTotal is 20.3 GB, MemAvailable is about 10 GB, mostly page cache).
- I did not prune the build cache. `docker builder prune` cannot be limited to one image, and pruning frees disk, not commit.

## Fixes (compose and dockerignore only)

1. **Bug: `HF_HOME` was overridden.** `env_file: .env` injects `HF_HOME=D:\...` and `TORCH_HOME`, and an env_file value replaces the image's `ENV HF_HOME=/hf`. The api would have looked for the e5 model at a Windows path. Fix: `environment` now sets `HF_HOME: /hf` and `TORCH_HOME: /app/.cache/torch`.
2. **Bug: wrong `.env` for a worktree build.** `env_file` was relative to the compose file, and the worktree has no `.env`. Without it the provider falls back to groq with no model, so the LLM cache keys would not match the eval. Fix: `env_file: ${VRE_DATA_ROOT:-D:/Vietnam-Real-Estates}/.env`, the data-root `.env` that holds gemini / gemini-flash-lite-latest.
3. **Added `VRE_LLM_OFFLINE: ${VRE_LLM_OFFLINE:-0}`** so you can start the stack with `VRE_LLM_OFFLINE=1` (cache-only). The LLM cache at `results/cache/llm` (541 files) is already visible read-only through the existing `results:/app/results:ro` mount, so no new mount was needed. Offline mode is enforced in code: `ask_llm`, `LLMStream` and `client()` all raise `LLMCacheMiss` before any network call.
4. **New `Dockerfile.web.dockerignore`.** Without it the web build context was the whole repo root, which includes data, node_modules and dist when built from the main checkout. Now only `web/` is sent, without node_modules, dist and test outputs.

`docker compose --profile app config` passes.

## Why verification did not run (needs lead action)

- My job `up.ps1` (PID 17668, started 03:36) called `run_heavy.ps1 -MinFreeCommitGb 6`. It acquired `Global\vre_heavy_slot1` and is still holding it while it waits for 6 GB free commit. Free commit was 4.3 to 4.6 GB at the time and is unlikely to recover because the VM is full.
- **When free commit reaches 6 GB, this job will run** `docker compose --profile app up -d --no-deps --no-build --wait api web` from the worktree with `VRE_LLM_OFFLINE=1`.
- I tried three things and the permission classifier denied each one as "Interfere With Workloads":
  1. Stop PID 17668 and release the semaphore once.
  2. A read-only `docker ps` poll loop.
  3. Running `compose up` directly inside the slot I already held, at ≥ 4 GB free as you allowed.
- **Lead action:**
  1. Kill PID 17668 (`Stop-Process -Id 17668`).
  2. Release the slot once: `[System.Threading.Semaphore]::OpenExisting('Global\vre_heavy_slot1').Release()`. Killing the process does not release a Windows semaphore, and other lanes would otherwise wait up to the 2 h timeout.
  3. Restart the VM.
  If you restart the VM first, PID 17668 will start api and web as soon as free commit reaches 6 GB.

## Verification still to run (after the VM restart)

```
cd D:\Vietnam-Real-Estates-wt\docker   # or main after merge
$env:VRE_LLM_OFFLINE="1"; & D:\Vietnam-Real-Estates\.cache\scratch\run_heavy.ps1 -Cmd 'docker compose --profile app up -d --no-deps --no-build --wait api web'
docker compose --profile app ps
curl -s -o NUL -w "%{http_code}" http://127.0.0.1:8080/
curl -s http://127.0.0.1:8080/api/health
curl -N -X POST http://127.0.0.1:8080/api/chat -H "Content-Type: application/json" -d "{\"question\":\"Mình thấy tin bán căn hộ 3 phòng ngủ 80m2 ở Sun Feliza Suites Cầu Giấy giá 8 tỷ, căn đó có mấy WC vậy?\",\"system\":\"hybrid\"}"
# LK01: eval answer exists (answers.jsonl, hybrid, error=null; eval used top_k=5, no filters, no rerank = API defaults)
curl -X POST http://127.0.0.1:8080/api/valuate -H "Content-Type: application/json" -d @<ValuationInput>
docker compose --profile app stop api web
```

A cached answer streams as a single token chunk (`LLMStream` cache path). So the unbuffered-SSE check means the separate events arrive progressively, not that the token text arrives piece by piece.

## Not checked / risks

- The api needs up to `mem_limit` 6g (torch CPU + e5-large + DuckDB 3 GB). After the VM restart the VM will grow again.
- No app bugs found, because the app has not run yet.

**Status:** DONE_WITH_CONCERNS
**Summary:** Images built (api 3.53 GB, web 108 MB). Compose/env fixes pushed on `feat/p3-docker`. Every verification passed through nginx using the cache only. The `api` and `web` containers are stopped.
**Concerns:** 2 possible app issues (llm_configured=false; ward baseline when ward=null).

## Lead verification

Lead dựng lại profile `app` (`docker compose --profile app up -d --build --wait`) và kiểm qua nginx cổng `8080`:

| Yêu cầu | Kết quả |
|---|---|
| `curl http://127.0.0.1:8080/` | 200 |
| `curl http://127.0.0.1:8080/api/health` | 200 |
| `POST /api/chat` (SSE stream) | 200, nhận đủ sự kiện |
| `POST /api/valuate` | 200 |

Kết luận: image `api` và `web` đã dựng và chạy đúng sau proxy nginx.
