# p3-mypy: CI lint job (ruff + mypy)

Branch `feat/p3-mypy`, worktree `D:\Vietnam-Real-Estates-wt\mypy`, commit `9f2ecfffbc86b0d96e7aeca4e279141bc9812dd2`, pushed to origin. No PR was opened because the brief did not ask for one.

## Result

| Check | Before | After |
|---|---|---|
| `mypy src/vre api` on CI (GitHub run 38781414749, feat/p3-integration) | 35 errors in 11 files | not re-run on CI yet |
| Local, emulating CI (`--platform linux`, pandas-stubs masked through MYPYPATH) | 36 errors in 11 files | **1 error in 1 file** (excluded file) |
| Local, as installed (win32, pandas-stubs 3.0.5) | 35 errors in 10 files | **1 error in 1 file** (excluded file) |
| `ruff check src tests scripts api` | passes | passes |
| `pytest -q tests/unit` (run through run_heavy.ps1) | n/a | **338 passed** |

## What's left: errors in files I was not allowed to edit

- `src/vre/index/embed.py:105`: `Unsupported operand types for + ("int" and "str")  [operator]` (the left operand is `str | int`). Whoever owns `embed.py` has to fix this before CI turns green.
- `src/vre/eval/**`, `src/vre/llm/**`, `src/vre/index/weights.py`, `src/vre/rag/query_encoder.py`, `notebooks/**`: **0 errors** on this branch. CI on feat/p3-integration also showed none in `eval/*`.

## Fixes (behavior unchanged)

- `pipeline/night.py`: the attributes are now typed (`failure: BaseException | None`, `ledger: Ledger | None`, `encoder: Encoder | None`). Encoder is imported under TYPE_CHECKING so torch does not load. `qc` is now typed `Any`. A new `require_ledger()` raises a clear RuntimeError if called before `stages.prepare`, and each stage now holds a narrowed local `ledger`. The queues are typed. `ledger` stays an Optional attribute because `report.finish` checks `job.ledger is None`.
- `data/duck.py`: new `one_row(cursor)` helper. It raises when `fetchone()` returns None and is used in `data/ingest.py:110`, `valuation/train.py:148` and `index/graph_export.py:158`.
- `pipeline/report.py`: `free_commit_gb()` now has a `sys.platform != "win32"` guard. Its only caller already branches on win32.
- `data/clean.py:109`: the "keep" value is now an object-dtype None array (`np.array(None, dtype=object)`) and the masks go through `.to_numpy(bool)`. I checked that the values and the object dtype match the old expression.
- `qa/cards.py:57`: `cast(...)` replaces a `type: ignore`. The ignore was "unused" on CI only because CI has no pandas-stubs.
- `valuation/search_cache.py`: annotated `SPACE: dict[str, list[int | float]]`.
- `index/qdrant_store.py`: the alias ops list is typed `list[DeleteAliasOperation | CreateAliasOperation]`.
- `valuation/features.py`: `_smoothed` now returns `dict[Any, float]` (the keys are not always str). The local in `target_encode_train` is renamed `is_valid` so it no longer reuses the Series parameter. These two errors only show up when pandas-stubs is installed.
- `api/services/backend.py:74`: `cast("QaSystemId", system)`. The router has already validated `system` against `api.schemas.SystemId`, which has the same 4 values.

No `# type: ignore` was added and the mypy config was not changed.

## Note

Local and CI environments are different: pandas-stubs is installed in the local `.venv` but is not listed in `requirements/*.txt`, so CI type-checks pandas as `Any`. The code now passes in both setups. Adding `pandas-stubs` to `requirements/dev.txt` would make the two match. I did not make that change.

**Status:** DONE_WITH_CONCERNS
**Summary:** Every mypy error in files this branch is allowed to touch is fixed. Ruff passes and 338 unit tests pass.
**Concerns:** CI lint stays red until another branch fixes `src/vre/index/embed.py:105`.
