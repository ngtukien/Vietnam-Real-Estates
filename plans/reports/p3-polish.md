# P3 polish: filter_relaxed badge + README notebooks

Branch `feat/p3-polish` (worktree `D:\Vietnam-Real-Estates-wt\polish`), pushed. No PR opened (not requested).
Commits: `c7ee856` web badge, `9a03650` README.

## 1. Web: filter_relaxed badge (c7ee856)
- New `web/src/lib/filterNote.ts`: `describeFilterNote(note)` returns `{kind:"dropped"}` for `filter_dropped`,
  `{kind:"relaxed", fields, label}` for `filter_relaxed:<a,b>`, and null otherwise (including an empty field list).
  Vietnamese field labels: project→dự án, street→đường, bedrooms→số phòng ngủ, price→giá, area→diện tích,
  province→tỉnh/thành, district→quận/huyện, ward→phường/xã. These are the labels from `_RELAX_STEPS` plus the place
  fields in `src/vre/rag/filters.py`. An unknown field keeps its raw name.
- `RunView.tsx` `FiltersLine`: dropped shows a warning badge "Đã bỏ lọc vì 0 kết quả" (`data-testid=filter-dropped-badge`).
  Relaxed shows an outline badge "Đã nới lọc: dự án, …" (`data-testid=filter-relaxed-badge`) with an explanatory `title`.
  The badge is plain text inside the filters line, so screen readers read it. It does not rely on color, and the
  outline tone uses `text-foreground`.
- `web/src/lib/filterNote.test.ts`: 4 vitest cases (null/unknown/empty, dropped, single field, multi + unknown field).
- I did not change e2e or fixtures. Adding one would need a new mock-api SSE route, which was not cheap, so e2e was not run.

## 2. README notebooks section (9a03650)
I replaced only the `## Notebook` section. It now covers: the 6 notebooks (00 to 05) with their contents, the build
from `notebooks/build_notebooks.py` (no hand edits), the cell kinds ẨN/CACHE/LIVE, `results/nb_cache/` (code root,
committed) vs `results/*.json` (`VRE_DATA_ROOT`), and a table for `VRE_NB_OFFLINE` / `VRE_NB_REBUILD` / `VRE_NB_STRICT`.
It states that there are no real LLM calls (`llm_cache_only()`, disk cache only) and gives the rebuild commands. It has
no result numbers. The `<!-- results:start/end -->` block is untouched.
- Not touched (out of scope): stale references elsewhere in the README to old notebook names `01_eda` / `02_rag`
  (lines in Kiến trúc, Pipeline, Hạn chế) and "cả năm notebook" in Kiểm thử. They need a follow-up.

## Verify (real output)
- `npm --prefix web ci` (the worktree had no node_modules), then `lint` passed (eslint, max-warnings 0), `typecheck`
  passed, and `test` reported `Test Files 4 passed (4) · Tests 26 passed (26)`. `build` ended with `✓ built in 3.73s`.
- `pytest -q tests/api tests/test_notebooks.py -m "not notebooks"` (after `vre_env.ps1`): **2 failed, 138 passed,
  1 deselected**. The failures are `test_saved_notebook_equals_regenerated[02_retrieval_generation]` and `[04_graphrag_eval]`.
  The diff is in headings that are filled from results, e.g. P2-11 "đúng 100,0% … (n = 1)" vs "(chưa có kết quả eval)"
  and P4-06 "60,0% (n = 5)" vs "3/5 câu …". The saved notebooks were built against a different `results/` from the one
  `VRE_DATA_ROOT` (main checkout) now holds. This is not from this branch, which touches no notebook, results or `src/`
  file. Fix: rebuild 02 and 04 on the intended `results/` and commit.

**Status:** DONE_WITH_CONCERNS
**Concerns:** 2 pre-existing notebook-regeneration test failures (data drift, see above). Stale old-notebook-name
references remain outside the README notebooks section.
