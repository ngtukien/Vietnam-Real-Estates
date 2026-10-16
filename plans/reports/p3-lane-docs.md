# Phase 3 — Lane D (notebook, README, CI), phần 1

**Status:** DONE_WITH_CONCERNS
team-tools-unavailable: applied-from-prompt

Nhánh `feat/p3-docs` (worktree `D:\Vietnam-Real-Estates-wt\docs`), commit `d9cd4a94aa3d28127f0da81d7d8162839a2fae5b`, đã push lên `origin/feat/p3-docs`. Chưa mở PR: lead tích hợp ở bước E.

## Đã làm

| # | Hạng mục | Trạng thái |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb`, `03_knowledge_graph.ipynb`, `05_valuation.ipynb`: đầy đủ, đã chạy và lưu kèm output. 03 đọc `examples.graph` nếu có, nếu không thì in ô "chưa có — sinh bằng `python scripts/vre_pipeline.py report examples`" (do code sinh) | ✓ |
| 2 | `02_rag.ipynb` (embedding benchmark + lý do chọn e5-large tính từ JSON, cấu hình cố định `model_slug`/`embed.token_budget`, night_report: các bước, counters, peaks, qdrant, errors; ablation + examples.rag có guard) và `04_graphrag_eval.ipynb` (overall, 9 nhóm × hệ kèm CI và error bar, pairwise, retrieval, ragas, router confusion, cost, error_analysis, examples.answers; mọi phần có guard theo hình §7.7/§7.8) | ✓ |
| 3 | Notebook cũ: trên nhánh này không còn (Phase 1 đã xoá; `git ls-files` không có `.ipynb` nào trước commit). Import chỉ `json, pathlib, os, pandas, matplotlib, IPython.display`. Ô markdown không chứa số; mọi số do code in từ JSON. Thư mục kết quả = `VRE_DATA_ROOT` hoặc gốc repo | ✓ |
| 4 | `tests/test_notebooks.py`: đủ 5 file; import cho phép + cấm magic `!`/`%`; không `[ĐIỀN SAU]`; markdown không có số gõ tay; đã lưu kèm output và không có output lỗi; output không chứa số điện thoại / khoá Groq, OpenAI, Anthropic, GitHub, AWS / đường dẫn máy. Test `notebooks` chạy 5 notebook bằng nbclient trên `results/` thật VÀ trên bản sao có thêm eval_summary/ablation/error_analysis/examples giả đúng khoá §7.7–§7.8 (để mọi nhánh "đã có dữ liệu" của 02/03/04 được chạy trước khi eval thật xong; ở bộ này không được còn ô "chưa có"), tổng < 120 s | ✓ |
| 5 | `README.md` viết lại bằng tiếng Việt theo D.3 (mục tiêu, kiến trúc Mermaid, dữ liệu + CC BY-NC 4.0 + ghi công geoBoundaries CC BY 4.0, ổ D + `vre_env.ps1` + `docker-data/`, lệnh pipeline, lượt chạy dài + resume, eval, API/web/Docker, notebook, khối `<!-- results:start -->`/`<!-- results:end -->` rỗng, kiểm thử/CI, hạn chế). Không có số kết quả gõ tay | ✓ |
| 6 | `.github/workflows/ci.yml` (đã `git rm` `tests.yml`): không `paths:`, concurrency cancel-in-progress, bỏ commit của `t1k-ci-sync[bot]`, `VRE_LOAD_DOTENV=0`; jobs `lint` (ruff `src tests scripts api` + mypy `src/vre api`, 5'), `unit` (dev.txt + `pip install -e . --no-deps`, `-m "not integration and not gpu and not notebooks"`, `--cov=vre --cov=api --cov-fail-under=70`, 10'), `notebooks` (10'), `web` (Node 24, npm ci/lint/typecheck/test/build, Playwright chromium, API fake `VRE_FAKE_BACKENDS=1 python -m uvicorn api.main:app --host 127.0.0.1 --port 8000` chờ `/api/health`, `npm run e2e`, 20'). `requirements/dev.txt` thêm `nbclient`, `ipykernel`, `xgboost` | ✓ |

Thêm (trong `notebooks/**`): `notebooks/build_notebooks.py` sinh nguồn 5 notebook + chạy + lưu output (bỏ dấu thời gian thực thi để diff ổn định).
Chạy lại sau khi eval xong: `& $VRE_PY notebooks/build_notebooks.py 02 03 04`.

## Verify (output thật)

- `& $VRE_PY notebooks/build_notebooks.py` → lưu đủ 5 notebook, toàn bộ ~16 s.
- `& $VRE_PY -m pytest -q tests/test_notebooks.py` → `27 passed, 2 warnings in 39.68s` (cảnh báo: marker `notebooks` chưa đăng ký; RuntimeWarning của zmq trên Windows).
- `& $VRE_PY -m pytest -q tests/unit` → `180 passed in 41.27s`.
- `& $VRE_PY -m ruff check tests notebooks` → `All checks passed!`.
- YAML: `yaml.safe_load(ci.yml)` OK; jobs `['lint', 'unit', 'notebooks', 'web']`, timeout `{lint: 5, unit: 10, notebooks: 10, web: 20}`, không có `paths`.
- Kiểm regex của test đi đỏ khi cần: `lh 0901234567` và `0901.234.567` bị bắt; `3.500.744`, id 16 chữ số, `0,711` không bị bắt; markdown `chiếm 70% … 1.000 tin` bị bắt `['70', '1.000']`.
- README, chữ số ngoài khối results / code span / code block / URL: chỉ còn `4.0` trong tên giấy phép "CC BY-NC 4.0" và "CC BY 4.0" (phiên bản giấy phép, không phải kết quả). Chữ số nằm trong từ (`e5`, `Neo4j`) không tính.

## Cần lead quyết / làm

1. **Đăng ký marker** trong `pyproject.toml` `[tool.pytest.ini_options].markers`: `"notebooks: chạy 5 notebook bằng nbclient (< 2 phút)"`. Hiện chỉ là cảnh báo `PytestUnknownMarkWarning`; nếu sau này bật `--strict-markers` sẽ thành lỗi.
2. **CI trên nhánh này một mình sẽ đỏ** ở `lint` (`ruff … api`, `mypy … api`), `unit` (`--cov=api`) và `web` cho tới khi gộp `api/` (Lane B) và `web/` (Lane C, cần `web/package-lock.json` cho cache npm). Chỉ đánh giá "4 job xanh" sau tích hợp E. Job `notebooks` không phụ thuộc lane khác.
3. Job `web` giả định `api.main` với `VRE_FAKE_BACKENDS=1` không nạp encoder / không cần torch, Qdrant hay Neo4j trong lifespan, và có `GET /api/health` (Lane B). `requirements/dev.txt` không có torch.
4. Sau `eval` + `report examples` + `ablation`: chạy `& $VRE_PY notebooks/build_notebooks.py 02 03 04`, rồi commit 3 notebook đó. Lead commit `results/*.json` mới để job `notebooks` trên CI đọc được.
5. Lựa chọn đã tự quyết (thận trọng): thêm `xgboost` vào `dev.txt` vì `tests/unit/test_valuation_search_cache.py` import nó và `tests.yml` cũ cài riêng (B9: CI chỉ cài từ file). Job `unit` thêm `and not notebooks` để không chạy notebook hai lần. Các test kiểm tĩnh của notebook không gắn marker nên chạy cả trong `unit`. README không ghi "~9h" (số gõ tay) mà trỏ tới `night_report.wall_hours` hiển thị ở notebook 02.
6. Ghi nhận từ dữ liệu: `night_report.git_sha = "dirty"`, `peaks` = null (ghi trong `errors[]` dạng warn); notebook 02 hiện đúng như vậy ("không ghi").

Không đụng file ngoài phạm vi sở hữu. Không có PR.
