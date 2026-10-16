# Phase 3 — Notebook theo bố cục bản cũ (phase-03-lanes §4c)

**Status:** DONE_WITH_CONCERNS

Nhánh `feat/p3-notebooks` (worktree `D:\Vietnam-Real-Estates-wt\notebooks`), đã push. HEAD `504f8cfde1b6119b8fa75290d1841cb88e8151dd`.
Chưa mở PR. Mỗi notebook một commit:

- `9d15f92` builder + test + `00_setup`, xoá `01_eda`/`02_rag`
- `1f9e3de` 01
- `fdbf139` 02
- `cdffa2b` 03
- `aab704c` 04
- `504f8cf` 05

## Đã làm

- **Bộ notebook** theo bố cục `b22c3cc^`:
  - `00_setup` (`00-01`…`00-03`, ẨN), mọi notebook gọi `%run ./00_setup.ipynb`
  - `01_rag_indexing` (P1-01…P1-13): EDA, làm sạch, chunk, embedding, Qdrant
  - `02_retrieval_generation` (P2-01…P2-11)
  - `03_knowledge_graph` (P3-01…P3-13)
  - `04_graphrag_eval` (P4-01…P4-14; P4-14 so sánh cặp là cell mới)
  - `05_valuation` (P5-01…P5-08, mới)

  Đã xoá `01_eda.ipynb` và `02_rag.ipynb`. Code dùng `vre.rag`, `vre.graph`, `vre.qa`, `vre.eval`, `vre.valuation`, `vre.index`, `vre.data`.
- **Dạng mỗi bước:** ô markdown `### Px-yy · <kết luận một câu>` + `**ẨN|CACHE|LIVE**` + đoạn giải thích (khái niệm, cách làm, cách đọc, lưu ý). Theo sau là ô code có dòng đầu `# Px-yy · <loại>`.
  - Số trong heading do `notebooks/build_notebooks.py` điền từ `results/*.json` (gốc `VRE_DATA_ROOT`) và `results/nb_cache/<id>.json`.
  - Khi số mẫu nhỏ, heading ghi kèm `n = …`.
  - Đoạn giải thích không có số gõ tay (test kiểm).
- **CACHE** đọc results JSON, hoặc tính một lần rồi ghi cache cell. Ví dụ: P1-01 schema/mẫu, P3-08 đếm tiện ích bằng degree, P2-11 bảng tạm.
- **LIVE** là demo nhẹ, chạy thật khi có dịch vụ/model, rồi ghi cache cell. Các demo:
  - P1-07 chunk một tin, so công thức với đếm thực
  - P1-09 cosine/PCA của 5 câu (encoder truy vấn CPU fp16; mỗi notebook nạp tối đa một lần)
  - P1-13, P2-04…P2-07: một lượt tìm trên Qdrant mỗi cell (dense, BM25, RRF, lọc trước/lọc sau)
  - P2-09 prompt + câu trả lời đọc từ `answers.jsonl`
  - P3-02, P3-09…P3-13, P4-04: Cypher (P3-10 kiểm chéo trung vị MarketStat với DuckDB: khớp)
  - P4-03 guard + Text2Cypher
  - P4-06 router
  - P5-07 một lần `valuate()`
- **Không gọi LLM thật:** `llm_cache_only()` thay `_ask_openai`/`_ask_anthropic` bằng hàm ném `LLMCacheMiss`, nên chỉ đọc cache đĩa LLM. Thiếu dịch vụ, hoặc `VRE_NB_OFFLINE=1`, thì cell đọc cache cell. `VRE_NB_STRICT=1` (builder bật) biến lỗi live thành lỗi build thay vì lặng lẽ lùi về cache.
- **Cache cell:** 27 file `results/nb_cache/*.json`, đã commit.
- **`tests/test_notebooks.py`** kiểm:
  - bộ file khớp builder
  - mỗi ô code có heading cùng mã/loại ngay trước
  - notebook đã lưu == bản sinh lại
  - giải thích không có số gõ tay
  - không có `[ĐIỀN SAU]`
  - đã chạy, không có output lỗi
  - output và cache cell không lộ số điện thoại, khoá, đường dẫn máy (ngoại lệ hẹp: chuỗi mẫu tự đặt `0901 234 567` của demo `mask_pii` P1-05)
  - marker `notebooks`: chạy cả 6 notebook offline bằng nbclient, không ghi cache, < 120 s

## Verify (output thật)

- `& D:\Vietnam-Real-Estates\.cache\scratch\run_heavy.ps1 -Cmd '...build_notebooks.py'`: cả 6 notebook chạy live và lưu, không có ô lỗi. Qdrant, Neo4j, HF, dữ liệu, cache LLM và `answers.jsonl` đều có trên máy.
- `pytest -q tests/test_notebooks.py`: `39 passed, 2 warnings in 28.61s`. Gồm cả lượt chạy offline 6 notebook; cảnh báo là marker chưa đăng ký và zmq.
- `ruff check tests notebooks/build_notebooks.py`: `All checks passed!`. Không chạy ruff trên `.ipynb` vì tên định nghĩa trong `00_setup` qua `%run` cho F821 — CI không lint notebook.
- Kết quả demo đáng chú ý, in trung thực:
  - P1-13: dense không lọc thoả đủ điều kiện `0/5`
  - P2-07: lọc trước giữ `5/5`, lọc sau còn `0/5`
  - P2-05: BM25 và dense cùng `5/5`

## Cần lead quyết / làm

1. **Test "notebook == bản sinh lại" sẽ ĐỎ trên CI cho tới khi commit results.** Notebook được sinh từ `D:\Vietnam-Real-Estates\results` (VRE_DATA_ROOT). Ở đó có `benchmark_gt.json`, `eval_summary.json`, `error_analysis.json` mà nhánh này không có. Sinh lại với kết quả của repo thì lệch 1 heading ở 02 (P2-11) và 9 heading ở 04.
   - Cách xử lý: commit các results đó (lead sở hữu), rồi chạy `& $VRE_PY notebooks/build_notebooks.py 02 04`.
   - Test cố ý đỏ khi notebook cũ hơn `results/`; không nới.
2. **Eval hiện tại là bản smoke (5/100 câu).** Heading 04 nói rõ "mới 5/100 câu (chưa đủ benchmark)" và kèm `n`. Sau eval đầy đủ, `ablation` và `report examples`: chạy `build_notebooks.py 02 04`, commit notebook cùng `results/nb_cache/P2-*`, `P4-*`.
3. **Đăng ký marker** `notebooks` trong `pyproject.toml` (vẫn còn `PytestUnknownMarkWarning`).
4. **README** (không sửa, vì chưa xác nhận được lane fix-up đã xong). Mục "Notebook" cần:
   - thay bảng 5 dòng `01_eda…05_valuation` bằng 6 dòng: `00_setup` (dùng chung, `%run`), `01_rag_indexing` (EDA + làm sạch + chunk + embed + Qdrant), `02_retrieval_generation`, `03_knowledge_graph`, `04_graphrag_eval`, `05_valuation`
   - thay câu "chỉ đọc results" bằng: ba loại cell ẨN/CACHE/LIVE; LIVE chạy thật khi có dịch vụ, không thì đọc `results/nb_cache/`; `VRE_NB_OFFLINE=1` buộc đọc cache (như CI); `VRE_NB_REBUILD=1` tính lại CACHE; notebook không gọi LLM thật
   - giữ lệnh `& $VRE_PY notebooks/build_notebooks.py [02 04]`
5. **Ghi nhận ngoài phạm vi** (`results/eval/raw/answers.jsonl`, LK01 hệ `graph`): `answer` bắt đầu bằng `<tai_lieu>\nKẾT QUẢ TRUY VẤN:` — có vẻ prompt/ngữ cảnh bị chép vào câu trả lời. Nên Lane A xem lại.
6. **Tài nguyên:** encoder truy vấn chỉ nạp ở 01 và 02 (CPU fp16, mỗi kernel một lần). Mọi lần chạy nặng đi qua `run_heavy.ps1`, và nó có lúc phải chờ commit trống ≥ 6 GB.

Không sửa file ngoài `notebooks/**`, `results/nb_cache/**`, `tests/test_notebooks.py`. `plans/…/phase-03-lanes.md` đang sửa dở trong worktree là của lead, không commit.
