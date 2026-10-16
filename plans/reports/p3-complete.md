# P3 complete: Phase 3 + Phase 4 outputs trên đủ 100 câu eval

Nhánh `feat/p3-complete` (worktree `D:\Vietnam-Real-Estates-wt\complete`). Không gọi LLM (`VRE_LLM_OFFLINE=1` cho pipeline, notebook, deck).

## Kết quả eval (đọc từ `results/eval_summary.json`)

- `n_questions` 100, `complete` true, chấm theo luật (`scoring: rule_based`), sinh `gemini-flash-lite-latest`.
- Độ chính xác tổng: Graph 0,75 · Hybrid 0,72 · Basic RAG 0,28 · LLM thuần 0,11. Router (hybrid) 0,96 trên 100 câu.
- `run_meta.json` đã có đủ 100 `question_ids` (không phải sửa); report chỉ đọc khối `llm` của file đó.
- Ghi chú RAGAS trong `eval_summary.notes` nay theo dữ liệu: không có faithfulness thì ghi "ragas: không chạy (chấm theo luật, không LLM giám khảo, để tiết kiệm chi phí)" (`RAGAS_OFF_NOTE`, có test).

## Thay đổi

1. **Eval/report**: `vre.eval.summary` chọn ghi chú RAGAS theo việc faithfulness có được đo hay không; `report readme` in thêm `ablation.notes` (cảnh báo c256/c400/c512 đo trên subcorpus). README: mô tả `eval score --offline`, mục Hạn chế thêm "không LLM giám khảo/RAGAS (tiết kiệm chi phí)", "c* chỉ so với nhau", "Docker đã dựng và kiểm qua nginx :8080".
2. **Selector ablation**: notebook P2-08 (headline + bảng) và slide `ret-prefilter`, `ev-ablation` chỉ dùng run `full_*` (theo tên). Thêm slide mới `ev-chunk` (quét chunk_size trên subcorpus, sinh từ JSON) và bảng pivot chunk_size trong notebook P2-08.
3. **Chữ sinh từ JSON**: các câu "rerank không đo / model không có trong cache" đã thay bằng số rerank thật (Recall@5, độ trễ p50); chữ "tập con / 36 câu" chuyển sang điều kiện `complete && n_questions == len(benchmark)` ⇒ "toàn bộ benchmark". Slide Hạn chế: không giám khảo LLM/RAGAS, Docker đã kiểm.
4. **HTML** `slide/hoc-phan-4-graphrag.html`: thêm khoá V cho 6 run rerank, `ckSizes`, `ck_hybrid`, `ck_hybrid_prefilter`, `subN` (map trong `numbers_map.d/html.yaml`), viết lại đoạn ablation, kịch bản ev-setup/ev-ablation/ev-results, thêm mục kịch bản `ev-chunk`, bảng rủi ro (bỏ hàng "chỉ chạy tập con", thêm hàng Docker), FAQ và quiz. `exAg03Ms/exAg03Gt` (ví dụ AG03 không còn trong examples.json) thay bằng `exLk01Ms`.
5. **Ví dụ**: examples.json mới không có AG03 (graph) và CS01 (rag). Slide `gr-patterns` chọn AG01 rồi ví dụ aggregate đầu tiên; `ret-cosine`/`ret-filters` dùng CS01 nếu có, không thì `rag[0]` (cũng là câu ràng buộc), notes không còn nhắc cứng CS01.
6. **Bug notebook 05**: P5-07 gọi `valuate(PAYLOAD, variant="tabular")`, hàm không còn tham số `variant` (luôn phục vụ tabular) ⇒ sửa thành `valuate(PAYLOAD)`.

## Dựng lại

- Notebook 02, 04, 05 chạy lại (cell LIVE dùng Qdrant/Neo4j), `results/nb_cache/**` cập nhật trong worktree. 01 và 03 không đọc eval/ablation nên không chạy lại.
- Deck 55 slide (`slide/build/build_deck.py`): `check` 0 unmapped / 0 broken / 0 wrong / 0 not found; `fonts --min-pt 14` 0 run < 14pt.
- HTML: `html --apply` rồi `html` check: 0 unmapped / 0 broken / 0 wrong / 0 not found.
- `results/{eval_summary,error_analysis,examples,ablation,benchmark_gt}.json` chép vào worktree (ablation, benchmark_gt không đổi). `plans/reports/*.md` chép vào worktree; `p3-docker.md` thêm mục "Lead verification".

## Kiểm tra

- `pytest -q tests/unit`: 400 passed. `tests/test_notebooks.py`: 39 passed. `tests/unit/test_slide_numbers.py`: 9 passed.
- `ruff check src tests scripts api` (phạm vi CI): sạch. `mypy src/vre api`: no issues (100 file).
- Ghi chú: `ruff check slide/build` còn 10 lỗi có sẵn từ trước (I001/UP037/B905/F401, baseline main là 11); ngoài phạm vi CI, không sửa.

## Hạn chế / cần thêm

- examples.json không còn ví dụ AG03/CS01 vì selector của report ưu tiên theo nhãn; slide đã tự lui về ví dụ khác cùng loại.
- "khoảng mười câu mỗi nhóm" trong notes/HTML là chữ, vì các nhóm có 10 đến 12 câu và `results/*.json` không có khoá min/max theo nhóm.
