# Phase 4 — Slide (phiên riêng)

> **Phạm vi khi cook file này: CHỈ phase này.** Dừng sau bước cuối của file (bước 7). Không làm việc của phase khác, không đọc `plan.md`
> để làm tiếp (ngoại lệ duy nhất: bước 0 đọc mục "Quyết định bổ sung" của `plan.md`).

Plan tổng: `plans/261015-vre-full-retrain/plan.md` · Ghi chú slide gốc: `plans/reports/brainstorm-261015-0007-full-retrain-redesign.md` §7
· Nguồn số liệu DUY NHẤT: `results/*.json` (schema: `plans/261015-vre-full-retrain/data-contract.md` §7, bảng writer/reader §7.9).

## Mục tiêu

Cập nhật bộ slide `slide/Từ-RAG-đến-GraphRAG-Trợ-lý-hỏi-đáp-thị-trường-bất-động-sản-Việt-Nam-có-lý-thuyết.pptx` (39 slide, 16:9, Calibri,
palette navy/teal/cam, mọi slide có speaker notes) cho dự án làm lại trên 3,5M tin: thay toàn bộ số stale (gắn 20k), thêm slide mới, giảm
wall-of-text, và xử lý `slide/hoc-phan-4-graphrag.html` (tài liệu ôn Người 4, slide 30–38). **Mọi con số trên slide (tiêu đề, thân, notes,
chart, mảng JS trong HTML) truy được về một khoá trong `results/*.json`** — có công cụ kiểm tự động, không nhập tay số.

## Điều kiện tiên quyết

- Phase 3 xong: có `results/{ingest_report,eda_summary,graph_report,embedding_benchmark,night_report,valuation_metrics,benchmark_gt,eval_summary,ablation,error_analysis,examples}.json`
  và `results/eda/*.png`; 5 notebook đã chạy (chart cùng theme).
- Quyết định về file HTML (tái sinh từ JSON hay bỏ) đã được ghi trong `plan.md` mục "Quyết định bổ sung". Chưa có ⇒ dừng ở bước 0 và hỏi người dùng
  bằng `AskUserQuestion`.
- Webapp chạy được (`docker compose --profile app up -d --wait`) để chụp ảnh/clip demo.
- Thư viện: `python-pptx` (cài vào `.venv`, `PIP_CACHE_DIR` trên D — thêm vào `requirements/dev.txt`), `@mermaid-js/mermaid-cli` qua `npx`
  (cache npm trên D) để xuất SVG/PNG sơ đồ.
- Mọi lệnh sau `. D:\Vietnam-Real-Estates\scripts\vre_env.ps1`.

## Prior art tái dùng (phạm vi: `slide/`, `notebooks/` mới, `results/`)

- Deck hiện có (chưa commit): bố cục, palette, notes → giữ khung, sửa nội dung. Sao lưu trước khi sửa: `slide/_backup/<tên>-<yyMMdd>.pptx` (gitignored).
- Notebook 01–05 (Phase 3) đã vẽ chart từ results với theme thống nhất → xuất PNG 1600×900 dùng trên slide.
- `results/examples.json` → ví dụ Cypher, ví dụ câu trả lời, ví dụ thất bại thật (không bịa ví dụ).
- Không có công cụ kiểm số trên slide (tìm `grep -ri "pptx" scripts src` → 0 kết quả) → MỚI.

## File ownership

**Tạo**: `scripts/slide_numbers.py` (≤ 200 dòng), `slide/numbers_map.yaml`, `slide/assets/charts/*.png`, `slide/assets/diagrams/*.{mmd,svg,png}`,
`slide/assets/demo/*.png` (ảnh chụp webapp), `tests/unit/test_slide_numbers.py`.
**Sửa**: file `.pptx` trong `slide/`; `slide/hoc-phan-4-graphrag.html` (nếu quyết định là tái sinh) hoặc xoá (nếu quyết định là bỏ);
`requirements/dev.txt` (+`python-pptx`); `.gitignore` (+`slide/_backup/`).
Không sửa code `src/`, `api/`, `web/` (sai số trong results ⇒ quay lại Phase 3 sửa pipeline, không sửa tay trên slide).

## Công cụ kiểm số (làm TRƯỚC khi sửa slide — replicate-and-automate)

`scripts/slide_numbers.py`:
- `extract <pptx>`: liệt kê mọi token số (regex số VN: `\d{1,3}(\.\d{3})+(,\d+)?|\d+(,\d+)?%?` …) trong tiêu đề, thân, bảng, speaker notes,
  và dữ liệu chart nhúng (`chart.plots[*].series[*].values`) — in `slide, shape, nơi, text, số`.
- `check <pptx> --map slide/numbers_map.yaml`: mỗi số phải (a) khớp một mục map `{slide, shape|notes|chart, text_pattern, key: "eval_summary.json:by_category.aggregate.hybrid.accuracy", format: "pct1"|"int_dot"|"float2"|…}`
  và giá trị định dạng từ JSON bằng đúng chuỗi trên slide; hoặc (b) nằm trong whitelist (số slide, năm, "RAG 2.0", k=60 hằng số thuật toán có ghi chú).
  Số không map ⇒ FAIL. Mục map trỏ khoá không tồn tại ⇒ FAIL.
- `apply <pptx> --map ...`: ghi giá trị từ JSON vào đúng shape/notes/chart (cập nhật cache dữ liệu chart), giữ định dạng run.
- `html <html> --map ...`: cùng kiểm/ghi cho mảng JS trong file HTML (nếu giữ HTML).
- `fonts <pptx> --min-pt 14`: liệt kê mọi run chữ nhỏ hơn ngưỡng (whitelist footer/số trang).
Test: `tests/unit/test_slide_numbers.py` trên một pptx tí hon dựng bằng python-pptx: số không map ⇒ fail; map sai khoá ⇒ fail; apply rồi check ⇒ pass.

## Checklist (từ report §7 → việc cụ thể)

### A. Sửa số stale (toàn bộ đi qua `numbers_map.yaml`, nguồn trong ngoặc)

- [ ] Số đếm dữ liệu: 20.000 → `raw_rows`; 19.925 → `dedup.kept`; 73 trùng → `dedup.exact_dup + dedup.prefix_dup`; 48 giá sửa → `price.fixed_thousand`;
      21.030 chunk → `night_report.counters.chunks`; 4,8% / 133s / 32,3MB → các số tương ứng trong `night_report` (thời gian từng bước, `peaks`) và
      kích thước collection (`night_report.qdrant`). (`ingest_report.json`, `night_report.json`)
- [ ] Tỷ lệ thiếu 50,8% / 75,2% / 14,5% → `ingest_report.missing_pct.{bedrooms,project,ward}` (dự kiến ~46/71/14%).
- [ ] Slide 7 EDA: histogram, số tỉnh 6888/6805…, "HN+HCM 69%" → `eda_summary.top_provinces`, `hn_hcm_share`; hình thay bằng `results/eda/*.png`.
- [ ] Mọi `Tin#ID` cũ (1, 1198, 14515, 8940, 12955, 15405, 3138, 18456, 1205) → id hash từ `examples.json` (id 16 chữ số: hiển thị đầy đủ hoặc
      rút gọn `Tin#…<6 số cuối>` NHƯNG notes ghi id đầy đủ).
- [ ] Graph: 22.116 node / 41.978 cạnh + mọi số con → `graph_report.nodes/relationships`; Leiden 460/13/Q=0,77/245,8/37,0 → `graph_report.communities` +
      bảng cụm (nếu cần số ngoài JSON ⇒ bổ sung vào writer ở Phase 3, không gõ tay).
- [ ] Toàn bộ kết quả slide 15, 33, 35, 36, 37, 38 + latency/token → `eval_summary.by_category`, `overall`, `cost`, `pairwise`, `retrieval`, `ragas`.
- [ ] Số trong **tiêu đề** slide 6, 11, 16, 19, 28, 29, 33, 35, 38; **speaker notes**; **7 chart nhúng** (cập nhật data cache chart hoặc thay bằng PNG
      xuất từ notebook); **mảng JS** trong HTML.
- [ ] Mâu thuẫn đã biết: "24 câu" → `eval_summary.n_questions` (~100); "3 hệ" → 4 hệ (`systems`); Thủ Đức 1.229 vs 1.367 → một số duy nhất từ
      `examples.json`/`graph_report` (hoặc bỏ); Cypher A2 thiếu `WHERE n >= 20` → lấy Cypher thật từ `examples.graph`; note slide 37 ngược chart → viết lại
      notes từ cùng dữ liệu chart; slide 15 chart trùng bảng slide 35 → bỏ một trong hai; "RTX 3050" → "RTX 4060 Laptop 8GB".

### B. Slide mới

- [ ] Sơ đồ kiến trúc end-to-end (Mermaid → SVG/PNG cùng palette): Parquet → DuckDB (clean/dedup) → embedding (`night_report.model_slug` → tên model qua `vre.config.EMBED_MODELS`) → Qdrant + Neo4j → Router → LLM → API → Web.
- [ ] Chọn embedding: bảng/biểu đồ chất lượng (Recall@1/@10, MRR@10) × tốc độ (docs/s) của 4 model đã đo, nhấn e5-large được chốt (`embedding_benchmark.json`).
- [ ] "Scale lên 3,5M": bảng thời gian từng bước của lượt chạy dài ~9h (`night_report.steps`, `night_report.wall_hours`, `ingest_report.seconds`, `graph_report.import_seconds`),
      tài nguyên đỉnh (`night_report.peaks`), kỹ thuật (dedup, checkpoint/resume, dry-run 2 part trước lượt dài, Phase 1 ∥ Phase 2 hai phiên,
      neo4j-admin import, int8 quantization).
- [ ] EDA thật (3–4 slide): giá/m² theo loại & vùng, xu hướng tháng, dữ liệu thiếu, outlier, trùng lặp (~23%: `(raw_rows − dedup.kept)/raw_rows`) — hình `results/eda/*.png`.
- [ ] Ablation (chỉ e5-large): dense/BM25/hybrid + pre-filter + rerank + chunk size (`ablation.runs`); so sánh model embedding lấy từ `embedding_benchmark.json`;
      bảng 9 nhóm câu có CI (`eval_summary.by_category[*].ci95`).
- [ ] Error analysis: số theo nguyên nhân (`error_analysis.counts`) + 2–3 ví dụ thất bại thật (`error_analysis.examples`).
- [ ] Module định giá (phần mở rộng, 1–2 slide): MAPE/MAE/R² tabular vs full vs baseline (`valuation_metrics.variants`), SHAP top (`shap_top`).
- [ ] Demo webapp: ảnh chụp so sánh 4 hệ, bản đồ, định giá, Cypher panel (`slide/assets/demo/*.png`, chụp bằng Playwright `page.screenshot` 1600×900 từ
      backend thật; clip tuỳ chọn).
- [ ] Hạn chế & hướng phát triển (địa giới cũ vs 2025, license CC-BY-NC, thiếu toạ độ, rate limit LLM…); Kết luận/takeaways trước Q&A.

### C. Redesign

- [ ] Slide lý thuyết 5, 8, 12, 17, 24, 30, 34: mỗi slide 1 ý, ≤ 12 shape, chữ thân ≥ 18pt; phần chữ thừa chuyển xuống notes.
- [ ] Caption tối thiểu 14pt (kiểm bằng script: liệt kê run < 14pt).
- [ ] Sơ đồ vẽ tay bằng shape → Mermaid xuất SVG (`npx -y @mermaid-js/mermaid-cli -i x.mmd -o x.svg`) cùng palette (navy `#1F3A5F`, teal `#2A9D8F`, cam `#F4A261` — đọc mã màu thật từ theme deck trước).
- [ ] Chart: chỉ dùng PNG xuất từ notebook (cùng theme) hoặc chart nhúng do `slide_numbers.py apply` ghi; không nhập tay số vào chart.
- [ ] Code Cypher: 1 khối/slide, highlight phần quan trọng (màu cam), lấy từ `examples.graph`.
- [ ] File HTML: theo quyết định đã ghi trong `plan.md` (tái sinh mảng JS bằng `slide_numbers.py html` và kiểm `check`, hoặc xoá file + bỏ tham chiếu).
- [ ] Viết lại speaker notes bằng giọng người trình bày; chạy skill `t1k-human-writing` (lint) trên notes trước khi chốt.

## Các bước

1. Bước 0 — kiểm điều kiện tiên quyết; sao lưu pptx; `& $VRE_PY -m pip install python-pptx` (cache trên D).
2. Viết `scripts/slide_numbers.py` + test; `extract` deck hiện tại → danh sách số stale (đối chiếu checklist A).
3. Soạn `slide/numbers_map.yaml` cho mọi số giữ lại; `apply`; `check` = 0 FAIL.
4. Sửa tay nội dung không phải số (mâu thuẫn, câu chữ, redesign C); thêm slide B với hình/sơ đồ từ assets.
5. Xử lý HTML theo quyết định.
6. Kiểm cuối: `check` 0 FAIL; liệt kê run < 14pt = 0; người dùng xem lại toàn bộ deck (mở PowerPoint) và xác nhận.
7. Commit (pptx, html nếu còn, `numbers_map.yaml`, `scripts/slide_numbers.py`, assets, test) theo pathspec; KHÔNG commit `slide/_backup/`.

## Verify

```powershell
& $VRE_PY -m pytest -q tests/unit/test_slide_numbers.py
& $VRE_PY scripts\slide_numbers.py check "slide\Từ-RAG-đến-GraphRAG-Trợ-lý-hỏi-đáp-thị-trường-bất-động-sản-Việt-Nam-có-lý-thuyết.pptx" --map slide\numbers_map.yaml
& $VRE_PY scripts\slide_numbers.py fonts "slide\Từ-RAG-đến-GraphRAG-Trợ-lý-hỏi-đáp-thị-trường-bất-động-sản-Việt-Nam-có-lý-thuyết.pptx" --min-pt 14
```
Đạt: test xanh; `check` in `0 unmapped, 0 broken keys`; `fonts` in 0 run dưới 14pt (trừ footer/số trang whitelist).

## Tiêu chí hoàn thành

- [ ] 0 số không truy được về `results/*.json` (công cụ `check`, không phải mắt thường).
- [ ] Mọi mục checklist A, B, C được đánh dấu kèm slide số.
- [ ] Không còn "RTX 3050", "24 câu", "3 hệ", id `Tin#` kiểu số thứ tự dòng.
- [ ] Notes đã lint human-writing; người dùng duyệt deck.
- [ ] Deck + assets + công cụ được commit (lần đầu deck vào git).

## Rủi ro

| Rủi ro | L | I | Điểm | Giảm thiểu |
|---|---|---|---|---|
| python-pptx không cập nhật được chart nhúng phức tạp | 3 | 2 | 6 | Thay chart nhúng bằng PNG từ notebook (cùng theme) |
| Định dạng số VN (dấu chấm nghìn, phẩy thập phân) lệch giữa JSON và slide | 3 | 2 | 6 | Formatter tập trung trong `slide_numbers.py` + test |
| Sửa slide làm vỡ bố cục/animation | 2 | 2 | 4 | Sao lưu trước; người dùng duyệt bằng PowerPoint |
| Số cần trên slide không có trong JSON | 3 | 3 | 9 | Bổ sung vào writer ở Phase 3 (pipeline), chạy lại report — cấm gõ tay |
| Phát hiện kết quả eval xấu hơn kỳ vọng | 3 | 2 | 6 | Trình bày trung thực + error analysis (report §9) |

## Timeline

| Bước | Agent | Người dùng |
|---|---|---|
| 0–1 chuẩn bị + công cụ | 3h | — |
| 2–3 map + apply số | 3h | — |
| 4 nội dung + slide mới + redesign | 5h | 30 phút góp ý |
| 5 HTML | 1h | — |
| 6–7 kiểm + commit | 1h | 30 phút duyệt |
| **Tổng** | **~13h agent (1–1,5 phiên)** | **~1h** |
