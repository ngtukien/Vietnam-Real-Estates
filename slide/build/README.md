# Dựng deck bằng script

Deck `slide/Từ-RAG-đến-GraphRAG-…-có-lý-thuyết.pptx` được **sinh** bởi `slide/build/build_deck.py` từ các module
`slide/build/parts/pN_*.py` (mỗi module một hàm `build(d)`), theo thứ tự `PARTS`. Không sửa pptx bằng tay.
Bản cũ (dữ liệu mẫu 20k, theme cũ) chỉ còn ở `slide/_backup/` để tham khảo nội dung; theme mới định nghĩa ở `kit.py`.

```powershell
. D:\Vietnam-Real-Estates\scripts\vre_env.ps1
& $VRE_PY slide\build\build_deck.py --draft --only p3_graph --out <scratch>\p3.pptx   # dựng một phần, rồi tự chạy check
& $VRE_PY scripts\slide_numbers.py fonts <scratch>\p3.pptx --min-pt 14 --map <scratch>\p3.map.yaml
& $VRE_PY slide\build\build_deck.py            # dựng cả deck (không --draft: khoá thiếu là lỗi)
```

## Quy tắc số (bắt buộc — `check` phải in `0 unmapped, 0 broken keys, 0 wrong, 0 not found`)

- Số dữ liệu: `d.N("file.json:path", "fmt")` — khoá trỏ vào `D:\Vietnam-Real-Estates\results\*.json`.
  Cú pháp đường dẫn: `a.b`, `[0]`, `[-1]`, `[id=LK01,system=hybrid]`, `[*]`. Biểu thức: `"=({ingest_report.json:raw_rows}-{ingest_report.json:dedup.kept})/{ingest_report.json:raw_rows}"`.
- `fmt`: `int` (2.537.920) · `decN` (0,656) · `pctN` (×100: 0,6561→65,6; dấu % tự gõ sau) · `milN` (÷1e6) · `bnN` (÷1e9, tỷ)
  · `kN` · `hN` (giây→giờ) · `minN` (giây→phút) · `str` · `last6`; tiền tố `s` thêm dấu (+/−): `spct1`.
- Số KHÔNG phải dữ liệu (đề bài, ví dụ minh hoạ công thức, "Nhóm 7", "k = 60"): `d.A("…", "lý do")`.
- Chuỗi nguyên văn từ results (Cypher, câu trả lời, tên model): `d.T("examples.json:graph[0].cypher")`.
- Câu hỏi benchmark: `q = d.Q("AG01")` rồi `d.A(q["question"], "đề bài AG01")`.
- Chart: `s.chart(x, y, w, h, "bar"|"column"|"line", categories_key_or_list, [(tên, khoá_list, màu[, scale])], number_format="#,##0")`
  — nhãn cố định không được chứa chữ số. Giá trị thô để rẽ nhánh: `d.V(key)` (không in ra).
- Số gõ thẳng trong chuỗi (kể cả trong notes) ⇒ `check` báo unmapped ⇒ sửa bằng `d.N`/`d.A`, không thêm whitelist.
- Số cần mà `results/*.json` không có ⇒ KHÔNG gõ tay: viết định tính, và ghi vào báo cáo cuối "cần thêm khoá X".

## Kit (`kit.py`) — khung 13,333 × 7,5 in, lề `MX` = 0,6, vùng nội dung y 1,75 → 6,85

`d.slide(name, section, title, notes=...)` — `name` duy nhất, tiền tố theo module (bảng dưới). Mọi slide có notes.
`s.text(x,y,w,h, content, size, bold, color, align, anchor, bullets)` · `s.card(x,y,w,h, title, body, accent)` ·
`s.stat(x,y,w, value, label, accent)` · `s.table(x,y,w, rows, col_w, size, highlight_row)` · `s.code(x,y,w,h, code, highlight=[...])`
· `s.image(path,x,y,w,h)` · `s.chart(...)` · `s.flow([(tiêu đề, mô tả)], y)` · `s.caption(text)` · `s.rect(...)`.
`content` là chuỗi hoặc list đoạn; đoạn là chuỗi hoặc list `(chữ, {"bold":True,"color":"teal"})`. Một chuỗi nhiều dòng
nên tách thành list đoạn (trừ trong `code`). Màu theo tên trong `C` (`ink, text, muted, blue, teal, amber, coral, violet`,
nền `*_bg`, `soft`); màu hệ thống: `SYSTEM` (LLM thuần xám, Basic RAG xanh dương, Graph cam, Hybrid teal).
Thiếu thành phần ⇒ viết helper trong module của mình; KHÔNG sửa `kit.py` (báo lead nếu cần chung).

## Thiết kế

- Một slide một ý. Tiêu đề là câu kết luận (≤ 2 dòng, 30pt). Slide lý thuyết: chữ thân ≥ 18pt (`SIZE["body"]` = 20),
  tối đa ~3 khối nội dung; chữ giải thích dài chuyển xuống notes. Slide dữ liệu: chữ ≥ 16pt, caption 14pt. Không chữ < 14pt.
- Nhiều khoảng trắng, căn lưới, ít khung viền. Không lặp hiệu ứng "phát biểu — kết luận" ở mọi thẻ.
- Notes: giọng người trình bày, nói được thành lời, 60–150 từ, kết bằng câu chuyển sang slide sau. Không dùng dấu "—" trong
  notes; không từ sáo rỗng. Nêu trung thực giới hạn của số liệu.
- Tiếng Việt có dấu đầy đủ. Ví dụ chỉ lấy từ results/examples.json hay benchmark (không bịa).

## Dàn ý toàn deck và chủ sở hữu module

| Module | Tiền tố tên | Nội dung | Chủ |
|---|---|---|---|
| `p0_intro` | `intro-` | bìa, lộ trình, 9 nhóm câu hỏi benchmark, LLM thuần vs RAG | lead |
| `p1_rag_data` | `rag-`, `data-`, `eda-`, `emb-`, `scale-`, `arch-` | lý thuyết RAG, hai pha, dữ liệu 3,5M + EDA, hai loại tri thức, chunking, embedding + chọn model, thực hành 1, scale lên 3,5M, kiến trúc end-to-end | agent A |
| `p2_retrieval` | `ret-` | ba loại yêu cầu, trích bộ lọc, cosine, dense/BM25/RRF, pre-filter, ba điểm lỗi RAG, prompt sinh, Basic RAG thất bại ở câu tổng hợp, thực hành 2 | lead |
| `p3_graph` | `kg-` | bảng→đồ thị, triple, property graph vs RDF vs SQL, nguyên tắc schema, schema thật, dựng đồ thị, Cypher, cộng đồng Leiden, thực hành 3 | agent B |
| `p4_graphrag_eval` | `gr-`, `ev-` | GraphRAG, so sánh 3 kiến trúc, mẫu truy xuất + lớp an toàn, router, hai tầng đánh giá, ablation truy xuất, chấm câu trả lời, chi phí, kết quả theo nhóm, error analysis | lead |
| `p5_ext` | `ext-`, `end-` | định giá, demo webapp, hạn chế, kết luận, hỏi đáp + tham khảo | agent C |

Mỗi agent chỉ tạo/sửa file module của mình (+ asset riêng trong `slide/assets/<module>/`). Commit bằng pathspec:
`git add <file mới>; git commit -m "..." -- <các file của mình>`. Không `git add -A`, không checkout/switch nhánh.
