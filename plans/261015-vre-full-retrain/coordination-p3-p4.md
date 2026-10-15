# Phối hợp Phase 3 (phiên p3) ∥ Phase 4 (phiên p4) — 2026-10-16

Người dùng yêu cầu Phase 3 + 4 xong 100% lúc **06:00 2026-10-16**, không ai thức kích hoạt lệnh. Hai phiên chạy song song, tự phối hợp
qua file tín hiệu (bền) + `SendMessage` (nhanh, tuỳ chọn). Mọi quyết định chính sách: tự chọn phương án thận trọng nhất, ghi lại, KHÔNG hỏi người dùng
(trừ bước duyệt deck cuối của Phase 4 — để người dùng duyệt sau 06:00).

## 1. Chỗ làm việc

| Phiên | Worktree | Nhánh |
|---|---|---|
| p3 (phiên này) | `D:\Vietnam-Real-Estates-wt\int` + các worktree lane | `feat/p3-integration` |
| p4 | `D:\Vietnam-Real-Estates-wt\slides` | `feat/p4-slides` (tạo từ `feat/p3-integration`) |

- Deck + HTML gốc (chưa từng commit) nằm ở `D:\Vietnam-Real-Estates\slide\`; p4 SAO CHÉP vào worktree `slides\slide\` ở bước 0 (sao lưu như phase-04 nói).
- `results/*.json` luôn đọc ở `D:\Vietnam-Real-Estates\results\` (data root `VRE_DATA_ROOT`), nơi p3 đang ghi kết quả eval. p4 KHÔNG ghi vào `results/`.
- p4 không sửa `src/`, `api/`, `web/`, `notebooks/` (phase-04 §File ownership). Cần số chưa có trong JSON ⇒ gửi `NOTE` cho p3 (p3 thêm vào writer).

## 2. Tài nguyên (máy ~40 GB, VM Docker giữ ~15 GB; commit là giới hạn thật)

- Mọi bước nặng (pytest lớn, Playwright/chromium, mermaid-cli, mở webapp thật) chạy qua khoá toàn máy:
  `& D:\Vietnam-Real-Estates\.cache\scratch\run_heavy.ps1 -Cmd '...'` (chờ khoá + commit trống ≥ 4 GB).
- p3 đang chạy eval đầy đủ (giữ khoá ~2,5 giờ, từ 00:13). p4 làm phần nhẹ (python-pptx, YAML, viết nội dung) không cần khoá.
- p4 chỉ `pip install python-pptx` (một lần, cache trên D). Không tải model nào.

## 3. Tín hiệu (file JSON trong `D:\Vietnam-Real-Estates\results\handoff\`, vỏ như data-contract §7.10, `from` = `p3`|`p4`)

| Tín hiệu | Ai gửi | Khi nào | p4 làm gì |
|---|---|---|---|
| `P3_RESULTS_READY` | p3 | `eval_summary`, `ablation`, `error_analysis`, `examples`, `benchmark_gt` đã ghi xong (đầy đủ 100 câu) | map + apply số eval/ablation/error/examples, slide 15/33/35–38, HTML |
| `P3_WEBAPP_READY` | p3 | API + web chạy với backend thật; payload `{web_url, api_url, until}` | chụp `slide/assets/demo/*.png` (Playwright 1600×900) trong khung giờ `until` |
| `P3_DONE` | p3 | Phase 3 đã merge vào `main` | p4 rebase/merge `main` vào `feat/p4-slides` trước khi mở PR |
| `NOTE` | cả hai | cần số mới / báo vấn đề | đọc và trả lời bằng `NOTE` |
| `P4_DONE` | p4 | deck `check` 0 FAIL, commit + push + PR | — |

Ghi tín hiệu: `& $VRE_PY scripts\vre_pipeline.py signal <NAME> --from p3|p4 --payload '<json>'` (hoặc ghi nguyên tử *.tmp → rename).
Chờ tín hiệu: chạy poller nền `run_in_background` (kiểm 5 phút/lần), KHÔNG ngồi chờ, làm phần khác trong lúc đó.

## 4. Giờ chốt

- ~03:30 p3 gửi `P3_RESULTS_READY`; ~04:00 `P3_WEBAPP_READY` (khung chụp ≥ 30 phút); ~05:00 `P3_DONE`.
- p4 dự kiến xong ~05:45: số từ results (tool `check` 0 FAIL), slide mới, redesign, HTML, notes lint human-writing; deck chờ người dùng duyệt sau 06:00.
- Trễ: phiên nào trễ ghi `NOTE` giờ dự kiến mới; không bỏ tiêu chí "0 số không truy được về results".
