# Phase 3 — chốt giữa các lane (lead, 2026-10-15)

Bổ sung cho `phase-03-system-eval-webapp.md` (spec từng lane) + `api-contract.md` + `data-contract.md` (SSOT dữ liệu/API).
File này chỉ ghi các điểm nối giữa lane mà hai hợp đồng chưa ghi. Không chép lại spec.

## 1. Worktree + nhánh (lead tạo trước khi fan-out; lane KHÔNG `git switch`/`checkout -b`)

| Lane | Nhánh | Worktree |
|---|---|---|
| A1 lõi QA (A1, sau đó A3–A5) | `feat/p3-core` | `D:\Vietnam-Real-Estates-wt\core` |
| A2 benchmark + GT | `feat/p3-bench` | `D:\Vietnam-Real-Estates-wt\bench` |
| B API | `feat/p3-api` | `D:\Vietnam-Real-Estates-wt\api` |
| C Web | `feat/p3-web` | `D:\Vietnam-Real-Estates-wt\web` |
| D Notebook/README/CI | `feat/p3-docs` | `D:\Vietnam-Real-Estates-wt\docs` |

- Dữ liệu dùng chung qua `VRE_DATA_ROOT=D:\Vietnam-Real-Estates` (`vre_env.ps1`): `results/`, `data/`, `index/`, `models/` là của
  checkout chính. Lane ghi `results/*.json` ⇒ file nằm ở checkout chính; lead commit chúng khi tích hợp.
- Python: `D:\Vietnam-Real-Estates\.venv\Scripts\python.exe` (dùng chung, đã cài sẵn fastapi/uvicorn/sse-starlette/httpx/nbclient/ipykernel).
  KHÔNG `pip install` thêm (tranh chấp venv); cần gói mới ⇒ ghi vào báo cáo, lead cài.
- Chạy pytest từ gốc worktree (`pythonpath=src` của worktree). Commit dạng pathspec, chỉ file mình sở hữu, nhánh của mình, push nhánh.

## 2. Sở hữu file (thay đổi so với bảng lane trong phase-03)

- `src/vre/config.py`, `pyproject.toml`, `.env.example`, `.gitignore`, `scripts/vre_pipeline.py`: lane A1 (đại diện lead).
  Lane khác cần khoá cấu hình mới ⇒ đọc env trong module của mình (B: `api/deps.py` có pydantic-settings riêng) hoặc báo lead.
- Lane A2 sở hữu: `benchmark/questions.jsonl`, xoá `benchmark/questions.csv`, `src/vre/eval/questions.py`, `src/vre/eval/ground_truth.py`,
  `tests/unit/test_questions.py`, `tests/unit/test_ground_truth.py`. A2 KHÔNG sửa `scripts/vre_pipeline.py`; A1 nối lệnh `eval gt`.
- Lane A1 sở hữu phần còn lại của `src/vre/eval/**`.

## 3. Chữ ký nối A1 ↔ A2

```python
# src/vre/eval/questions.py  (A2)
Category = Literal["lookup","constraint","aggregate","multihop","temporal","comparison","project_street","valuation","unanswerable"]
class Question(BaseModel):            # đúng các khoá data-contract §7.7, extra="forbid"
    id: str; category: Category; route_gold: Literal["basic_rag","graph","hybrid","valuation"]
    question: str; gt_kind: Literal["ids","value","refusal"]; gt_sql: str | None
    value_tolerance: float | None = None; tags: list[str] = []
    valuation_input: dict | None = None   # chỉ category=valuation: payload ValuationInput (api-contract §3) mô tả BĐS giả định
def load_questions(path: Path | None = None) -> list[Question]     # mặc định CODE_ROOT/benchmark/questions.jsonl; validate + id duy nhất

# src/vre/eval/ground_truth.py  (A2)
def run(settings: Settings) -> dict          # chạy mọi gt_sql trên v_listings, ghi results/benchmark_gt.json (data-contract §7.7), trả dict đó
def load_gt(settings: Settings) -> dict[str, dict]   # id -> item của benchmark_gt.json
```

`valuation_input` là trường bổ sung của hợp đồng §7.7 (A2 ghi thêm dòng vào data-contract §7.7 trong commit của mình).

## 4. Tài nguyên máy (rule resource-budget)

RAM vật lý ~40 GB, đang dùng ~25 GB (Docker Qdrant ≤ 12 GB + Neo4j heap 4G/pagecache 4G + VS Code). Năm lane chạy song song:
- DuckDB trong lane: `VRE_DUCKDB_MEMORY_GB=3`, `VRE_DUCKDB_THREADS=4`.
- Chỉ lane A1 được nạp encoder e5-large (GPU, ~2,5 GB) và chỉ cho test/thủ công ngắn; giải phóng ngay. Không chạy eval đầy đủ, ablation hay embed khi chưa có lệnh của lead.
- Trước bước nặng: đọc `Win32_OperatingSystem` FreePhysicalMemory/FreeVirtualMemory; free commit < 6 GB ⇒ chờ, không chạy.
- Lời gọi Groq thật: A1 ≤ 20 lời gọi để kiểm thủ công; các lane khác 0 (dùng fake/cache).

## 5. Báo cáo

Mỗi lane kết thúc (hoặc chạm ngân sách) ⇒ commit + push nhánh TRƯỚC, rồi ghi `D:\Vietnam-Real-Estates\plans\reports\p3-lane-<core|bench|api|web|docs>.md`
(đã làm, lệnh verify + kết quả thật, chưa làm, cần lead quyết) và trả cùng nội dung trong tin nhắn cuối.
