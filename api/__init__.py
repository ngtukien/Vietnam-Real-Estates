"""FastAPI + SSE backend của Vietnam-Real-Estates (Phase 3, api-contract.md là SSOT).

Package `vre` nằm ở `<repo>/src` và không được cài vào venv (mỗi worktree một bản code) ⇒ thêm `src/` của CHÍNH repo này
vào `sys.path` để `python -m api.export_openapi` và `uvicorn api.main:app` chạy từ gốc repo mà không cần PYTHONPATH.
"""

import sys
from pathlib import Path

API_VERSION = "0.3.0"

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
