"""Ghi `api/openapi.json` từ app (web sinh `web/src/api/types.ts` bằng openapi-typescript từ file này).

    & $VRE_PY -m api.export_openapi            # ghi file
    & $VRE_PY -m api.export_openapi --check    # exit 1 nếu file đã commit lệch với app
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

OUT = Path(__file__).with_name("openapi.json")


def render() -> str:
    os.environ.setdefault("VRE_FAKE_BACKENDS", "1")  # sinh schema không cần dịch vụ
    from api.main import create_app

    schema: dict[str, Any] = create_app().openapi()
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        if current != text:
            print(f"{OUT.name} lệch với app — chạy: python -m api.export_openapi", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"Đã ghi {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
