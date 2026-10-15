"""Dựng deck từ các module `slide/build/parts/*.py` (mỗi module có `build(d: Deck)`), theo thứ tự PARTS.

  python slide/build/build_deck.py [--draft] [--only p1_rag_data,p3_graph] [--out X.pptx]

Ghi deck + `slide/numbers_map.d/generated.yaml` (map số sinh tự động), rồi chạy `slide_numbers.py check`.
`--draft`: khoá results chưa có thì in ⟨?⟩ thay vì dừng (để dựng bố cục trước khi có kết quả eval).
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from kit import Deck  # noqa: E402

PARTS = ["p0_intro", "p1_rag_data", "p2_retrieval", "p3_graph", "p4_graphrag_eval", "p5_ext"]
DECK = ROOT / "slide" / "Từ-RAG-đến-GraphRAG-Trợ-lý-hỏi-đáp-thị-trường-bất-động-sản-Việt-Nam-có-lý-thuyết.pptx"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    results = Path(os.environ.get("VRE_DATA_ROOT", "D:/Vietnam-Real-Estates")) / "results"
    d = Deck(results, draft=a.draft)
    only = a.only.split(",") if a.only else PARTS
    for name in PARTS:
        if name in only and (HERE / "parts" / f"{name}.py").is_file():
            importlib.import_module(f"parts.{name}").build(d)
    out = a.out or DECK
    gen = (out.parent / (out.stem + ".map.yaml")) if a.out else ROOT / "slide" / "numbers_map.d" / "generated.yaml"
    gen.parent.mkdir(parents=True, exist_ok=True)
    d.save(out, gen)
    print(f"đã ghi {out} ({len(d.prs.slides)} slide), map {gen}")
    for m in d.missing:
        print(f"DRAFT thiếu khoá: {m}")
    base = (ROOT / "slide" / "numbers_map.yaml").read_text(encoding="utf-8")
    combined = gen.parent / (gen.stem + ".check.yaml")
    keep = "".join(ln for ln in base.splitlines(keepends=True) if not ln.startswith("include:"))
    combined.write_text(keep + f"include: [{gen.name}]\n", encoding="utf-8")
    rc = subprocess.call([sys.executable, str(ROOT / "scripts" / "slide_numbers.py"), "check", str(out),
                          "--map", str(combined), "--results", str(results)])
    combined.unlink()
    return rc


if __name__ == "__main__":
    sys.exit(main())
