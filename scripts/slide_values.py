"""Giải khoá `file.json:path` trong results/ và định dạng số kiểu Việt Nam cho slide.

Cú pháp khoá: `eval_summary.json:by_category.lookup.hybrid.accuracy`, chỉ số `[0]`, chọn phần tử
`[id=LK01,system=hybrid]`, trải mảng `[*]` (trả list). Biểu thức số bắt đầu bằng `=`, tham chiếu
khoá trong ngoặc nhọn: `=({ingest_report.json:raw_rows}-{ingest_report.json:dedup.kept})/{ingest_report.json:raw_rows}`.
"""
from __future__ import annotations

import ast
import json
import operator
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

_TOKEN = re.compile(r"\.?([^.\[\]]+)|\[([^\]]*)\]")
_REF = re.compile(r"\{([^{}]+)\}")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_SCALES = {"pct": 100, "mil": 1e-6, "bn": 1e-9, "k": 1e-3, "dec": 1, "h": 1 / 3600, "min": 1 / 60}
MINUS = "−"


class KeyBroken(KeyError):
    """Khoá trỏ tới file/trường không tồn tại."""


class Resolver:
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self._cache: dict[str, object] = {}

    def _load(self, name: str):
        if name not in self._cache:
            path = self.results_dir / name
            if not path.is_file():
                raise KeyBroken(f"không có file {path}")
            self._cache[name] = json.loads(path.read_text(encoding="utf-8"))
        return self._cache[name]

    def get(self, key: str):
        if key.startswith("="):
            return self._expr(key[1:])
        name, _, path = key.partition(":")
        node = self._load(name.strip())
        return _walk(node, _TOKEN.findall(path.strip()), key) if path.strip() else node

    def _expr(self, text: str) -> float:
        refs: dict[str, float] = {}

        def sub(m: re.Match) -> str:
            name = f"_v{len(refs)}"
            refs[name] = float(self.get(m.group(1)))
            return name

        return _eval(ast.parse(_REF.sub(sub, text), mode="eval").body, refs)


def _walk(node, tokens: list[tuple[str, str]], key: str):
    for i, (field, bracket) in enumerate(tokens):
        if field:
            if not isinstance(node, dict) or field not in node:
                raise KeyBroken(f"{key}: thiếu trường '{field}'")
            node = node[field]
        elif bracket == "*":
            if not isinstance(node, list):
                raise KeyBroken(f"{key}: [*] trên giá trị không phải list")
            return [_walk(x, tokens[i + 1:], key) for x in node]
        elif bracket.lstrip("-").isdigit():
            idx = int(bracket)
            if not isinstance(node, list) or not -len(node) <= idx < len(node):
                raise KeyBroken(f"{key}: chỉ số [{idx}] ngoài phạm vi")
            node = node[idx]
        else:
            conds = [c.split("=", 1) for c in bracket.split(",")]
            hits = [x for x in (node if isinstance(node, list) else [])
                    if isinstance(x, dict) and all(str(x.get(k.strip())) == v.strip() for k, v in conds)]
            if len(hits) != 1:
                raise KeyBroken(f"{key}: bộ chọn [{bracket}] khớp {len(hits)} phần tử (cần đúng 1)")
            node = hits[0]
    return node


def _eval(node, names: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left, names), _eval(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand, names))
    raise ValueError(f"biểu thức không hỗ trợ: {ast.dump(node)}")


def _group(int_part: str) -> str:
    return re.sub(r"(?<=\d)(?=(\d{3})+$)", ".", int_part)


def fmt(value, spec: str) -> str:
    """int | dec2 | pct1 | mil1 | bn2 | k0 | h1 | min0 | str | last6; tiền tố `s` thêm dấu (+/−)."""
    if spec == "str":
        return str(value)
    if spec == "last6":
        return str(int(value))[-6:]
    if value is None:
        raise ValueError("giá trị null không định dạng được thành số")
    signed = spec.startswith("s")
    spec = spec[1:] if signed else spec
    m = re.fullmatch(r"(int|[a-z]+)(\d*)", spec)
    if not m or (m.group(1) != "int" and m.group(1) not in _SCALES):
        raise ValueError(f"format không hỗ trợ: {spec}")
    kind, digits = m.group(1), int(m.group(2) or 0)
    scaled = Decimal(repr(float(value))) * Decimal(repr(_SCALES.get(kind, 1)))
    q = scaled.quantize(Decimal(1) if kind == "int" else Decimal(10) ** -digits, rounding=ROUND_HALF_UP)
    sign = MINUS if q < 0 else ("+" if signed and q > 0 else "")
    int_part, _, frac = f"{abs(q):f}".partition(".")
    return sign + _group(int_part) + ("," + frac if frac else "")
