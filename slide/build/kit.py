"""Bộ dựng slide dùng chung: theme, thành phần, và sổ ghi nguồn gốc mọi con số.

Quy tắc: mọi con số dữ liệu chèn bằng `d.N("file.json:path", "fmt")`; số không phải dữ liệu (đề bài, ví dụ
công thức) bọc bằng `d.A("5 tỷ", "lý do")`. Số gõ thẳng vào chuỗi sẽ bị `slide_numbers.py check` báo unmapped.
Toạ độ tính bằng inch trên khung 13,333 × 7,5.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from slide_values import KeyBroken, Resolver, fmt  # noqa: E402

FONT, FONT_HEAD, FONT_MONO = "Calibri", "Calibri", "Consolas"
C = {  # theme — đổi ở đây là đổi cả deck
    "ink": "0F172A", "text": "334155", "muted": "64748B", "line": "CBD5E1", "soft": "F1F5F9", "white": "FFFFFF",
    "navy": "0B2545", "blue": "2563EB", "teal": "0D9488", "amber": "D97706", "coral": "E11D48", "violet": "7C3AED",
    "blue_bg": "DBEAFE", "teal_bg": "CCFBF1", "amber_bg": "FEF3C7", "coral_bg": "FFE4E6", "violet_bg": "EDE9FE",
}
SYSTEM = {"llm_only": ("LLM thuần", "muted"), "basic_rag": ("Basic RAG", "blue"),
          "graph": ("Graph", "amber"), "hybrid": ("Hybrid", "teal")}
SIZE = {"title": 30, "body": 20, "small": 16, "caption": 14, "footer": 12, "stat": 40}
W, H, MX = 13.333, 7.5, 0.6
BODY_TOP, BODY_BOTTOM = 1.75, 6.85
_SENT = re.compile("(\\d+)")


def rgb(name: str) -> RGBColor:
    return RGBColor.from_string(C.get(name, name))


class Deck:
    def __init__(self, results_dir: Path, draft: bool = False):
        self.res, self.draft = Resolver(results_dir), draft
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self._blank = self.prs.slide_layouts[6]
        self.marks: list[dict] = []
        self.entries: list[dict] = []
        self.fonts_allow: list[dict] = []
        self.missing: list[str] = []
        self.names: set[str] = set()

    # ---- nguồn gốc con số ----
    def N(self, key: str, spec: str = "int") -> str:
        """Số lấy từ results; trả chuỗi có dấu mốc, được thay bằng giá trị khi đặt vào shape."""
        try:
            value = fmt(self.res.get(key), spec)
        except (KeyBroken, ValueError) as exc:
            if not self.draft:
                raise
            self.missing.append(f"{key}: {exc}")
            value = "⟨?⟩"
        self.marks.append({"kind": "num", "key": key, "fmt": spec, "value": value})
        return f"{len(self.marks) - 1}"

    def V(self, key: str):
        """Giá trị thô (để rẽ nhánh/sắp xếp trong code dựng slide, KHÔNG để in ra)."""
        try:
            return self.res.get(key)
        except KeyBroken:
            if not self.draft:
                raise
            self.missing.append(key)
            return None

    def Q(self, qid: str) -> dict:
        """Một câu trong benchmark/questions.jsonl (SSOT câu hỏi). In đề bài ra slide bằng d.A(q["question"], …)."""
        import json
        for line in (ROOT / "benchmark" / "questions.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line)["id"] == qid:
                return json.loads(line)
        raise KeyBroken(f"không có câu {qid} trong benchmark/questions.jsonl")

    def T(self, key: str, reason: str = "") -> str:
        """Chuỗi nguyên văn từ results (câu trả lời, Cypher, tên model): chèn như d.A, lý do ghi khoá nguồn."""
        value = self.V(key)
        return self.A("⟨?⟩" if value is None else str(value), reason or f"nguyên văn {key}")

    def A(self, text: str, reason: str) -> str:
        """Chuỗi chứa số không phải dữ liệu (đề bài, ví dụ minh hoạ, tên phiên bản); bắt buộc ghi lý do."""
        self.marks.append({"kind": "allow", "value": text, "reason": reason})
        return f"{len(self.marks) - 1}"

    def resolve(self, s: str, slide: "Slide", where: str, shape_id: int) -> str:
        """Thay dấu mốc, ghi mục map cho shape; trả chuỗi cuối cùng."""
        ids = [int(i) for i in _SENT.findall(s)]
        if not ids:
            return s
        final = _SENT.sub(lambda m: self.marks[int(m.group(1))]["value"], s)
        nums = [self.marks[i] for i in ids if self.marks[i]["kind"] == "num"]
        if nums:
            tpl = _SENT.sub(lambda m: "{}" if self.marks[int(m.group(1))]["kind"] == "num"
                            else self.marks[int(m.group(1))]["value"], s)
            self.entries.append({"slide": slide.name, "in": where, "shape": shape_id, "text": tpl,
                                 "keys": [n["key"] for n in nums], "fmt": [n["fmt"] for n in nums]})
        for i in ids:
            mk = self.marks[i]
            if mk["kind"] == "allow" and any(ch.isdigit() for ch in mk["value"]):
                self.entries.append({"slide": slide.name, "in": where, "shape": shape_id,
                                     "text": mk["value"],
                                     "allow": mk["reason"]})
        return final

    def slide(self, name: str, section: str, title: str, notes: str | None = None, dark: bool = False) -> "Slide":
        if name in self.names:
            raise ValueError(f"trùng tên slide {name}")
        self.names.add(name)
        s = Slide(self, self.prs.slides.add_slide(self._blank), name, dark)
        s.header(section, title)
        if notes:
            s.notes(notes)
        return s

    def save(self, deck_path: Path, map_path: Path) -> None:
        import yaml
        for i, s in enumerate(self.prs.slides, 1):
            Slide.footer_number(self, s, i)
        self.prs.save(deck_path)
        data = {"entries": self.entries, "fonts_allow": self.fonts_allow}
        map_path.write_text("# SINH TỰ ĐỘNG bởi slide/build/build_deck.py — không sửa tay\n"
                            + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")


class Slide:
    def __init__(self, deck: Deck, slide, name: str, dark: bool):
        self.d, self.s, self.name, self.dark = deck, slide, name, dark
        slide._element.cSld.set("name", name)
        if dark:
            self.rect(0, 0, W, H, fill="navy", line=None)

    # ---- khung ----
    def header(self, section: str, title: str) -> None:
        if section:
            self.rect(MX, 0.42, 0.08, 0.32, fill="teal", line=None)
            self.text(MX + 0.18, 0.36, 8, 0.42, section.upper(), size=SIZE["caption"], bold=True,
                      color="teal", name="section")
        if title:
            self.text(MX, 0.78, W - 2 * MX, 0.95, title, size=SIZE["title"], bold=True,
                      color="white" if self.dark else "ink", font=FONT_HEAD, anchor="top", name="title")

    @staticmethod
    def footer_number(deck: Deck, slide, idx: int) -> None:
        sl = Slide.__new__(Slide)
        sl.d, sl.s, sl.dark = deck, slide, False
        sl.name = slide._element.cSld.get("name")
        box = sl.text(W - MX - 1.0, H - 0.48, 1.0, 0.3, deck.A(str(idx), "số trang"),
                      size=SIZE["footer"], color="muted", align="right", name="footer-no")
        foot = sl.text(MX, H - 0.48, 7, 0.3, deck.A("Nhóm 7 · Từ RAG đến GraphRAG", "tên nhóm"), size=SIZE["footer"],
                       color="muted", name="footer")
        for b in (box, foot):
            deck.fonts_allow.append({"slide": sl.name, "shape": b.shape_id, "reason": "footer/số trang"})

    def notes(self, text: str) -> None:
        tf = self.s.notes_slide.notes_text_frame
        tf.text = self.d.resolve(text, self, "notes", 0)

    # ---- thành phần cơ bản ----
    def rect(self, x, y, w, h, fill="soft", line="line", radius: float | None = None, shape=None):
        kind = shape or (MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE)
        sh = self.s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
        if radius and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
            sh.adjustments[0] = radius
        if fill:
            sh.fill.solid()
            sh.fill.fore_color.rgb = rgb(fill)
        else:
            sh.fill.background()
        if line:
            sh.line.color.rgb = rgb(line)
            sh.line.width = Pt(1)
        else:
            sh.line.fill.background()
        sh.shadow.inherit = False
        sh.text_frame.text = ""
        return sh

    def text(self, x, y, w, h, content, size=SIZE["body"], bold=False, color="text", align="left",
             anchor="top", font=FONT, bullets=False, name=None, shape=None, spacing=1.1):
        """content: str | list[đoạn]; đoạn: str | list[(chữ, {bold, color, size, font, italic})]."""
        sh = shape or self.s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        if name:
            sh.name = name
        tf = sh.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[anchor]
        for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
            setattr(tf, side, Inches(0.04))
        paras = [[(p, {})] if isinstance(p, str) else p for p in (content if isinstance(content, list) else [content])]
        if bullets:
            paras = [[("•  ", {"color": "teal", "bold": True})] + p for p in paras]
        flat = "\n".join("".join(t for t, _ in p) for p in paras)
        if size < SIZE["caption"] and name not in ("footer", "footer-no"):
            raise ValueError(f"{self.name}: cỡ chữ {size} < {SIZE['caption']}pt")
        self.d.resolve(flat, self, "body", sh.shape_id)  # ghi map một lần cho cả shape
        for i, p in enumerate(paras):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
            para.line_spacing = spacing
            if bullets:
                para.space_after = Pt(6)
            for t, st in p:
                r = para.add_run()
                r.text = _SENT.sub(lambda m: self.d.marks[int(m.group(1))]["value"], t)
                f = r.font
                f.name = st.get("font", font)
                f.size = Pt(st.get("size", size))
                f.bold = st.get("bold", bold)
                f.italic = st.get("italic", False)
                f.color.rgb = rgb(st.get("color", color))
        return sh

    # ---- thành phần ghép ----
    def card(self, x, y, w, h, title, body=None, accent="teal", fill="soft", size=SIZE["small"], title_size=20):
        self.rect(x, y, w, h, fill=fill, line=None, radius=0.06)
        self.rect(x, y, 0.08, h, fill=accent, line=None)
        self.text(x + 0.25, y + 0.15, w - 0.4, 0.5, title, size=title_size, bold=True, color="ink")
        if body is not None:
            self.text(x + 0.25, y + 0.15 + title_size / 40, w - 0.4, h - 0.3 - title_size / 40, body, size=size)

    def stat(self, x, y, w, value, label, accent="teal", h=1.5):
        self.rect(x, y, w, h, fill="soft", line=None, radius=0.06)
        self.text(x + 0.2, y + 0.12, w - 0.4, 0.75, value, size=SIZE["stat"], bold=True, color=accent)
        self.text(x + 0.2, y + 0.85, w - 0.4, h - 0.9, label, size=SIZE["caption"], color="text")

    def caption(self, text, y=6.55):
        return self.text(MX, y, W - 2 * MX, 0.4, text, size=SIZE["caption"], color="muted")

    def table(self, x, y, w, rows, col_w=None, size=SIZE["small"], row_h=0.45, header_fill="navy", highlight_row=None):
        nr, nc = len(rows), len(rows[0])
        gt = self.s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
        tbl = gt.table
        if col_w:
            for i, cw in enumerate(col_w):
                tbl.columns[i].width = Inches(cw)
        for r, row in enumerate(rows):
            tbl.rows[r].height = Inches(row_h)
            for c, val in enumerate(row):
                cell = tbl.cell(r, c)
                cell.fill.solid()
                is_head = r == 0
                cell.fill.fore_color.rgb = rgb(header_fill if is_head else ("teal_bg" if r == highlight_row else
                                                                            ("white" if r % 2 else "soft")))
                cell.margin_left = cell.margin_right = Inches(0.08)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                tf = cell.text_frame
                self.d.resolve(val, self, "body", gt.shape_id)
                tf.paragraphs[0].text = ""
                run = tf.paragraphs[0].add_run()
                run.text = _SENT.sub(lambda m: self.d.marks[int(m.group(1))]["value"], val)
                run.font.size, run.font.name = Pt(size), FONT
                run.font.bold = is_head
                run.font.color.rgb = rgb("white" if is_head else "ink")
                tf.paragraphs[0].alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        return gt

    def code(self, x, y, w, h, code: str, highlight=(), size=SIZE["caption"]):
        """Khối code nền tối; chuỗi trong `highlight` tô màu amber."""
        self.rect(x, y, w, h, fill="navy", line=None, radius=0.03)
        marks = [self.d.marks[int(i)] for i in _SENT.findall(code)]
        if any(mk["kind"] == "num" for mk in marks):
            raise ValueError(f"{self.name}: code() không nhận d.N — dùng d.T/d.A cho chuỗi nguyên văn")
        code = _SENT.sub(lambda m: self.d.marks[int(m.group(1))]["value"], code)
        paras = []
        for line in code.split("\n"):
            parts, rest = [], line
            pat = "|".join(re.escape(hh) for hh in highlight) if highlight else None
            for seg in (re.split(f"({pat})", rest) if pat else [rest]):
                if seg:
                    hot = bool(pat) and re.fullmatch(pat, seg) is not None
                    parts.append((seg, {"color": "FBBF24" if hot else "E2E8F0", "bold": hot, "font": FONT_MONO}))
            paras.append(parts or [(" ", {"font": FONT_MONO})])
        sh = self.text(x + 0.2, y + 0.15, w - 0.4, h - 0.3, paras, size=size, spacing=1.0)
        for mk in marks:
            if any(ch.isdigit() for ch in mk["value"]):
                self.d.entries.append({"slide": self.name, "in": "body", "shape": sh.shape_id,
                                       "text": mk["value"], "allow": mk["reason"]})
        return sh

    def image(self, path, x, y, w=None, h=None):
        return self.s.shapes.add_picture(str(path), Inches(x), Inches(y),
                                         Inches(w) if w else None, Inches(h) if h else None)

    def chart(self, x, y, w, h, kind, categories, series, number_format="0", legend=True, labels=True,
              font_size=SIZE["caption"], max_value=None):
        """categories: khoá results (list) hoặc list nhãn cố định; series: [(tên, khoá, màu, scale=1)]."""
        cats = (categories if isinstance(categories, list) else self.d.V(categories)) or ["⟨?⟩"]
        data = CategoryChartData()
        data.categories = [str(c) for c in cats]
        rec = []
        for item in series:
            sname, key, _color, *rest = item
            scale = rest[0] if rest else 1
            vals = ([self.d.V(k) for k in key] if isinstance(key, list) else self.d.V(key)) or [0] * len(cats)
            data.add_series(sname, [None if v is None else float(v) * scale for v in vals])
            rec.append({"name": sname, "key": key, **({"scale": scale} if scale != 1 else {})})
        ct = {"bar": XL_CHART_TYPE.BAR_CLUSTERED, "column": XL_CHART_TYPE.COLUMN_CLUSTERED,
              "line": XL_CHART_TYPE.LINE_MARKERS}[kind]
        gf = self.s.shapes.add_chart(ct, Inches(x), Inches(y), Inches(w), Inches(h), data)
        ch = gf.chart
        ch.font.size, ch.font.name = Pt(font_size), FONT
        ch.has_legend = legend and len(series) > 1
        if ch.has_legend:
            ch.legend.position, ch.legend.include_in_layout = XL_LEGEND_POSITION.TOP, False
        for item, s in zip(series, ch.plots[0].series):
            col = rgb(item[2])
            if kind == "line":
                s.format.line.color.rgb = col
                s.marker.format.fill.solid()
                s.marker.format.fill.fore_color.rgb = col
            else:
                s.format.fill.solid()
                s.format.fill.fore_color.rgb = col
        plot = ch.plots[0]
        if kind != "line":
            plot.gap_width = 60
        if labels:
            plot.has_data_labels = True
            plot.data_labels.number_format, plot.data_labels.number_format_is_linked = number_format, False
            plot.data_labels.font.size = Pt(font_size)
        va = ch.value_axis
        va.has_major_gridlines, va.visible = False, not labels
        va.tick_labels.number_format, va.tick_labels.number_format_is_linked = number_format, False
        if max_value is not None:
            va.maximum_scale = max_value
        ch.category_axis.tick_labels.font.size = Pt(font_size)
        ch.category_axis.format.line.color.rgb = rgb("line")
        entry = {"slide": self.name, "chart": gf.shape_id, "series": rec,
                 "categories": categories if isinstance(categories, list) else categories}
        if isinstance(categories, list):
            for c in categories:
                if any(ch_.isdigit() for ch_ in str(c)):
                    raise ValueError(f"{self.name}: nhãn chart cố định chứa số ({c}) — dùng khoá results")
        self.d.entries.append(entry)
        return gf

    def flow(self, steps, y, h=1.2, x0=MX, x1=W - MX, fills=None, size=SIZE["small"]):
        """Chuỗi hộp nối mũi tên; steps: [(tiêu đề, mô tả)]."""
        n, gap = len(steps), 0.35
        bw = (x1 - x0 - gap * (n - 1)) / n
        for i, (t, desc) in enumerate(steps):
            x = x0 + i * (bw + gap)
            fill = (fills or ["blue_bg"] * n)[i]
            self.rect(x, y, bw, h, fill=fill, line=None, radius=0.08)
            self.text(x + 0.1, y + 0.1, bw - 0.2, 0.45, t, size=size + 2, bold=True, color="ink", align="center")
            if desc:
                self.text(x + 0.1, y + 0.55, bw - 0.2, h - 0.6, desc, size=size, align="center")
            if i < n - 1:
                self.rect(x + bw + 0.05, y + h / 2 - 0.12, gap - 0.1, 0.24, fill="muted", line=None,
                          shape=MSO_SHAPE.RIGHT_ARROW)
