"""Vẽ 8 hình EDA (data-contract §7.2) bằng matplotlib Agg — palette slide, 1600×900, chữ ≥ 14pt."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAVY, TEAL, ORANGE = "#1F3A5F", "#2A9D8F", "#F4A261"
GRID = "#D9D9D9"
FIGSIZE, DPI = (16, 9), 100  # 1600×900 px
TITLE_PT, BASE_PT, CAPTION_PT = 22, 14, 15

FIGURES: dict[str, tuple[str, str]] = {
    "price_m2_by_type.png": ("Giá/m² theo loại bất động sản",
                             "Trung vị và khoảng tứ phân vị giá/m² (triệu đồng) của tin bán, đã loại outlier giá và diện tích."),
    "price_m2_by_region.png": ("Giá/m² theo tỉnh/thành",
                               "Trung vị giá/m² (triệu đồng) của tin bán ở các tỉnh/thành nhiều tin nhất."),
    "monthly_trend.png": ("Số tin và giá/m² theo tháng",
                          "Cột: số tin đăng mỗi tháng; đường: trung vị giá/m² (triệu đồng) của tin bán."),
    "missing_values.png": ("Tỷ lệ thiếu dữ liệu theo cột", "Phần trăm giá trị rỗng của từng cột trong bảng sạch."),
    "outliers.png": ("Tỷ lệ outlier theo loại bất động sản",
                     "Phần trăm tin bị gắn cờ outlier giá/m² và outlier diện tích theo từng loại."),
    "duplicates.png": ("Mức độ trùng lặp tin đăng",
                       "Số tin đại diện theo số dòng thô đã gộp vào (trùng chính xác và trùng tiền tố)."),
    "top_provinces.png": ("Tỉnh/thành nhiều tin nhất", "Số tin và tỷ trọng của các tỉnh/thành đứng đầu (địa giới cũ)."),
    "top_districts_hcm_hn.png": ("Quận/huyện nhiều tin nhất ở Hà Nội và TP. Hồ Chí Minh",
                                 "Mười quận/huyện có nhiều tin đăng nhất ở mỗi thành phố."),
}
FIGURE_FILES = tuple(FIGURES)

plt.rcParams.update({"font.size": BASE_PT, "axes.titlesize": BASE_PT + 2, "axes.labelsize": BASE_PT,
                     "xtick.labelsize": BASE_PT, "ytick.labelsize": BASE_PT, "legend.fontsize": BASE_PT,
                     "axes.edgecolor": NAVY, "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "axes.spines.top": False, "axes.spines.right": False})


def _num(v: Any) -> float:
    return float("nan") if v is None else float(v)


def _barh(ax: Any, labels: list[str], values: list[float], color: str, xlabel: str) -> None:
    ax.barh(labels[::-1], values[::-1], color=color)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)


def _empty(ax: Any) -> None:
    ax.text(0.5, 0.5, "Không có dữ liệu", ha="center", va="center", fontsize=BASE_PT + 4, color=NAVY, transform=ax.transAxes)


def plot_price_by_type(fig: Any, d: dict[str, Any]) -> None:
    ax = fig.add_subplot(111)
    rows = d["type_quartiles"]
    if not rows:
        return _empty(ax)
    labels = [r["property_type"] for r in rows]
    med = [_num(r["q2"]) for r in rows]
    err = [[m - _num(r["q1"]) for m, r in zip(med, rows, strict=True)], [_num(r["q3"]) - m for m, r in zip(med, rows, strict=True)]]
    ax.bar(labels, med, color=TEAL, yerr=err, ecolor=NAVY, capsize=8)
    ax.set_ylabel("Triệu đồng/m²")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)


def plot_price_by_region(fig: Any, d: dict[str, Any]) -> None:
    ax = fig.add_subplot(111)
    rows = d["province_price"]
    if not rows:
        return _empty(ax)
    _barh(ax, [r["province"] for r in rows], [_num(r["median_price_m2_mil"]) for r in rows], NAVY, "Trung vị triệu đồng/m²")


def plot_monthly(fig: Any, d: dict[str, Any]) -> None:
    ax = fig.add_subplot(111)
    rows = d["monthly"]
    if not rows:
        return _empty(ax)
    months = [r["month"] for r in rows]
    ax.bar(months, [r["n"] for r in rows], color=TEAL, label="Số tin")
    ax.set_ylabel("Số tin")
    ax.set_ylim(0, 1.25 * max(r["n"] for r in rows))  # chừa chỗ cho chú giải
    ax.tick_params(axis="x", rotation=45)
    ax2 = ax.twinx()
    ax2.plot(months, [_num(r["median_price_m2_mil"]) for r in rows], color=ORANGE, marker="o", linewidth=3, label="Giá/m²")
    ax2.set_ylabel("Trung vị triệu đồng/m²")
    ax2.spines["right"].set_visible(True)
    fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.86))


def plot_missing(fig: Any, d: dict[str, Any]) -> None:
    ax = fig.add_subplot(111)
    items = sorted(d["missing_pct"].items(), key=lambda kv: kv[1], reverse=True)
    _barh(ax, [k for k, _ in items], [v for _, v in items], ORANGE, "% giá trị rỗng")
    ax.set_xlim(0, 100)


def plot_outliers(fig: Any, d: dict[str, Any]) -> None:
    ax = fig.add_subplot(111)
    rows = d["outliers"]
    if not rows:
        return _empty(ax)
    xs = range(len(rows))
    width = 0.4
    ax.bar([x - width / 2 for x in xs], [_num(r["price_pct"]) for r in rows], width, color=NAVY, label="Outlier giá/m²")
    ax.bar([x + width / 2 for x in xs], [_num(r["area_pct"]) for r in rows], width, color=ORANGE, label="Outlier diện tích")
    ax.set_xticks(list(xs), [r["property_type"] for r in rows])
    ax.set_ylabel("% số tin")
    ax.legend()
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)


def plot_duplicates(fig: Any, d: dict[str, Any]) -> None:
    ax = fig.add_subplot(111)
    labels = [lab for lab, _ in d["dup_buckets"]]
    ax.bar(labels, [n for _, n in d["dup_buckets"]], color=TEAL)
    ax.set_yscale("symlog")
    ax.set_xlabel("Số dòng thô gộp vào 1 tin")
    ax.set_ylabel("Số tin (thang log)")
    ax.set_title(f"{d['n_listings']:,} tin sạch từ {d['raw_rows_total']:,} dòng thô".replace(",", "."), color=NAVY)


def plot_top_provinces(fig: Any, d: dict[str, Any]) -> None:
    ax = fig.add_subplot(111)
    rows = d["top_provinces"]
    if not rows:
        return _empty(ax)
    _barh(ax, [r["province"] for r in rows], [r["n"] for r in rows], NAVY, "Số tin")
    for i, r in enumerate(rows[::-1]):
        ax.text(r["n"], i, f" {100 * _num(r['share']):.1f}%", va="center", color=NAVY)
    ax.set_xlim(0, 1.12 * rows[0]["n"])


def plot_top_districts(fig: Any, d: dict[str, Any]) -> None:
    for i, (prov, color) in enumerate(zip(d["districts"], (NAVY, TEAL), strict=True)):
        ax = fig.add_subplot(1, 2, i + 1)
        rows = d["districts"][prov]
        ax.set_title(prov, color=NAVY)
        if not rows:
            _empty(ax)
            continue
        _barh(ax, [r["district"] for r in rows], [r["n"] for r in rows], color, "Số tin")


PLOTTERS: dict[str, Callable[[Any, dict[str, Any]], None]] = {
    "price_m2_by_type.png": plot_price_by_type, "price_m2_by_region.png": plot_price_by_region,
    "monthly_trend.png": plot_monthly, "missing_values.png": plot_missing, "outliers.png": plot_outliers,
    "duplicates.png": plot_duplicates, "top_provinces.png": plot_top_provinces, "top_districts_hcm_hn.png": plot_top_districts,
}


def render(name: str, data: dict[str, Any], out_dir: Path) -> dict[str, str]:
    """Vẽ một hình, ghi nguyên tử (`.tmp` rồi `os.replace`), trả `{file,title,caption}`."""
    title, caption = FIGURES[name]
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    try:
        PLOTTERS[name](fig, data)
        fig.suptitle(title, fontsize=TITLE_PT, color=NAVY, fontweight="bold")
        fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=CAPTION_PT, color=NAVY, wrap=True)
        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
        target = out_dir / name
        tmp = target.with_name(target.name + ".tmp")
        fig.savefig(tmp, format="png", dpi=DPI)
        os.replace(tmp, target)
    finally:
        plt.close(fig)
    return {"file": name, "title": title, "caption": caption}


def render_all(data: dict[str, Any], out_dir: Path) -> list[dict[str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [render(name, data, out_dir) for name in FIGURE_FILES]
