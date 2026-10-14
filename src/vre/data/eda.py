"""EDA trên bảng sạch (data-contract §7.2): truy vấn DuckDB trên `v_listings` → dict + 8 hình trong `results/eda/`."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vre.common import result_header, scalar, write_json_atomic
from vre.config import EMBED_MODEL_SLUG, EMBED_MODELS, Settings
from vre.data import duck

SUMMARY_NAME = "eda_summary.json"
FIG_DIR_NAME = "eda"
TOKEN_SAMPLE = 20_000
SAMPLE_SEED = 42
TOP_PROVINCES = 15
TOP_DISTRICTS = 10
HN_HCM = ("Hà Nội", "Hồ Chí Minh")
PCTS = (0.5, 0.9, 0.99)
MISSING_COLS = ("title", "description", "province", "province_2025", "district", "ward", "street", "project",
                "price_vnd", "area_m2", "floors", "frontage_m", "depth_m", "road_width_m", "bedrooms", "bathrooms",
                "house_direction", "balcony_direction", "published_at")
DUP_BUCKETS = (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4–5", 4, 5), ("6–10", 6, 10), (">10", 11, None))
SALE = "deal_type = 'sale'"

Con = Any  # duckdb.DuckDBPyConnection


def _rows(con: Con, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    cur = con.execute(sql, list(params))
    names = [d[0] for d in cur.description]
    return [{k: _clean(v) for k, v in zip(names, row, strict=True)} for row in cur.fetchall()]


def _clean(value: Any) -> Any:
    value = scalar(value)
    return round(value, 4) if isinstance(value, float) else value


def _pcts(values: Sequence[float] | np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"p50": None, "p90": None, "p99": None}
    return {f"p{round(q * 100)}": round(float(np.quantile(arr, q)), 1) for q in PCTS}


def count_listings(con: Con) -> int:
    return int(con.execute("SELECT count(*) FROM v_listings").fetchone()[0])


def summary_queries(con: Con, n: int) -> dict[str, Any]:
    """Các trường số liệu của eda_summary.json (trừ tokens_per_listing và figures)."""
    period = _rows(con, "SELECT min(month) AS min_month, max(month) AS max_month FROM v_listings")[0]
    by_type = _rows(con, f"""SELECT property_type, count(*) AS n,
        median(price_m2_mil) FILTER (WHERE {SALE}) AS median_price_m2_mil
        FROM v_listings GROUP BY 1 ORDER BY n DESC, 1""")
    top_prov = _rows(con, """SELECT province, count(*) AS n, count(*) / ?::DOUBLE AS share
        FROM v_listings WHERE province IS NOT NULL GROUP BY 1 ORDER BY n DESC, 1 LIMIT ?""", [max(n, 1), TOP_PROVINCES])
    hn_hcm = con.execute("SELECT count(*) FILTER (WHERE province IN (?, ?)) / greatest(count(*), 1)::DOUBLE FROM v_listings",
                         list(HN_HCM)).fetchone()[0]
    monthly = _rows(con, f"""SELECT month, count(*) AS n,
        median(price_m2_mil) FILTER (WHERE {SALE}) AS median_price_m2_mil
        FROM v_listings WHERE month IS NOT NULL GROUP BY 1 ORDER BY 1""")
    miss_sql = ", ".join(f"100 * avg(({c} IS NULL)::INT) AS {c}" for c in MISSING_COLS)
    missing = _rows(con, f"SELECT {miss_sql} FROM v_listings")[0]
    q = ", ".join(f"quantile_cont(length(description), {p})" for p in PCTS)
    desc = con.execute(f"SELECT {q} FROM v_listings WHERE description IS NOT NULL").fetchone()
    return {
        "n_listings": n,
        "period": period,
        "by_property_type": by_type,
        "top_provinces": top_prov,
        "hn_hcm_share": round(float(hn_hcm or 0.0), 4),
        "monthly": monthly,
        "missing_pct": {k: round(float(v or 0.0), 2) for k, v in missing.items()},
        "description_chars": {f"p{round(p * 100)}": (None if v is None else round(float(v), 1)) for p, v in zip(PCTS, desc, strict=True)},
    }


def plot_queries(con: Con) -> dict[str, Any]:
    """Dữ liệu phụ chỉ dùng để vẽ (không ghi vào JSON)."""
    type_q = _rows(con, f"""SELECT property_type, quantile_cont(price_m2_mil, 0.25) AS q1, median(price_m2_mil) AS q2,
        quantile_cont(price_m2_mil, 0.75) AS q3 FROM v_listings WHERE {SALE} AND price_m2_mil IS NOT NULL
        GROUP BY 1 ORDER BY q2 DESC""")
    region = _rows(con, f"""WITH top AS (SELECT province FROM v_listings WHERE province IS NOT NULL
        GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT {TOP_PROVINCES})
        SELECT province, median(price_m2_mil) AS median_price_m2_mil FROM v_listings
        WHERE province IN (SELECT province FROM top) AND {SALE} AND price_m2_mil IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC""")
    outliers = _rows(con, """SELECT property_type, 100 * avg(is_price_outlier::INT) AS price_pct,
        100 * avg(is_area_outlier::INT) AS area_pct FROM v_listings GROUP BY 1 ORDER BY 1""")
    cases = " ".join(f"WHEN n_raw_rows BETWEEN {lo} AND {hi if hi else 2**31 - 1} THEN '{lab}'" for lab, lo, hi in DUP_BUCKETS)
    dups = {r["bucket"]: r["n"] for r in _rows(con, f"SELECT CASE {cases} END AS bucket, count(*) AS n FROM v_listings GROUP BY 1")}
    raw_total = con.execute("SELECT coalesce(sum(n_raw_rows), 0) FROM v_listings").fetchone()[0]
    districts = {p: _rows(con, """SELECT district, count(*) AS n FROM v_listings WHERE province = ? AND district IS NOT NULL
        GROUP BY 1 ORDER BY n DESC, 1 LIMIT ?""", [p, TOP_DISTRICTS]) for p in HN_HCM}
    return {"type_quartiles": type_q, "province_price": region, "outliers": outliers,
            "dup_buckets": [(lab, int(dups.get(lab, 0))) for lab, _, _ in DUP_BUCKETS],
            "raw_rows_total": int(raw_total), "districts": districts}


def load_tokenizer(settings: Settings) -> Callable[[list[str]], list[int]]:
    """Chỉ tokenizer của e5-large (không nạp model), đọc từ cache HF trên ổ D, không tải mạng."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(EMBED_MODELS[EMBED_MODEL_SLUG][0], local_files_only=True,
                                        cache_dir=str(settings.cache_dir / "huggingface" / "hub"))

    def count(texts: list[str]) -> list[int]:
        return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]

    return count


def token_stats(con: Con, settings: Settings, batch: int = 1000) -> dict[str, float | None]:
    rows = con.execute(f"""SELECT coalesce(title, '') || chr(10) || coalesce(description, '') FROM v_listings
        USING SAMPLE reservoir({TOKEN_SAMPLE} ROWS) REPEATABLE ({SAMPLE_SEED})""").fetchall()
    texts = [r[0] for r in rows]
    if not texts:
        return _pcts([])
    count = load_tokenizer(settings)
    lengths: list[int] = []
    for i in range(0, len(texts), batch):
        lengths.extend(count(texts[i:i + batch]))
    return _pcts(lengths)


def _paths(settings: Settings) -> tuple[Path, Path]:
    return settings.results_dir / SUMMARY_NAME, settings.results_dir / FIG_DIR_NAME


def _existing(summary_path: Path, fig_dir: Path, n: int) -> dict[str, Any] | None:
    from vre.data.eda_plots import FIGURE_FILES

    if not summary_path.exists() or not all((fig_dir / f).exists() for f in FIGURE_FILES):
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return data if data.get("n_listings") == n else None


def run(settings: Settings) -> dict[str, Any]:
    """Ghi `results/eda_summary.json` + 8 PNG; đã có và khớp số dòng bảng sạch ⇒ in "đã xong" và trả dict cũ."""
    from vre.data.eda_plots import render_all

    summary_path, fig_dir = _paths(settings)
    con = duck.connect(settings)
    try:
        n = count_listings(con)
        done = _existing(summary_path, fig_dir, n)
        if done is not None:
            print("đã xong")
            return done
        stats = summary_queries(con, n)
        stats["tokens_per_listing"] = token_stats(con, settings)
        extra = plot_queries(con)
    finally:
        con.close()
    figures = render_all({**stats, **extra}, fig_dir)
    summary = {**result_header(), **stats, "figures": figures}
    write_json_atomic(summary_path, summary)
    return summary
