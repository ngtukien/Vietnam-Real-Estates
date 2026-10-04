"""Nạp, làm sạch và chunk dữ liệu (Notebook 1). Người 2, 3, 4 dùng lại qua data/listings_clean.parquet."""

import math
import re
import unicodedata
from itertools import islice

import numpy as np
import pandas as pd

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, COL, DATASET, NUMERIC_COLS, TEXT_COLS

# ---------------------------------------------------------------- nạp


def load_raw(n_rows: int, dataset: str = DATASET) -> pd.DataFrame:
    """Đọc theo luồng n_rows dòng đầu (không tải cả ~3 GB). listing_id = số thứ tự dòng, từ 1."""
    from datasets import load_dataset

    stream = load_dataset(dataset, split="train", streaming=True)
    df = pd.DataFrame(list(islice(iter(stream), n_rows)))
    df.insert(0, "listing_id", np.arange(1, len(df) + 1))
    return df


def quality_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Kiểu dữ liệu, tỷ lệ thiếu và số giá trị khác nhau của từng cột."""
    blank = raw.isna() | raw.astype(str).isin(["", "None", "nan"])
    return pd.DataFrame({
        "kiểu": raw.dtypes.astype(str),
        "% thiếu": (blank.mean() * 100).round(1),
        "số giá trị khác nhau": raw.nunique(),
    })


def description_length_stats(raw: pd.DataFrame) -> pd.Series:
    words = raw[COL["description"]].fillna("").str.split().str.len()
    return words.describe(percentiles=[0.5, 0.9, 0.99]).round(0)


def duplicate_count(raw: pd.DataFrame) -> int:
    key = raw[COL["title"]].fillna("").str.lower() + "|" + raw[COL["description"]].fillna("").str.lower()
    return int(key.duplicated().sum())


# ---------------------------------------------------------------- làm sạch

PHONE = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PROVINCE_ALIASES = {"hcm": "Hồ Chí Minh", "tphcm": "Hồ Chí Minh", "tp hcm": "Hồ Chí Minh",
                     "sài gòn": "Hồ Chí Minh", "hn": "Hà Nội"}


def normalize_text(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = unicodedata.normalize("NFC", re.sub(r"\s+", " ", str(value))).strip()
    return None if text in {"", "None", "nan"} else text


def mask_pii(text: str | None) -> str | None:
    """Ẩn số điện thoại và email; dataset gốc đã thay một phần bằng [phone_number]."""
    if not isinstance(text, str):
        return text
    return EMAIL.sub("[email]", PHONE.sub("[phone_number]", text))


def normalize_place(value, kind: str) -> str | None:
    """Entity resolution cho địa danh: 'Q.7', 'q7', '7' -> 'Quận 7'; 'P.5' -> 'Phường 5';
    'TP. Hồ Chí Minh' -> 'Hồ Chí Minh'; 'Huyện Củ Chi' -> 'Củ Chi'."""
    text = normalize_text(value)
    if not text:
        return None
    if kind == "province":
        text = re.sub(r"^(tp\.?|thành phố|tỉnh)\s*", "", text, flags=re.I).strip()
        return _PROVINCE_ALIASES.get(text.lower(), text)
    if kind == "district":
        number = re.fullmatch(r"(?:q\.?|quận)?\s*0*(\d+)", text, flags=re.I)
        if number:
            return f"Quận {int(number.group(1))}"
        return re.sub(r"^(quận|huyện|thị xã|thành phố|q\.|h\.|tx\.?|tp\.?)\s*", "", text, flags=re.I).strip()
    if kind == "ward":
        number = re.fullmatch(r"(?:p\.?|phường)?\s*0*(\d+)", text, flags=re.I)
        if number:
            return f"Phường {int(number.group(1))}"
        return re.sub(r"^(phường|xã|thị trấn|p\.|x\.|tt\.?)\s*", "", text, flags=re.I).strip()
    return text


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Trả về (bảng sạch với tên cột nội bộ, số dòng sau từng bước)."""
    df = raw.rename(columns={v: k for k, v in COL.items()})[["listing_id", *COL]].copy()
    steps = {"ban đầu": len(df)}
    for col in TEXT_COLS:
        df[col] = df[col].map(normalize_text)
    df["title"] = df["title"].map(mask_pii)
    df["description"] = df["description"].map(mask_pii)
    df = df[df["title"].notna() | df["description"].notna()]
    steps["bỏ tin thiếu cả tiêu đề và mô tả"] = len(df)

    key = df["title"].fillna("").str.lower() + "|" + df["description"].fillna("").str.lower()
    df = df[~key.duplicated()]
    steps["bỏ tin trùng tiêu đề + mô tả"] = len(df)

    for col in NUMERIC_COLS:
        values = pd.to_numeric(df[col], errors="coerce")
        df[col] = values.where(np.isfinite(values) & (values > 0))
    for kind in ("province", "district", "ward"):
        df[kind] = df[kind].map(lambda v, k=kind: normalize_place(v, k))

    # Một số tin nhập giá theo nghìn đồng ("10 tỷ" lưu thành 10.000.000). Giá bán dưới 100 triệu:
    # nhân 1.000 nếu khi đó giá/m² hợp lý; tin cho thuê (giá theo tháng) hoặc không sửa được thì bỏ giá.
    low = df["price"] < 1e8
    rental = df["title"].fillna("").str.lower().str.startswith("cho thuê")
    fixable = low & ~rental & (df["price"] * 1000 / df["area"] / 1e6).between(5, 2000)
    df.loc[fixable, "price"] *= 1000
    df.loc[low & ~fixable, "price"] = np.nan
    steps[f"sửa giá nhập theo nghìn đồng ({int(fixable.sum())} tin, bỏ giá {int((low & ~fixable).sum())} tin)"] = len(df)

    # Chuẩn hoá đơn vị: giá gốc là VND; thêm tỷ và triệu/m² để đọc và so sánh.
    df["price_bn"] = df["price"] / 1e9
    df["price_m2_mil"] = df["price"] / df["area"] / 1e6
    # Giá/m² ngoài [0,05; 2000] triệu thường là nhập sai đơn vị hoặc giá thuê; không dùng để thống kê.
    df.loc[~df["price_m2_mil"].between(0.05, 2000), "price_m2_mil"] = np.nan
    steps["sau chuẩn hoá giá và địa danh"] = len(df)
    return df.reset_index(drop=True), steps


# ---------------------------------------------------------------- chunk


def chunk_starts(L: int, S: int, O: int) -> list[int]:
    """Vị trí bắt đầu của từng chunk khi cắt L token, cỡ S, chồng lấn O (Phụ lục A.1)."""
    starts, s = [0], 0
    while s + S < L:
        s += S - O
        starts.append(s)
    return starts


def n_chunks_formula(L: int, S: int, O: int) -> int:
    """n = ⌈(L − O) / (S − O)⌉ khi L > S, ngược lại 1."""
    return 1 if L <= S else math.ceil((L - O) / (S - O))


def text(value) -> str | None:
    """Giá trị chữ hoặc None: pandas đọc ô chữ bị thiếu thành NaN (float, và NaN là truthy)."""
    return value if isinstance(value, str) and value else None


def listing_header(row) -> str:
    place = ", ".join(p for p in (text(row.get(k)) for k in ("ward", "district", "province")) if p)
    parts = [text(row.get("property_type")), place]
    if pd.notna(row.get("price")):
        parts.append(f"giá {row['price'] / 1e9:.2f} tỷ")
    if pd.notna(row.get("area")):
        parts.append(f"{row['area']:g} m²")
    if pd.notna(row.get("bedrooms")):
        parts.append(f"{row['bedrooms']:g} phòng ngủ")
    return f"{text(row.get('title')) or ''}\n" + " · ".join(p for p in parts if p)


def body_budget(row, tokenizer, size: int = CHUNK_SIZE) -> int:
    """Số token còn cho phần mô tả sau khi trừ header lặp lại ở mỗi chunk."""
    return size - len(tokenizer.encode(listing_header(row), add_special_tokens=False)) - 1


def chunk_listing(row, tokenizer, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
                  strategy: str = "fixed") -> list[dict]:
    """strategy='fixed': cửa sổ token cỡ `size`, chồng lấn `overlap`, header lặp ở mọi chunk.
    strategy='whole': 1 tin = 1 chunk (mô hình embedding sẽ cắt phần vượt 512 token)."""
    header = listing_header(row)
    body = text(row.get("description")) or ""
    if strategy == "whole":
        texts = [header + "\n" + body]
    else:
        budget = body_budget(row, tokenizer, size)
        if budget <= overlap:
            raise ValueError("Header dài hơn ngân sách token của chunk")
        offsets = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        texts = []
        for start in chunk_starts(len(offsets), budget, overlap):
            window = offsets[start:start + budget]
            texts.append(header + ("\n" + body[window[0][0]:window[-1][1]] if window else ""))
    lid = int(row["listing_id"])
    return [dict(chunk_id=f"{lid}:{i}", listing_id=lid, chunk_index=i, text=text,
                 n_tokens=len(tokenizer.encode(text, add_special_tokens=False)))
            for i, text in enumerate(texts)]


META_COLS = ["province", "district", "ward", "property_type", "price", "area", "bedrooms", "price_m2_mil"]


def build_chunks(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """Chunk toàn bộ tin và gắn metadata để lọc (pre-filter) trong vector DB."""
    rows = []
    for row in df.to_dict("records"):
        meta = {col: row[col] for col in META_COLS}
        rows.extend({**chunk, **meta} for chunk in chunk_listing(row, tokenizer))
    return pd.DataFrame(rows)
