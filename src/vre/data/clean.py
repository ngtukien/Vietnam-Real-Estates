"""Làm sạch một shard thô (data-contract §2–§3): định danh, mask PII, giá, loại giao dịch, số đo.

Mọi hàm vector hoá trên pandas Series; `clean_frame` trả bảng trung gian của một shard (giữ cả dòng bị loại,
có `drop_reason`) để tầng dedup toàn cục ghi `dup_map.parquet` đủ 3.500.744 dòng.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from vre.data.amenities import amenity_lists
from vre.data.places import normalize_place, normalize_text, province_2025

PROPERTY_TYPES = ("Nhà", "Đất", "Căn hộ chung cư", "Biệt thự/Nhà liền kề", "Shophouse")
ID_MASK = (1 << 53) - 1
PREFIX_CHARS = 300
RENT_TITLE = re.compile(r"^\s*(?:cần\s+)?cho\s+thuê")
RENT_DESC_CHARS = 200
RENT_PRICE_MAX = 1e8  # giá < 100 triệu + "cho thuê" ở đầu mô tả ⇒ tin thuê
THOUSAND_PRICE_MAX = 1e8  # giá bán < 100 triệu: nghi nhập theo nghìn đồng
THOUSAND_M2_RANGE = (5.0, 2000.0)  # tr/m² hợp lý sau khi nhân 1000
SMALLINT_MAX = 32767
PHONE = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

# cột thô → cột sạch
NUMERIC = {"area": "area_m2", "floor_count": "floors", "frontage_width": "frontage_m", "house_depth": "depth_m",
           "road_width": "road_width_m", "bedroom_count": "bedrooms", "bathroom_count": "bathrooms"}
INT_COLS = ("floors", "bedrooms", "bathrooms")
NONNULL_COLS = ("title", "description", "province", "district", "ward", "street", "project", "price_vnd", "area_m2",
                "floors", "frontage_m", "depth_m", "road_width_m", "bedrooms", "bathrooms", "house_direction",
                "balcony_direction", "published_at")


def norm(values: pd.Series) -> pd.Series:
    """norm(s) = lower(NFC(collapse_whitespace(s or ""))).strip() — data-contract §2.1."""
    out = values.fillna("").astype(str).str.replace(r"\s+", " ", regex=True)
    return out.str.normalize("NFC").str.lower().str.strip()


def _hash64(text: str, person: bytes = b"") -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8, person=person).digest(), "big")


def listing_ids(name_norm: pd.Series, desc_norm: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(listing_id 53-bit, hash phụ 63-bit để phát hiện va chạm) từ key_exact = norm(name) + "\\x1f" + norm(desc)."""
    keys = (name_norm + "\x1f" + desc_norm).tolist()
    ids = np.fromiter((_hash64(k) & ID_MASK for k in keys), dtype=np.int64, count=len(keys))
    check = np.fromiter((_hash64(k, b"vre-check") >> 1 for k in keys), dtype=np.int64, count=len(keys))
    return ids, check


def prefix_keys(name: pd.Series, description: pd.Series) -> np.ndarray:
    """blake2b(norm(name + "|" + description)[:300]) — khoá trùng gần tầng 2 (giống `.cache/scratch/profile.py`)."""
    joined = norm(name.fillna("") + "|" + description.fillna("")).str.slice(0, PREFIX_CHARS)
    return np.fromiter((_hash64(k) >> 1 for k in joined.tolist()), dtype=np.int64, count=len(joined))


def mask_pii(values: pd.Series) -> pd.Series:
    """Ẩn số điện thoại và email; dataset gốc đã thay một phần bằng [phone_number]."""
    return values.str.replace(PHONE, "[phone_number]", regex=True).str.replace(EMAIL, "[email]", regex=True)


def parse_price(raw: pd.Series) -> pd.Series:
    """TRY_CAST(price AS DOUBLE), chỉ giữ giá > 0 hữu hạn."""
    values = pd.to_numeric(raw, errors="coerce").astype("float64")
    return values.where(np.isfinite(values) & (values > 0))


def positive(raw: pd.Series) -> pd.Series:
    values = pd.to_numeric(raw, errors="coerce").astype("float64")
    return values.where(np.isfinite(values) & (values > 0))


def deal_types(title: pd.Series, description: pd.Series, price: pd.Series) -> pd.Series:
    title_rent = title.fillna("").str.lower().str.contains(RENT_TITLE, regex=True)
    head = description.fillna("").str.slice(0, RENT_DESC_CHARS).str.lower()
    desc_rent = head.str.contains("cho thuê", regex=False) & (price < RENT_PRICE_MAX)
    return pd.Series(np.where(title_rent | desc_rent, "rent", "sale"), index=title.index)


def fix_thousand(price: pd.Series, area: pd.Series, deal: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Giá bán < 1e8: ×1000 nếu giá/m² sau sửa ∈ [5, 2000] tr; không sửa được ⇒ null. Giá thuê giữ nguyên."""
    low = (deal == "sale") & (price < THOUSAND_PRICE_MAX)
    fixable = low & (price * 1000 / area / 1e6).between(*THOUSAND_M2_RANGE)
    fixed = price.where(~low, price * 1000).where(~low | fixable)
    return fixed, fixable.fillna(False).astype(bool)


def _small_int(values: pd.Series) -> pd.Series:
    return values.where(values <= SMALLINT_MAX).round().astype("Int16")


def clean_frame(raw: pd.DataFrame, shard: int) -> pd.DataFrame:
    """Bảng trung gian của một shard: định danh + cột sạch + `drop_reason` (null nếu giữ lại)."""
    out = pd.DataFrame({"source_shard": np.int16(shard), "source_row": np.arange(len(raw), dtype=np.int32)})
    name_norm, desc_norm = norm(raw["name"]), norm(raw["description"])
    out["listing_id"], out["key_hash2"] = listing_ids(name_norm, desc_norm)
    out["prefix_key"] = prefix_keys(raw["name"], raw["description"])
    title = raw["name"].map(normalize_text).astype("string")
    description = raw["description"].map(normalize_text).astype("string")
    out["title"], out["description"] = mask_pii(title), mask_pii(description)
    ptype = raw["property_type_name"].map(normalize_text)
    no_text = title.isna() & description.isna()
    out["drop_reason"] = np.where(no_text, "dropped_no_text", np.where(~ptype.isin(PROPERTY_TYPES), "dropped_bad_type", None))
    out["property_type"] = ptype
    price = parse_price(raw["price"])
    for col, new in NUMERIC.items():
        out[new] = positive(raw[col])
    out["deal_type"] = deal_types(out["title"], out["description"], price)
    out["price_vnd"], out["price_fixed_thousand"] = fix_thousand(price, out["area_m2"], out["deal_type"])
    for col in INT_COLS:
        out[col] = _small_int(out[col])
    for kind, col in (("province", "province_name"), ("district", "district_name"), ("ward", "ward_name")):
        uniques = raw[col].dropna().unique()
        out[kind] = raw[col].map({u: normalize_place(u, kind) for u in uniques})
    out["province_2025"] = out["province"].map(lambda p: province_2025(p) if isinstance(p, str) else None)
    for col, new in (("street_name", "street"), ("project_name", "project"),
                     ("house_direction", "house_direction"), ("balcony_direction", "balcony_direction")):
        uniques = raw[col].dropna().unique()
        out[new] = raw[col].map({u: normalize_text(u) for u in uniques})
    out["published_at"] = pd.to_datetime(raw["published_at"], errors="coerce", format="ISO8601")
    out["amenities"] = amenity_lists(out["title"].fillna("") + "\n" + out["description"].fillna(""))
    out["n_nonnull"] = out[list(NONNULL_COLS)].notna().sum(axis=1).astype(np.int16)
    return out
