"""vre.data.clean: giá chuỗi, sửa nghìn đồng, deal_type, mask PII, clean_frame trên DataFrame mẫu."""

import numpy as np
import pandas as pd

from vre.data.clean import clean_frame, deal_types, fix_thousand, mask_pii, parse_price, positive


def test_parse_price_string_digits():
    out = parse_price(pd.Series(["7450000000", "abc", None, "0", "-5"]))
    assert out.iloc[0] == 7.45e9
    assert out.iloc[1:].isna().all()


def test_positive_drops_nonpositive_and_inf():
    out = positive(pd.Series([80.0, 0.0, -1.0, np.inf, None]))
    assert out.iloc[0] == 80.0 and out.iloc[1:].isna().all()


def test_fix_thousand_sale_fixable_and_not():
    price = pd.Series([5_000_000.0, 5_000.0, 15_000_000.0, 3e9])
    area = pd.Series([100.0, 100.0, 70.0, 50.0])
    deal = pd.Series(["sale", "sale", "rent", "sale"])
    fixed, flag = fix_thousand(price, area, deal)
    assert fixed.iloc[0] == 5e9 and flag.iloc[0]  # 50 tr/m² sau sửa ⇒ sửa
    assert pd.isna(fixed.iloc[1]) and not flag.iloc[1]  # 0,05 tr/m² ⇒ không sửa được ⇒ null
    assert fixed.iloc[2] == 15e6 and not flag.iloc[2]  # giá thuê giữ nguyên
    assert fixed.iloc[3] == 3e9 and not flag.iloc[3]


def test_deal_type_rules():
    title = pd.Series(["Cho thuê căn hộ", "  cần cho thuê nhà", "Bán nhà", "Bán nhà"])
    desc = pd.Series(["", "", "cho thuê lại mặt bằng", "cho thuê lại mặt bằng"])
    price = pd.Series([1e7, 1e7, 5e7, 5e9])
    assert deal_types(title, desc, price).tolist() == ["rent", "rent", "rent", "sale"]


def test_mask_pii():
    out = mask_pii(pd.Series(["gọi 0901 234 567 hoặc a.b@x.vn", None], dtype="string"))
    assert out.iloc[0] == "gọi [phone_number] hoặc [email]"
    assert pd.isna(out.iloc[1])


def test_clean_frame_drops_and_types(raw_frame):
    out = clean_frame(raw_frame, shard=3, row_offset=100)
    assert out["source_row"].tolist() == list(range(100, 110))
    assert (out["source_shard"] == 3).all()
    reasons = out["drop_reason"].tolist()
    assert reasons[7] == "dropped_no_text" and reasons[8] == "dropped_bad_type"
    assert out.loc[0, "listing_id"] == out.loc[1, "listing_id"]  # trùng chính xác
    assert out.loc[2, "price_vnd"] == 5e9 and bool(out.loc[2, "price_fixed_thousand"])
    assert out.loc[3, "deal_type"] == "rent" and out.loc[3, "price_vnd"] == 15e6
    assert pd.isna(out.loc[6, "price_vnd"])  # "abc"
    assert "[phone_number]" in out.loc[2, "description"] and "[email]" in out.loc[9, "description"]
    assert out.loc[0, "province_2025"] == "Hồ Chí Minh"
    assert "hem_xe_hoi" in out.loc[0, "amenities"] and "so_hong" in out.loc[0, "amenities"]


def test_mask_pii_keeps_prices_and_masks_real_phones():
    """Đo trên 350k tin thô (2026-10-15): mọi khớp '84…' không có '+' đều là số điện thoại thật."""
    texts = pd.Series(["giá 8,4 tỷ", "giá 84 tỷ/căn", "LH 84977917692 xem nhà", "gọi 087887676",
                       "giá 1.250.000.000 đồng"], dtype="string")
    out = mask_pii(texts).tolist()
    assert out[0] == "giá 8,4 tỷ" and out[1] == "giá 84 tỷ/căn" and out[4] == "giá 1.250.000.000 đồng"
    assert out[2] == "LH [phone_number] xem nhà" and out[3] == "gọi [phone_number]"
