"""Từ điển amenity (data-contract §3.2): mỗi khoá khớp ≥ 2 câu dương và không khớp câu âm."""

import pandas as pd
import pytest

from vre.data.amenities import amenity_lists, amenity_table, match_amenities

SAMPLES = {
    "thang_may": (["Nhà 6 tầng có thang máy", "nhà có TM riêng"], "tầng trệt thoáng mát"),
    "ho_boi": (["Có hồ bơi tràn", "bể bơi bốn mùa"], "gần hồ Tây"),
    "hem_xe_hoi": (["hẻm xe hơi 6m", "HXH thông"], "hẻm ba gác"),
    "so_hong": (["sổ hồng riêng", "đã có sổ đỏ", "SHR"], "sổ tay"),
    "mat_tien": (["nhà mặt tiền đường", "MT kinh doanh"], "tiền điện rẻ"),
    "gan_truong": (["gần trường học", "Gần  trường quốc tế"], "trường thọ"),
    "view_song": (["view sông Sài Gòn", "view hồ đẹp"], "xem sông"),
    "gara": (["có gara ô tô", "ô tô vào nhà"], "ô tô đỗ cửa"),
    "noi_that": (["full nội thất", "đầy đủ nội thất cao cấp", "full NT"], "nội thất cơ bản"),
    "an_ninh": (["bảo vệ 24/7", "an ninh 24h"], "an ninh tốt"),
}


def test_dictionary_size_and_keys_covered():
    table = amenity_table()
    assert 10 <= len(table) <= 30
    assert set(SAMPLES) <= set(table)


@pytest.mark.parametrize("key", sorted(SAMPLES))
def test_positive_and_negative(key):
    positives, negative = SAMPLES[key]
    for sentence in positives:
        assert key in match_amenities(sentence), sentence
    assert key not in match_amenities(negative)


def test_vectorized_matches_scalar():
    texts = pd.Series(["Nhà có thang máy, sổ hồng", None, "đất trống"])
    assert amenity_lists(texts) == [match_amenities(t) for t in ["Nhà có thang máy, sổ hồng", "", "đất trống"]]
