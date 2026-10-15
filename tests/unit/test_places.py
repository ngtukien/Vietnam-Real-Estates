"""vre.data.places: chuẩn hoá chữ + địa danh (giữ quy tắc cũ) và map tỉnh 2025."""

import math

import pytest

from vre.data.places import normalize_place, normalize_text, province_2025, text


@pytest.mark.parametrize(("raw", "kind", "expected"), [
    ("Q.7", "district", "Quận 7"), ("q7", "district", "Quận 7"), ("07", "district", "Quận 7"),
    ("Huyện Củ Chi", "district", "Củ Chi"), ("Thành phố Thủ Đức", "district", "Thủ Đức"),
    ("P.5", "ward", "Phường 5"), ("phường 05", "ward", "Phường 5"), ("Xã Tân Phú", "ward", "Tân Phú"),
    ("TP. Hồ Chí Minh", "province", "Hồ Chí Minh"), ("tphcm", "province", "Hồ Chí Minh"), ("Tỉnh Long An", "province", "Long An"),
    ("  Phố   Huế ", "street", "Phố Huế"),
])
def test_normalize_place(raw, kind, expected):
    assert normalize_place(raw, kind) == expected


@pytest.mark.parametrize("blank", [None, "", "  ", "None", "nan", float("nan")])
def test_blank_values_become_none(blank):
    assert normalize_text(blank) is None
    assert normalize_place(blank, "ward") is None


def test_normalize_text_nfc_and_whitespace():
    decomposed = "Hòa  Bình"  # dấu tổ hợp
    assert normalize_text(decomposed) == "Hòa Bình"


def test_text_helper():
    assert text("a") == "a"
    assert text("") is None
    assert text(math.nan) is None


def test_province_2025_merges():
    assert province_2025("Bình Dương") == "Hồ Chí Minh"
    assert province_2025("Hà Nội") == "Hà Nội"
    assert province_2025(None) is None
