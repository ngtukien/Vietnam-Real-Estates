"""Map tỉnh 63 → 34 (data-contract §3.4)."""

from vre.data.places import admin_2025

UNCHANGED = {"Hà Nội", "Huế", "Lai Châu", "Điện Biên", "Sơn La", "Lạng Sơn", "Quảng Ninh", "Thanh Hóa", "Nghệ An",
             "Hà Tĩnh", "Cao Bằng"}


def test_63_rows_34_new_provinces():
    mapping = admin_2025()
    assert len(mapping) == 63
    assert len(set(mapping.values())) == 34


def test_unchanged_provinces_map_to_themselves():
    mapping = admin_2025()
    assert all(mapping[p] == p for p in UNCHANGED)


def test_every_new_province_is_also_an_old_name():
    mapping = admin_2025()
    assert set(mapping.values()) <= set(mapping)
