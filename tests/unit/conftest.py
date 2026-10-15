"""Từ vựng tên (dự án/đường) cố định cho unit test: không đọc bảng sạch thật (`vre.rag.names`)."""

import pandas as pd
import pytest

from vre.rag import names

PROJECTS = [
    ("Vinhomes Grand Park", "Thủ Đức", "Hồ Chí Minh", 5000),
    ("Sun Feliza Suites Cầu Giấy", "Cầu Giấy", "Hà Nội", 2277),
    ("Hà Nội Paragon (Paragon Tower)", "Cầu Giấy", "Hà Nội", 300),
    ("Khu đô thị Ecopark", "Văn Giang", "Hưng Yên", 2036),
    ("Sky Forest Residences Ecopark", "Văn Giang", "Hưng Yên", 1319),
    ("Aqua City", "Biên Hòa", "Đồng Nai", 1833),
    ("Aqua City Hạ Long", "Hạ Long", "Quảng Ninh", 132),
    ("Khu dân cư Kara Garden Biên Hòa", "Biên Hòa", "Đồng Nai", 40),
    ("City", "Quận 7", "Hồ Chí Minh", 5),
]
STREETS = [
    ("Nguyễn Văn Linh", "Quận 7", "Hồ Chí Minh", 10570),
    ("Cầu Giấy", "Cầu Giấy", "Hà Nội", 900),
    ("Linh Đông", "Thủ Đức", "Hồ Chí Minh", 50),
]

PLACES = {
    "province": ["Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Hưng Yên", "Đồng Nai", "Quảng Ninh", "Bình Dương"],
    "district": ["Quận 1", "Quận 7", "Quận 9", "Cầu Giấy", "Thủ Đức", "Gò Vấp", "Tây Hồ", "Hải Châu", "Dĩ An",
                 "Văn Giang", "Biên Hòa", "Hạ Long", "Bình Tân", "Củ Chi"],
    "ward": ["Tân Phong", "Dịch Vọng", "Phường 5", "Quảng An", "Thạch Thang"],
}


def vocab(rows):
    return pd.DataFrame(rows, columns=["name", "district", "province", "n"])


@pytest.fixture(autouse=True)
def name_vocabulary():
    names.use_vocabulary("project", vocab(PROJECTS))
    names.use_vocabulary("street", vocab(STREETS))
    for field, values in PLACES.items():
        names.use_vocabulary(field, vocab([(v, None, None, 1) for v in values]))
    yield
    names.reset()
