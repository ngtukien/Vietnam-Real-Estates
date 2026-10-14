"""Cấu hình pytest dùng chung: không đọc `.env` local, thư mục tạm trên ổ D, DataFrame mẫu nhỏ."""

import os
from pathlib import Path

import pytest

os.environ["VRE_LOAD_DOTENV"] = "0"  # test không phụ thuộc `.env` của máy

REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path(os.environ.get("VRE_DATA_ROOT", str(REPO_ROOT))) / ".cache" / "tmp" / "pytest"


@pytest.fixture
def tmp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("vre")


def pytest_configure(config: pytest.Config) -> None:
    if os.name == "nt" and not config.option.basetemp:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(TMP_ROOT / f"run-{os.getpid()}")


@pytest.fixture
def raw_frame():
    """10 dòng thô theo schema data-contract §1 (đủ ca: trùng chính xác, giá nghìn đồng, thuê, thiếu ward)."""
    import pandas as pd

    rows = [
        ("Bán nhà Quận 7", "Nhà đẹp hẻm xe hơi, sổ hồng", "Nhà", "Hồ Chí Minh", "Quận 7", "Phường Tân Phong", "7450000000", 80.0),
        ("Bán nhà Quận 7", "Nhà đẹp hẻm xe hơi, sổ hồng", "Nhà", "Hồ Chí Minh", "Quận 7", "Phường Tân Phong", "7450000000", 80.0),
        ("Bán đất Thủ Đức", "Đất nền 100m2 liên hệ 0901234567", "Đất", "Hồ Chí Minh", "Thủ Đức", None, "5000000", 100.0),
        ("Cho thuê căn hộ", "Căn hộ full nội thất, hồ bơi", "Căn hộ chung cư", "Hà Nội", "Cầu Giấy", "Dịch Vọng", "15000000", 70.0),
        ("Bán biệt thự", "Biệt thự view sông, thang máy", "Biệt thự/Nhà liền kề", "Hà Nội", "Tây Hồ", "Quảng An", "45000000000", 300.0),
        ("Bán shophouse", "Shophouse mặt tiền", "Shophouse", "Đà Nẵng", "Hải Châu", "Thạch Thang", None, 120.0),
        ("Bán nhà Gò Vấp", "Nhà 2 tầng gần trường", "Nhà", "Hồ Chí Minh", "Gò Vấp", "Phường 5", "abc", 60.0),
        (None, None, "Nhà", "Hồ Chí Minh", "Quận 1", None, "1000000000", 50.0),
        ("Bán kho", "Kho xưởng", "Kho xưởng", "Bình Dương", "Dĩ An", None, "9000000000", 500.0),
        ("Bán căn hộ", "Căn hộ 2PN, email a@b.com", "Căn hộ chung cư", "Hà Nội", "Cầu Giấy", "Dịch Vọng", "3200000000", 65.0),
    ]
    cols = ["name", "description", "property_type_name", "province_name", "district_name", "ward_name", "price", "area"]
    df = pd.DataFrame(rows, columns=cols)
    for col in ("street_name", "project_name", "house_direction", "balcony_direction"):
        df[col] = None
    for col in ("floor_count", "frontage_width", "house_depth", "road_width", "bedroom_count", "bathroom_count"):
        df[col] = float("nan")
    df.loc[4, "bedroom_count"] = 5.0
    df["published_at"] = "2025-06-01T05:12:56.941000"
    return df
