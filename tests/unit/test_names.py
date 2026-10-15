"""Phân giải tên dự án/đường (p3-filter-drop fix 1), nới bộ lọc theo bậc (fix 2), luật deal_type (fix 3)."""

import pandas as pd
import pytest

from vre.rag import filters as F
from vre.rag import names, retrieve


@pytest.mark.parametrize(("text", "expected"), [
    ("Khu đô thị Ecopark", "khu do thi ecopark"),
    ("Hà Nội Paragon (Paragon Tower)", "ha noi paragon"),
    ("  Biên  Hòa ", "bien hoa"),
    ("Đường Nguyễn-Văn-Linh", "duong nguyen van linh"),
])
def test_fold(text, expected):
    assert names.fold(text) == expected


@pytest.mark.parametrize(("value", "district", "expected_first"), [
    ("Sun Feliza Suites", "Cầu Giấy", "Sun Feliza Suites Cầu Giấy"),  # LK01: câu hỏi nằm trong tên thật
    ("sun feliza suites", None, "Sun Feliza Suites Cầu Giấy"),
    ("Thung Lũng Mùa Xuân Ecopark", None, "Khu đô thị Ecopark"),  # LK05: tên thật nằm trong câu hỏi
    ("Aqua City Biên Hòa", None, "Aqua City"),  # LK07: hậu tố tỉnh/quận bị bỏ
    ("Paragon Tower", None, None),  # bí danh trong ngoặc không dùng để khớp
    ("chung cư hà nội paragon", "Cầu Giấy", "Hà Nội Paragon (Paragon Tower)"),
    ("Vinhomes Grand Park", None, "Vinhomes Grand Park"),
])
def test_project_resolution(value, district, expected_first):
    found = names.resolve("project", value, district, None)
    if expected_first is None:
        assert found == ()
    else:
        assert found and found[0] == expected_first


def test_resolution_ranks_and_bounds():
    found = names.resolve("project", "Aqua City", None, None)
    assert found[:2] == ("Aqua City", "Aqua City Hạ Long")  # bằng nhau, nhiều tin trước
    assert "City" not in names.resolve("project", "Aqua City Biên Hòa", None, None)  # tên chỉ gồm từ chung chung
    assert "Sky Forest Residences Ecopark" in names.resolve("project", "Ecopark", None, None)
    assert names.resolve("project", "Ecopark", None, "Hưng Yên")[0] == "Khu đô thị Ecopark"
    big = names.NameIndex("project", pd.DataFrame(
        [(f"Ecopark Lô {i}", "Văn Giang", "Hưng Yên", i) for i in range(50)], columns=["name", "district", "province", "n"]))
    assert len(big.resolve("Ecopark")) == names.MAX_CANDIDATES


def test_street_resolution_is_stricter():
    assert names.resolve("street", "đường Nguyễn Văn Linh", None, None) == ("Nguyễn Văn Linh",)
    assert names.resolve("street", "Linh Đông Thủ Đức", None, None) == ("Linh Đông",)  # hậu tố quận bỏ được
    assert names.resolve("street", "Nguyễn Văn Linh Đông", None, None) == ()  # không khớp "nằm trong câu hỏi"


def test_conditions_use_resolved_names():
    f = F.QueryFilters(district="Cầu Giấy", project="Sun Feliza Suites", street="Không Có Đường Này")
    conds = F.conditions(f)
    assert F.Condition("project", "in", ["Sun Feliza Suites Cầu Giấy"]) in conds
    assert all(c.field != "street" for c in conds)
    assert F.unresolved(f) == ["street"]


def test_no_vocabulary_falls_back_to_exact_match():
    names.use_vocabulary("project", None)
    assert F.conditions(F.QueryFilters(project="X")) == [F.Condition("project", "eq", "X")]


def test_relaxation_order():
    f = F.QueryFilters(district="Cầu Giấy", property_type="Căn hộ chung cư", bedrooms=3, project="Sun Feliza Suites",
                       min_price_mil=8000, max_price_mil=8000, min_area_m2=70, max_area_m2=90)
    steps = F.relaxations(f)
    assert [labels for _, labels in steps] == [["project"], ["project", "bedrooms", "price"]]
    first, second = steps[0][0], steps[1][0]
    assert first.project is None and first.bedrooms == 3
    assert second.bedrooms is None and second.min_price_mil is None and second.max_price_mil is None
    assert (second.min_area_m2, second.max_area_m2, second.district) == (70, 90, "Cầu Giấy")  # khoảng thật giữ nguyên
    assert F.relaxations(F.QueryFilters(project="Sun Feliza Suites")) == []  # bậc duy nhất = bỏ hết
    assert F.relaxed_note(["project", "price"]) == "filter_relaxed:project,price"


def _fake_search(ok):
    calls = []

    def search(question, k, filters):
        calls.append(filters)
        hit = ok(filters)
        return pd.DataFrame({"listing_id": [1]}) if hit else pd.DataFrame({"listing_id": []})
    return search, calls


def test_search_relaxed_notes():
    f = F.QueryFilters(district="Cầu Giấy", bedrooms=3, project="Sun Feliza Suites")
    search, calls = _fake_search(lambda q: q is not None and q.project is None and q.bedrooms is not None)
    hits, note = retrieve.search_relaxed(search, "q", 5, f)
    assert not hits.empty and note == "filter_relaxed:project" and len(calls) == 2
    search, calls = _fake_search(lambda q: q is None)
    hits, note = retrieve.search_relaxed(search, "q", 5, f)
    assert note == F.FILTER_DROPPED and calls[-1] is None
    search, _ = _fake_search(lambda q: True)
    assert retrieve.search_relaxed(search, "q", 5, f)[1] is None
    unknown = F.QueryFilters(district="Cầu Giấy", project="Không Có Dự Án Này")
    assert retrieve.search_relaxed(search, "q", 5, unknown)[1] == "filter_relaxed:project"
    assert retrieve.search_relaxed(search, "q", 5, None)[1] is None


def test_relaxed_note_reaches_answer_prompt():
    from vre.rag import generate

    prompt = generate.build_prompt("q", ["[Tin#1] x"], "filter_relaxed:project,price")
    assert prompt.startswith("LƯU Ý") and "project,price" in prompt


def test_deal_type_prompt_rule():
    prompt = F.FILTER_PROMPT
    assert "deal_type là loại TIN cần tìm" in prompt and "'sale'" in prompt and "triệu/tháng" in prompt
    assert "tỷ" in prompt.split("deal_type là loại TIN cần tìm", 1)[1]


def test_places_resolve_by_exact_fold():
    assert names.resolve("district", "Thu Duc") == ("Thủ Đức",)  # CS08: LLM bỏ dấu
    assert names.resolve("district", "tang duc") == ()  # LK10: LLM bịa tên quận ⇒ bỏ điều kiện
    f = F.QueryFilters(district="Thu Duc", province="Hồ Chí Minh")
    assert F.conditions(f) == [F.Condition("province", "eq", "Hồ Chí Minh"), F.Condition("district", "in", ["Thủ Đức"])]
    assert F.unresolved(F.QueryFilters(district="tang duc")) == ["district"]
    assert names.resolve("district", "Thủ") == ()  # không khớp kiểu chứa


def test_total_price_in_ty_means_sale():
    from unittest import mock

    from vre.llm.client import LLMResult

    raw = {"district": "Quận 1", "deal_type": "rent"}
    with mock.patch.object(F, "ask_llm", return_value=LLMResult("", raw, 1, 1, True)):
        assert F.parse_filters("Nhà mặt tiền q1 giá 55 tỷ đang cho thuê bao nhiêu một tháng?")[0].deal_type == "sale"
        assert F.parse_filters("Cho thuê căn hộ 15 triệu/tháng ở q1")[0].deal_type == "rent"
