"""Trích dẫn [Tin#id] với listing_id 16 chữ số (data-contract §2.1)."""

from vre.rag import citations


def test_sixteen_digit_ids_are_kept_whole():
    lid = 8353835631106358
    text = citations.normalize_citations(f"Giá 1,2 tỷ【Tin#{lid}】 và ［Tin # 7］")
    assert text == f"Giá 1,2 tỷ[Tin#{lid}] và [Tin#7]"
    assert citations.cited_ids(text) == [lid, 7]
    assert citations.check_citations(text, [lid]) == ([lid], [7])


def test_seventeen_digits_is_not_a_citation():
    assert citations.cited_ids("[Tin#12345678901234567]") == []
