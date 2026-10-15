"""Trích dẫn [Tin#ID] (data-contract §2.1: `listing_id` tối đa 16 chữ số): chuẩn hoá và kiểm tra nguồn."""

from __future__ import annotations

import re
from collections.abc import Iterable

# Mô hình đôi khi viết 【Tin#ID】, ［Tin # ID］ hay [Tin # ID]: đưa về dạng chuẩn để giao diện tạo được liên kết.
_LOOSE = re.compile(r"[\[【［]\s*Tin\s*#\s*(\d{1,16})\s*[\]】］]")
CITATION = re.compile(r"\[Tin#(\d{1,16})\]")


def normalize_citations(answer: str) -> str:
    return _LOOSE.sub(r"[Tin#\1]", answer)


def cited_ids(answer: str) -> list[int]:
    """Id được trích theo thứ tự xuất hiện, không trùng (chỉ dạng chuẩn `[Tin#<1..16 chữ số>]`)."""
    return [int(x) for x in dict.fromkeys(CITATION.findall(answer))]


def check_citations(answer: str, sources: Iterable) -> tuple[list[int], list[int]]:
    """(trích dẫn có trong nguồn đã đưa vào ngữ cảnh, trích dẫn không có: dấu hiệu LLM bịa nguồn).
    Nguồn hợp lệ so theo `listing_id`."""
    allowed = {int(s) for s in sources}
    cited = cited_ids(answer)
    return [i for i in cited if i in allowed], [i for i in cited if i not in allowed]
