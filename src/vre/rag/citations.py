"""Trích dẫn [Tin#ID]: chuẩn hoá và kiểm tra nguồn (chuyển nguyên từ `src/rag.py`)."""

from __future__ import annotations

import re

_CITATION = re.compile(r"[\[【［]\s*Tin\s*#\s*(\d+)\s*[\]】］]")


def normalize_citations(answer: str) -> str:
    """Đưa trích dẫn về đúng dạng [Tin#ID]: mô hình đôi khi viết 【Tin#ID】 hoặc [Tin # ID], khi đó
    giao diện không tạo được liên kết tới thẻ tin nguồn."""
    return _CITATION.sub(r"[Tin#\1]", answer)


def cited_ids(answer: str) -> list[int]:
    return [int(x) for x in dict.fromkeys(re.findall(r"Tin#(\d+)", answer))]


def check_citations(answer: str, sources) -> tuple[list[int], list[int]]:
    """(trích dẫn có trong nguồn đã đưa vào ngữ cảnh, trích dẫn không có: dấu hiệu LLM bịa nguồn)."""
    allowed = {int(s) for s in sources}
    cited = cited_ids(answer)
    return [i for i in cited if i in allowed], [i for i in cited if i not in allowed]
