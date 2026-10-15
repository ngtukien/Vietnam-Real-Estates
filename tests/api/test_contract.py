"""OpenAPI sinh từ app ↔ bảng hợp đồng `api-contract.md` (enum §1, trường §2/§3, payload SSE §4, tên schema §8)."""

from __future__ import annotations

import re
from typing import Any

import pytest
from api.export_openapi import OUT, render
from api.schemas import EVENT_MODELS
from api_helpers import CONTRACT, make_app

TEXT = CONTRACT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def schemas() -> dict[str, Any]:
    out: dict[str, Any] = make_app().openapi()["components"]["schemas"]
    return out


def _section(title_prefix: str) -> str:
    start = TEXT.index(f"## {title_prefix}")
    nxt = TEXT.find("\n## ", start + 4)
    return TEXT[start: nxt if nxt != -1 else None]


def contract_enums() -> dict[str, list[str]]:
    out = {}
    for name, values in re.findall(r"^\| `(\w+)` \| (.+?) \|$", _section("1. Enums"), flags=re.M):
        out[name] = re.findall(r"`([^`]+)`", values.split(" (")[0])  # bỏ ghi chú trong ngoặc
    return out


def _strip(body: str) -> str:
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    while re.search(r"\{[^{}]*\}", body):
        body = re.sub(r"\{[^{}]*\}", "", body)
    return re.sub(r"Record<[^>]*>|Array<[^>]*>", "", body)


def contract_types() -> dict[str, set[str]]:
    """Kiểu TS trong các khối ```ts: `Name { a: ...; b?: ... }` (một hoặc nhiều dòng) và `Name = Base & { ... }`."""
    found: dict[str, set[str]] = {}
    for block in re.findall(r"```ts\n(.*?)```", TEXT, flags=re.S):
        for chunk in re.split(r"(?m)^(?=\w+ (?:= \w+ & )?\{)", block):
            head = re.match(r"(\w+) (?:= (\w+) & )?\{", chunk)
            if not head:
                continue
            name, base = head.groups()
            body = chunk[head.end(): chunk.rindex("}")]
            fields = set(re.findall(r"(\w+)\??\s*:", _strip(body)))
            found[name] = fields | (found.get(base, set()) if base else set())
    return found


def test_enums_match_contract(schemas: dict[str, Any]) -> None:
    enums = contract_enums()
    assert len(enums) == 9
    for name, values in enums.items():
        assert schemas[name]["enum"] == values, name


def test_field_names_match_contract(schemas: dict[str, Any]) -> None:
    types = contract_types()
    assert {"QueryFilters", "ListingCard", "ListingDetail", "CypherResult", "ChatRequest", "ValuationInput", "ValuationResult"} <= set(types)
    for name, fields in types.items():
        assert set(schemas[name]["properties"]) == fields, name


def test_nullable_response_fields_are_required(schemas: dict[str, Any]) -> None:
    """`T | null` trong response luôn có mặt ⇒ required trong OpenAPI."""
    for name in ("ListingCard", "ListingDetail", "ValuationResult", "CypherResult", "InfoResponse", "GeoItem"):
        assert set(schemas[name]["required"]) == set(schemas[name]["properties"]), name


def test_sse_payloads_match_contract(schemas: dict[str, Any]) -> None:
    rows = re.findall(r"^\| `(\w+)` \| (.+?) \| .+ \|$", _section("4. SSE"), flags=re.M)
    table = {name: payload for name, payload in rows}
    assert set(table) == {*EVENT_MODELS, "compare_done"}
    by_type = {"CypherResult": "CypherResult", "ValuationResult": "ValuationResult"}
    for name, payload in table.items():
        model = "EventCompareDone" if name == "compare_done" else EVENT_MODELS[name].__name__
        props = set(schemas[model]["properties"])
        ref = re.fullmatch(r"`(\w+)`", payload.strip())
        if ref:
            expected = set(schemas[by_type[ref.group(1)]]["properties"])
        else:
            inner = re.search(r"`\{(.*)\}`", payload).group(1)  # type: ignore[union-attr]
            expected = set(re.findall(r"(\w+)\s*:", _strip(inner)))
        extra = {"request_id"} if name == "compare_done" else {"request_id", "system"}
        assert props == expected | extra, name


def test_schema_names_section_8(schemas: dict[str, Any]) -> None:
    names = set()
    for row in re.findall(r"^\| [^|]+ \| (.+) \|$", _section("8. Tên schema"), flags=re.M):
        names |= set(re.findall(r"`(\w+)`", row))
    names = {n for n in names if n[0].isupper()}  # bỏ tên trường trong ghi chú (`system`, `request_id`)
    assert len(names) > 50
    assert names <= set(schemas), sorted(names - set(schemas))
    assert "HTTPValidationError" not in schemas  # 422 được map về 400


def test_committed_openapi_is_current() -> None:
    assert OUT.read_text(encoding="utf-8") == render(), "chạy: python -m api.export_openapi"
